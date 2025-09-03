"""
Some notes about the sensing process:

Agents independently sample s_i entries (without replacement) from the global known set G.
Mu is the mean per-agent endowment. Overlap is purely random and increases as mu increases.

Rho controls mu via mu(rho) = (1-rho) G/N + rho G (where N is the number of agents).
When rho = 0, mu = G/N (each agent on average receives the equal share). 
When rho = 1, mu = G (each agent on average requests the whole global set). 
By controlling mu, rho implicitly controls the degree of overlap between agents. The formula 
interpolates between an equal partition with no overlap, and full sharing of known entries 
by all agents.

Gamma controls the inequality of agent endowments around the mean mu. Gamma is mapped to a Dirichlet 
concentration alpha (gamma_curve controls the nonlinearity of the mapping; <1 increases sensitivity 
at low gamma).
1. Derive alpha.
2. Draw a Dirichlet random proportion vector p = (p_1...p_N) with concentration parameter alpha.
3. Convert to integer sample sizes: s_i = round( p_i * (N * mu) ) then clip s_i to [0, G].
When gamma ≈ 0  => alpha large => p almost uniform         => s_i ≈ μ for all i 
When gamma >> 0 => alpha small => p is sparse/heavy-tailed => some s_i much larger than others

Interaction of rho and gamma:
* When gamma is near zero:
	- rho simply shifts every agent's expected size from G/N toward G.
	- As rho increases, all agents increase their s_i in lock-step. 
    - Overlap increases smoothly and uniformly across all agent pairs.

* When gamma is near 0.5 (some inequality):
	- Agents differ in endowments. Increasing rho raises the average size, but because some agents 
      are already larger, overlap becomes heterogeneous.
    - The mean pairwise overlap AND the variance of the pairwise overlap increase with rho.
	- Pairs that include a large agent quickly show high overlap (large agent covers a big fraction 
      of G), while small-small pairs may still have low overlap.

When gamma is near one (very unequal/heavy-tailed):
	- A few agents get a very large share of G.
    - Even for small rho, overlap is already be significant among large agents and negligible among 
      small ones.
    - As rho increases, small agents' s_i increases toward mu, while the largest agents quickly 
      saturate at G. Once large agents saturate, pairwise overlap among large agents approaches 1; 
      while the overlap with small agents increases but remains asymmetric.
	- Clusters of very high overlap between large agents may appear, along with many near-zero 
      overlaps. The overlap distribution becomes multi-modal.
	- Many s_i will be clipped to G if mu is close to G, causing repeated identical masks and large 
      jumps in overlap.

Stats track the mean pairwise overlap, the variance of pairwise overlap, and the fraction of agents 
clipped at G.

Note that if G is set too low, there may not be meaningful variation when changing rho and gamma.
Note that agents with zero counts are possible, especially when gamma is high.
Note that the 100 and 0.01 alpha endpoints can be tweaked to compress/expand the inequality range.
"""

import numpy as np
import torch


class SensingMasks(object):
    """
    Class to generate boolean sensing masks for all agents, and a boolean global mask.

    Hyperparameters:
        rho (float): ρ in [0,1]. Interpolates the mean per-agent endowment between G/N (ρ=0)
                     and G (ρ=1). Higher ρ -> higher expected overlap because each agent
                     requests more entries on average.
        gamma (float): γ in [0,1]. Controls inequality of agent endowments. γ=0 -> near-equal,
                       γ=1 -> highly unequal (heavy-tailed) via a Dirichlet with small concentration.

    Args:
        TemporalData (torch.utils.data.Dataset): Dataset of ground truth matrices (must have .t and .m).
        rank (int): Rank of the matrices.
        num_agents (int): Number of agents.
        density (float): Target density of global known entries.
        future_only (bool): hide the last row from the global_known set (same semantics as before).
        rho (float): overlap-control hyperparameter (default 0.0).
        gamma (float): inequality-control hyperparameter (default 0.0).
        gamma_curve (float): controls the mapping of gamma to a Dirichlet alpha.
        ndim (int): if 2, the input is a t x m vectorized matrix. if 1, the input is a matrix row
                    of length m (default 2).
    """
    def __init__(self, TemporalData, rank, num_agents, density, *,
                 hide_future: bool = True, 
                 rho: float = 0.0, gamma: float = 0.0, gamma_curve: float = 1.2,
                 ndim: int = 2,
                 verbose: bool = True):
        assert 0.0 <= rho <= 1.0, "rho must be in [0,1]"
        assert 0.0 <= gamma <= 1.0, "gamma must be in [0,1]"
        if ndim > 2:
            raise NotImplementedError("Masks must be for a 1D or 2D tensor.")

        self.num_matrices = len(TemporalData)
        self.num_agents = num_agents
        self.density = density
        self.ndim = ndim
        if self.ndim == 2:
            self.t, self.m = TemporalData.t, TemporalData.m
            self.total_entries = self.t * self.m
        else:
            self.t, self.m = 1, TemporalData.m
            self.total_entries = self.m
            
        self.r = rank
        self.verbose = verbose
        self.stats = {
            "agent_overlap": None,
            "agent_overlap_var": None,
            "agent_endowments": None,
            "actual_knowns": None,
            "oversample_flags": 0,
            "fraction_clipped": 0.0
        }
        self.hide_future = hide_future

        self.rho = float(rho)
        self.gamma = float(gamma)
        self.gamma_curve = float(gamma_curve)

        self.masks, self.global_known = self._generate()

    def __getitem__(self, idx):
        assert idx < self.num_agents, f"Agent index {idx} is out of range"
        return self.masks[idx, :] if idx != -1 else self.global_known

    def __call__(self, X: torch.Tensor, global_mask: bool = False):
        if global_mask:
            global_known = self.global_known[None,...]
            assert X.shape == global_known.shape
            return X * global_known
        else:
            if len(X.shape) == 2:
                if X.shape[0] == self.num_agents:
                    assert X.shape[1] == self.total_entries
                    return X * self.masks
                elif X.shape[0] == 1 and X.shape[1] == self.total_entries:
                    return X.repeat(self.num_agents, 1) * self.masks
            elif len(X.shape) == 3:
                # Batched implementation
                assert X.shape[1] == 1
                assert X.shape[2] == self.total_entries
                batch_size = X.shape[0]
                return X * self.masks[None,...].repeat(batch_size, 1, 1)
            else:
                raise ValueError(f"X is of unexpected shape {X.shape}")

    def _generate(self):
        global_known_idx = self._robust_sample_global_known_idx()
        masks, samplesizes, oversampled = self._build_agent_masks(global_known_idx)
        global_actual_known = masks.sum(dim = 0) > 0

        # update stats
        self.stats["agent_endowments"] = samplesizes.clone() if isinstance(samplesizes, torch.Tensor) else torch.tensor(samplesizes)
        self.stats["oversample_flags"] = int(oversampled)
        self.stats["actual_knowns"] = int(global_actual_known.sum().item())
        if self.verbose:
            self.__summary(masks)

        return masks, global_actual_known

    def _sample_global_known_idx(self):
        global_known_count = max(int(self.density * self.total_entries), 1)
        all_indices = torch.randperm(self.total_entries)
        global_known_indices = all_indices[:global_known_count]
        global_mask = torch.zeros(self.total_entries, dtype=torch.bool)
        global_mask[global_known_indices] = True
        if self.hide_future and self.t > 1:
            # Hide the entire last row
            global_mask[-self.m:] = False
            keep = global_known_indices < self.total_entries - self.m
            global_known_indices = global_known_indices[keep]
        return global_mask, global_known_indices

    def _robust_sample_global_known_idx(self, max_attempts: int = 20):
        if self.r <= min(self.m, self.t) // 2 and self.t > 1:
            for attempt in range(1, max_attempts + 1):
                global_mask, global_known_idx = self._sample_global_known_idx()
                mask_2d = global_mask.view(self.t, self.m)
                rows_ok = (mask_2d[:-1, :].sum(dim=1) >= self.r).all()
                cols_ok = (mask_2d.sum(dim=0) >= self.r).all()
                if rows_ok and cols_ok:
                    return global_known_idx
                if self.verbose and attempt < max_attempts:
                    print(f"Warning: Retrying sampling (attempt {attempt}) due to sparse rows/cols.")
            raise RuntimeError(
                f"Failed to sample a valid mask after {max_attempts} attempts. "
                f"Could not ensure at least {self.r} known entries in each row and column. "
                f"Density: {self.density}, size: {self.t}x{self.m}"
            )
        else:
            _, global_known_idx = self._sample_global_known_idx()
            unseen = self.m - global_known_idx.shape[0]
            if self.verbose:
                if self.t > 1:
                    print("Warning: matrices are not low-rank. r is high compared to t or m.")
                elif self.t == 1:
                    print(f"Warning: {unseen} columns cannot be sampled from and will remain secret.")
            return global_known_idx

    def _agent_samplesizes(self, num_global_known: int):
        """
        Produce per-agent sample sizes given gamma (γ) and rho (ρ).
        
        μ(ρ) = (1 - ρ) * (G / N) + ρ * G
        We draw Dirichlet proportions (concentration determined by γ) and scale to N * μ so that
        the mean per-agent size ≈ μ.

        Geometric/log interpolation between alpha_max and alpha_min, with an exponent
        self.gamma_curve applied to gamma to control sensitivity.

        alpha = 10 ** ( (1 - gamma_adj) * log10(alpha_max) + gamma_adj * log10(alpha_min) )
        gamma_adj = gamma ** gamma_curve
        """
        G = num_global_known
        N = self.num_agents

        if G == 0:
            return np.zeros(N, dtype=int), 0

        # target mean per-agent endowment
        mu = (1.0 - self.rho) * (G / float(max(1, N))) + self.rho * float(G)

        # mapping parameters (tunable)
        alpha_max = 50.0   # concentration corresponding to near-uniform proportions
        alpha_min = 0.01   # concentration corresponding to highly skewed proportions
        gamma_adj = float(self.gamma) ** float(self.gamma_curve)

        # geometric (log-space) interpolation
        log10_alpha = (1.0 - gamma_adj) * np.log10(alpha_max) + gamma_adj * np.log10(alpha_min)
        concentration = float(10.0 ** log10_alpha)

        # build Dirichlet alpha vector and sample proportions
        alpha = np.ones(N) * concentration
        p = np.random.dirichlet(alpha)  # sums to 1

        total_target = N * mu
        raw = p * total_target
        samplesizes = np.round(raw).astype(int)

        # Count oversample attempts and clip
        oversampled = int(np.sum(samplesizes > G))
        samplesizes = np.minimum(samplesizes, G)

        return samplesizes, oversampled

    def _build_agent_masks(self, global_known_idx):
        # convert torch tensor to numpy indices if necessary
        if isinstance(global_known_idx, torch.Tensor):
            global_known_idx = global_known_idx.cpu().numpy()

        num_global_known = len(global_known_idx)
        masks = torch.zeros((self.num_agents, self.total_entries), dtype=torch.bool)

        samplesizes, oversampled_total = self._agent_samplesizes(num_global_known)

        # For sampling per-agent independently from G (no common pool enforced)
        # Clip sizes already handled in _agent_samplesizes; ensure integer
        samplesizes = samplesizes.astype(int)

        for i in range(self.num_agents):
            k = int(samplesizes[i])
            if k <= 0:
                continue
            # sample without replacement from global_known_idx for this agent
            # different agents can pick same indices (desired behavior)
            chosen = np.random.choice(global_known_idx, size=k, replace=False)
            masks[i, chosen] = True

        # compute fraction clipped
        fraction_clipped = float(oversampled_total) / float(self.num_agents) if self.num_agents > 0 else 0.0
        self.stats["fraction_clipped"] = fraction_clipped

        return masks, torch.tensor(samplesizes), oversampled_total

    def __summary(self, masks):
        # compute pairwise overlaps list (Jaccard)
        overlaps = _compute_pairwise_overlaps(masks) if self.num_agents > 1 else []
        mean_overlap = float(np.mean(overlaps)) if overlaps else ( -np.inf if self.num_agents == 1 else 0.0 )
        var_overlap = float(np.var(overlaps, ddof=0)) if overlaps else 0.0

        self.stats["agent_overlap"] = mean_overlap
        self.stats["agent_overlap_var"] = var_overlap

        endowment_vec = self.stats["agent_endowments"].float() if isinstance(self.stats["agent_endowments"], torch.Tensor) else torch.tensor(self.stats["agent_endowments"])
        endowment_mean = float(torch.mean(endowment_vec).item()) if endowment_vec.numel() > 0 else 0.0
        actual_known_mean = float(self.stats["actual_knowns"])  # union size

        print("--------------------------")
        print(f"Target density: {self.density:.2f}.")
        print(f"Target known entries: {self.total_entries * self.density:.1f}, Actual union known entries: {actual_known_mean:.1f}")
        print(f"{self.num_agents} agents, Avg overlap (Jaccard): {mean_overlap:.3f}, Var overlap: {var_overlap:.6f}")
        print(f"Avg entries per agent: {endowment_mean:.1f}")
        print(f"Fraction clipped at G: {self.stats['fraction_clipped']:.3f}")
        if self.stats["oversample_flags"] > 0:
            print(f"WARNING ⚠️  {self.stats['oversample_flags']} matrices had agents sampling all known entries.")
        print("--------------------------")

def _compute_pairwise_overlaps(masks: torch.Tensor):
    """
    Return list of pairwise Jaccard overlaps (intersection / union) across agent pairs.
    """
    assert len(masks.shape) == 2 and masks.shape[0] > 1, "masks must be [num_agents, total_entries] with num_agents > 1"
    A, D = masks.shape
    overlaps = []
    for i in range(A):
        for j in range(i + 1, A):
            inter = int((masks[i] & masks[j]).sum().item())
            union = int((masks[i] | masks[j]).sum().item())
            if union > 0:
                overlaps.append(inter / union)
            else:
                overlaps.append(0.0)
    return overlaps


class SensingMasksTemporal(object):
    """
    Column-wise sensing for temporal matrices.

    Each agent observes a random fraction rho of the m columns of a t x m matrix.
    Application semantics:
      - Input X can be a flattened single matrix [1, D] with D = t*m, or a batched
        collection [B, 1, D].
      - This mask zeros out entire columns per agent.
      - Returns masked tensors with shapes [A, t, m] (single) or [B, A, t, m] (batched).

    Args:
        TemporalData: dataset providing .t and .m attributes.
        num_agents: number of agents A.
        rho: fraction of columns per agent to keep (0.0 to 1.0).
        seed: optional torch RNG seed (for reproducible column choices).
    """
    def __init__(self, TemporalData, num_agents: int, rho: float, *, seed: int | None = None):
        assert 0.0 <= rho <= 1.0, "rho must be in [0, 1]"
        self.t, self.m = int(TemporalData.t), int(TemporalData.m)
        self.num_agents = int(num_agents)
        self.rho = float(rho)
        self.total_entries = self.t * self.m
        if seed is not None:
            torch.manual_seed(int(seed))
        # Build per-agent column masks: [A, m]
        k = int(round(self.rho * self.m))
        k = max(min(k, self.m), 0)  # clamp to [0, m]
        masks = torch.zeros((self.num_agents, self.m), dtype=torch.bool)
        if k > 0:
            for a in range(self.num_agents):
                cols = torch.randperm(self.m)[:k]
                masks[a, cols] = True
        self.col_masks = masks  # [A, m], True where observed

    def __getitem__(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < self.num_agents, f"Agent index {idx} out of range"
        return self.col_masks[idx]

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply agent-specific column masks to input matrices.

        Accepts:
          - [1, D] with D = t*m  -> returns [A, t, m]
          - [B, 1, D]           -> returns [B, A, t, m]
          - [t, m]              -> returns [A, t, m]
          - [B, t, m]           -> returns [B, A, t, m]
        """
        device = X.device
        t, m, A = self.t, self.m, self.num_agents
        D = t * m
        col_masks = self.col_masks.to(device=device)

        if X.dim() == 2 and X.shape[0] == 1 and X.shape[1] == D:
            # [1, D] -> [A, t, m]
            X_tm = X.view(1, t, m)
            X_A_tm = X_tm.repeat(A, 1, 1)
            return X_A_tm * col_masks[:, None, :]
        elif X.dim() == 3 and X.shape[0] >= 1 and X.shape[1] == 1 and X.shape[2] == D:
            # [B, 1, D] -> [B, A, t, m]
            B = X.shape[0]
            X_B_tm = X.view(B, 1, t, m)
            X_BA_tm = X_B_tm.repeat(1, A, 1, 1)
            return X_BA_tm * col_masks[None, :, None, :]
        elif X.dim() == 2 and X.shape == (t, m):
            # [t, m] -> [A, t, m]
            X_tm = X.unsqueeze(0)
            X_A_tm = X_tm.repeat(A, 1, 1)
            return X_A_tm * col_masks[:, None, :]
        elif X.dim() == 3 and X.shape[1:] == (t, m):
            # [B, t, m] -> [B, A, t, m]
            B = X.shape[0]
            X_B_tm = X.unsqueeze(1)  # [B, 1, t, m]
            X_BA_tm = X_B_tm.repeat(1, A, 1, 1)
            return X_BA_tm * col_masks[None, :, None, :]
        else:
            raise ValueError(f"Unexpected input shape {tuple(X.shape)}; expected [1,D], [B,1,D], [t,m], or [B,t,m] with D=t*m")


# ========= Sample usage (commented) =========
"""
from datautils.datagen_temporal import GTMatrices, TemporalData
import matplotlib.pyplot as plt
import torch

# Parameters
NUM_MATRICES = 1
T = 50
M = 25
A = 4
R = 10
RHO = 0.3  # fraction of columns each agent observes

with torch.no_grad():
    gt = GTMatrices(N=NUM_MATRICES, t=T, m=M, r=R)
    td = TemporalData(gt)
    smt = SensingMasksTemporal(td, num_agents=A, rho=RHO, seed=42)

    sample = td[0]['matrix']          # [1, T*M]
    masked = smt(sample)              # [A, T, M]

    fig, axs = plt.subplots(1, A + 1, figsize=(3*(A+1), 4))
    axs[0].imshow(sample.view(T, M))
    axs[0].set_title('Ground Truth')
    axs[0].set_xticks([]); axs[0].set_yticks([])
    for a in range(A):
        axs[a+1].imshow(masked[a])
        axs[a+1].set_title(f'Agent {a} (rho={RHO})')
        axs[a+1].set_xticks([]); axs[a+1].set_yticks([])
    plt.tight_layout(); plt.show()
"""
