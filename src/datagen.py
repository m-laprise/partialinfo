
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


def generate_low_rank_matrix(n: int, m: int, r: int, sigma=0.0):
    """
    Returns: matrix M of shape [n, m] = U @ V.T + noise
    """
    U = np.random.randn(n, r)
    V = np.random.randn(m, r) 
    M = U @ V.T
    if sigma > 0:
        M += sigma * np.random.randn(n, m)
    M *= 1.0 / np.sqrt(r)
    return torch.tensor(M, dtype=torch.float32)


def sample_global_known_entries(n:int, m:int, density:float):
    total_entries = n * m
    global_known_count = max(int(density * total_entries), 1)
    all_indices = torch.randperm(total_entries)
    global_known_indices = all_indices[:global_known_count]
    global_mask = torch.zeros(total_entries, dtype=torch.bool)
    global_mask[global_known_indices] = True
    return global_mask, global_known_indices


def robust_sample_global_known_entries(
    n: int, m: int, r: int, density: float, 
    total_entries: int,
    idx: int, max_attempts: int = 10,
    verbose: bool = True
):
    for attempt in range(1, max_attempts + 1):
        global_mask, global_known_idx = sample_global_known_entries(n, m, density)
        mask_2d = global_mask.view(n, m)
        rows_ok = (mask_2d.sum(dim=1) >= r).all()
        cols_ok = (mask_2d.sum(dim=0) >= r).all()
        if rows_ok and cols_ok:
            return global_known_idx
        if verbose and attempt < max_attempts:
            print(f"Warning: Retrying sampling for matrix {idx} (attempt {attempt}) due to sparse rows/cols.")
    raise RuntimeError(
        f"Failed to sample a valid mask after {max_attempts} attempts. "
        f"Could not ensure at least {r} known entries in each row and column. "
        f"Matrix index: {idx}, density: {density}, size: {n}x{m}"
    )


def define_agent_samplesize(num_agents:int, num_global_known:int):
    base = max(1, num_global_known // num_agents)
    low = min(int(2.0 * base), num_global_known)
    high = min(int(4.0 * base), num_global_known)
    samplesize = np.random.randint(low, high) if high > low else low
    oversampled = int(samplesize > num_global_known)
    samplesize = min(samplesize, num_global_known)
    return samplesize, oversampled


def build_agent_masks(num_matrices: int,
                      num_agents: int,
                      total_entries: int,
                      global_known_idx_list: list[torch.Tensor],
                      mode: str = 'uniform', 
                      scheme: str = 'constant'):
    """
    Returns:
        - mask_tensor: shape [num_matrices, num_agents, total_entries]
        - agent_endowments: list of list of ints [num_matrices][num_agents]
        - oversampled: total oversampled flags
    """
    masks = []
    endowments_all = []
    oversampled_total = 0
    
    if scheme == 'constant':
        assert len(global_known_idx_list) == 1
        global_known_idx = global_known_idx_list[0]
        num_global_known = len(global_known_idx)
        agent_masks = []
        agent_endowments = []
        for _ in range(num_agents):
            mask = torch.zeros(total_entries, dtype=torch.bool)
            sample_size, agent_oversampled = define_agent_samplesize(num_agents, num_global_known)
            oversampled_total += agent_oversampled
            sample_idx = global_known_idx[torch.randint(num_global_known, (sample_size,))]
            mask[sample_idx] = True
            agent_masks.append(mask)
            agent_endowments.append(len(sample_idx))
        masks = torch.stack(agent_masks)
        endowments_all = agent_endowments
        
    elif scheme == 'random':
        assert len(global_known_idx_list) == num_matrices
        for global_known_idx in global_known_idx_list:
            agentmatrix_masks = []
            agentmatrix_endowments = []
            num_global_known = len(global_known_idx)

            for _ in range(num_agents):
                mask = torch.zeros(total_entries, dtype=torch.bool)

                if mode == 'all-see-all':
                    sample_idx = global_known_idx

                elif mode == 'uniform':
                    if num_global_known == 0:
                        sample_idx = []
                    else:
                        sample_size, agent_oversampled = define_agent_samplesize(num_agents, num_global_known)
                        oversampled_total += agent_oversampled
                        sample_idx = global_known_idx[torch.randint(num_global_known, (sample_size,))]
                        
                else:
                    raise NotImplementedError(f"Unknown mode: {mode}")

                mask[sample_idx] = True
                agentmatrix_masks.append(mask)
                agentmatrix_endowments.append(len(sample_idx))

            masks.append(torch.stack(agentmatrix_masks)) 
            endowments_all.append(np.mean(agentmatrix_endowments))
        masks = torch.stack(masks)
    else:
        raise NotImplementedError(f"Unknown scheme: {scheme}")
    
    return masks, endowments_all, oversampled_total  # [num_matrices, num_agents, total_entries]


def compute_avg_agent_overlap(agent_views: torch.Tensor) -> float:
    """
    Computes average Jaccard overlap (intersection / union) 
    across all pairs of agents.
    """
    A, D = agent_views.shape
    binary = (agent_views > 0).float()
    overlaps = []
    for i in range(A):
        for j in range(i + 1, A):
            inter = (binary[i] * binary[j]).sum().item()
            union = (binary[i] + binary[j]).clamp(0, 1).sum().item()
            if union > 0:
                overlaps.append(inter / union)
    return float(np.mean(overlaps)) if overlaps else 0.0


class AgentMatrixReconstructionDataset(InMemoryDataset):
    def __init__(self, 
                 num_matrices: int, 
                 n: int = 20, m: int = 20, r: int = 4, 
                 num_agents: int = 30, 
                 agentdistrib: str = 'uniform',
                 sampling_scheme: str = 'constant',
                 density: float = 0.2, 
                 sigma: float = 0.0, 
                 verbose: bool = True,
                 masks_cache = None):
        self.num_matrices = num_matrices
        self.n, self.m, self.r = n, m, r
        self.input_dim = self.total_entries = n * m
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.agentdistrib = agentdistrib
        self.sampling_scheme = sampling_scheme
        self.verbose = verbose
        self._masks_cache = masks_cache  # use external cache (shared across datasets)
        super().__init__('.')
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        matrices = []
        
        stats = {
            "nuclear_norms": [],
            "gaps": [],
            "variances": [],
            "agent_overlaps": [],
            "agent_endowments": [],
            "actual_knowns": [],
            "oversample_flags": 0,
        }

        # Use external cache if available, otherwise build
        if self._masks_cache is not None:
            if self.sampling_scheme == "constant":
                # Safe reuse — all matrices share the same global_known_idx
                agent_masks = self._masks_cache
                agent_endowments = (self._masks_cache > 0).sum(-1).tolist()
                oversampled = 0
            else:
                raise ValueError("Mask cache reuse is only supported for 'constant' sampling_scheme")
        else:
            # A: matrix-level knowable entries
            if self.sampling_scheme == 'random':
                global_known_idx_list = [
                    robust_sample_global_known_entries(
                        self.n, self.m, self.r, self.density, self.total_entries, idx
                    )
                    for idx in range(self.num_matrices)
                ]
            elif self.sampling_scheme == 'constant':
                global_known_idx = robust_sample_global_known_entries(
                    self.n, self.m, self.r, self.density, self.total_entries, 0
                )
                global_known_idx_list = [global_known_idx]
            else:
                raise NotImplementedError(f"Unknown sampling scheme: {self.sampling_scheme}")
            
            # B: Agent-level masks
            agent_masks, agent_endowments, oversampled = build_agent_masks(
                self.num_matrices, self.num_agents, self.total_entries,
                global_known_idx_list, mode=self.agentdistrib, scheme=self.sampling_scheme
            )
            if self.sampling_scheme == 'constant':
                self._masks_cache = agent_masks
            stats["oversample_flags"] = oversampled
            stats["agent_endowments"] = agent_endowments
        
        # Generate low rank matrices
        for _ in range(self.num_matrices):
            M = generate_low_rank_matrix(self.n, self.m, self.r, self.sigma)
            M_vec = M.view(-1)
            matrices.append(M_vec)
            
            stats["nuclear_norms"].append(torch.linalg.norm(M, ord='nuc').item())
            S = torch.linalg.svdvals(M)
            stats["gaps"].append((S[self.r - 1] - S[self.r]).item())
            stats["variances"].append(M_vec.var().item())
        
        # Apply masks to create final data objects
        for i in range(self.num_matrices):
            M_stack = matrices[i].repeat(self.num_agents, 1)
            agent_masks_i = agent_masks[i, :, :] if self.sampling_scheme == 'random' else agent_masks
            assert M_stack.shape == agent_masks_i.shape
            # If constant, apply the unique mask. If random, apply the varying masks for each matrix.
            agent_views = M_stack * agent_masks_i
            
            #if self.sampling_scheme == 'random':
            #    stats["agent_endowments"].extend(agent_endowments[i])
                
            if self.num_agents > 1:
                avg_overlap = compute_avg_agent_overlap(agent_views)
                stats["agent_overlaps"].append(avg_overlap)

            global_actual_known = agent_masks_i.sum(dim = 0) > 0
            actual_known = global_actual_known.sum()
            stats["actual_knowns"].append(actual_known)
            
            data = Data(x=agent_views, y=matrices[i], mask=global_actual_known)
            data_list.append(data)

        # Save summary statistics
        self.nuclear_norm_mean = np.mean(stats["nuclear_norms"])
        self.gap_mean = np.mean(stats["gaps"])
        self.variance_mean = np.mean(stats["variances"])
        self.agent_overlap_mean = (
            np.mean(stats["agent_overlaps"]) if stats["agent_overlaps"] else -np.inf
        )
        self.agent_endowment_mean = np.mean(stats["agent_endowments"]) if stats["agent_endowments"] else -np.inf
        self.actual_known_mean = np.mean(stats["actual_knowns"])

        if self.verbose:
            self._print_summary(stats["oversample_flags"])

        return self.collate(data_list)

    def _print_summary(self, oversampled):
        print("--------------------------")
        print(f"Generated {self.num_matrices} rank-{self.r} matrices of size {self.n}x{self.m}")
        print(f"Mean nuclear norm: {self.nuclear_norm_mean:.4f}, Spectral gap: {self.gap_mean:.4f}")
        print(f"Observed density: {self.density:.2f}, Noise level: {self.sigma:.4f}")
        print(f"Total entries: {self.total_entries}, Target known entries: {self.total_entries * self.density:.1f}, Mean known entries: {self.actual_known_mean:.1f}")
        print(f"{self.num_agents} agents, Avg overlap: {self.agent_overlap_mean:.3f}")
        print(f"Avg entries per agent: {self.agent_endowment_mean:.1f}")
        if oversampled > 0:
            print(f"WARNING ⚠️  {oversampled} matrices had agents sampling all known entries.")
        print("--------------------------")

    def get_masks_cache(self):
        """Return internal cache for external reuse."""
        return self._masks_cache
