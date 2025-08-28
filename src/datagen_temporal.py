from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from torch.utils.data import Dataset


@torch.no_grad()
def matern_covariance(t: int,
                      length_scale: float,
                      nu: float,
                      *,
                      sigma2: float = 1.0,
                      jitter: float = 1e-8) -> torch.Tensor:
    """
    Return a t x t Matérn covariance matrix over integer index positions 0 to t-1.
    """
    # Build kernel with free variance
    kernel = ConstantKernel(constant_value=sigma2) * Matern(length_scale=length_scale, nu=nu)
    # Locations in index space → shape (n,1)
    X = np.arange(t, dtype=float).reshape(-1, 1)
    # Evaluate and stabilize (to later draw samples, need a positive-definite Σ) 
    vcov = kernel(X)              # equivalent to kernel(X, X)
    np.fill_diagonal(vcov, vcov.diagonal() + jitter) # type: ignore
    return torch.as_tensor(vcov)


@torch.no_grad()
def _random_matern_params(mode: int, *, nu_low: float = 0.5, nu_high: float = 8.0) -> Dict[str, float]:
    """
    Generate random time series parameters for slow, medium, or fast dynamics.
    """    
    if mode not in {0, 1, 2}:
        raise ValueError("mode must be 0 (slow), 1 (medium) or 2 (fast)")
    scale_ranges = {
        0: (20.0, 40.0),   # slow
        1: (5.0, 15.0),    # medium
        2: (1.0, 4.0),      # fast
    }
    scale_low, scale_high = scale_ranges[mode]
    scale = (torch.rand(()) * (scale_high - scale_low) + scale_low).item()
    nu    = (torch.rand(()) * (nu_high - nu_low) + nu_low).item() 
    return {'length_scale': scale, 'nu': nu}


@torch.no_grad()
def _random_cauchy_params(mode: int) -> Dict[str, float]:
    """
    Generate random time series parameters for very long, long, or no memory.
    
    Note the following reference combinations:
     - α=2.0, β=1.0: Gaussian-like
     - α=0.5, β=2.0: fast decay, rough
     - α=2.0, β=2.0: fast decay, smooth
     - α=0.5, β=0.2: long memory, rough
     - α=2.0, β=0.2: long memory, smooth
    """    
    if mode not in {0, 1, 2}:
        raise ValueError("mode must be 0 (long-range), 1 (medium-range) or 2 (fast decay)")
    beta_ranges = {
        0: (1e-3, 0.5),
        1: (0.5, 0.9),
        2: (1.0, 1.5),
    }
    beta_low, beta_high = beta_ranges[mode]
    alpha_low, alpha_high = (0.75, 2)
    beta  = (torch.rand(()) * (beta_high - beta_low) + beta_low).item()
    alpha = (torch.rand(()) * (alpha_high - alpha_low) + alpha_low).item() 
    return {'alpha': alpha, 'beta': beta}


@torch.no_grad()
def cauchy_correlation(h, *, alpha, beta):
    if alpha <= 0 or alpha > 2:
        raise ValueError("alpha ∈ (0, 2]")
    if beta < 0:
        raise ValueError("beta > 0")
    
    return (1 + np.abs(h)**alpha)**(-beta/alpha)


@torch.no_grad()
def cauchy_covariance(t, *, alpha, beta, sigma2=1.0):
    """
    Build covariance matrix for Cauchy class process.
    
    Parameters:
    t : int, number of time points
    alpha : float, in (0, 2], controls smoothness
    beta : float, > 0, controls decay rate
    sigma2 : float, marginal variance
    """
    vcov = torch.zeros((t, t))
    for i in range(t):
        for j in range(t):
            if i <= j:
                h = i - j
                vcov[i, j] = vcov[j,i] = sigma2 * cauchy_correlation(h / 4, alpha=alpha, beta=beta)
    # Add small diagonal term for numerical stability
    vcov += 1e-8 * torch.eye(t)
    return vcov


def hurst_coef(beta):
    if beta < 1:
        return 1 - beta/2
    else:
        return None


def fractal_dim(alpha, ndim=1):
    return ndim + 1 - alpha/2


def _even_partition(total: int, k: int) -> List[int]:
    """
    Split `total` into `k` integer parts whose sizes differ by at most 1. Assumes 1 ≤ k ≤ total.
    """
    base, extra = divmod(total, k)
    return [base + (i < extra) for i in range(k)]


@torch.no_grad()
def _generate_V(r: int,
                m: int,
                row_sizes: Optional[List[int]] = None,
                *,
                strong_mean: float = 3.0,
                strong_std:  float = 0.75,
                weak_mean:  float = 0.0,
                weak_std:   float = 1.0,
                shuffle_columns: bool = True,
                vtype: str = 'block') -> torch.Tensor:
    if r <= 0 or m <= 0:
        raise ValueError("r and m must be positive integers")
    if r > m:
        raise ValueError("r should not exceed m for block construction")
    if vtype not in ['block', 'random']:
        raise ValueError("vtype must be 'block' or 'random'")
    if vtype == 'block':
        # determine block sizes
        if row_sizes is None:
            row_sizes = _even_partition(r, min(r, 3))
        col_sizes = _even_partition(m, min(m, 3))   # three column-blocks
        # build logits tensor
        logits = torch.normal(mean=weak_mean, std=weak_std, size=(r, m))
        # mark the starts of each column block once for efficiency
        col_starts = [0]                    
        for sz in col_sizes[:-1]:             
            col_starts.append(col_starts[-1] + sz)
        # overwrite the diagonal blocks with “high” logits
        row_start = 0
        for block in range(len(row_sizes)):
            row_end = row_start + row_sizes[block]
            col_start = col_starts[block]
            col_end   = col_start + col_sizes[block]
            # Set strong logits for the diagonal block
            logits[row_start:row_end, 
                col_start:col_end] = torch.normal(
                    mean=strong_mean,
                    std=strong_std,
                    size=(row_sizes[block], col_sizes[block])
                )
            row_start = row_end
        # optional column shuffling
        if shuffle_columns:
            perm = torch.randperm(m)
            logits = logits[:, perm]
        # softmax over rows
        V = torch.softmax(logits, dim=0) 
    elif vtype == 'random':
        V = torch.randn(r, m) + torch.ones(r, m) 
        V /= V.sum(dim=0, keepdim=True)
    return V.T


@torch.no_grad()
def _generate_DGP(
    t: int, m: int, r: int, kernel: str, vtype: str, **kernel_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Define a random data-generating process comprised of the following parameters:
    - r autocovariance matrices of size t x t (one for each column of U)
    - A mixing weight matrix V of size r x m
    with a division of the rank r into up to three blocks representing different
    dynamics (ie, a list of integers r1, r2, r3 such that r = r1 + r2 + r3)
    """
    if kernel not in ['matern', 'cauchy', 'whitenoise']:
        raise NotImplementedError(f"Kernel {kernel} not implemented")
    nblocks = min(r, 3)           # 1, 2, or 3 blocks
    block_sizes: List[int] = _even_partition(r, nblocks)
    modes = torch.repeat_interleave(torch.arange(nblocks), torch.tensor(block_sizes))
    if kernel == 'matern':
        vcov_list = [
            matern_covariance(t, **_random_matern_params(int(mode)), **kernel_kwargs)
            for mode in modes
        ]
        vcovU = torch.stack(vcov_list, dim=0)  # r x t x t 
    elif kernel == 'cauchy':
        vcov_list = [
            cauchy_covariance(t, **_random_cauchy_params(int(mode)), **kernel_kwargs)
            for mode in modes
        ]
        vcovU = torch.stack(vcov_list, dim=0)  # r x t x t 
    elif kernel == 'whitenoise':
        vcovU = torch.eye(t).expand(r, t, t)
    V = _generate_V(r, m, block_sizes, vtype=vtype)
    return vcovU, V


@torch.no_grad()
def _gen_U_col(vcovU: torch.Tensor, *, offset: float) -> torch.Tensor:
    t = vcovU.shape[1]
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.full(
            (t,), offset, dtype=vcovU.dtype
        ),
        covariance_matrix=vcovU
    )
    U_col = mvn.rsample() 
    return U_col


@torch.no_grad()
def _generate_U(vcovsU: torch.Tensor, *, offset: float = 5.0):
    r, t, _ = vcovsU.shape
    U = torch.zeros(t, r)
    for col in range(r):
        U[:, col] = _gen_U_col(vcovsU[col], offset=offset)
    return U


def _fin_return(val_start, val_end):
    return (val_end - val_start) / val_start


class GTMatrices(Dataset):
    """
    Generates ground truth matrices. Task-agnostic and does not contain any labels.
    """
    @torch.no_grad()
    def __init__(self, 
                 N: int,                    # Number of examples M = U @ V.T
                 t: int, m: int, r: int,    # M is a t by m matrix of rank r
                 #sigma: float = 0.0, 
                 structured: bool = True,   # Use iid or structured U factors
                 realizations: int = 10,    # If structured, nb of realizations for each DGP
                 mode: str = 'value',
                 kernel: str = 'matern',
                 vtype: str = 'random',
                 seed: Optional[int] = None,
                 **kernel_kwargs):
        assert N % realizations == 0
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.mode = mode
        self.kernel = kernel
        self.vtype = vtype
        self.N = N
        self.num_dgps = N // realizations
        self.t, self.m, self.r = t, m, r 
        self.structured = structured
        self.realizations = realizations
        #self.sigma = sigma
        self.U, self.V, self.vcovU = self._generate_factors(N, **kernel_kwargs)

    def __len__(self):
        return self.U.shape[0]
    
    def _generate_factors(self, N: int, **kernel_kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.structured:
            t = self.t
            if self.mode == 'return':
                t += 1
            vcovsUs = torch.zeros((self.num_dgps, self.r, t, t))
            Us = torch.zeros((self.realizations * self.num_dgps, t, self.r))
            Vs = torch.zeros((self.realizations * self.num_dgps, self.m, self.r))
            for dgp in range(self.num_dgps):
                vcovsU, V = _generate_DGP(
                    t = t, m = self.m, r = self.r, 
                    kernel=self.kernel, vtype=self.vtype,
                    **kernel_kwargs
                )
                vcovsUs[dgp] = vcovsU
                for i in range(self.realizations):
                    idx = dgp * self.realizations + i
                    Us[idx] = _generate_U(vcovsU)
                    Vs[idx] = V
        else:
            Us = torch.rand(N, self.t, self.r, dtype=torch.float32) * 2
            Vs = torch.rand(N, self.m, self.r, dtype=torch.float32)
            vcovsUs = torch.eye(self.t).reshape(1, 1, self.t, self.t)
        return Us, Vs, vcovsUs

    def generate_matrices(self, idx = None):
        idx = np.arange(self.U.shape[0]) if idx is None else idx
        M = torch.einsum('...ij,...kj->...ik', self.U[idx], self.V[idx])
        #M += self.sigma * torch.randn(M.shape, dtype=torch.float32)
        if not self.structured:
            M /= np.sqrt(self.r)
            return M / M.std(dim=1, keepdim=True)
        elif self.mode == 'return':
            M_r = _fin_return(M[:, :-1, :], M[:, 1:, :])
            return M_r / M_r.std(dim=1, keepdim=True)
        else:
            for idx in range(M.shape[0]):
                M[idx] /= M[idx].std(dim=0)
            return M
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.generate_matrices(idx)


class RandomLinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 2*self.in_dim, bias=True),
            nn.SiLU(),
            nn.Linear(2*self.in_dim, self.out_dim, bias=False)
        ) 
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.mlp(x))
    

class TemporalData(Dataset):
    """
    Takes ground truth matrices and a task statement, and generates labeled data that can
    be provided to a DataLoader.
    """
    def __init__(self, matrices: GTMatrices, task: str = 'argmax', verbose=True):
        super().__init__()
        if task not in ['argmax', 'nonlinear']:
            raise NotImplementedError(f"Task {task} not implemented")
        
        self.data = matrices[:len(matrices)]
        self.t, self.m, self.r = matrices.t, matrices.m, matrices.r
        self.task = task
        if self.task == 'nonlinear':
            self.random_mlp = RandomLinearHead(self.m, 1)
        
        self.verbose = verbose
        self.stats = self.__summary()

    def __len__(self):
        return self.data.shape[0]
    
    def _mlp_apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_mlp(x)
    
    def _generate_ycol(self, matrix):
        if matrix.ndimension() == 2:
            matrix = matrix.unsqueeze(0)
        vectorized_mlp = torch.func.vmap(self._mlp_apply)
        y_column = vectorized_mlp(matrix).squeeze(0)
        return y_column
    
    def _generate_label(self, matrix, y_column):
        if self.task == 'argmax':
            # Label is the index of the maximum entry in the last row
            label = torch.argmax(matrix[-1, :])
        elif self.task == 'nonlinear':
            # Label is the last row and the last element of y
            last_row = matrix[-1, :]
            label = torch.cat((last_row, y_column[-1]))
        return label

    def __getitem__(self, idx: int):
        _, t, m = self.data.shape
        matrix = self.data[idx, :, :]
        y_column = self._generate_ycol(matrix) if self.task == 'nonlinear' else None
        sample = {
            'matrix': matrix.view(1, t*m),
            'label': self._generate_label(matrix, y_column)
        }
        
        return sample
    
    def __summary(self):
        # TO DEBUG
        S = torch.linalg.svdvals(self.data)
        if self.r < min(self.t, self.m):
            gap = np.mean((S[:, self.r - 1] - S[:, self.r]).tolist())
        else:
            gap = 0.0
        nuc = np.mean(S.sum(dim=1).tolist())
        var = np.mean(self.data.var(dim=1).tolist())
        if self.verbose:
            print("--------------------------")
            print(f"Dataset of {len(self)} labelled rank-{self.r} matrices of size {self.t}x{self.m}")
            print(f"Mean nuclear norm: {nuc:.4f}, Spectral gap: {gap:.4f}")
            print(f"Mean variance of entries: {var:.4f}")
            print(f"Total entries: {self.t * self.m}")
            print("--------------------------")
        return { "nuclear_norms": nuc, "gaps": gap, "variances": var }


def _compute_avg_overlap(masks: torch.Tensor) -> float:
    assert len(masks.shape) == 2 and masks.shape[0] > 1, "masks must be [num_agents, total_entries] with num_agents > 1"
    # (intersection / union) across all pairs of agents.
    A, D = masks.shape
    overlaps = []
    for i in range(A):
        for j in range(i + 1, A):
            inter = (masks[i] * masks[j]).sum().item()
            union = (masks[i] + masks[j]).clamp(0, 1).sum().item()
            if union > 0:
                overlaps.append(inter / union)
    return float(np.mean(overlaps)) if overlaps else 0.0

class SensingMasks(object):
    """
    Class to generate boolean sensing masks for all agents, and a boolean global mask for 
    backpropagation to train only on entries known by at least one agent.

    Args:
        TemporalData (torch.utils.data.Dataset): Dataset of ground truth matrices.
        rank (int): Rank of the matrices.
        num_agents (int): Number of agents.
        density (float): Target density of global known entries.
        
    Attributes:
        num_matrices (int): Number of matrices in the dataset.
        num_agents (int): Number of agents.
        density (float): Target density of global known entries.
        t (int): Time dimension of the matrices.
        m (int): Feature dimension of the matrices.
        r (int): Rank of the matrices.
        total_entries (int): Total number of entries in the matrices.
        stats (dict): Statistics about the generated masks.
        masks (torch.Tensor): Boolean masks for each agent.
        global_known (torch.Tensor): Boolean mask for global known entries.
    
    Methods:
        __init__(): Initialize object of class SensingMasks.
        __call__(torch.Tensor, global_mask: bool): Apply local or global masks to a tensor.
        __getitem__(int): Index a specific agent's mask. -1 returns the global mask.
        _generate(): Generate masks for SensingMasks initialization.
        _sample_global_known_idx(): Sample indices of global known entries.
        _robust_sample_global_known_idx(): Robustly sample indices of global known entries.
        _agent_samplesizes(): Draw sample sizes for each agent.
        _build_agent_masks(): Build masks for each agent.
        _summary(): Print summary of the generated masks.
    """
    def __init__(self, TemporalData, rank, num_agents, density, *, future_only: bool = True):
        self.num_matrices = len(TemporalData)
        self.num_agents = num_agents
        self.density = density
        self.t, self.m = TemporalData.t, TemporalData.m
        self.r = rank
        self.total_entries = self.t * self.m
        self.stats = {
            "agent_overlap": [], "agent_endowments": [],
            "actual_knowns": [], "oversample_flags": 0
        }
        self.future_only = future_only
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
                    # create stacked vectors of Xs, so result is num_agents x total_entries,
                    # then apply masks
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

        self.stats["agent_endowments"] = samplesizes
        self.stats["oversample_flags"] = oversampled
        self.stats["actual_knowns"] = global_actual_known.sum()
        self.__summary(masks)
        
        return masks, global_actual_known

    def _sample_global_known_idx(self):
        global_known_count = max(int(self.density * self.total_entries), 1)
        all_indices = torch.randperm(self.total_entries)
        global_known_indices = all_indices[:global_known_count]
        global_mask = torch.zeros(self.total_entries, dtype=torch.bool)
        global_mask[global_known_indices] = True
        if self.future_only:
            # Hide the entire last row
            global_mask[-self.m:] = False
            keep = global_known_indices < self.total_entries - self.m
            global_known_indices = global_known_indices[keep]
        return global_mask, global_known_indices

    def _robust_sample_global_known_idx(self, max_attempts: int = 20, verbose: bool = True):
        if self.r <= min(self.m, self.t) // 2:
            for attempt in range(1, max_attempts + 1):
                global_mask, global_known_idx = self._sample_global_known_idx()
                mask_2d = global_mask.view(self.t, self.m)
                rows_ok = (mask_2d[:-1, :].sum(dim=1) >= self.r).all()
                cols_ok = (mask_2d.sum(dim=0) >= self.r).all()
                if rows_ok and cols_ok:
                    return global_known_idx
                if verbose and attempt < max_attempts:
                    print(f"Warning: Retrying sampling (attempt {attempt}) due to sparse rows/cols.")
            raise RuntimeError(
                f"Failed to sample a valid mask after {max_attempts} attempts. "
                f"Could not ensure at least {self.r} known entries in each row and column. "
                f"Density: {self.density}, size: {self.t}x{self.m}"
            )
        else:
            _, global_known_idx = self._sample_global_known_idx()
            print("Warning: matrices are not low-rank. r is high compared to t or m.")
            return global_known_idx
            
    def _agent_samplesizes(self, num_global_known: int):
        base = max(1, num_global_known // self.num_agents)
        low = min(int(2.0 * base), num_global_known)
        high = min(int(4.0 * base), num_global_known)
        samplesizes = np.random.randint(low, high, size=self.num_agents) if high > low else np.full(self.num_agents, low)
        oversampled = np.sum(samplesizes > num_global_known)
        samplesizes = np.minimum(samplesizes, num_global_known)
        return samplesizes, oversampled
    
    def _build_agent_masks(self, global_known_idx):
        num_global_known = len(global_known_idx)
        masks = torch.zeros((self.num_agents, self.total_entries), dtype=torch.bool)
        samplesizes, oversampled_total = self._agent_samplesizes(num_global_known)
        for i in range(self.num_agents):
            sample_idx = np.random.permutation(global_known_idx)[:samplesizes[i]]
            masks[i, sample_idx] = True
            
        return masks, torch.tensor(samplesizes), oversampled_total  

    def __summary(self, masks):
        self.stats["agent_overlap"] = _compute_avg_overlap(masks) if self.num_agents > 1 else -np.inf
        endowment_mean = np.mean(self.stats["agent_endowments"].tolist()) 
        actual_known_mean = np.mean(self.stats["actual_knowns"].tolist())
        print("--------------------------")
        print(f"Target density: {self.density:.2f}.")
        print(f"Target known entries: {self.total_entries * self.density:.1f}, Mean known entries: {actual_known_mean:.1f}")
        print(f"{self.num_agents} agents, Avg overlap: {self.stats['agent_overlap']:.3f}")
        print(f"Avg entries per agent: {endowment_mean:.1f}")
        if self.stats["oversample_flags"] > 0:
            print(f"WARNING ⚠️  {self.stats['oversample_flags']} matrices had agents sampling all known entries.")
        print("--------------------------")




#=========#
"""
NUM_MATRICES = 20
NUM_AGENTS = 128
T = 100
M = 25
R = 25

groundtruth = GTMatrices(NUM_MATRICES, T, M, R)
groundtruth[:NUM_MATRICES]

trainingdata = TemporalData(groundtruth)
sensingmasks = SensingMasks(trainingdata, R, NUM_AGENTS, 0.05)

example = trainingdata[0]['matrix']
test = sensingmasks(example, global_mask=True)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(9.5, 6), gridspec_kw={
        'width_ratios': [3, 3, 1.5]
    })
axs[0].imshow(example.reshape(T, M))
axs[0].set_title("Ground Truth Matrix")
axs[0].set_xticks([])
axs[0].set_xlabel("Features")
axs[0].set_ylabel("Time")
axs[1].imshow(test.reshape(T, M))
axs[1].set_title("Info Available to Agents")
axs[1].set_xlabel("Features")
axs[1].set_xticks([])
axs[1].set_ylabel("Time")
axs[2].imshow(groundtruth.U[0], aspect = 0.25)
axs[2].set_title("Latent dynamics")
axs[2].set_xlabel("Factors")
axs[2].set_xticks([])
axs[2].set_ylabel("Time")
fig.suptitle(
    f"Example data point: {T} by {M} matrix of rank {R}, Sampling = {sensingmasks.density}", fontsize=14)
fig.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
for col in range(M):
    ax2.plot(example.reshape(T, M)[:, col])
plt.show()

from torch.utils.data import DataLoader

train_loader = DataLoader(
        trainingdata, batch_size=20, shuffle=True,
        pin_memory=torch.cuda.is_available(), 
    )

print("...")
"""