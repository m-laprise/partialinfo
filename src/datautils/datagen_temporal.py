
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
    # Vectorized Toeplitz-style construction using pairwise distances
    idx = torch.arange(t, dtype=torch.get_default_dtype())
    # Absolute pairwise lag matrix, with optional scaling (historical /4 kept)
    h = (idx[:, None] - idx[None, :]).abs() / 4
    corr = (1 + h.pow(alpha)).pow(-beta / alpha)
    vcov = sigma2 * corr
    # Add small diagonal term for numerical stability
    vcov = vcov + (1e-8) * torch.eye(t, dtype=vcov.dtype, device=vcov.device)
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
def _generate_U(vcovsU: torch.Tensor, *, offset: float = 0.0):
    r, t, _ = vcovsU.shape
    # Build a robust Cholesky with adaptive jitter to ensure PD
    def _stable_cholesky(S: torch.Tensor, init_jitter: float = 1e-6, max_tries: int = 7) -> torch.Tensor:
        I = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        jitter = init_jitter
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(S + jitter * I)
            except RuntimeError:
                jitter *= 10.0
        # Last resort: symmetrize and add strong jitter
        S_sym = 0.5 * (S + S.transpose(-1, -2))
        return torch.linalg.cholesky(S_sym + jitter * I)

    scale_tril = _stable_cholesky(vcovsU)
    # Batched multivariate normal using scale_tril (more numerically stable)
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.full((r, t), offset, dtype=vcovsU.dtype, device=vcovsU.device),
        scale_tril=scale_tril,
    )
    U_rt = mvn.rsample()  # shape: (r, t)
    return U_rt.T  # shape: (t, r)


@torch.no_grad()
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
                 U_only: bool = False,
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
        self.U_only = U_only
        if self.U_only:
            if r < m:
                print(f"WARNING: U_only option is activated but rank {r} was requested.",
                    "Full rank matrices will be returned.")
                r = m
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
        
        if self.U_only:
            U = self.U[idx]
            for i in range(U.shape[0]):
                U[i] /= U[i].std(dim=0)
            return U
        
        M = torch.einsum('...ij,...kj->...ik', self.U[idx], self.V[idx])
        #M += self.sigma * torch.randn(M.shape, dtype=torch.float32)
        if not self.structured:
            M /= np.sqrt(self.r)
            return M / M.std(dim=1, keepdim=True)
        elif self.mode == 'return':
            M_r = _fin_return(M[:, :-1, :], M[:, 1:, :])
            return M_r / M_r.std(dim=1, keepdim=True)
        else:
            for i in range(M.shape[0]):
                M[i] /= M[i].std(dim=0)
            return M
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.generate_matrices(idx)


""" class RandomLinearHead(nn.Module):
    @torch.no_grad()
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim, bias=True),
            nn.SiLU(),
            nn.Linear(2 * self.in_dim, self.out_dim, bias=False)
        ) 
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.mlp(x) / 3.0) * 3.0
 """    

class TemporalData(Dataset):
    """
    Takes ground truth matrices and a task statement, and generates labeled data that can
    be provided to a DataLoader.
    """
    def __init__(self, matrices: GTMatrices, task: str = 'argmax', *, 
                 target_source: str = 'observed', verbose=True, rho_out: float = 0.4):
        super().__init__()
        if task not in ['argmax', 'lastrow', 'nextrow', 'nonlin_function', 'lrmc']:
            raise NotImplementedError(f"Task {task} not implemented")
        
        # Keep a reference to GT to optionally build targets from latent factors
        #self.gt = matrices
        self.data = matrices[:len(matrices)]
        self.t, self.m, self.r = matrices.t, matrices.m, matrices.r
        self.task = task
        self.u_only = bool(getattr(matrices, 'U_only', False))
        self.rho_out = float(rho_out)
        # Setup for nonlin_function task: fixed random weights over all columns (size m)
        if self.task == 'nonlin_function':
            self.W_out = torch.randn(self.m, dtype=torch.get_default_dtype())
            # Build a fixed per-row selection mask with fraction rho_out of columns
            k = int(round(self.rho_out * self.m))
            k = max(1, min(k, self.m))
            sel_mask = torch.zeros(self.t, self.m, dtype=torch.bool)
            cols = torch.randperm(self.m)[:k]
            sel_mask[:, cols] = True
            self._row_sel_mask = sel_mask
        
        self.verbose = verbose
        self.S = torch.linalg.svdvals(self.data)
        if self.r < min(self.t, self.m):
            self.gap = np.mean((self.S[:, self.r - 1] - self.S[:, self.r]).tolist())
        else:
            self.gap = 0.0
        self.nuc = np.mean(self.S.sum(dim=1).tolist())
        self.var = np.mean(self.data.var(dim=1).tolist())
        self.stats = self.__summary()

    def __len__(self):
        return self.data.shape[0]
    
    def _generate_label(self, matrix): 
        if self.task == 'argmax':
            # Label is the index of the maximum entry in the last row
            label = torch.argmax(matrix[-1, :])
        elif self.task == 'lastrow':
            # Label is the last row 
            label = matrix[-1, :]
        elif self.task == 'nextrow':
            label = matrix[1:, :]
        elif self.task == 'nonlin_function':
            # Produce a time series label y ∈ R^{T×1} with one-step shift:
            # y[t] = tanh(W_out · matrix[t-1, :]) for t >= 1, and y[0] = 0.
            t, m = matrix.shape
            y = torch.zeros(t, dtype=matrix.dtype, device=matrix.device)
            if t > 1:
                prev_rows = matrix[:-1, :]          # [T-1, M]
                # Apply per-row selection mask on previous rows
                row_mask = self._row_sel_mask[:-1, :].to(device=matrix.device)
                prev_rows_masked = prev_rows * row_mask.to(dtype=prev_rows.dtype)
                w = self.W_out.to(device=matrix.device, dtype=prev_rows.dtype)
                # weighted sum across selected columns -> [T-1]
                z = prev_rows_masked @ w
                # Center and normalize z to unit variance to control tanh saturation
                z = z - z.mean()
                z_std = z.std()
                if torch.isfinite(z_std) and z_std > 0:
                    z_norm = z / (z_std + 1e-8)
                    # Find a gain g such that Var[tanh(g * z_norm)] ≈ target_var
                    # Note: due to tanh bounds, variance cannot exceed 1.
                    target_var = 0.9  # "close to 1" without hard saturation
                    # Bisection over g in [g_low, g_high]
                    g_low, g_high = 0.1, 10.0
                    def tanh_var(g: float) -> float:
                        return torch.tanh(g * z_norm).var(unbiased=False).item()
                    v_low = tanh_var(g_low)
                    v_high = tanh_var(g_high)
                    # Ensure target is within reachable range; adjust if necessary
                    target = min(max(target_var, v_low), v_high)
                    g = g_high
                    # Quick bisection (fixed iters)
                    for _ in range(12):
                        g_mid = 0.5 * (g_low + g_high)
                        v_mid = tanh_var(g_mid)
                        if v_mid < target:
                            g_low = g_mid
                        else:
                            g_high = g_mid
                        g = g_mid
                    # Reduce preactivations to avoid saturation: halve the effective gain
                    eff = (g * z_norm) / 2.0
                    # Also normalize by number of selected columns per row: 1/sqrt(m*rho_out*5)
                    k_sel = max(1, int(round(self.rho_out * self.m)))
                    col_scale = 1.0 / math.sqrt(k_sel * 2.0)
                    y[1:] = torch.tanh(eff * col_scale)
                else:
                    # Degenerate case: keep zeros (already initialized)
                    pass
            label = y.view(t, 1)
        return label

    def __getitem__(self, idx: int):
        _, t, m = self.data.shape
        matrix = self.data[idx, :, :]
        if self.task == 'lrmc':
            sample = {
                'matrix': matrix.view(1, t*m),
                'label': matrix.view(t*m)
            }
        else:
            sample = {
                'matrix': matrix.view(1, t*m),
                'label': self._generate_label(matrix) 
            }
        return sample
    
    def __summary(self):
        if self.verbose:
            print("--------------------------")
            print(f"Dataset of {len(self)} labelled rank-{self.r} matrices of size {self.t}x{self.m}")
            print(f"Mean nuclear norm: {self.nuc:.4f}, Spectral gap: {self.gap:.4f}")
            print(f"Mean variance of entries: {self.var:.4f}")
            print(f"Total entries: {self.t * self.m}")
            print("--------------------------")
        return { "nuclear_norms": self.nuc, "gaps": self.gap, "variances": self.var }

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