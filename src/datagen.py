
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


def define_agent_samplesize(num_agents:int, num_global_known:int) -> tuple[int, int]:
    oversampled = 0
    base = max(1, num_global_known // num_agents)
    low = min(int(2.0 * base), num_global_known)
    high = min(int(4.0 * base), num_global_known)
    if high <= low:
        samplesize = low
    else:
        samplesize = np.random.randint(low, high)
    if samplesize > num_global_known:
        oversampled = 1
    samplesize = min(samplesize, num_global_known)
    return samplesize, oversampled


def build_agent_views(num_agents, global_known_idx, observed, 
                      total_entries, mode='uniform'):
    views = []
    agent_endowments = []
    oversampled = 0
    num_global_known = len(global_known_idx)

    for _ in range(num_agents):
        view = torch.zeros(total_entries, dtype=torch.float32)

        if mode == 'all-see-all':
            view[global_known_idx] = observed[global_known_idx]
            endowment = len(global_known_idx)

        elif mode == 'uniform':
            if len(global_known_idx) == 0:
                sample_size = 0
                sample_idx = []
            else:
                sample_size, agent_oversampled = define_agent_samplesize(num_agents, num_global_known)
                oversampled += agent_oversampled
                sample_idx = global_known_idx[
                    torch.randint(num_global_known, (sample_size,))
                ]
            view = torch.zeros(total_entries, dtype=torch.float32)
            if sample_size > 0:
                view[sample_idx] = observed[sample_idx]
            endowment = sample_size
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")

        views.append(view)
        agent_endowments.append(endowment)

    return torch.stack(views), agent_endowments, oversampled


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
                 num_matrices:int, 
                 n:int=20, m:int=20, r:int=4, 
                 num_agents:int=30, 
                 agentdistrib:str='uniform',
                 density:float=0.2, 
                 sigma:float=0.0, 
                 verbose:bool=True):
        self.num_matrices = num_matrices
        self.n, self.m, self.r = n, m, r
        self.input_dim = self.total_entries = n * m
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.constant_global_known_idx = None
        self.constant_global_mask = None
        self.agentdistrib = agentdistrib
        self.verbose = verbose
        super().__init__('.')
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        stats = {
            "nuclear_norms": [],
            "gaps": [],
            "variances": [],
            "agent_overlaps": [],
            "agent_endowments": [],
            "actual_knowns": [],
            "oversample_flags": 0,
        }

        for idx in range(self.num_matrices):
            M = generate_low_rank_matrix(self.n, self.m, self.r, self.sigma)
            M_vec = M.view(-1)
            S = torch.linalg.svdvals(M)

            stats["nuclear_norms"].append(torch.linalg.norm(M, ord='nuc').item())
            stats["gaps"].append(S[self.r - 1] - S[self.r])
            stats["variances"].append(M_vec.var().item())
            
            global_mask = torch.zeros(self.total_entries, dtype=torch.bool)
            known_idx = []

            max_attempts = 10
            for attempt in range(1, max_attempts + 1):
                global_mask, known_idx = sample_global_known_entries(self.n, self.m, self.density)
                mask_2d = global_mask.view(self.n, self.m)
                rows_ok = (mask_2d.sum(dim=1) >= self.r).all()
                cols_ok = (mask_2d.sum(dim=0) >= self.r).all()
                if rows_ok and cols_ok:
                    break
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Failed to sample a valid mask after {max_attempts} attempts. "
                        f"Could not ensure at least {self.r} known entries in each row and column. "
                        f"Matrix index: {idx}, density: {self.density}, size: {self.n}x{self.m}"
                    )
                if self.verbose:
                    print(f"Warning: Retrying sampling for matrix {idx} (attempt {attempt}) due to sparse rows/cols.")
            
            observed = M_vec.clone()
            observed[~global_mask] = 0.0

            if self.num_agents == 1 or self.agentdistrib == 'all-see-all':
                agent_views, endowments, _ = build_agent_views(
                    self.num_agents, known_idx, observed, self.total_entries, mode='all-see-all'
                )
            else:
                agent_views, endowments, oversampled = build_agent_views(
                    self.num_agents, known_idx, observed, self.total_entries, mode='uniform'
                )
                if oversampled > 0:
                    stats["oversample_flags"] += 1
                    
            assert all(e <= len(known_idx) for e in endowments), "Agent oversampling cap failed."
            mask_tensor = (agent_views != 0).any(dim=0)
            stats["actual_knowns"].append(mask_tensor.sum().item())
            stats["agent_endowments"].extend(endowments)

            if self.num_agents > 1:
                avg_overlap = compute_avg_agent_overlap(agent_views)
                stats["agent_overlaps"].append(avg_overlap)

            data = Data(x=agent_views, y=M_vec, mask=mask_tensor)
            data_list.append(data)

        # Save summary statistics
        self.nuclear_norm_mean = np.mean(stats["nuclear_norms"])
        self.gap_mean = np.mean(stats["gaps"])
        self.variance_mean = np.mean(stats["variances"])
        self.agent_overlap_mean = (
            np.mean(stats["agent_overlaps"]) if stats["agent_overlaps"] else -np.inf
        )
        self.agent_endowment_mean = np.mean(stats["agent_endowments"])
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

