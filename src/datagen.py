
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


def generate_low_rank_matrix(n, m, r, sigma=0.0):
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


def sample_known_entries(n, m, density):
    total_entries = n * m
    known_count = max(int(density * total_entries), 1)
    all_indices = torch.randperm(total_entries)
    known_indices = all_indices[:known_count]
    mask = torch.zeros(total_entries, dtype=torch.bool)
    mask[known_indices] = True
    return mask, known_indices


def build_agent_views(num_agents, known_indices, observed, 
                      total_entries, mode='uniform'):
    views = []
    agent_endowments = []
    oversampled = 0

    for _ in range(num_agents):
        view = torch.zeros(total_entries, dtype=torch.float32)

        if mode == 'all-see-all':
            view[known_indices] = observed[known_indices]
            endowment = len(known_indices)

        elif mode == 'uniform':
            if len(known_indices) == 0:
                sample_size = 0
                sample_idx = []
            else:
                base = max(1, len(known_indices) // num_agents)
                low = min(int(2.0 * base), len(known_indices))
                high = min(int(4.0 * base), len(known_indices))
                if high <= low:
                    sample_size = low  
                else:
                    sample_size = np.random.randint(low, high)
                if sample_size > len(known_indices):
                    oversampled += 1
                sample_size = min(sample_size, len(known_indices))
                sample_idx = known_indices[
                    torch.randint(len(known_indices), (sample_size,))
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
    def __init__(self, num_matrices, n=20, m=20, r=4, 
                 num_agents=30, agentdistrib='uniform',
                 density=0.2, sigma=0.0, verbose=True):
        self.num_matrices = num_matrices
        self.n, self.m, self.r = n, m, r
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.agentdistrib = agentdistrib
        self.verbose = verbose
        self.input_dim = n * m
        super().__init__('.')
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        total_entries = self.n * self.m

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

            global_mask, known_idx = sample_known_entries(self.n, self.m, self.density)
            observed = M_vec.clone()
            observed[~global_mask] = 0.0

            if self.num_agents == 1 or self.agentdistrib == 'all-see-all':
                agent_views, endowments, _ = build_agent_views(
                    self.num_agents, known_idx, observed, total_entries, mode='all-see-all'
                )
            else:
                agent_views, endowments, oversampled = build_agent_views(
                    self.num_agents, known_idx, observed, total_entries, mode='uniform'
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
            self._print_summary(total_entries, stats["oversample_flags"])

        return self.collate(data_list)

    def _print_summary(self, total_entries, oversampled):
        print("--------------------------")
        print(f"Generated {self.num_matrices} rank-{self.r} matrices of size {self.n}x{self.m}")
        print(f"Mean nuclear norm: {self.nuclear_norm_mean:.4f}, Spectral gap: {self.gap_mean:.4f}")
        print(f"Observed density: {self.density:.2f}, Noise level: {self.sigma:.4f}")
        print(f"Total entries: {total_entries}, Target known entries: {total_entries * self.density:.1f}, Mean known entries: {self.actual_known_mean:.1f}")
        print(f"{self.num_agents} agents, Avg overlap: {self.agent_overlap_mean:.3f}")
        print(f"Avg entries per agent: {self.agent_endowment_mean:.1f}")
        if oversampled > 0:
            print(f"WARNING ⚠️  {oversampled} matrices had agents sampling all known entries.")
        print("--------------------------")

