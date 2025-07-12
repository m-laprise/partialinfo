import numpy as np
import torch
from torch.utils.data import Dataset


class GTMatrices(Dataset):
    def __init__(self, num_matrices: int, 
                 t: int, m: int, r: int, 
                 sigma: float = 0.0):
        super().__init__()
        self.t, self.m, self.r = t, m, r
        self.sigma = sigma
        self.U, self.V = self._generate_factors(num_matrices)
        self.scale = np.sqrt(r)

    def __len__(self):
        return self.U.shape[0]
    
    def _generate_factors(self, num_matrices: int):
        U = torch.rand(num_matrices, self.t, self.r, dtype=torch.float32) * 2
        V = torch.rand(num_matrices, self.m, self.r, dtype=torch.float32) * 2
        return U, V

    def generate_matrices(self, idx = None):
        idx = np.arange(self.U.shape[0]) if idx is None else idx
        M = torch.einsum('...ij,...kj->...ik', self.U[idx], self.V[idx])
        M += self.sigma * torch.randn(M.shape, dtype=torch.float32)
        M /= self.scale
        return M 
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.generate_matrices(idx)


class TemporalData(Dataset):
    def __init__(self, matrices: GTMatrices):
        super().__init__()
        self.data = matrices[:len(matrices)]
        self.t, self.m, self.r = matrices.t, matrices.m, matrices.r
        self.stats = self.__summary()

    def __len__(self):
        return self.data.shape[0]
    
    def _generate_label(self, matrix):
        # Label is the index of the maximum entry in the last row
        label = torch.argmax(matrix[-1, :])
        return label

    def __getitem__(self, idx: int):
        _, t, m = self.data.shape
        matrix = self.data[idx, :, :]
        sample = {
            'matrix': matrix.view(1, t*m),
            'label': self._generate_label(matrix)
        }
        
        return sample
    
    def __summary(self):
        # TO DEBUG
        S = torch.linalg.svdvals(self.data)
        gap = np.mean((S[:, self.r - 1] - S[:, self.r]).tolist())
        nuc = np.mean(S.sum(dim=1).tolist())
        var = np.mean(self.data.var(dim=1).tolist())

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
    def __init__(self, TemporalData, rank, num_agents, density):
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
        return global_mask, global_known_indices

    def _robust_sample_global_known_idx(self, max_attempts: int = 20, verbose: bool = True):
        for attempt in range(1, max_attempts + 1):
            global_mask, global_known_idx = self._sample_global_known_idx()
            mask_2d = global_mask.view(self.t, self.m)
            rows_ok = (mask_2d.sum(dim=1) >= 3).all()
            cols_ok = (mask_2d.sum(dim=0) >= 3).all()
            if rows_ok and cols_ok:
                return global_known_idx
            if verbose and attempt < max_attempts:
                print(f"Warning: Retrying sampling (attempt {attempt}) due to sparse rows/cols.")
        raise RuntimeError(
            f"Failed to sample a valid mask after {max_attempts} attempts. "
            f"Could not ensure at least {self.r} known entries in each row and column. "
            f"Density: {self.density}, size: {self.t}x{self.m}"
        )

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
NUM_MATRICES = 1000
NUM_AGENTS = 125
T = 100
M = 50
R = 5

groundtruth = GTMatrices(NUM_MATRICES, T, M, R)
groundtruth[:NUM_MATRICES]

trainingdata = TemporalData(groundtruth)
sensingmasks = SensingMasks(trainingdata, R, NUM_AGENTS, 0.5)

sensingmasks(trainingdata[0]['matrix'], global_mask=True)

from torch.utils.data import DataLoader

train_loader = DataLoader(
        trainingdata, batch_size=20, shuffle=True,
        pin_memory=torch.cuda.is_available(), 
    )

print("...")
"""