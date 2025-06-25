
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class AgentMatrixReconstructionDataset(InMemoryDataset):
    def __init__(self, num_graphs=1000, n=20, m=20, r=4, 
                 num_agents=30, density=0.2, sigma=0, verbose=True):
        self.num_graphs = num_graphs
        self.n = n
        self.m = m
        self.r = r
        self.density = density
        self.sigma = sigma
        self.num_agents = num_agents
        self.verbose = verbose
        self.input_dim = n * m
        self.nuclear_norm_mean = 0.0
        self.gap_mean = 0.0
        self.variance_mean = 0.0
        self.agent_overlap_mean = 0.0
        self.agent_endowment_mean = 0.0
        self.actual_known_mean = 0.0
        super().__init__('.')
        self.data, self.slices = self._generate()
    
    def _generate(self):
        data_list = []
        norms = []
        gaps = []
        variances = []
        agent_overlaps = []
        agent_endowments = []
        actual_knowns = []
        total_entries = self.n * self.m
        target_known = int(self.density * total_entries)
        for _ in range(self.num_graphs):
            # Low-rank matrix with Gaussian noise
            U = np.random.randn(self.n, self.r) / np.sqrt(self.r)
            V = np.random.randn(self.m, self.r) / np.sqrt(self.r)
            M = U @ V.T + self.sigma * np.random.randn(self.n, self.m)
            M_tensor = torch.tensor(M, dtype=torch.float32).view(-1)
            norms.append(torch.linalg.norm(M_tensor.view(self.n, self.m), ord='nuc').item())
            S = torch.linalg.svdvals(M_tensor.view(self.n, self.m))
            gaps.append(S[self.r - 1] - S[self.r])
            variances.append(torch.var(M_tensor).item())
            # Controlled global known vs secret mask
            all_indices = torch.randperm(total_entries)
            known_global_idx = all_indices[:target_known]
            global_mask = torch.zeros(total_entries, dtype=torch.bool)
            global_mask[known_global_idx] = True
            # Build observed tensor with zeros at unknowns
            observed_tensor = M_tensor.clone()
            observed_tensor[~global_mask] = 0.0
            # Agent-specific views
            features = []
            for i in range(self.num_agents):
                # Each agent samples (with replacement) a variable-sized subset 
                # of the global known entries
                # control expected agent overlap by tweaking agent_sample_size range
                agent_sample_size = np.random.randint(
                    int(2.0 * (target_known // self.num_agents)),
                    int(4.0 * (target_known // self.num_agents)) 
                )
                if agent_sample_size > target_known:
                    agent_sample_size = target_known
                    print(f"Warning: agent {i} sampled all known entries.")
                agent_endowments.append(agent_sample_size)
                sample_idx = known_global_idx[
                    torch.randint(len(known_global_idx), (agent_sample_size,))
                ]
                agent_view = torch.zeros(total_entries, dtype=torch.float32)
                agent_view[sample_idx] = observed_tensor[sample_idx]
                features.append(agent_view)
            x = torch.stack(features)
            # Create mask tensor from agent views reflecting entries actually seen by any agent
            mask_tensor = (x != 0).sum(dim=0) > 0
            
            overlap_matrix = (torch.stack(features) > 0).float() @ (torch.stack(features) > 0).float().T
            overlap_matrix /= overlap_matrix.diagonal().view(-1, 1)  # normalize
            avg_overlap = (overlap_matrix.sum() - overlap_matrix.trace()) / (self.num_agents * (self.num_agents - 1))
            agent_overlaps.append(avg_overlap)
            # Count how many entries known by any agent, avoid double counting
            actual_known = mask_tensor.sum().item()
            actual_knowns.append(actual_known)
            # Create Data object
            data = Data(x=x, y=M_tensor, mask=mask_tensor, nb_known_entries=actual_known)
            data_list.append(data)
        self.nuclear_norm_mean = np.mean(norms)
        self.gap_mean = np.mean(gaps)
        self.variance_mean = np.mean(variances)
        self.agent_overlap_mean = np.mean(agent_overlaps)
        self.agent_endowment_mean = np.mean(agent_endowments)
        self.actual_known_mean = np.mean(actual_knowns)
        output = self.collate(data_list)
        if self.verbose:
            print(f"Generated {self.num_graphs} rank-{self.r} matrices of size {self.n}x{self.m} " +
                f"with mean nuclear norm {self.nuclear_norm_mean:.4f} and mean gap {self.gap_mean:.4f}.")
            print(f"Global observed density {self.density:.4f} and noise level {self.sigma:.4f}.")
            print(f"Total entries {total_entries}; target known entries {target_known}; mean actual known entries {self.actual_known_mean}.")
            print(f"Entries distributed among {self.num_agents} agents with mean overlap {self.agent_overlap_mean:.4f}.")
            print(f"Average number of entries per agent: {self.agent_endowment_mean:.4f}.")
        return output
