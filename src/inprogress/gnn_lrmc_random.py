import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregator
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # linear transform and propagate
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class GNN_LRMC(torch.nn.Module):
    def __init__(self, num_rows, num_cols, embed_dim=32, num_layers=3):
        super().__init__()
        self.row_embed = torch.nn.Embedding(num_rows, embed_dim)
        self.col_embed = torch.nn.Embedding(num_cols, embed_dim)
        self.layers = torch.nn.ModuleList([
            GNNLayer(embed_dim, embed_dim) for _ in range(num_layers)
        ])

    def forward(self, edge_index, edge_weight):
        # Node embedding tensor: first rows, then columns
        row_x = self.row_embed.weight               # shape [n, d]
        col_x = self.col_embed.weight               # shape [m, d]
        x = torch.cat([row_x, col_x], dim=0)       # [n + m, d]

        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)

        n = self.row_embed.num_embeddings
        return x[:n], x[n:]

    def predict(self, row_ids, col_ids, row_emb, col_emb):
        return (row_emb[row_ids] * col_emb[col_ids]).sum(dim=-1)
    
def generate_low_rank(n, m, r, density=0.2, sigma=0.1):
    U = np.random.randn(n, r)
    V = np.random.randn(m, r)
    M = U @ V.T + sigma * np.random.randn(n, m)
    mask = np.random.rand(n, m) < density
    return M, mask

def train_demo():
    # Config
    n, m, r = 100, 80, 5
    density = 0.2
    embed_dim = 32
    num_layers = 3
    lr = 1e-3
    epochs = 200

    # 1. Generate low-rank matrix with noise and observation mask
    U = np.random.randn(n, r)
    V = np.random.randn(m, r)
    M = U @ V.T + 0.1 * np.random.randn(n, m)
    mask = np.random.rand(n, m) < density
    obs = np.argwhere(mask)
    vals = M[mask].astype(np.float32)

    # 2. Build bipartite graph: rows [0..n-1] and cols [n..n+m-1]
    adj = sp.coo_matrix((vals, (obs[:,0], obs[:,1])), shape=(n, m))
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index[1] += n  # offset column indices

    data = Data(edge_index=edge_index, edge_weight=edge_weight)

    # 3. Initialize model and optimizer
    model = GNN_LRMC(n, m, embed_dim, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Full-batch GNN training
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        row_emb, col_emb = model(data.edge_index, data.edge_weight)
        preds = model.predict(
            data.edge_index[0].long(),
            data.edge_index[1].long() - n,
            row_emb,
            col_emb
        )
        loss = F.mse_loss(preds, data.edge_weight)
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} — train MSE: {loss.item():.4f}")

    # 5. Evaluate on missing entries
    with torch.no_grad():
        full_est = row_emb @ col_emb.T
        M_true = torch.tensor(M, dtype=full_est.dtype)
        missing_mask = ~torch.tensor(mask)
        mse_missing = F.mse_loss(full_est[missing_mask], M_true[missing_mask])
        rmse_missing = torch.sqrt(mse_missing)
        print(f"Missing-entry RMSE: {rmse_missing.item():.4f}")

if __name__ == "__main__":
    train_demo()
    