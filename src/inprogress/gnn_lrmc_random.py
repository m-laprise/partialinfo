import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregator
        self.lin = torch.nn.Linear(in_channels, out_channels)
        #torch.nn.init.xavier_uniform_(self.lin.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.kaiming_uniform_(self.lin.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.lin.bias)


    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class GNN_LRMC(torch.nn.Module):
    def __init__(self, num_rows, num_cols, embed_dim=32, num_layers=3):
        super().__init__()
        self.row_embed = torch.nn.Embedding(num_rows, embed_dim)
        self.col_embed = torch.nn.Embedding(num_cols, embed_dim)
        bound = 1.0 / np.sqrt(embed_dim)
        torch.nn.init.uniform_(self.row_embed.weight, -bound, bound)
        torch.nn.init.uniform_(self.col_embed.weight, -bound, bound)

        self.layers = torch.nn.ModuleList([
            GNNLayer(embed_dim, embed_dim) for _ in range(num_layers)
        ])

    def forward(self, edge_index):
        N = self.row_embed.num_embeddings
        M = self.col_embed.num_embeddings
        device = edge_index.device

        # Build properly tracked features
        row_x = self.row_embed(torch.arange(N, device=device))
        col_x = self.col_embed(torch.arange(M, device=device))
        x = torch.cat([row_x, col_x], dim=0)

        # Add self-loops for both halves to preserve parameter flow
        edge_index_with_self_loops = add_self_loops(edge_index, num_nodes=N + M)[0]

        # Message passing
        for layer in self.layers:
            x = F.relu(layer(x, edge_index_with_self_loops))

        # Return row & column embeddings
        return x[:N], x[N:]

    def predict(self, row_ids, col_ids, row_emb, col_emb):
        return (row_emb[row_ids] * col_emb[col_ids]).sum(dim=-1)
    
def generate_low_rank(n, m, r, density=0.2, sigma=0.01):
    U = np.random.randn(n, r) / np.sqrt(r)
    V = np.random.randn(m, r) / np.sqrt(r)
    M = U @ V.T + sigma * np.random.randn(n, m)
    mask = np.random.rand(n, m) < density
    return M, mask

def train_demo():
    torch.manual_seed(42)
    np.random.seed(42)
    
    n, m, r = 80, 64, 2
    density, embed_dim, num_layers = 0.5, 32, 2
    lr, epochs = 1e-3, 300

    # Generate synthetic low-rank matrix
    M, mask = generate_low_rank(n, m, r, density=density, sigma=0.1)
    obs = np.argwhere(mask)
    vals = M[mask].astype(np.float32)

    # Build graph
    adj = sp.coo_matrix((vals, (obs[:, 0], obs[:, 1])), shape=(n, m))
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index[1] += n

    data = Data(edge_index=edge_index, edge_weight=edge_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN_LRMC(n, m, embed_dim, num_layers).to(device)
    data = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float('inf')
    best_state = None
    mask_tensor = torch.tensor(mask, device=device)
    M_true = torch.tensor(M, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        row_emb, col_emb = model(data.edge_index)
        ri = data.edge_index[0]
        ci = data.edge_index[1] - n
        preds = model.predict(ri, ci, row_emb, col_emb)
        loss = F.mse_loss(preds, data.edge_weight)
        loss.backward()

        # Gradient flow diagnostics
        #if epoch % 100 == 0 or epoch == 1:
        #    for name, param in model.named_parameters():
        #        if param.grad is not None:
        #            print(f"Grad [{name}]: {param.grad.norm().item():.4f}")

        opt.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            full_mat = row_emb @ col_emb.T
            mse_obs = F.mse_loss(full_mat[mask_tensor], M_true[mask_tensor])
            mse_all = F.mse_loss(full_mat, M_true)
            rmse_obs = torch.sqrt(mse_obs)
            rmse_all = torch.sqrt(mse_all)

        if rmse_all < best_rmse:
            best_rmse = rmse_all.item()
            best_state = model.state_dict()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE: {loss.item():.4f} | "
                  f"RMSE_obs: {rmse_obs:.4f}, RMSE_all: {rmse_all:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n🎯 Best checkpoint restored: RMSE_all = {best_rmse:.4f}")

    with torch.no_grad():
        row_emb, col_emb = model(edge_index)
        full_mat = row_emb @ col_emb.T
        mse_obs = F.mse_loss(full_mat[mask_tensor], M_true[mask_tensor])
        mse_all = F.mse_loss(full_mat, M_true)
        print(f"Final Metrics — RMSE_obs: {torch.sqrt(mse_obs):.4f}, RMSE_all: {torch.sqrt(mse_all):.4f}")


def matrix_to_graph(M, mask):
    n, m = M.shape
    obs = np.argwhere(mask)
    vals = M[mask].astype(np.float32)
    adj = sp.coo_matrix((vals, (obs[:, 0], obs[:, 1])), shape=(n, m))
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index[1] += n  # shift col indices
    return Data(edge_index=edge_index, edge_weight=edge_weight), mask, M

def train_on_stream(model, optimizer, device, n, m, r, density, sigma, steps):
    model.train()
    for step in range(1, steps + 1):
        M, mask = generate_low_rank(n, m, r, density, sigma)
        data, _, _ = matrix_to_graph(M, mask)
        data = data.to(device)

        mask_tensor = torch.tensor(mask, device=device)
        M_true = torch.tensor(M, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        row_emb, col_emb = model(data.edge_index)

        ri = data.edge_index[0]
        ci = data.edge_index[1] - n
        preds = model.predict(ri, ci, row_emb, col_emb)

        # Known entries only for training
        loss = F.mse_loss(preds, data.edge_weight)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                full = row_emb @ col_emb.T
                rmse_obs = torch.sqrt(F.mse_loss(full[mask_tensor], M_true[mask_tensor]))
                rmse_all = torch.sqrt(F.mse_loss(full, M_true))
                print(f"[Train step {step:03d}] loss: {loss.item():.4f} | "
                      f"RMSE_obs: {rmse_obs:.4f} | RMSE_all: {rmse_all:.4f}")


def evaluate_on_stream(model, device, n, m, r, density, sigma, trials=10):
    model.eval()
    rmse_all_list = []
    rmse_obs_list = []

    for _ in range(trials):
        M, mask = generate_low_rank(n, m, r, density, sigma)
        data, _, _ = matrix_to_graph(M, mask)
        data = data.to(device)

        mask_tensor = torch.tensor(mask, device=device)
        M_true = torch.tensor(M, dtype=torch.float32, device=device)

        with torch.no_grad():
            row_emb, col_emb = model(data.edge_index)
            full = row_emb @ col_emb.T

            rmse_all = torch.sqrt(F.mse_loss(full, M_true))
            rmse_obs = torch.sqrt(F.mse_loss(full[mask_tensor], M_true[mask_tensor]))

            rmse_all_list.append(rmse_all.item())
            rmse_obs_list.append(rmse_obs.item())

    print(f"\n📊 Evaluation over {trials} test matrices")
    print(f"   Avg RMSE_obs: {np.mean(rmse_obs_list):.4f}")
    print(f"   Avg RMSE_all: {np.mean(rmse_all_list):.4f}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    n, m, r = 80, 64, 2
    embed_dim, num_layers = 32, 2
    density, sigma = 0.5, 0.1
    train_steps = 500
    test_trials = 20
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN_LRMC(n, m, embed_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("🔧 Training...")
    train_on_stream(model, optimizer, device, n, m, r, density, sigma, train_steps)

    print("\n🧪 Testing...")
    evaluate_on_stream(model, device, n, m, r, density, sigma, test_trials)


if __name__ == "__main__":
    main()
    