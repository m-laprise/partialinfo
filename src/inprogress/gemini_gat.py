import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, to_dense_adj


class GraphAttentionLayer(nn.Module):
    """
    An efficient implementation of a single Graph Attention Layer.
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        """
        The efficient forward pass. It operates on the sparse edge_index.
        
        Args:
            h (torch.Tensor): Input node features, shape [N, in_features].
            edge_index (torch.Tensor): Graph connectivity in COO format, shape [2, E].
        
        Returns:
            torch.Tensor: Output node features, shape [N, out_features].
        """
        # 1. Apply linear transformation to all nodes
        Wh = torch.mm(h, self.W) # shape [N, out_features]
        
        # --- Efficiency Improvement Start ---
        # Instead of creating a dense NxN matrix, we only compute attention for existing edges.
        
        # 2. Compute attention scores for each edge
        source_nodes, target_nodes = edge_index
        
        # Get the feature vectors for the source and target nodes of each edge
        Wh_source = Wh.index_select(0, source_nodes) # shape [E, out_features]
        Wh_target = Wh.index_select(0, target_nodes) # shape [E, out_features]
        
        # Concatenate the features: [Wh_i || Wh_j] for each edge (i, j)
        a_input = torch.cat([Wh_source, Wh_target], dim=1) # shape [E, 2 * out_features]

        # Compute the unnormalized attention score 'e' for each edge
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(1) # shape [E]

        # 3. Normalize attention scores using softmax over each node's neighborhood
        # This requires a scatter operation to group scores by target node.
        # We'll implement a simplified sparse softmax.
        attention = self.sparse_softmax(e, target_nodes, h.shape[0])
        
        # Apply dropout to attention scores
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 4. Aggregate features (message passing)
        # Create weighted messages from source nodes
        weighted_messages = Wh_source * attention.unsqueeze(1) # [E, out_features]
        
        # Aggregate messages at target nodes
        h_prime = torch.zeros_like(Wh)
        h_prime.index_add_(0, target_nodes, weighted_messages)
        # --- Efficiency Improvement End ---

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def sparse_softmax(self, scores, index, num_nodes):
        """
        Computes softmax for sparse attention scores.
        Args:
            scores (torch.Tensor): Raw attention scores for each edge, shape [E].
            index (torch.Tensor): The target node indices for each edge, shape [E].
            num_nodes (int): The total number of nodes N.
        """
        # To prevent numerical instability, subtract the max score for each node's neighborhood.
        scores_max = scores.max()
        scores = scores - scores_max
        
        # Compute exp and sum for the denominator
        exp_scores = torch.exp(scores)
        exp_sum = torch.zeros(num_nodes, device=scores.device)
        exp_sum.index_add_(0, index, exp_scores)
        
        # The denominator for each score is the sum of exp_scores in its neighborhood
        exp_sum = exp_sum.index_select(0, index) # shape [E]
        
        return exp_scores / (exp_sum + 1e-16)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, features, edge_index, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()

def test(model, features, edge_index, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, edge_index)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test.item(), acc_test.item()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def main():
    lr = 0.005
    weight_decay = 5e-4
    epochs = 200
    patience = 100
    nhid = 8
    nheads = 8
    dropout = 0.6
    alpha = 0.2

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    features = data.x
    labels = data.y
    idx_train = data.train_mask.nonzero(as_tuple=False).squeeze()
    idx_val = data.val_mask.nonzero(as_tuple=False).squeeze()
    idx_test = data.test_mask.nonzero(as_tuple=False).squeeze()
    
    # Add self-loops to the edge index
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    model = GAT(nfeat=features.shape[1],
                nhid=nhid,
                nclass=int(labels.max()) + 1,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print("Starting training with efficient implementation...")
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        loss_train, acc_train = train(model, optimizer, features, edge_index, labels, idx_train)
        
        model.eval()
        with torch.no_grad():
            output = model(features, edge_index)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, '
                  f'Loss: {loss_train:.4f}, '
                  f'Train Acc: {acc_train:.4f}, '
                  f'Val Acc: {acc_val:.4f}')

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_gat_cora_efficient.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping!")
            break
            
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")

    print("\nLoading best model and testing...")
    model.load_state_dict(torch.load('best_gat_cora_efficient.pth'))
    loss_test, acc_test = test(model, features, edge_index, labels, idx_test)
    print(f"Test set results: loss= {loss_test:.4f}, accuracy= {acc_test:.4f}")

if __name__ == '__main__':
    main()
    