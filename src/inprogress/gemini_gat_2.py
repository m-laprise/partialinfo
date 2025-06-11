import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Graph Attention Network model using torch_geometric.nn.GATConv layers.
    This architecture is based on the one used for the Cora dataset in the paper.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        Args:
            nfeat (int): Number of input features.
            nhid (int): Number of hidden units per head.
            nclass (int): Number of output classes.
            dropout (float): Dropout rate.
            alpha (float): LeakyReLU negative slope (not directly used by GATConv,
                           but good to keep for consistency).
            nheads (int): Number of attention heads.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # First GATConv layer: Multi-head attention.
        # GATConv handles multi-head attention and concatenation internally.
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, dropout=dropout)

        # Second GATConv layer: The output layer.
        # It takes the concatenated output of the first layer as input.
        # heads=1 and concat=False is standard for the final classification layer.
        self.conv2 = GATConv(nhid * nheads, nclass, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.
        
        Returns:
            torch.Tensor: Log-probabilities for each class.
        """
        # Apply dropout to the input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GATConv layer followed by ELU activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Apply dropout to the hidden layer features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GATConv layer (output layer)
        x = self.conv2(x, edge_index)
        
        # Return log-probabilities
        return F.log_softmax(x, dim=1)

def train(model, optimizer, data):
    """Training function."""
    model.train()
    optimizer.zero_grad()
    # The model forward pass needs node features and edge index
    output = model(data.x, data.edge_index)
    # The loss is calculated only on the training nodes
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()

def test(model, data, mask):
    """Testing/Validation function."""
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])
        return loss.item(), acc.item()

def accuracy(output, labels):
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def main():
    """Main function to run the GAT model on Cora."""
    # --- Hyperparameters ---
    lr = 0.005
    weight_decay = 5e-4
    epochs = 200
    patience = 100
    nhid = 8
    nheads = 8
    dropout = 0.6
    alpha = 0.2

    # --- Load Data ---
    # Using torch_geometric's Planetoid dataset loader
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # --- Model and Optimizer Setup ---
    model = GAT(nfeat=dataset.num_features,
                nhid=nhid,
                nclass=dataset.num_classes,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # --- Training Loop ---
    print("Starting training using torch_geometric.nn.GATConv...")
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        loss_train, acc_train = train(model, optimizer, data)
        loss_val, acc_val = test(model, data, data.val_mask)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, '
                  f'Loss: {loss_train:.4f}, '
                  f'Train Acc: {acc_train:.4f}, '
                  f'Val Acc: {acc_val:.4f}')

        # Early stopping
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_gat_cora_torch_geometric.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}!")
            break
            
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")

    # --- Final Test ---
    print("\nLoading best model and testing...")
    model.load_state_dict(torch.load('best_gat_cora_torch_geometric.pth'))
    loss_test, acc_test = test(model, data, data.test_mask)
    print(f"Test set results: loss= {loss_test:.4f}, accuracy= {acc_test:.4f}")

if __name__ == '__main__':
    main()