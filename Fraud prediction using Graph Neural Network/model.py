import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch.nn import BatchNorm1d


# Credit Card Fraud GNN Model (using SAGEConv)
class FraudGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x


# Bitcoin Fraud GNN Model (using GATv2Conv)
# Adjust model architecture to match saved model's architecture
class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super(GAT, self).__init__()
        self.norm1 = BatchNorm1d(165)  # Update to the correct number based on saved model
        self.gat1 = GATv2Conv(165, 128, heads=heads, dropout=0.3)  # Match the saved sizes
        self.norm2 = BatchNorm1d(1024)  # Update to match the saved model
        self.gat2 = GATv2Conv(1024, 1, heads=heads, concat=False, dropout=0.6)  # Update based on saved model
        
    def forward(self, x, edge_index):
        h = self.norm1(x)
        h = self.gat1(h, edge_index)
        h = self.norm2(h)
        h = F.leaky_relu(h)
        out = self.gat2(h, edge_index)
        return out



# Accuracy Calculation Function
def accuracy(y_pred, y_test, prediction_threshold=0.5):
    y_pred_label = (torch.sigmoid(y_pred) > prediction_threshold).float()
    correct_results_sum = (y_pred_label == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc
