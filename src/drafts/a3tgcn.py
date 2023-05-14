import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN, DCRNN
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import METRLADatasetLoader


class TemporalGCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 output_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size

        # Temporal Graph Convolutional Network
        self.tgnn = TGCN(in_channels=in_channels, 
                         out_channels=out_channels,
                         add_self_loops=False)
        
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, output_size)

    def forward(self, x, edge_index):
        """

        :param x: node features for T time steps
        :type x: torch tensor
        :param edge_index: torch array
        :type edge_index: torch tensor
        """

        # x: (num_nodes, num_feeatures, T)
        # edge_index: (2, num_nonzero_adj_entries)
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)

        return h

class AttentionTemporalGNN(torch.nn.Module):
    def __init__(self, node_features, out_channels, periods):
        super().__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

# loader = METRLADatasetLoader()
# dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)       

# GPU support
device = torch.device('cpu') # cuda
subset = 2000

# Create model and optimizers
# model = TemporalGCN(in_channels=2, out_channels=32, output_size=1).to(device)
model = RecurrentGCN(node_features=2)
# model = AttentionTemporalGNN(node_features=2, out_channels=32, periods=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Running training...")
for epoch in range(10): 
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot.to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        loss = loss + torch.mean((y_hat-snapshot.y)**2) 
        step += 1
        if step > subset:
          break

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))