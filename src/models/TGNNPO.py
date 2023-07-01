import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

class TGNNPO(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TGNNPO, self).__init__()

        self.tgnn = A3TGCN2(in_channels=node_features, 
                            out_channels=32, 
                            periods=periods,
                            batch_size=batch_size)
        
        # equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

        self.linear = torch.nn.Linear(32, periods)
        self.softmax = torch.nn.Softmax(0)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)

        # apply softmax function to respect the contraint $w_i \in [0, 1]$
        w = self.softmax(h)

        return w