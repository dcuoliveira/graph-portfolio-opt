import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

class TGNNPO(torch.nn.Module):
    def __init__(self, node_features, num_nodes, periods):
        super(TGNNPO, self).__init__()
        mid_channels = num_nodes * 5

        self.tgnn = A3TGCN2(in_channels=node_features, 
                            out_channels=mid_channels, 
                            periods=periods,
                            batch_size=periods)
        
        # equals single-shot prediction
        self.linear = torch.nn.Linear(mid_channels, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)

        # apply sigmoid function to respect the contraint $w_i \in [0, 1]$
        h = torch.special.expit(h)
        #  h = F.normalize(h, p=1.0, dim=1)

        return h