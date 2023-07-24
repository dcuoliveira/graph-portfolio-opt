import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TGNNPO(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TGNNPO, self).__init__()
        self.periods = periods
        mid_channels = 32
        
        # models
        self.sage = SAGEConv(node_features, mid_channels, normalize=True)
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device='cpu'))
        # equals single-shot prediction
        self.linear = torch.nn.Linear(mid_channels, 1)
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self._attention)
        self.linear.reset_parameters()
        self.sage.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = 0
        probs = F.gumbel_softmax(self._attention, dim=0)
        for period in range(self.periods):
            h = h + probs[period] * self.sage(x[0, :, :, period], edge_index, edge_weight)
        h = F.leaky_relu(h)
        h = self.linear(h)

        # apply sigmoid function to respect the contraint $w_i \in [0, 1]$
        # h = torch.special.expit(h)
        # h = F.relu(h)
        # h = F.normalize(h, p=1.0, dim=0)
        h = F.gumbel_softmax(h, dim=0)

        return h