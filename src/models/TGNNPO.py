import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TGNNPO(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TGNNPO, self).__init__()
        self.periods = periods
        mid_channels = 3
        
        # models
        self.sages = torch.nn.ModuleList([SAGEConv(node_features, mid_channels, normalize=True) for i in range(periods)])
        #self._attention = torch.nn.Parameter(torch.ones(self.periods, device='cpu'))
        # equals single-shot prediction
        self.linear = torch.nn.Linear(mid_channels * periods, 1)
    
    def reset_parameters(self):
        #torch.nn.init.uniform_(self._attention)
        self.linear.reset_parameters()
        for sage in self.sages:
            sage.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = []
        #probs = F.gumbel_softmax(self._attention, dim=0)
        for period in range(self.periods):
            h.append(((period + 1)/self.periods) * self.sages[period](x[0, :, :, period], edge_index, edge_weight))
        h = torch.cat(h, 1)
        h = F.leaky_relu(h)
        h = self.linear(h)

        # apply sigmoid function to respect the contraint $w_i \in [0, 1]$
        # h = torch.special.expit(h)
        # h = F.relu(h)
        # h = F.normalize(h, p=1.0, dim=0)
        h = F.gumbel_softmax(h, dim=0)

        return h