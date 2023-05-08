import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, returns, vol, weights):
        
        realized_sharpe_ratio = torch.divide(torch.mul(weights, returns), vol)

        return realized_sharpe_ratio.sum()