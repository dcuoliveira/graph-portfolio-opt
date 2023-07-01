import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prices, weights, ascent=True):
        
        # portfolio values
        portfolio_values = torch.mul(weights, prices)

        # portfolio returns
        portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]

        # portfolio sharpe
        sharpe_ratio = torch.mean(portfolio_returns) / torch.std(portfolio_returns)

        return sharpe_ratio * (-1 if ascent else 1)