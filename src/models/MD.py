import torch
import numpy as np
import cvxopt as opt
from cvxopt import solvers

from estimators.Estimators import Estimators

class MD(Estimators):
    def __init__(self,
                 covariance_estimator: str="mle") -> None:
        """"
        This function impements the maximum diversification portfolio (MD) method proposed by Choueifaty and Coignard (2008).

        Args:
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Choieifaty, Y., and Y. Coignard, (2008). Toward Maximum Diversification. Journal of Portfolio Management.
        """
        super().__init__()
        
        self.covariance_estimator = covariance_estimator

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                verbose: bool=True) -> torch.Tensor:
        
       pass
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        
        # covariance estimator
        if self.covariance_estimator == "mle":
            cov_t = self.MLECovariance(returns)
        else:
            raise NotImplementedError
        
        # compute volatilities from each asset
        vol_t = torch.sqrt(torch.diag(cov_t))[:, None]

        # compute weights
        wt = torch.divide(torch.matmul(torch.inverse(cov_t), vol_t), torch.matmul(torch.matmul(vol_t.T, torch.inverse(cov_t)), vol_t))
        wt = wt.repeat(num_timesteps_out, 1)

        return wt
