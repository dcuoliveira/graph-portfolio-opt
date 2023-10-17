import torch
import numpy as np
import scipy.optimize as opt

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

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
   
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_t, weights)))
        weighted_volatilities = np.dot(weights.T, self.vol_t)
        diversification_ratio = - weighted_volatilities / portfolio_volatility
        return diversification_ratio * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool) -> torch.Tensor:

        K = returns.shape[1]

        # covariance estimator
        if self.covariance_estimator == "mle":
            cov_t = self.MLECovariance(returns)
        else:
            raise NotImplementedError
        
        self.cov_t = cov_t.numpy()
        self.vol_t = torch.sqrt(torch.diag(cov_t))[:, None].numpy()

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # The weights sum to one
            ]
            bounds = [(0, None) for _ in range(K)]
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # the weights sum to zero
            ]
            bounds = [(None, None) for _ in range(K)]

        # initial guess for the weights (equal distribution)
        w0 = self.random_weights_with_constraints(K=K)

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
