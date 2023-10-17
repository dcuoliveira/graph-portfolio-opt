import torch
import numpy as np

class Estimators:
    """
    This class implements the estimators for all the unknown quantites we have on the optimization problems.

    """
    def __init__(self) -> None:
        pass

    def random_weights(self,
                       K: int) -> torch.Tensor:
        """
        Sample random weights from a uniform distribution with the given constraints.

        Args:
            K (int): number of assets.
        Returns:
            weights (torch.tensor): random weights.
        """

        wt = torch.tensor(np.random.uniform(-2, 2, size=K))

        return wt


    def random_weights_with_constraints(self,
                                        K: int) -> torch.Tensor:
        """
        Sample random weights from a uniform distribution with the given constraints.

        Args:
            K (int): number of assets.
        Returns:
            weights (torch.tensor): random weights.
        """

        wt = torch.tensor(np.random.uniform(-2, 2, size=K))

        wt[wt < 0] = wt[wt < 0] / wt[wt < 0].abs().sum()
        wt[wt > 0] = wt[wt > 0] / wt[wt < 0].abs().sum()

        return wt

    def MLEMean(self,
                returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the mean of the returns.

        Args:
            returns (torch.tensor): returns tensor.
        
        Returns:
            mean_t (torch.tensor): MLE estimates for the mean of the returns.
        """
        mean_t = torch.mean(returns, axis = 0)

        return mean_t
    
    def MLECovariance(self,
                      returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the covariance of the returns.

        Args:
            returns (torch.tensor): returns tensor.

        Returns:
            cov_t (torch.tensor): MLE estimates for the covariance of the returns.
        """
        
        cov_t = torch.cov(returns.T,correction = 0)

        return cov_t
    