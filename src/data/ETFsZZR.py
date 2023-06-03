import os
import numpy as np
import torch
import pandas as pd

from utils.dataset_utils import concatenate_prices_returns

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class ETFsZZR(object):
    """
    This class implements the dataset used in Zhang, Zohren, and Roberts (2021)
    https://arxiv.org/abs/2005.13665 in to torch geomatric data loader format.
    The data consists of daily observatios of four etf prices and returns 
    concatenated together, from January 2000 to February 2023.
    
    """
    
    def __init__(self,):
        super().__init__()
        self._read_data()

    def _read_data(self):

        prices = pd.read_excel(os.path.join(os.path.dirname(__file__), "inputs", "etfs-zhang-zohren-roberts.xlsx"))

        # prepare dataset
        prices.set_index("date", inplace=True)
        prices = prices.shift(-1)

        # compute returns and subset data
        returns = np.log(prices).diff().dropna()
        prices = prices.loc[returns.index].values.astype('float32')
        returns = returns.values.astype('float32')

        # TODO: scale data

        # fully connected adjaneccny matrix
        A = torch.ones(prices.shape[1], prices.shape[1])

        # features array
        X = torch.zeros(prices.shape[0], prices.shape[1], 2)
        for i in range(prices.shape[1]):
             # fix dimensios
            prices_tmp = torch.tensor(prices[:, i]).reshape(prices.shape[0], 1)
            returns_tmp = torch.tensor(returns[:, i]).reshape(returns.shape[0], 1)

            # stack prices and returns
            X[:, i, :] = torch.hstack((prices_tmp, returns_tmp))

        # features array
        y = torch.zeros(prices.shape[0], prices.shape[1], 1)
        for i in range(prices.shape[1]):
             # fix dimensios
            prices_tmp = torch.tensor(prices[:, i]).reshape(prices.shape[0], 1)

            # stack prices and returns
            y[:, i, :] = prices_tmp

        # reshape all
        X = X.reshape((X.shape[1], X.shape[2], X.shape[0]))
        y = y.reshape((y.shape[1], y.shape[2], y.shape[0]))

        # outputs
        self.A = A
        self.X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        self.y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """
        Uses the node features of the graph and generates a feature/target relationship
        of the shape (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out).

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[1] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, i : i + num_timesteps_in]).numpy())
            target.append((self.y[:, i : i + num_timesteps_in]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(self,
                    num_timesteps_in: int = 12,
                    num_timesteps_out: int = 12) -> StaticGraphTemporalSignal:
        """
        Returns data iterator for Zhang, Zohren, and Roberts (2021) ETFs dataset
        as an instance of the static graph temporal signal class.

        
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(self.edges,
                                            self.edge_weights,
                                            self.features,
                                            self.targets)

        return dataset