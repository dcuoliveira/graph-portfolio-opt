import os
import numpy as np
import torch
import pandas as pd
import glob
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from utils.dataset_utils import create_rolling_window_ts_for_graphs

class NewETFs(object):
    """
    This class implements the dataset used in Zhang, Zohren, and Roberts (2021)
    https://arxiv.org/abs/2005.13665 in to torch geomatric data loader format.
    The data consists of daily observatios of four etf prices and returns 
    concatenated together, from January 2000 to February 2023.
    
    """
    
    def __init__(self,
                 use_last_data: bool = True,
                 fields=["close"],
                 years=["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]):
        super().__init__()

        self.inputs_path = os.path.join(os.getcwd(), "src", "data", "inputs")
        self.use_last_data = use_last_data
        self.fields = fields
        self.years = years

        self._read_data(years=self.years, fields=self.fields)


    def _read_data(self,
                   fields: list,
                   years: list):

        if self.use_last_data:
            etfs_df = pd.read_csv(os.path.join(self.inputs_path, "etfs-new.csv"))
            etfs_df["date"] = pd.to_datetime(etfs_df["date"])
            etfs_df.set_index("date", inplace=True)
        else:
            etfs = []
            for y in tqdm(years, total=len(years), desc="Loading All ETFs Data"):
                files = glob.glob(os.path.join(self.inputs_path , "new_etfs", y, "*.csv.gz"))

                for f in files:
                    tmp_df = pd.read_csv(f,
                                        compression='gzip',
                                        on_bad_lines='skip')
                    tmp_df = tmp_df[["ticker"] + fields]
                    tmp_df["date"] = pd.to_datetime(f.split(os.sep)[-1].split(".")[0])

                    pivot_tmp_df = tmp_df.pivot_table(index=["date"], columns=["ticker"], values=["close"])
                    pivot_tmp_df.index.name = None
                    pivot_tmp_df.columns = pivot_tmp_df.columns.droplevel(0)

                    etfs.append(pivot_tmp_df)
            etfs_df = pd.concat(etfs, axis=0)
            etfs_df = etfs_df.sort_index().dropna(axis=1, how="any")
            etfs_df.index.name = "date"

            etfs_df.to_csv(os.path.join(self.inputs_path, "etfs-new.csv"))

        # compute returns and subset data
        returns = np.log(etfs_df).diff().dropna()

        # subset all
        idx = returns.index
        returns = returns.loc[idx]
        prices = etfs_df.loc[idx]

        # create tensor with (num_nodes, num_features_per_node, num_timesteps)
        num_nodes = prices.shape[1]
        num_features_per_node = len(fields)
        num_timesteps = prices.shape[0]

        X = torch.zeros(num_nodes, num_features_per_node, num_timesteps)
        y = torch.zeros(num_nodes, num_timesteps)
        for i in range(num_nodes):
            # features
            X[i, :, :] = torch.from_numpy(returns.loc[:, returns.columns[i]].values)

            # target
            y[i, :] = torch.from_numpy(prices.loc[:, prices.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.X = X
        self.y = y

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self,  num_timesteps_in: int = 12, num_timesteps_out: int = 12, fix_start: bool = False):

        features, target = create_rolling_window_ts_for_graphs(target=self.y,
                                                               features=self.X,
                                                               num_timesteps_in=num_timesteps_in,
                                                               num_timesteps_out=num_timesteps_out,
                                                               fix_start=fix_start)

        self.features = features
        self.targets = target

    def get_dataset(self, num_timesteps_in: int = 12,
                    num_timesteps_out: int = 12,
                    fix_start: bool = False) -> StaticGraphTemporalSignal:
        
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out, fix_start) 
        dataset = StaticGraphTemporalSignal(self.edges,
                                            self.edge_weights,
                                            self.features,
                                            self.targets)
        
        return dataset
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = NewETFs()
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
