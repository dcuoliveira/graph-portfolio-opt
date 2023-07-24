import os
import numpy as np
import torch
import pandas as pd
import glob
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class CRSPSimple(object):
    """
    This class implements the dataset used in Zhang, Zohren, and Roberts (2021)
    https://arxiv.org/abs/2005.13665 in to torch geomatric data loader format.
    The data consists of daily observatios of four etf prices and returns 
    concatenated together, from January 2000 to February 2023.
    
    """
    
    def __init__(self,
                 use_sample_data: bool = True,
                 fields: list=["close"],
                 all_years: bool = False,
                 years: list=["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]):
        super().__init__()

        self.inputs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inputs")
        self.use_sample_data = use_sample_data
        self.fields = fields
        self.all_years = all_years
        self.years = years

        self._read_data(years=self.years, fields=self.fields)


    def _read_data(self,
                   fields: list,
                   years: list):
        
        if self.all_years:
            years = os.listdir(os.path.join(self.inputs_path, "crsp"))
            years = [val for val in years if val != ".DS_Store"]
            years.sort()

        if self.use_sample_data:
    
            crsp_df = pd.read_csv(os.path.join(self.inputs_path, "crsp_simple_sample.csv"))

            crsp_df["date"] = pd.to_datetime(crsp_df["date"])
            crsp_df.set_index("date", inplace=True)
        else:

            crsp = []
            for y in tqdm(years, total=len(years), desc="Loading All CRSP Data"):
                files = glob.glob(os.path.join(self.inputs_path , "crsp", y, "*.csv.gz"))

                for f in files:
                    tmp_df = pd.read_csv(f,
                                        compression='gzip',
                                        on_bad_lines='skip')
                    tmp_df = tmp_df[["ticker"] + fields]
                    tmp_df["date"] = pd.to_datetime(f.split(os.sep)[-1].split(".")[0])

                    pivot_tmp_df = tmp_df.pivot_table(index=["date"], columns=["ticker"], values=["close"])
                    pivot_tmp_df.index.name = None
                    pivot_tmp_df.columns = pivot_tmp_df.columns.droplevel(0)

                    crsp.append(pivot_tmp_df)
            crsp_df = pd.concat(crsp, axis=0)

            crsp_df = crsp_df.sort_index().dropna(axis=1, how="any")
            crsp_df.index.name = "date"

            # check if file exists
            if not os.path.exists(os.path.join(self.inputs_path, "crsp_simple_sample.csv")):
                crsp_df.iloc[:, 0:50].to_csv(os.path.join(self.inputs_path, "crsp_simple_sample.csv"))

        # compute returns and subset data
        returns = np.log(crsp_df).diff().dropna()

        # save indexes
        self.index = list(returns.index)
        self.columns = list(returns.columns)

        # subset all
        idx = returns.index
        returns_df = returns.loc[idx]
        prices_df = crsp_df.loc[idx]

        # create tensor with (num_nodes, num_features_per_node, num_timesteps)
        num_nodes = prices_df.shape[1]
        num_features_per_node = len(fields)
        num_timesteps = prices_df.shape[0]

        features = torch.zeros(num_nodes, num_features_per_node, num_timesteps)
        prices = torch.zeros(num_nodes, num_timesteps)
        returns = torch.zeros(num_nodes, num_timesteps)
        for i in range(num_nodes):
            # features
            features[i, :, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)

            # target
            prices[i, :] = torch.from_numpy(prices_df.loc[:, prices_df.columns[i]].values)

            # returns
            returns[i, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.features = features
        self.returns = returns
        self.prices = prices
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = CRSPSimple(use_sample_data=False,
                            fields=["close"],
                            all_years=True,
                            years=["2011", "2012", "2013", "2014", "2015", "2016"])