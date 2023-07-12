import os
from typing import Optional, List
from tqdm import tqdm
from pandas import DataFrame, Timestamp, Timedelta, read_csv, concat, get_dummies
import torch
from torch import Tensor

from torch_geometric.utils import dense_to_sparse
from data.ForecastBatch import ForecastBatch

class CRSPLoader(object):
    """
    This class implements the loader to the US CRSP NYSE data.
    """

    def __init__(
        self,
        load_data: bool = False,
        load_path: Optional[str] = None,
    ):
        super().__init__()
        self.load_data = load_data
        self.load_path = load_path
        if self.load_path is None:
            self.load_path = os.path.join(os.path.dirname(__file__), "US_CRSP_NYSE_master.csv")
        if self.load_data:
            self._load_data(self.load_path)
        else:
            # Store path to raw yearly data
            self.yearly_path = os.path.join(os.path.dirname(__file__), "US_CRSP_NYSE", "Yearly")
            self._read_raw_data(self.load_path)
        self.edge_raw_path = os.path.join(os.path.dirname(__file__), "correlation_networks", "monthly", "corr_networks_MR")
        self._read_raw_edge_data()

    def _read_raw_data(
        self,
        save_path: str,
    ):
        print('Generating CRSP loader from raw data...')
        self.master_data = []
        cur_year = 2000
        while cur_year <= 2021:
            year_path = os.path.join(self.yearly_path, f'{cur_year}')
            cur_files = os.listdir(year_path)
            cur_files.sort()
            cur_files = list(filter(lambda x: x.endswith('.csv.gz'), cur_files))
            pbar = tqdm(cur_files, total=len(cur_files))
            for file in pbar:
                pbar.set_description(f'Loading year: {cur_year}')
                cur_data = read_csv(os.path.join(year_path, file)).iloc[: , 1:]
                cur_data['date'] = Timestamp(f'{file[0:4]}-{file[4:6]}-{file[6:8]}')
                self.master_data.append(cur_data)
            cur_year = cur_year + 1
        self.master_data = concat(self.master_data)
        self._categorize_master_data()
        self.master_data.to_csv(path_or_buf=save_path)
        self._update_ticker_index()

    def _load_data(
        self,
        load_path: str,
    ):
        print('Loading in saved CRSP data...')
        self.master_data = read_csv(load_path)
        self.master_data['date'] = self.master_data['date'].astype('datetime64[ns]')
        self._categorize_master_data()
        self._update_ticker_index()

    def _read_raw_edge_data(
        self,
    ):
        print('Generating edge weights from raw data...')
        self.edge_data = []
        self.edge_data_index = 0
        cur_files = os.listdir(self.edge_raw_path)
        cur_files.sort()
        pbar = tqdm(cur_files, total=len(cur_files))
        for file in pbar:
            pbar.set_description('Loading edge files')
            self.edge_data.append(
                (Timestamp(f'{file[0:4]}-{file[4:6]}-{file[6:8]}'), read_csv(os.path.join(self.edge_raw_path, file)).iloc[: , 1:])
            )
        self._update_edge_ticker_index()

    def _categorize_master_data(
        self,
    ):
        # Stringify all first
        self.master_data['ticker'] = self.master_data['ticker'].astype('string')
        self.master_data['PERMNO'] = self.master_data['PERMNO'].astype('string')
        self.master_data['SICCD'] = self.master_data['SICCD'].astype('string')
        self.master_data['PERMCO'] = self.master_data['PERMCO'].astype('string')
        # Then we categorize
        self.master_data['PERMNO'] = self.master_data['PERMNO'].astype('category')
        self.master_data['SICCD'] = self.master_data['SICCD'].astype('category')
        self.master_data['PERMCO'] = self.master_data['PERMCO'].astype('category')

    def _update_ticker_index(
        self,
        ticker_list: Optional[List[str]] = None,
    ):
        if ticker_list is None:
            cats = self.master_data['ticker'].unique()
        else :
            cats = ticker_list
        self.ticker_index = {}
        self.rev_ticker_index = {}
        for step, ticker in enumerate(cats):
            self.ticker_index[ticker] = step
            self.rev_ticker_index[step] = ticker
        self.num_nodes = len(cats)

    def _update_edge_ticker_index(
        self,
    ):
        cols = list(self.edge_data[0][1].columns)
        self.edge_ticker_index = {}
        self.rev_edge_ticker_index = {}
        for step, ticker in enumerate(cols):
            self.edge_ticker_index[ticker] = step
            self.rev_edge_ticker_index[step] = ticker
        self.num_nodes = len(cols)
    
    def select_tickers(
        self,
        tickers: List[str],
        data: Optional[DataFrame] = None,
    ) -> DataFrame:
        if data is None:
            data = self.master_data
        return data[data['ticker'].isin(tickers)]

    def get_full_time_window(
        self,
        train_length: int,
        test_day: Timestamp,
        data: Optional[DataFrame] = None,
    ) -> DataFrame:
        if data is None:
            data = self.master_data
        time_delta = Timedelta(days=train_length)
        window_start = test_day - time_delta
        return data[(data['date'] >= window_start) & (data['date'] <= test_day)]

    def _get_edge_weights(
        self,
        data: DataFrame,
    ) -> Tensor:
        out_tensor = torch.ones(1, self.num_nodes**2, dtype=torch.float)
        for ticker_index in range(self.num_nodes - 1):
            for bottom_index in range(ticker_index + 1, self.num_nodes):
                ticker1 = self.rev_ticker_index[ticker_index]
                ticker2 = self.rev_ticker_index[bottom_index]
                if ticker1 in data and ticker2 in data:
                    value = data.loc[:, ticker1][self.edge_ticker_index[ticker2]]
                else:
                    value = 0.5
                out_tensor[0][(ticker_index * self.num_nodes) + bottom_index] = value
                out_tensor[0][(bottom_index * self.num_nodes) + ticker_index] = value
        return out_tensor

    def get_feature_matrix(
        self,
        data: Optional[DataFrame] = None,
    ) -> Tensor:
        print('Generating feature matrix...')
        self.edge_weights = []
        def ticker_index(ticker):
            return self.ticker_index[ticker]
        if data is None:
            data = self.master_data
        num_feat_names = ['open',
                          'high',
                          'low',
                          'close',
                          'volume',
                          'OPCL',
                          'pvCLCL',
                          'prevAdjClose',
                          'SPpvCLCL',
                          'sharesOut',
                          'prevRawOpen',
                          'prevRawClose',
                          'prevAdjOpen',
                          ]
        cat_feat_names = ['SICCD',
                          ]
        all_feats = []
        dates = data['date'].unique()
        pbar = tqdm(dates, total=len(dates))
        self.num_dates = len(dates)
        for date in pbar:
            # Get data from current date
            cur_data = data[data['date'] == date]

            # Get numerical features
            cur_feat = torch.tensor(cur_data[num_feat_names].values, dtype=torch.float)

            # Get categorical features
            cur_feat = torch.cat((cur_feat, torch.tensor(get_dummies(cur_data[cat_feat_names]).values, dtype=torch.float)), dim=1)

            # Get indices of tickers of the current date
            cur_tick_ind = torch.tensor(cur_data['ticker'].apply(ticker_index).values)

            # Calculate final feature matrix of this date and append it to list
            num_feat = torch.zeros(self.num_nodes, cur_feat.shape[1], dtype=torch.float)
            num_feat[cur_tick_ind, :] = cur_feat
            all_feats.append(num_feat)

            # Handle edge weights
            cur_edge_date = self.edge_data[self.edge_data_index][0]
            while self.edge_data_index + 1 < len(self.edge_data) and abs((date - cur_edge_date).total_seconds()) > abs((date - self.edge_data[self.edge_data_index + 1][0]).total_seconds()):
                self.edge_data_index = self.edge_data_index + 1
                cur_edge_date = self.edge_data[self.edge_data_index][0]
            cur_corr_mat = self.edge_data[self.edge_data_index][1]
            self.edge_weights.append(self._get_edge_weights(cur_corr_mat))
        self.num_features = cur_feat.shape[1]
        self.edge_weights = torch.squeeze(torch.stack(self.edge_weights, dim=2))
        return torch.stack(all_feats, dim=2)

    def get_target_matrix(
        self,
        data: Optional[DataFrame] = None,
    ) -> Tensor:
        print('Generating target matrix...')
        def ticker_index(ticker):
            return self.ticker_index[ticker]
        if data is None:
            data = self.master_data
        num_feat_names = ['open',
                          ]
        all_feats = []
        dates = data['date'].unique()
        pbar = tqdm(dates, total=len(dates))
        for date in pbar:
            # Get data from current date
            cur_data = data[data['date'] == date]

            # Get numerical features
            cur_feat = torch.tensor(cur_data[num_feat_names].values, dtype=torch.float)

            # Get indices of tickers of the current date
            cur_tick_ind = torch.tensor(cur_data['ticker'].apply(ticker_index).values)

            # Calculate final feature matrix of this date and append it to list
            num_feat = torch.zeros(self.num_nodes, cur_feat.shape[1], dtype=torch.float)
            num_feat[cur_tick_ind, :] = cur_feat
            all_feats.append(num_feat)
        return torch.squeeze(torch.stack(all_feats, dim=2))

    def _get_edges_and_weights(
        self,
    ):
        return dense_to_sparse(torch.ones(self.num_nodes, self.num_nodes))[0].numpy(), self.edge_weights.numpy()

    def _generate_task(
        self,
        data: Optional[DataFrame] = None,
    ):
        print('Generating CRSP dataset...')
        if data is None:
            data = self.master_data
        X = self.get_feature_matrix(data)
        y = self.get_target_matrix(data)
        return X, y

    def get_dataset(
        self,
        data: Optional[DataFrame] = None,
        window_length: int = 50,
        step_length: int = 5,
    ) -> ForecastBatch:
        if data is None:
            data = self.master_data
        features, targets = self._generate_task(data)
        edges, edge_weights = self._get_edges_and_weights()
        dataset = ForecastBatch(edges,
                                edge_weights,
                                features.numpy(),
                                targets.numpy(),
                                window_length,
                                step_length)
        return dataset