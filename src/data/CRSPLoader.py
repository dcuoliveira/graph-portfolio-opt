import os
from typing import Optional, List
from tqdm import tqdm
from pandas import DataFrame, Timestamp, Timedelta, read_csv, concat, get_dummies
import torch
from torch import Tensor
import pandas as pd
import datetime as dt

# import sys
# sys.path.append(os.path.join(os.getcwd(), 'src'))

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
        load_edge_data: bool = False,
        num_feat_names: List[str] = ['open', 'high', 'low', 'close', 'volume', 'OPCL', 'pvCLCL', 'prevAdjClose', 'SPpvCLCL', 'sharesOut', 'prevRawOpen', 'prevRawClose', 'prevAdjOpen',],
        cat_feat_names: List[str] = ['SICCD'],
    ):
        super().__init__()
        self.load_data = load_data
        self.load_path = load_path
        self.load_edge_data = load_edge_data
        self.num_feat_names = num_feat_names
        self.cat_feat_names = cat_feat_names

        if self.load_path is None:
            self.load_path = os.path.join(os.path.dirname(__file__),  "inputs", "US_CRSP_NYSE_master.csv")
        
        # load crsp data
        if self.load_data:
            self._load_data(self.load_path)
        else:
            # Store path to raw yearly data
            self.yearly_path = os.path.join(os.path.dirname(__file__), "inputs", "US_CRSP_NYSE", "Yearly")
            self._read_raw_data(self.load_path)

        # load edge data
        if self.load_edge_data:
            print("Monthly Edge data only!")
            self.edge_raw_path = os.path.join(os.path.dirname(__file__), "inputs", "correlation_networks", "monthly", "corr_networks_MR")
            self._read_raw_edge_data()
        else:
            raise ValueError("Edge data must be loaded")

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
        self.master_data.to_csv(path_or_buf=os.path.join(save_path, "US_CRSP_NYSE_master.csv"))
        self._update_ticker_index()

    def _load_data(
        self,
        load_path: str,
    ):
        print('Loading in saved CRSP data...')
        self.master_data = read_csv(os.path.join(load_path, "US_CRSP_NYSE_master.csv"))
        self.master_data['date'] = pd.to_datetime(self.master_data['date'])
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
        
        self.num_tickers = len(tickers)
        
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
        out_tensor = torch.zeros(1, self.num_nodes**2, dtype=torch.float)
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

        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        resampled_data = []
        for ticker in data['ticker'].unique():
            
            # Subset data to current ticker
            ticker_data = data[data['ticker'] == ticker]

            # Pivot data
            ticker_data = ticker_data.pivot_table(index=["date"], columns=["ticker"], values=self.num_feat_names)

            # Drop level 1 of column index
            ticker_data.columns = ticker_data.columns.droplevel(1)

            # Resample data to business days
            ticker_data = ticker_data.resample('B').last()

            # Forward fill missing values
            ticker_data = ticker_data.fillna(method='ffill')

            # Melt data
            ticker_data = ticker_data.reset_index()
            ticker_data["ticker"] = ticker

            # Add categorical features
            ticker_data = pd.merge(ticker_data, data[data['ticker'] == ticker][["date"] + self.cat_feat_names], on="date", how="left")

            resampled_data.append(ticker_data)

        resampled_data = concat(resampled_data)
        
        all_feats = []
        self.dates = pd.to_datetime(resampled_data['date'].unique())
        first_edge_date = self.edge_data[0][0]
        pbar = tqdm(self.dates, total=len(self.dates))
        self.num_dates = len(self.dates)
        for date in pbar:

            # While no edge data, continue
            if date <= first_edge_date:
                continue

            # Get data from current date
            cur_data = resampled_data[resampled_data['date'] == date]            

            # Get numerical features
            cur_feat = torch.tensor(cur_data[self.num_feat_names].values, dtype=torch.float)

            # Get categorical features
            if len(self.cat_feat_names) > 0:
                cur_feat = torch.cat((cur_feat, torch.tensor(get_dummies(cur_data[self.cat_feat_names]).values, dtype=torch.float)), dim=1)

            # Get indices of tickers of the current date
            cur_tick_ind = torch.tensor(cur_data['ticker'].apply(ticker_index).values)

            # Calculate final feature matrix of this date and append it to list
            num_feat = torch.zeros(self.num_nodes, cur_feat.shape[1], dtype=torch.float)
            num_feat[cur_tick_ind, :] = cur_feat
            all_feats.append(num_feat)

            # Get edge date
            first_day_of_cur_month = date.replace(day=1)
            target_edge_date = first_day_of_cur_month - dt.timedelta(days=1)

            # find index in edge_data that matches year and month of cur_edge_month
            for i in range(len(self.edge_data)):
                cur_edge_year, cur_edge_month = self.edge_data[i][0].year, self.edge_data[i][0].month
                if (cur_edge_year == target_edge_date.year) and (cur_edge_month == target_edge_date.month):
                    self.edge_data_index = i
                    break

            # get index of edge data
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
        num_feat_names = ['pvCLCL']
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

        self.dates = dates
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

        self.num_timesteps = y.shape[1]
        self.num_features = X.shape[1]

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

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        # Full 664 tickers
        #etf_tickers = ['AA', 'ABM', 'ABT', 'ADI', 'ADM', 'ADX', 'AEE', 'AEG', 'AEM', 'AEP', 'AES', 'AFG', 'AFL', 'AIG', 'AIN', 'AIR', 'AIV', 'AJG', 'ALB', 'ALK', 'ALL', 'ALV', 'AMD', 'AME', 'AMG', 'AMT', 'AN', 'ANF', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ARW', 'ASA', 'ASG', 'ASGN', 'ASH', 'ATI', 'ATO', 'ATR', 'AU', 'AVA', 'AVB', 'AVT', 'AVY', 'AWF', 'AWR', 'AXL', 'AXP', 'AZN', 'AZO', 'B', 'BA', 'BAC', 'BAX', 'BBY', 'BC', 'BCE', 'BCS', 'BDN', 'BDX', 'BEN', 'BF', 'BFS', 'BHE', 'BHP', 'BIO', 'BK', 'BKE', 'BKH', 'BKN', 'BKT', 'BLK', 'BLL', 'BMO', 'BMY', 'BOH', 'BPT', 'BRC', 'BRK', 'BRO', 'BSX', 'BTI', 'BTO', 'BWA', 'BXP', 'BYD', 'CAG', 'CAH', 'CAT', 'CB', 'CBT', 'CCK', 'CCL', 'CDE', 'CFR', 'CHD', 'CHH', 'CHN', 'CI', 'CIA', 'CIEN', 'CIF', 'CIK', 'CL', 'CLB', 'CLF', 'CLS', 'CLX', 'CMA', 'CMC', 'CMS', 'CMU', 'CNA', 'CNI', 'CNX', 'COF', 'COHU', 'COO', 'CP', 'CPB', 'CPE', 'CPK', 'CPT', 'CR', 'CRD', 'CRK', 'CRS', 'CRY', 'CSL', 'CSX', 'CTS', 'CUZ', 'CVS', 'CW', 'CWT', 'CX', 'CXE', 'D', 'DBD', 'DCI', 'DDF', 'DDS', 'DE', 'DEO', 'DGX', 'DHF', 'DHI', 'DHR', 'DHY', 'DIS', 'DLX', 'DNP', 'DOV', 'DOX', 'DRE', 'DRI', 'DRQ', 'DSM', 'DSU', 'DTE', 'DUK', 'DVN', 'DY', 'E', 'EAT', 'EBF', 'ECL', 'ED', 'EFX', 'EGP', 'EIX', 'EL', 'ELY', 'EMF', 'EMN', 'EMR', 'ENZ', 'EOG', 'EPD', 'EPR', 'EQR', 'EQT', 'ESE', 'ESS', 'ETN', 'ETR', 'EVF', 'EWH', 'EWJ', 'EWQ', 'EWS', 'EWU', 'F', 'FAF', 'FBP', 'FC', 'FCF', 'FCN', 'FCX', 'FDP', 'FDS', 'FDX', 'FE', 'FIX', 'FLS', 'FMC', 'FMS', 'FNF', 'FOE', 'FR', 'FRT', 'FSS', 'FT', 'FUN', 'GAB', 'GAM', 'GBL', 'GCI', 'GCO', 'GD', 'GE', 'GES', 'GF', 'GFF', 'GGG', 'GGT', 'GIB', 'GIL', 'GIM', 'GIS', 'GLT', 'GLW', 'GPC', 'GPI', 'GPS', 'GS', 'GT', 'GUT', 'GVA', 'GWW', 'HAE', 'HAL', 'HAS', 'HCN', 'HD', 'HE', 'HEI', 'HIG', 'HIO', 'HIW', 'HIX', 'HL', 'HMC', 'HMN', 'HNI', 'HON', 'HP', 'HQH', 'HQL', 'HR', 'HRB', 'HRL', 'HSC', 'HSY', 'HUM', 'HVT', 'HXL', 'HYB', 'IBM', 'IDA', 'IEX', 'IFF', 'IFN', 'IGT', 'IIF', 'IMAX', 'INCY', 'INFY', 'ING', 'INT', 'IO', 'IP', 'IPG', 'IQI', 'IR', 'IRM', 'IT', 'ITW', 'IVC', 'JBL', 'JCI', 'JNJ', 'JNPR', 'JOE', 'JOF', 'JPM', 'JW', 'JWN', 'K', 'KBH', 'KEP', 'KEX', 'KEY', 'KFY', 'KGC', 'KIM', 'KMB', 'KMT', 'KMX', 'KO', 'KOF', 'KR', 'KRC', 'KSM', 'KSS', 'KTF', 'LEE', 'LEG', 'LEN', 'LEO', 'LGF', 'LH', 'LII', 'LLY', 'LMT', 'LNC', 'LNN', 'LNT', 'LOW', 'LPX', 'LTC', 'LUB', 'LUV', 'LXP', 'LZB', 'MAA', 'MAC', 'MAN', 'MAR', 'MAS', 'MAT', 'MBI', 'MCA', 'MCD', 'MCK', 'MCR', 'MCS', 'MCY', 'MDC', 'MDT', 'MDU', 'MFC', 'MFL', 'MFM', 'MGA', 'MGF', 'MHF', 'MHK', 'MHN', 'MIN', 'MIY', 'MKC', 'MKL', 'MLI', 'MLM', 'MMC', 'MMM', 'MMS', 'MMT', 'MMU', 'MO', 'MOG', 'MPA', 'MQT', 'MQY', 'MRK', 'MRO', 'MSD', 'MSM', 'MTB', 'MTD', 'MTG', 'MTN', 'MTW', 'MTX', 'MTZ', 'MU', 'MUA', 'MUC', 'MUJ', 'MUR', 'MVF', 'MVT', 'MXF', 'MYC', 'MYD', 'MYE', 'MYI', 'MYJ', 'MYN', 'NAC', 'NAD', 'NBR', 'NC', 'NCA', 'NCR', 'NEM', 'NFG', 'NHI', 'NI', 'NJR', 'NKE', 'NL', 'NLY', 'NNN', 'NOC', 'NOK', 'NPK', 'NR', 'NSC', 'NSL', 'NUE', 'NUS', 'NUV', 'NVO', 'NVR', 'NWL', 'NX', 'NXP', 'NYT', 'O', 'OCN', 'ODP', 'OFG', 'OGE', 'OHI', 'OI', 'OIA', 'OII', 'OKE', 'OLN', 'OMC', 'OMI', 'ORCL', 'ORI', 'OXY', 'PAA', 'PBI', 'PCF', 'PCG', 'PCH', 'PEG', 'PEI', 'PEO', 'PEP', 'PFD', 'PFE', 'PFO', 'PG', 'PGR', 'PH', 'PHG', 'PHI', 'PHM', 'PII', 'PIM', 'PKE', 'PKI', 'PKX', 'PLD', 'PMM', 'PMO', 'PNC', 'PNM', 'PNR', 'PNW', 'PPG', 'PPL', 'PPT', 'PRGO', 'PSA', 'PSB', 'PVH', 'PWR', 'PXD', 'R', 'RAD', 'RCL', 'RCS', 'RDN', 'RE', 'REG', 'REV', 'RGA', 'RGR', 'RHI', 'RIG', 'RJF', 'RL', 'RLI', 'RMD', 'RNR', 'ROG', 'ROK', 'ROL', 'ROP', 'RPM', 'RRC', 'RS', 'RSG', 'RVT', 'RY', 'RYN', 'SAH', 'SAM', 'SBR', 'SCS', 'SEE', 'SFE', 'SGU', 'SHW', 'SJI', 'SJM', 'SJR', 'SJT', 'SKM', 'SKT', 'SKX', 'SLB', 'SLG', 'SLM', 'SMG', 'SNA', 'SNV', 'SO', 'SON', 'SOR', 'SPG', 'SPH', 'SPY', 'SRE', 'SRI', 'SSD', 'SSP', 'STE', 'STM', 'STT', 'SU', 'SUI', 'SUP', 'SWK', 'SWM', 'SWN', 'SWX', 'SWZ', 'SXI', 'SYK', 'SYY', 'TD', 'TDF', 'TDS', 'TDW', 'TDY', 'TEF', 'TEI', 'TEN', 'TEO', 'TER', 'TEVA', 'TEX', 'TFX', 'TG', 'TGI', 'THC', 'THO', 'TJX', 'TK', 'TKR', 'TLK', 'TM', 'TMO', 'TOL', 'TR', 'TRN', 'TRP', 'TSM', 'TSN', 'TTC', 'TTI', 'TUP', 'TV', 'TWI', 'TXN', 'TXT', 'TY', 'TYL', 'UDR', 'UFI', 'UGI', 'UHS', 'UHT', 'UIS', 'UL', 'UNFI', 'UNH', 'UNM', 'UNP', 'UPS', 'URI', 'USA', 'USB', 'USM', 'UVV', 'VBF', 'VFC', 'VGM', 'VKQ', 'VLO', 'VLY', 'VMC', 'VMO', 'VNO', 'VOD', 'VSH', 'VTR', 'VVI', 'VVR', 'WAB', 'WABC', 'WAT', 'WCC', 'WDC', 'WEC', 'WEN', 'WFC', 'WGO', 'WHR', 'WMB', 'WMK', 'WMT', 'WNC', 'WRE', 'WSM', 'WSO', 'WST', 'WTS', 'WWW', 'WY', 'X', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XOM', 'XRX', 'YUM', 'ZTR']
        
        window_length = 50
        step_length = 1
        etf_tickers = ['SPY', 'XLF', 'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP', 'XLE']
        num_feat_names = ['pvCLCL']
        cat_feat_names = []

        # load and prepare dataset
        loader = CRSPLoader(load_data=True,
                            load_path=os.path.join(os.getcwd(), "src", "data", "inputs"),
                            load_edge_data=True,
                            num_feat_names=num_feat_names,
                            cat_feat_names=cat_feat_names)
        loader._update_ticker_index(ticker_list=etf_tickers)
        dataset = loader.get_dataset(data=loader.select_tickers(tickers=etf_tickers),
                                     window_length=window_length,
                                     step_length=step_length)