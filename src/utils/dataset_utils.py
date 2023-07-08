import pandas as pd
import torch

def timeseries_train_test_split(X, y, train_ratio):
    train_ratio = int(len(X) * train_ratio)
    
    X_train = X[:train_ratio, :]
    y_train = y[:train_ratio, :]
    X_test = X[train_ratio:, :]
    y_test = y[train_ratio:, :]

    return X_train, X_test, y_train, y_test

def create_rolling_indices(num_timesteps_in, num_timesteps_out, n_timesteps, fix_start):
    
    # generate rolling window indices
    indices = [
        (0 if fix_start else i, i + (num_timesteps_in + num_timesteps_out))
        for i in range(n_timesteps - (num_timesteps_in + num_timesteps_out) + 1)
    ]

    return indices

def create_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False):
    """"
    This function is used to create the rolling window time series to be used on DL ex-GNN.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
        
    if features.shape[0] != target.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[0]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)
    
    # use rolling window indices to subset data
    window_features, window_target = torch.zeros((len(indices), num_timesteps_in, features.shape[1])), torch.zeros((len(indices), num_timesteps_out + 1, target.shape[1]))
    batch = 0
    for i, j in indices:
        window_features[batch, :, :] = torch.tensor(features[i:(i + num_timesteps_in), :])
        window_target[batch, :, :] = torch.tensor(target[(i + num_timesteps_in - 1):j, :])

        batch += 1

    return window_features, window_target

def create_rolling_window_ts_for_graphs(target, features, num_timesteps_in, num_timesteps_out, fix_start=False):
    """"
    This function is used to create the rolling window time series to be used on GNNs.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
    if features.shape[-1] != target.shape[-1]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[2]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)

    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:
        window_features.append((features[:, :, i : i + num_timesteps_in]).numpy())
        window_target.append((target[:, (i + num_timesteps_in - 1):j]).numpy())

    return window_features, window_target

def concatenate_prices_returns(prices, returns):
    prices_names = list(prices.columns)
    returns_names = list(returns.columns)

    names = list(set(prices_names) & set(returns_names))
    
    all = []
    for name in names:
        all.append(prices[[name]].rename(columns={name: "{} price".format(name)}))
        all.append(returns[[name]].rename(columns={name: "{} ret".format(name)}))
    all_df = pd.concat(all, axis=1)

    return all_df, names

DEBUG = False

if __name__ == "__main__":

    if DEBUG:

        import os
        import numpy as np

        num_timesteps_in = 100
        num_timesteps_out = 1
        test_ratio = 0.2

        # relevant paths
        source_path = os.getcwd()
        inputs_path = os.path.join(source_path, "src", "data", "inputs")

        # prepare dataset
        prices = pd.read_excel(os.path.join(inputs_path, "etfs-zhang-zohren-roberts.xlsx"))
        prices.set_index("date", inplace=True)
        returns = np.log(prices).diff().dropna()
        prices = prices.loc[returns.index]
        features, names = concatenate_prices_returns(prices=prices, returns=returns)
        idx = features.index
        returns = returns[names].loc[idx].values.astype('float32')
        prices = prices[names].loc[idx].values.astype('float32')
        features = features.loc[idx].values.astype('float32')  

        # define train and test datasets
        X_train, X_test, prices_train, prices_test = timeseries_train_test_split(features, prices, test_ratio=test_ratio)
        X_train, X_val, prices_train, prices_val = timeseries_train_test_split(X_train, prices_train, test_ratio=test_ratio) 

        X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                         target=prices_train,
                                                         num_timesteps_in=num_timesteps_in,
                                                         num_timesteps_out=num_timesteps_out)        

        