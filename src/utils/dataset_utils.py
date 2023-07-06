import pandas as pd
import torch

def create_rolling_indices(num_timesteps_in, num_timesteps_out, n_timesteps):
    
    # generate rolling window indices
    indices = [
        (i, i + (num_timesteps_in + num_timesteps_out))
        for i in range(n_timesteps - (num_timesteps_in + num_timesteps_out) + 1)
    ]

    return indices

def create_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out):
    
    if features.shape[0] != target.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[0]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps)

    # use rolling window indices to subset data
    window_features, window_target = torch.zeros((len(indices), num_timesteps_in, features.shape[1])), torch.zeros((len(indices), num_timesteps_out, target.shape[1]))
    batch = 0
    for i, j in indices:
        window_features[batch, :, :] = torch.tensor(features[i : i + num_timesteps_in, :])
        window_target[batch, :, :] = torch.tensor(target[ i + num_timesteps_in : j, :])

        batch += 1

    return window_features, window_target

def create_rolling_window_ts_for_graphs(target, features, num_timesteps_in, num_timesteps_out):
    
    if features.shape[-1] != target.shape[-1]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[2]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps)

    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:
        window_features.append((features[:, :, i : i + num_timesteps_in]).numpy())
        window_target.append((target[:, i + num_timesteps_in : j]).numpy())

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

    return all_df

DEBUG = False

if __name__ == "__main__":

    if DEBUG:

        import os
        import numpy as np
        from sklearn.model_selection import train_test_split

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
        features = concatenate_prices_returns(prices=prices, returns=returns)
        idx = features.index
        returns = returns.loc[idx].values.astype('float32')
        prices = prices.loc[idx].values.astype('float32')
        features = features.loc[idx].values.astype('float32')  

        # define train and test datasets
        X_train, X_test, prices_train, prices_test = train_test_split(features, prices, test_size=test_ratio, random_state=1)
        X_train, X_val, prices_train, prices_val = train_test_split(X_train, prices_train, test_size=test_ratio, random_state=1) 

        X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                        target=prices_train,
                                                        num_timesteps_in=num_timesteps_in,
                                                        num_timesteps_out=num_timesteps_out)
        

        