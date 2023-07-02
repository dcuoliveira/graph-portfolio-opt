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