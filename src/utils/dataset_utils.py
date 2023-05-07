import pandas as pd
import torch
import numpy as np


def create_rolling_window_array(data, seq_length):
    
    X = []
    y = []
    for i in range(len(data) - seq_length - 1):
        _X = data[i:(i + seq_length)]
        _y = data[(i + seq_length)]
        X.append(_X)
        y.append(_y)

    y_out = torch.Tensor(y)
    X_out = torch.Tensor(X)

    return X_out, y_out

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