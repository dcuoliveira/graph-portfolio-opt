import pandas as pd
import torch


def create_rolling_window_rets_vol_array(return_prices, prices, seq_length):
    
    return_prices_out = []
    prices_out = []
    for i in range(len(return_prices) - seq_length - 1):

        # features
        return_prices_tmp = return_prices[i:(i + seq_length)]

        # targets
        prices_tmp = prices[(i + seq_length)]

        # append all
        return_prices_out.append(return_prices_tmp)
        prices_out.append(prices_tmp)

    return_prices_out = torch.Tensor(return_prices_out)
    prices_out = torch.Tensor(prices_out)

    return return_prices_out, prices_out

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