import torch
import torch.nn as nn
import numpy as np

class SharpeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prices, weights, ascent=True, annualize=True):
        
        # asset returns
        asset_returns = torch.diff(torch.log(prices), dim=1)

        # portfolio returns
        portfolio_returns = torch.mul(weights, asset_returns)

        # portfolio sharpe
        sharpe_ratio = (torch.mean(portfolio_returns) / torch.std(portfolio_returns)) * (np.sqrt(252) if annualize else 1)

        return sharpe_ratio * (-1 if ascent else 1)
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        import os
        import sys
        import torch.utils.data as data

        sys.path.append(os.path.join(os.getcwd(), "src"))

        from utils.dataset_utils import concatenate_prices_returns, create_rolling_window_ts, timeseries_train_test_split
        from models.DLPO import DLPO

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
        X_train, X_test, prices_train, prices_test = timeseries_train_test_split(features, prices, test_ratio=0.2)
        X_train, X_val, prices_train, prices_val = timeseries_train_test_split(X_train, prices_train, test_ratio=0.2) 

        num_timesteps_out = 1
        X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                         target=prices_train,
                                                         num_timesteps_in=50,
                                                         num_timesteps_out=num_timesteps_out)

        # define data loaders
        train_loader = data.DataLoader(data.TensorDataset(X_train, prices_train), shuffle=False, batch_size=10, drop_last=True)

        # (1) model
        model = DLPO(input_size=4 * 2,
                     output_size=4,
                     hidden_size=64,
                     num_layers=1,
                     num_timesteps_out=num_timesteps_out,
                     batch_first=True)

        # (2) loss fucntion
        lossfn = SharpeLoss()
        
        (X_batch, prices_batch) = next(iter(train_loader))
                    
        # compute forward propagation
        # NOTE - despite num_timesteps_out=1, the predictions are being made on the batch_size(=10) dimension. Need to fix that.
        weights_pred = model.forward(X_batch)

        # compute loss
        loss = lossfn(prices_batch, weights_pred, ascent=True)