import torch.nn as nn
import torch

"""
Implementation of the LSTM version of the deep learning for 
portfolio optimization from https://arxiv.org/abs/2005.13665
"""

class GCN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 batch_first=True) -> None:
        super().__init__()

    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import os
        import sys
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import torch.utils.data as data
        import matplotlib.pyplot as plt

        # temporally add repo to path
        sys.path.append(os.path.join(os.getcwd(), "src"))

        from utils.dataset_utils import create_rolling_window_rets_vol_array, concatenate_prices_returns
        from utils.conn_data import compute_realized_ewma_vol
        from loss_functions.SharpeLoss import SharpeLoss

        # load data
        source_path = os.path.dirname(os.path.dirname(__file__))
        inputs_path = os.path.join(source_path, "data", "inputs")
        prices = pd.read_excel(os.path.join(inputs_path, "etfs-zhang-zohren-roberts.xlsx"))

        # prepare dataset
        prices.set_index("date", inplace=True)
        prices = prices.shift(-1)

        returns = np.log(prices).diff().dropna()
        prices = prices.loc[returns.index]
        vols = compute_realized_ewma_vol(returns=returns, window=50)
        features = concatenate_prices_returns(prices=prices, returns=returns)

        idx = vols.index
        returns = returns.loc[idx].values.astype('float32')
        prices = prices.loc[idx].values.astype('float32')
        vols = vols.loc[idx].values.astype('float32')
        features = features.loc[idx].values.astype('float32')  

        if returns.shape[0] == prices.shape[0] == vols.shape[0] == features.shape[0]:
            pass
        else:
            raise Exception("Some of the arrays have different sizes")

        # scale data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(features)

        # hyperparameter
        input_size = features.shape[1]
        output_size = int(features.shape[1] / 2)
        hidden_size = 64
        num_layers = 1
        learning_rate = 1e-3

        seq_length = 90
        train_size_perc = 0.6
        train_size = int(features.shape[0] * train_size_perc)
        batch_size = 10
        n_epochs = 500
        print_every = 10

        # (1) model
        model = DLPO(input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True)
        
        # (2) loss fucntions
        lossfn = SharpeLoss()

        # (3) optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # define arrays of rolling window observations
        X, prices = create_rolling_window_rets_vol_array(return_prices=features, 
                                                        prices=prices,
                                                        seq_length=seq_length)

        # define train and test datasets
        X_train, prices_train = X[0:train_size], prices[0:train_size]
        X_test, prices_test = X[train_size:], prices[train_size:]

        # define data loaders
        train_loader = data.DataLoader(data.TensorDataset(X_train, prices_train), shuffle=True, batch_size=batch_size)

        # (4) training procedure
        training_loss_values = []
        for epoch in range(n_epochs + print_every):
        
            model.train()
            for X_batch, prices_batch in train_loader:

                optimizer.zero_grad()
                # compute forward probagation
                weights_pred = model.forward(X_batch)

                # compute loss
                loss = lossfn(prices_batch, weights_pred, ascent=True)
                
                # compute gradients and backpropagate
                loss.backward()
                optimizer.step()

            if epoch % print_every == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item() * -1))
                training_loss_values.append(loss.item() * -1)
        
        training_loss_df = pd.DataFrame(training_loss_values, columns=["sharpe_ratio"])

        print("Average training sharpe {}".format(training_loss_df.mean().item()))

        training_loss_df.hist(bins=10)
        plt.show()

        end = 1



