import torch.nn as nn
import torch

class DLPO(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 batch_first=True) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        # init hidden state
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)

        # init cell state
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        
        # propagate input through LSTM
        o1t, (ht, _) = self.lstm(x, (h0, c0))
        
        # collapse array first dim: (num_layers, num_batches, num_features) => (num_batches, num_features) 
        ht = ht.view(-1, self.hidden_size)
        
        # linear aggregate hidden states to match output
        o2t = self.linear(ht)

        # apply softmax function to respect the contraint $w \in [0, \inf]$
        wt = self.softmax(o2t)

        return wt
    

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import torch.utils.data as data

    # temporally add repo to path
    sys.path.append(os.path.join(os.getcwd(), "src"))

    from utils.dataset_utils import concatenate_prices_returns, create_rolling_window_array

    # load data
    source_path = os.path.dirname(os.path.dirname(__file__))
    inputs_path = os.path.join(source_path, "data", "inputs")
    prices = pd.read_excel(os.path.join(inputs_path, "etfs-zhang-zohren-roberts.xlsx"))

    # prepare dataset
    prices.set_index("date", inplace=True)
    returns = np.log(prices).diff().dropna()
    prices = prices.loc[returns.index]
    all_df = concatenate_prices_returns(prices=prices, returns=returns)

    # scale data
    timeseries = all_df.values.astype('float32')
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(timeseries)

    # hyperparameter
    input_size = all_df.shape[1]
    output_size = int(all_df.shape[1] / 2)
    hidden_size = 2
    num_layers = 1
    learning_rate = 0.01

    seq_length = 5
    train_size_perc = 0.6
    train_size = int(timeseries.shape[0] * train_size_perc)
    batch_size = 10
    n_epochs = 500
    print_every = 10

    # (1) model
    model = DLPO(input_size=input_size,
                 output_size=output_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 batch_first=True)
    
    # (2) loss fucntions - TODO: need to change it to the sharpe ratio
    lossfn = torch.nn.MSELoss()

    # (3) optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define arrays of rolling window observations
    X, y = create_rolling_window_array(data=training_data, seq_length=seq_length)

    # define train and test datasets
    X_train, y_train = X[0:train_size], y[0:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # define data loaders
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

    # (4) training procedure
    for epoch in range(n_epochs + print_every):
       
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # compute forward probagation
            y_pred = model.forward(X_batch)

            # compute loss
            loss = lossfn(y_pred, y_batch)
            
            # compute gradients and backpropagate
            loss.backward()
            optimizer.step()

        if epoch % print_every == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



