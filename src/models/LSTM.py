import torch.nn as nn
    
class LSTM(torch.nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, 1)

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
        
        return o2t
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:

        import os
        import sys
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch.utils.data as data
        import torch
        from sklearn.preprocessing import MinMaxScaler

        # temporally add repo to path
        sys.path.append(os.path.join(os.getcwd(), "src"))

        from utils.dataset_utils import create_rolling_window_array

        # load toy data
        df = pd.read_csv(os.path.join(os.getcwd(), "src", "data", "inputs", "lstm_toy.csv"))
        timeseries = df[["Passengers"]].values.astype('float32')

        # scale data
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(timeseries)

        # hyperparameter
        hidden_size = 2
        num_layers = 1
        num_classes = 1
        learning_rate = 0.01

        seq_length = 4
        train_size_perc = 0.6
        train_size = int(timeseries.shape[0] * train_size_perc)
        batch_size = 10
        n_epochs = 500
        print_every = 10

        # (1) model
        model = LSTM(input_size=timeseries.shape[1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # (2) loss fucntions
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
            
            """""
            model.train() tells your model that you are training the model. This helps inform
            layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation. 
            
            For instance, in training mode, BatchNorm updates a
            moving average on each new batch; whereas, for evaluation mode, these updates are frozen
            """
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

    model.eval()
    y_pred = model(X).data.numpy()
    y_true = y.reshape(y.shape[0], y.shape[1]).data.numpy()

    # inverse transform
    y_pred = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(y_true)

    plt.axvline(x=train_size, c='r', linestyle='--')

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.suptitle('Time-Series Prediction')
    plt.show()



