import torch.nn as nn
import torch
from torch_geometric_temporal.nn import TGCN
import torch.nn.functional as F


class TGCNPO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 prediction_steps):
        """
        TGCN applied to portfolio optimization.

        :param in_channels: number of features per node
        :type in_channels: int
        :param out_channels: number of hidden units per hidden layer
        :type out_channels: int
        :param prediction_steps: number of steps ahead to predict
        :type prediction_steps: int
        """

        super().__init__()

        # Temporal Graph Convolutional Network
        self.tgcn = TGCN(in_channels=in_channels, 
                         out_channels=out_channels, 
                         periods=prediction_steps)
        
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index):
        """

        :param x: node features for T time steps
        :type x: torch tensor
        :param edge_index: torch array
        :type edge_index: torch tensor
        """

        h = self.tgcn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import sys
        import os
        from torch_geometric_temporal.signal import temporal_signal_split
        import pandas as pd
        import matplotlib.pyplot as plt

        # temporally add repo to path
        sys.path.append(os.path.join(os.getcwd(), "src"))

        from data.ETFsZZR import ETFsZZR
        from loss_functions.SharpeLoss import SharpeLoss

        # hyperparameter
        learning_rate = 1e-3
        hidden_size = 64
        prediction_steps = 1
        train_size_perc = 0.6
        seq_length = 90
        print_every = 10

        # build dataset loader, and its train/test split
        loader = ETFsZZR()
        dataset = loader.get_dataset(num_timesteps_in=seq_length, num_timesteps_out=prediction_steps)
        train_loader, test_loader = temporal_signal_split(dataset, train_ratio=train_size_perc)

        # (1) define model
        model = TGCNPO(in_channels=2, out_channels=hidden_size, prediction_steps=prediction_steps)

        # (2) define loss function
        lossfn = SharpeLoss()

        # (3) define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # (4) training procedure
        training_loss_values = []
        model.train()
        for epoch in range(10): 

            for grap_data_batch in train_loader:
                
                optimizer.zero_grad()
                # Get model predictions
                weights_pred = model(grap_data_batch.x, grap_data_batch.edge_index)
                
                # compute loss
                loss = lossfn(grap_data_batch.y, weights_pred, ascent=True)

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

        model.eval()