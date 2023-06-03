import torch.nn as nn
import torch
from torch_geometric_temporal.nn import TGCN
import torch.nn.functional as F


class TGCNPO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 output_size):
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size

        # Temporal Graph Convolutional Network
        self.tgcn = TGCN(in_channels=in_channels, 
                         out_channels=out_channels)
        
        # Equals single-shot prediction
        self.linear = nn.Linear(out_channels, output_size)
        self.softmax = nn.Softmax(1)

    def forward(self, x, edge_index):
        """

        :param x: node features for T time steps
        :type x: torch tensor
        :param edge_index: torch array
        :type edge_index: torch tensor
        """

        # x: (num_nodes, num_features, T)
        # edge_index: (2, num_nonzero_adj_entries)
        h = self.tgcn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)

        # apply softmax function to respect the contraint $w_i \in [0, 1]$
        w = self.softmax(h)

        return w

DEBUG = True

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
        output_size = 4
        train_size_perc = 0.6
        seq_length = 90
        prediction_steps = 1
        print_every = 10
        n_epochs = 500

        # build dataset loader, and its train/test split
        loader = ETFsZZR()
        dataset = loader.get_dataset(num_timesteps_in=seq_length, num_timesteps_out=prediction_steps)
        train_loader, test_loader = temporal_signal_split(dataset, train_ratio=train_size_perc)

        # (1) define model
        model = TGCNPO(in_channels=8, out_channels=hidden_size, output_size=output_size)

        # (2) define loss function
        lossfn = SharpeLoss()

        # (3) define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # (4) training procedure
        training_loss_values = []
        model.train()
        for epoch in range(n_epochs + print_every): 

            for graph_data_batch in train_loader:
                
                optimizer.zero_grad()
                # compute forward probagation
                weights_pred = model(graph_data_batch.x.T, graph_data_batch.edge_index)
                
                # compute loss
                loss = lossfn(graph_data_batch.y, weights_pred, ascent=True)

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