import torch.nn as nn
import torch

"""
Implementation of the LSTM version of the deep learning for 
portfolio optimization from https://arxiv.org/abs/2005.13665
"""

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

        # apply softmax function to respect the contraint $w_i \in [0, 1]$
        wt = self.softmax(o2t)

        return wt