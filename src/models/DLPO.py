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
                 num_timesteps_out,
                 batch_first=True,
                 bidirectional=False,
                 proj_size=0) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps_out = num_timesteps_out
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional,
                            proj_size=proj_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x, long_only):

        if self.batch_first:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[1]

        if self.bidirectional:
            D = 2
        else:
            D = 1

        if self.proj_size > 0:
            hout = self.proj_size
        else:
            hout = self.hidden_size

        hcell = self.hidden_size

        # init hidden state
        # (D * num_layers, N, hout)​
        h0 = torch.zeros(D * self.num_layers, batch_size, hout)

        # init cell state
        # (D * num_layers, N, Hcell)​
        c0 = torch.zeros(D * self.num_layers, batch_size, hcell)
        
        # propagate input through LSTM
        o1t, (ht, _) = self.lstm(x, (h0, c0))

        # linear aggregate hidden states to match output
        o2t = self.linear(o1t)

        # subset prediction steps ahead
        pred_start = -self.num_timesteps_out - 1
        pred_end = -1
        o2t = o2t[:, pred_start:pred_end, :]

        wts = []
        for i in range(o2t.shape[0]):
            if long_only:
                wts.append(self.softmax(o2t[i, :, :]))
            else:
                wt_star = self.tanh(o2t[i, :, :])

                pos_mask = (wt_star > 0)
                neg_mask = (wt_star < 0)

                wt_star_pos = wt_star * pos_mask.float()
                wt_star_neg = wt_star * neg_mask.float()

                wt_star_pos_sum = wt_star_pos.sum()
                wt_star_neg_sum = wt_star_neg.sum()

                wt_star = wt_star_pos / (wt_star_pos_sum + 1e-9) - wt_star_neg / (wt_star_neg_sum + 1e-9)

                wts.append(wt_star)

        wt = torch.stack(wts, dim=0)

        return wt