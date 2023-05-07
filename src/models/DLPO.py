import torch.nn as nn

class DLPO(nn.Module):
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

