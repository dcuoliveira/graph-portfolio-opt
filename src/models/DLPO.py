import torch.nn as nn

class DLPO(nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            output_size=input_size)
        self.softmax = nn.Softmax(input_size=input_size, output_size=input_size)

    def forward(self, x):
        x, h = self.lstm(x)
        w = self.softmax(x)

        return w