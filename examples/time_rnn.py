import torch
import torch.nn as nn


class TimeRNN(nn.Module):

    def __init__(self, input_size, hidden_size=2, num_layers=1):
        super(TimeRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.state = None
        self.sigmoid = nn.Sigmoid()

    def reset(self):
        self.state = None

    #returns mu, sigma
    def forward(self, x):
        # [[t0, y0, f0], [t1, y1, f1], ...]
        #+ [xi[2][0].view(xi[2][0].size(0)
        x = [torch.cat([xi[i].view(xi[i].size(0), -1) for i in range(3)], dim=1) for xi in x]
        x = torch.stack(x, dim=1)
        out, self.state = self.rnn(x, self.state)
        out[:, -1, 0] = self.sigmoid(out[:, -1, 0])
        # out[:, -1, 1] = torch.exp(0.5*out[:, -1, 1])
        return out[:, -1, 0], torch.exp(0.5*out[:, -1, 1])
