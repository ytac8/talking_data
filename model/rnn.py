import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=2):
        super(RNN, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers=3)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid(hidden_size, 2)

    def foward(self, x, h, y):
        output, hidden = self.lstm(x, h)
        output = self.linear(output)
        output = F.relu(output)
        output = self.sigmoid(output)
        return output
