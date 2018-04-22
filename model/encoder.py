import torch.functional as F
import torch.nn as nn


class Encoder():

    def __init__(self, input_size, hidden_size, num_layers=3, output_size=2, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid(hidden_size, output_size)

    def foward(self, x, h, y):
        output, hidden = self.lstm(x, h)
        output = self.linear(output)
        output = F.relu(output)
        output = self.sigmoid(output)
        return output
