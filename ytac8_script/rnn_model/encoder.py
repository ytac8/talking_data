import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_layers=3, output_size=2, dropout_p=0.1):

        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        torch.backends.cudnn.benchmark = True

    def forward(self, input):
        outputs, hidden = self.gru(input)
        outputs = self.linear(outputs)
        outputs = F.relu(outputs)
        outputs = self.softmax(outputs)
        return outputs
