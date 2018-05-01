import torch.functional as F
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_layers=3, output_size=2, dropout_p=0.1):

        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def foward(self, input_variable, input_length):
        input_length = input_length.squeeze().data.tolist()
        input_variable = nn.utils.rnn.pack_padded_sequence(
            input_variable, input_length, batch_first=True)

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(input_variable)
        outputs = self.linear(outputs)
        outputs = F.relu(outputs)
        outputs = self.sigmoid(outputs)

        return outputs
