import torch
import torch.optim as optim
from torch.optim import lr_scheduler


class Optimizer():

    def __init__(self, encoder, decoder=None, lr=0.01, model_name='rnn'):
        self.encoder = encoder
        self.decoder = decoder
        self.model_name = model_name
        self.encoder_optimizer = optim.SGD(
            encoder.parameters(), lr=lr, momentum=0.9)
        self.decoder_optimizer = optim.SGD(
            decoder.parameters(), lr=lr, momentum=0.9)
        self.encoder_scheduler = lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=10, gamma=0.8)
        self.decoder_scheduler = lr_scheduler.StepLR(
            self.decoder_optimizer, step_size=10, gamma=0.8)

    def zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.model_name == 'seq2seq':
            self.decoder_optimizer.zero_grad()

    def step(self):
        self.encoder_optimizer.step()
        if self.model_name == 'seq2seq':
            self.decoder_optimizer.step()

    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.model_name == 'seq2seq':
            self.decoder_scheduler.step()

    def gradient_clip(self, clip):
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        if self.model_name == 'seq2seq':
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
