import torch


class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer, cuda, clip=5):

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = cuda
        self.clip = clip

    def train(self):

        self.optimzer.scheduler_step()

        for batch_i, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            input_var = batch
