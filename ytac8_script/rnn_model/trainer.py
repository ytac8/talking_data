from tqdm import tqdm
import torch


class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer, cuda,
                 clip=5, mini_set_size=500):

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = cuda
        self.clip = clip
        self.mini_set_size = mini_set_size

    def train(self):

        self.optimizer.scheduler_step()
        device = torch.device("cuda" if self.use_cuda else "cpu")

        for batch_i, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            batch_len = batch['feature'].size(1)
            # batch_size = batch['feature'].size(0)
            length = batch['length']
            loss = 0
            # for i in range(int(batch_len / self.mini_set_size) + 1):
            #     start_point = i * self.mini_set_size
            #     if batch_len > self.mini_set_size * i:
            #         input_variable = batch['feature'][:,
            #                                           start_point:start_point + self.mini_set_size, :].cuda()
            #         target = batch['label'][:, start_point:start_point +
            #                                 self.mini_set_size].view(-1).long().cuda()
            #     else:
            #         input_variable = batch['feature'][:, i *
            #                                           self.mini_set_size:batch_len, :].cuda()
            #         target = batch['label'][:, i *
            #                                 self.mini_set_size:batch_len].long().cuda()
            #     predict = self.model(input_variable, length)
            #     predict = predict.view(-1, 2)

            #     # lossの計算
            #     loss += self.criterion(predict, target)

            input_variable = batch['feature'].to(device)
            target = batch['label'].long().to(device)
            input_variable.requires_grad_()
            target.requires_grad_()
            predict = self.model(input_variable, length).view(-1, 2)
            loss += self.criterion(predict, target)

            loss.backward()
            self.optimizer.gradient_clip(self.clip)
            self.optimizer.step()

        return loss.data[0]
