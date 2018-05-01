class Trainer():

    def __init__(self, data_loader, model, criterion, optimizer, cuda, clip=5):

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = cuda
        self.clip = clip

    def train(self):

        self.optimizer.scheduler_step()

        for batch_i, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            loss = 0
            input_variable = batch['feature']
            target = batch['target']
            length = batch['length']
            predict = self.model(input_variable, length)

            # lossの計算
            loss += self.criterion(predict, target)
            loss.backward()
            self.optimizer.gradient_clip(self.clip)
            self.optimizer.step()

        return loss.data[0]
