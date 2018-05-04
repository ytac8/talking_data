from sklearn.metrics import roc_auc_score


class Predictor():

    def __init__(self, data_loader, model, criterion, cuda):
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.use_cuda = cuda

    def predict(self):
        self.optimizer.scheduler_step()

        for batch_i, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            input_variable = batch['feature']
            length = batch['length']
            target = batch['label']
            predict = self.model(input_variable, length)

        auc = roc_auc_score(target, predict)

        return predict, auc
