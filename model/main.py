import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import datetime
import pandas as pd
from torch.utils.data import DataLoader
from trainer import Trainer
from predictor import Predictor
from encoder import Encoder
from dataset import UserData
from optimizer import Optimizer


def main(dataset, epochs, is_train=True, use_time=False):
    # initialize
    is_train = True
    n_epochs = epochs
    save_epoch = 10
    hidden_size = 200
    dropout_p = 0.1
    learning_rate = 0.1
    use_time = use_time
    use_cuda = True
    batch_size = 20
    checkpoint_path = None

    train_user_count = pd.read_csv(
        '../data/preprocesssed/train_user_counts.csv', header=None)
    test_user_count = pd.read_csv(
        '../data/preprocesssed/test_user_counts.csv', header=None)

    train_max_len = max(train_user_count.iloc[:, 1])
    test_max_len = max(test_user_count.iloc[:, 1])

    train_data, train_loader = dataset(dataset, batch_size, train_max_len)
    test_data, test_loader = dataset(dataset, batch_size, test_max_len)

    # データの特徴量の数を定義
    input_size = train_data.get_column_num()
    output_size = 2

    encoder = Encoder(input_size, hidden_size,
                      output_size=output_size, dropout_p=dropout_p)

    optimizer = Optimizer(encoder, lr=learning_rate)
    criterion = nn.NLLLoss()

    if torch.cuda.is_available():
        encoder.cuda()

    predictor = Predictor(encoder, test_loader,)

    if is_train:
        trainer = Trainer(train_loader, encoder, criterion,
                          optimizer, cuda=use_cuda)

        # training
        for epoch in tqdm(range(1, n_epochs + 1)):
            loss = trainer.train()
            save_model(encoder, optimizer, epoch, save_epoch)
            if epoch % 10 == 0:
                auc, rmse = pred_and_print(
                    predictor, encoder, test_loader, checkpoint_path, epoch)
            print(epoch, loss)

        print('finished training')

    else:
        checkpoint_path = '../output/save_point/' + epochs + 'epoch.pth.tar'

    return auc, rmse


def pred_and_print(predictor, model, data_loader, checkpoint, epochs=None):
    epochs = '' if epochs is None else ' ' + str(epochs) + ' epochs '
    prediction, auc = predictor.predict()
    print('========' + epochs + '=========')
    print(auc)
    return auc


def save_model(model, optimizer, epoch, save_epoch):
    if epoch % save_epoch == 0:
        model_filename = '../../../output/save_point/' + \
            'model_' + str(epoch) + 'epochs.pth.tar'

        state = {
            'state_dict': model.state_dict(),
            'encoder_optimizer': optimizer.encoder_optimizer.state_dict(),
            'decoder_optimizer': optimizer.decoder_optimizer.state_dict()
        }
        torch.save(state, model_filename)


def dataset(user_list, batch_size, max_seq_len):

        # datasetの読み込み
    dataset = UserData(user_list, max_seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    return dataset, data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting model and dataset')
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='epochs')

    args = parser.parse_args()
    now = datetime.datetime.now().strftime('%s')
    output_dir_name = args.dir_name + '/'
    epochs = args.epochs
    file_name = now + '.csv'

    try:
        os.makedirs('../log/' + output_dir_name)
    except FileExistsError as e:
        pass

    output_path = '../../../log/' + output_dir_name + file_name
    sum_aucs = 0

    with open(output_path, 'w') as f:
        f.write(str(rmse) + ',' + str(sum_aucs))
