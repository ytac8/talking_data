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
from dataset import LogData
import gc
from optimizer import Optimizer


def main(epochs, is_train=True, use_time=False):
    # initialize
    is_train = True
    n_epochs = epochs
    save_epoch = 10
    hidden_size = 100
    dropout_p = 0.1
    learning_rate = 0.1
    use_time = use_time
    use_cuda = True
    batch_size = 1
    checkpoint_path = None

    # data_path = '../../data/'
    # train_df = pd.read_csv(data_path + 'raw/train.csv', chunksize=chunk_size)
    # test_df = pd.read_csv(data_path + 'raw/test.csv', chunksize=chunk_size)
    df = pd.read_hdf(
        '../../data/featureh5/concated_data_for_dl.h5', 'table')
    train_df = df[df['is_train'] == 1]
    test_df = df[df['is_train'] == 0]
    train_user = pd.read_csv(
        '../../data/preprocessed/ip_device_os_train_user_list.csv')
    test_user = pd.read_csv(
        '../../data/preprocessed/test_ip_device_os_list.csv')

    # debug
    # train_user = train_user[train_user['ip'] == 5348]
    # print(len(train_user))

    train_data, train_loader = dataset(
        train_df, train_user, batch_size, True)
    test_data, test_loader = dataset(
        test_df, test_user, batch_size, False)
    del df, train_df, test_df, train_user, test_user
    gc.collect()

    # データの特徴量の数を定義
    input_size = 200
    output_size = 2

    encoder = Encoder(input_size, hidden_size,
                      output_size=output_size, dropout_p=dropout_p)

    optimizer = Optimizer(encoder, lr=learning_rate)
    criterion = nn.NLLLoss()

    if torch.cuda.is_available():
        print('use cuda')
        encoder.cuda()

    predictor = Predictor(test_loader, encoder, criterion, use_cuda)

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
        auc, predictions = pred_and_print(
            predictor, encoder, test_loader, checkpoint_path)
        predictions.to_csv('../output/results/rnn/predictions.csv', index=None)

    return auc


def pred_and_print(predictor, model, data_loader, checkpoint, epochs=None):
    epochs = '' if epochs is None else ' ' + str(epochs) + ' epochs '
    predict, index, auc = predictor.predict()
    predictions = torch.cat([index, predict], dim=1)
    predictions = pd.DataFrame(
        predictions.numpy(), columns=['index', 'predict'])

    print('========' + epochs + '=========')
    print(auc)
    return auc, predictions


def save_model(model, optimizer, epoch, save_epoch):
    if epoch % save_epoch == 0:
        model_filename = '../output/save_point/' + \
            'model_' + str(epoch) + 'epochs.pth.tar'

        state = {
            'state_dict': model.state_dict(),
            'encoder_optimizer': optimizer.encoder_optimizer.state_dict(),
            'decoder_optimizer': optimizer.decoder_optimizer.state_dict()
        }
        torch.save(state, model_filename)


def dataset(data, ip_list, batch_size, is_train):

    # datasetの読み込み
    dataset = LogData(data, ip_list, is_train)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataset, data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting model and dataset')
    parser.add_argument('-ep', '--epochs', default=10, metavar='epochs', type=int,
                        help='epochs')
    # parser.add_argument('-t', '--train', default=, metavar='epochs', type=int,
    #                     help='epochs')

    args = parser.parse_args()
    now = datetime.datetime.now().strftime('%s')
    output_dir_name = 'log/'
    epochs = args.epochs
    file_name = now + '.csv'

    try:
        os.makedirs('../log/' + output_dir_name)
    except FileExistsError as e:
        pass

    output_path = '../' + output_dir_name + file_name
    main(epochs)
    sum_aucs = 0

    with open(output_path, 'w') as f:
        f.write(str(sum_aucs))
