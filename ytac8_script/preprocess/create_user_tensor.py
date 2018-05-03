import pandas as pd
import pickle
import torch
import gc


def create(os, app, channel, device, user_list):
    for user in user_list:
        df = pd.read_csv(
            '../data/preprocessed/user_data/data_' + str(user) + '.csv')
        df = df.sort_values(by='click_time')
        dummied_app = torch.Tensor(app.transform(df.app.tolist()))
        dummied_os = torch.Tensor(os.transform(df.os.tolist()))
        dummied_channel = torch.Tensor(channel.transform(df.channel.tolist()))
        dummied_device = torch.Tensor(device.transform(df.device.tolist()))
        tensor = torch.cat((dummied_app, dummied_channel,
                            dummied_os, dummied_device), dim=1)
        with open("../data/preprocessed/dummied_user_data" + str(user) + ".pkl", 'wb') as f:
            pickle.dump(tensor, f, pickle.HIGHEST_PROTOCOL)
        gc.collect()


def list_split(list, nsplit, i):
    block = int(len(list) / nsplit)
    return list[i - 1 * block:i * block]


if __name__ == "__main__":

    users = pd.read_csv('../data/preprocessed/user_list.csv', header=None)
    user_list = users.iloc[:, 0].tolist()

    with open('../data/pickle/os.pkl', mode='rb') as f:
        os = pickle.load(f)
    with open('../data/pickle/app.pkl', mode='rb') as f:
        app = pickle.load(f)
    with open('../data/pickle/channel.pkl', mode='rb') as f:
        channel = pickle.load(f)
    with open('../data/pickle/device.pkl', mode='rb') as f:
        device = pickle.load(f)

    user_list = pd.read_csv('../data/preprocessed/user_list.csv')
