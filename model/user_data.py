from torch.utils import Dataset
import torch
import pandas as pd
import numpy as np


class UserData(Dataset):

    def __init__(self, user_list, max_len):
        self.user_list = pd.read_csv(user_list)
        self.user_list = self.user_list.iloc[:, 0].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.user_lis
        user_data, label = self._preprocess(user_id)
        padded = torch.from_numpy(self._padding(user_data))
        return {"feature": padded, "label": label}

    def _preprocess(self, user_id):
        df = pd.read_csv(
            '../data/preprocessed/user_data/data_' + str(self.user) + '.csv')
        df = df.sort_values(by='click_time')
        label = df.is_attribute
        dummied_app = self.app.transform(df.app.tolist())
        dummied_os = self.os.transform(df.os.tolist())
        dummied_channel = self.channel.transform(df.channel.tolist())
        dummied_device = self.device.transform(df.device.tolist())
        mat = np.concatenate((dummied_app, dummied_channel,
                              dummied_os, dummied_device), axis=1)
        return mat, label

    def _padding(self, feature, pad_value=-1):
        pad_len = self.max_len - len(feature)
        if(feature.shape) == 1:
            padded = np.pad(feature, [[0, pad_len]],
                            'constant', constant_values=pad_value)
        elif len(feature.shape) == 2:
            padded = np.pad(feature, [[0, pad_len], [0, 0]],
                            'constant', constant_values=pad_value)
        else:
            print('feature dimension must be 1 or 2')
        return padded
