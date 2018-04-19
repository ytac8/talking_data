import pandas as pd
import torch


class Preprocesser():

    def __init__(self, os_label, app_label, device_label, channel_label):
        self.os = os_label
        self.app = app_label
        self.device = device_label
        self.channel = channel_label

    def train_val_split(self, train_val_rate=0.8):
        data_num = len(self.df)
        train_data = self.data.iloc[:data_num * train_val_rate, :]
        val_data = self.data.iloc[data_num * train_val_rate:, :]
        return train_data, val_data

    def preprocess(self, user_id):
        df = pd.read_csv(
            '../data/preprocessed/user_data/data_' + str(self.user) + '.csv')
        df = df.sort_values(by='click_time')
        dummied_app = torch.Tensor(self.app.transform(df.app.tolist()))
        dummied_os = torch.Tensor(self.os.transform(df.os.tolist()))
        dummied_channel = torch.Tensor(
            self.channel.transform(df.channel.tolist()))
        dummied_device = torch.Tensor(
            self.device.transform(df.device.tolist()))
        tensor = torch.cat((dummied_app, dummied_channel,
                            dummied_os, dummied_device), dim=1)
        return tensor
