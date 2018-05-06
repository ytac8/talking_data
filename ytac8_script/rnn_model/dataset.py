from torch.utils.data import Dataset
import pickle
import torch
import gc


class LogData(Dataset):

    def __init__(self, data, user_list, is_train):
        self.is_train = is_train
        self.dictionaries = {}
        self.user_list = user_list
        if is_train:
            self.data = data[['index', 'ip', 'app', 'os', 'device',
                              'channel', 'click_time', 'is_attributed']]
        else:
            self.data = data[['index', 'ip', 'app', 'os',
                              'device', 'channel', 'click_time']]

        # load dictionary
        # with open('../../data/pickle/Embedding/ip_dict.pkl', mode='rb') as f:
        #     self.dictionaries['ip'] = pickle.load(f)
        with open('../../data/pickle/Embedding/device_dict.pkl', mode='rb') as f:
            self.dictionaries['device'] = pickle.load(f)
        with open('../../data/pickle/Embedding/os_dict.pkl', mode='rb') as f:
            self.dictionaries['os'] = pickle.load(f)
        with open('../../data/pickle/Embedding/app_dict.pkl', mode='rb') as f:
            self.dictionaries['app'] = pickle.load(f)
        with open('../../data/pickle/Embedding/channel_dict.pkl', mode='rb') as f:
            self.dictionaries['channel'] = pickle.load(f)

        # load embedding
        # with open('../../data/pickle/Embedding/ip_emb.pkl', mode='rb') as f:
        #     self.ip_emb = pickle.load(f)
        with open('../../data/pickle/Embedding/device_emb.pkl', mode='rb') as f:
            self.device_emb = pickle.load(f)
        with open('../../data/pickle/Embedding/os_emb.pkl', mode='rb') as f:
            self.os_emb = pickle.load(f)
        with open('../../data/pickle/Embedding/app_emb.pkl', mode='rb') as f:
            self.app_emb = pickle.load(f)
        with open('../../data/pickle/Embedding/channel_emb.pkl', mode='rb') as f:
            self.channel_emb = pickle.load(f)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list.iloc[idx, :]
        feature, label, index = self._preprocess(user)
        return {"feature": feature, "label": label, "index": index}

    def _preprocess(self, user):
        ip = user['ip']
        os = user['os']
        device = user['device']
        df = self.data[(self.data['ip'] == ip) & (self.data['device'] == device) & (
            self.data['os'] == os)].sort_values(by='click_time')
        index = torch.LongTensor(df['index'].astype(int).tolist())

        # inputになるテンソルの作成
        # ip_input = torch.LongTensor(self._convert(df.ip.tolist(), 'ip'))
        app_input = torch.LongTensor(self._convert(df.app.tolist(), 'app'))
        os_input = torch.LongTensor(self._convert(df.os.tolist(), 'os'))
        device_input = torch.LongTensor(
            self._convert(df.device.tolist(), 'device'))
        channel_input = torch.LongTensor(
            self._convert(df.channel.tolist(), 'channel'))

        # embedding
        # ip_input = self.ip_emb(ip_input).data
        app_input = self.app_emb(app_input).data
        os_input = self.os_emb(os_input).data
        device_input = self.device_emb(device_input).data
        channel_input = self.channel_emb(channel_input).data

        # mat = torch.cat([ip_input, app_input, os_input,
        #                  channel_input, device_input], dim=1)
        mat = torch.cat([app_input, os_input,
                         channel_input, device_input], dim=1)

        label = None
        if self.is_train:
            label = torch.FloatTensor(
                df.is_attributed.tolist())

        del app_input, os_input, device_input, channel_input
        gc.collect()

        return mat, label, index

    def _convert(self, input_list, name):
        conved_list = []
        dictionary = self.dictionaries[name]
        for i in input_list:
            conved_list.append(dictionary[i])
        return conved_list

    def _padding(self, feature, pad_value=-1):
        length = feature.size()[0]
        width = feature.size()[1]
        padded = torch.zeros(self.max_len, width)
        padded[:length, :] = feature
        return padded
