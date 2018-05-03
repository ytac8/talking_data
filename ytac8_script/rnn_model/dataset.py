from torch.utils.data import Dataset
from torch.autograd import Variable
import pickle
import torch


class UserData(Dataset):

    def __init__(self, data_reader, batch_size, is_train):
        self.data = data_reader
        self.is_train = is_train
        self.batch_size = batch_size

        # load dictionary
        with open('../data/pickle/Embedding/device_dict.pkl', mode='rb') as f:
            self.dictionaries['device'] = pickle.load(f)
        with open('../data/pickle/Embedding/os_dict.pkl', mode='rb') as f:
            self.dictionaries['os'] = pickle.load(f)
        with open('../data/pickle/Embedding/app_dict.pkl', mode='rb') as f:
            self.dictionaries['app'] = pickle.load(f)
        with open('../data/pickle/Embedding/channel_dict.pkl', mode='rb') as f:
            self.dictionaries['channel'] = pickle.load(f)

        # load embedding
        with open('../data/pickle/Embedding/device_emb.pkl', mode='rb') as f:
            self.device_emb = pickle.load(f)
        with open('../data/pickle/Embedding/os_emb.pkl', mode='rb') as f:
            self.os_emb = pickle.load(f)
        with open('../data/pickle/Embedding/app_emb.pkl', mode='rb') as f:
            self.app_emb = pickle.load(f)
        with open('../data/pickle/Embedding/channel_emb.pkl', mode='rb') as f:
            self.channel_emb = pickle.load(f)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        user_id = self.user_list[idx]
        user_data, label = self._preprocess(user_id)
        return {"feature": user_data, "label": label}

    def _preprocess(self):
        df = self.data.get_chunk(self.batch_size)
        df = df.sort_values(by='click_time')

        # inputになるテンソルの作成
        app_input = torch.LongTensor(self._convert(df.app.tolist(), 'app'))
        os_input = torch.LongTensor(self._convert(df.os.tolist(), 'os'))
        device_input = torch.LongTensor(self._convert(df.device.tolist(), 'device'))
        channel_input = torch.LongTensor(self._convert(df.channel.tolist(), 'channel'))

        app_embedded = self.app_emb(app_input)
        device_embedded = self.device_emb(device_input)
        os_embedded = self.os_emb(os_input)
        channel_embedded = self.channel_emb(channel_input)

        mat = torch.cat([app_embedded, os_embedded,
                         channel_embedded, device_embedded], dim=1)

        label = None
        if self.is_train:
            label = df.is_attributed.tolist()

        return mat.data, label

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
