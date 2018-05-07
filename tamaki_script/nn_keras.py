import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '12'
import gc
from sklearn.metrics import roc_auc_score
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('load train....')
train_df = pd.read_hdf('../h5/X_train_add_supplement.h5', 'table')
print('load test....')
test_df = pd.read_hdf('../h5/X_test_add_supplement.h5', 'table')

len_train = len(train_df)
val_size = 2500000
val_df = train_df[(len_train-val_size):]
train_df = train_df[:(len_train-val_size)]
y_train = train_df['is_attributed'].values
y_val = val_df['is_attributed'].values
drop_cols = ['click_time', 'ip', 'minute', 'second', 'ip_day_hour_minute_second_count', 'is_attributed']
train_df.drop(drop_cols, axis=1, inplace=True)
val_df.drop(drop_cols, axis=1, inplace=True)

print ('neural network....')
max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'idhc': np.array(dataset.ip_day_hour_count),
        'iac': np.array(dataset.ip_app_count),
        'ipoc': np.array(dataset.ip_app_os_count),
        'idc': np.array(dataset.ip_device_count),
        'acc': np.array(dataset.app_channel_count),
        'idhmc': np.array(dataset.ip_day_hour_minute_count),
        'idoau': np.array(dataset.ip_device_os_app_unique),
        'idu': np.array(dataset.ip_device_unique),
        'iau': np.array(dataset.ip_app_unique),
        'acu': np.array(dataset.app_channel_unique),
        'icu': np.array(dataset.ip_channel_unique),
        'nc': np.array(dataset.nextClick)
    }
    return X

train_df = get_keras_data(train_df)
val_df = get_keras_data(val_df)

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
in_d = Input(shape=[1], name = 'd')
in_idhc = Input(shape=[1], name = 'idhc')
in_iac = Input(shape=[1], name = 'iac')
in_ipoc = Input(shape=[1], name = 'ipoc')
in_idc = Input(shape=[1], name = 'idc')
in_acc = Input(shape=[1], name = 'acc')
in_idhmc = Input(shape=[1], name = 'idhmc')
in_idoau = Input(shape=[1], name = 'idoau')
in_idu = Input(shape=[1], name = 'idu')
in_iau = Input(shape=[1], name = 'iau')
in_acu = Input(shape=[1], name = 'acu')
in_icu = Input(shape=[1], name = 'icu')
in_nc = Input(shape=[1], name = 'nc')
cat_fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os)])
neu_fe = concatenate([(in_h), (in_d), (in_idhc), (in_iac), (in_ipoc), (in_idc),
                 (in_acc), (in_idhmc), (in_idoau), (in_idu), (in_iau),
                 (in_acu), (in_icu), (in_nc)])
cat_fe = SpatialDropout1D(0.2)(cat_fe)
cat_fe = Flatten()(cat_fe)
s_dout = concatenate([cat_fe, neu_fe])

x = Dropout(0.2)(Dense(512, activation='relu')(s_dout))
x = Dropout(0.2)(Dense(256, activation='relu')(x))
x = Dropout(0.2)(Dense(128, activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_idhc,in_iac,in_ipoc,in_idc,in_acc,in_idhmc,in_idoau,in_idu, in_iau,in_acu,in_icu,in_nc], outputs=outp)

batch_size = 1000000
epochs = 5
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.002, 0.0002
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.002, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()

class_weight = {0:.01,1:.99} # magic
model.fit(train_df, y_train, batch_size=batch_size, epochs=5, class_weight=class_weight, shuffle=False, verbose=1,
callbacks=[roc_callback(training_data=(train_df, y_train), validation_data=(val_df, y_val))])
del train_df, y_train, val_df, y_val
gc.collect()
model.save_weights('imbalanced_data.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
drop_cols = ['click_id', 'click_time', 'ip', 'minute', 'second', 'ip_day_hour_minute_second_count', 'is_attributed']
test_df.drop(drop_cols, axis=1, inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print("writing....")
sub.to_csv('imbalanced_data_v2.csv.gz', index=False, compression='gzip')
