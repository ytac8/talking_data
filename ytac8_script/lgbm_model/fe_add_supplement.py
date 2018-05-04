import pandas as pd
import time
import numpy as np
import gc
from contextlib import contextmanager
import pickle


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def df_add_counts(df, cols, tag="_count"):
    print('df_add_counts', cols)
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols) + tag] = counts[unqtags]
    gc.collect()
    return df


def df_add_uniques(df, cols, tag="_unique"):
    print('df_add_uniques', cols)
    gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols) + tag})
    df = df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    gc.collect()
    return df


def make_count_features(df):
    with timer("add count features"):
        df['click_time'] = pd.to_datetime(df['click_time'])
        dt = df['click_time'].dt
        df['day'] = dt.day.astype('uint8')
        df['hour'] = dt.hour.astype('uint8')
        df['minute'] = dt.minute.astype('uint8')
        df['second'] = dt.second.astype('uint8')
        del(dt)
        df = df_add_counts(df, ['ip', 'day', 'hour', 'minute', 'second'])
        df = df_add_counts(df, ['ip', 'day', 'hour', 'minute'])
        df = df_add_counts(df, ['ip', 'day', 'hour'])
        df = df_add_counts(df, ['ip', 'app'])
        df = df_add_counts(df, ['ip', 'app', 'os'])
        df = df_add_counts(df, ['ip', 'device'])
        df = df_add_counts(df, ['app', 'channel'])
        df = df_add_counts(df, ['ip', 'day', 'hour', 'channel'])


def make_next_click_feature(df):
    with timer("Adding next click times"):
        df['click_time'] = (df['click_time'].astype(
            np.int64) // 10 ** 9).astype(np.int32)
        df['nextClick'] = (df.groupby(['ip', 'app', 'device', 'os']
                                      ).click_time.shift(-1) - df.click_time).astype(np.float32)
        gc.collect()


path = '../input/'

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

with timer("load training data"):
    with open('../data/pickle/train.csv.pkl', mode='rb') as f:
        train_df = pickle.load(f)

with timer("load test supplement data"):
    with open('../data/pickle/test_supplement.csv.pkl', mode='rb') as f:
        test_supplement_df = pickle.load(f)

len_train = train_df.shape[0]
print('len_train:', len_train)

concat_df = train_df.append(test_supplement_df)

del train_df, test_supplement_df
gc.collect()

make_count_features(concat_df)
make_next_click_feature(concat_df)

with timer("add uniques features"):
    concat_df = df_add_uniques(concat_df, ['ip', 'channel'])  # X0
    concat_df = df_add_uniques(concat_df, ['ip', 'app'])  # X3
    concat_df = df_add_uniques(concat_df, ['ip', 'device', 'os', 'app'])  # X8
    concat_df = df_add_uniques(concat_df, ['ip', 'device'])  # X5
    concat_df = df_add_uniques(concat_df, ['app', 'channel'])  # X6

test_supplement_df = concat_df[len_train:]
train_df = concat_df[:len_train]

with timer("train to_hdf"):
    train_df.to_hdf("X_train_add_supplement.h5", 'table',
                    complib='blosc', complevel=9)


with timer("test to_hdf"):
    test_supplement_df.to_hdf("test_supplement.h5",
                              'table', complib='blosc', complevel=9)

print('done.')
