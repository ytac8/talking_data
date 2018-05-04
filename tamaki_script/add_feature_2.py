import pandas as pd
import time
import numpy as np
import gc
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def df_add_uniques(df, cols, tag="_unique"):
    print('df_add_uniques', cols)
    gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
    df = df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    gc.collect()
    return df

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

with timer("load training data"):
    train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

with timer("load test supplement data"):
    test_supplement_df = pd.read_csv(path+"test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = train_df.shape[0]
print('len_train:', len_train)

concat_df = train_df.append(test_supplement_df)

del train_df, test_supplement_df
gc.collect()

with timer("add unique features"):
    concat_df['click_time'] = pd.to_datetime(concat_df['click_time'])
    dt = concat_df['click_time'].dt
    concat_df['day'] = dt.day.astype('uint8')
    del(dt)
    concat_df = df_add_uniques(concat_df, ['ip', 'day'])

with timer("log trans"):
    concat_df['ip_log'] = np.log2(1 + concat_df.ip.values)

test_supplement_df = concat_df[len_train:]
train_df = concat_df[:len_train]

del concat_df
gc.collect()

df = pd.read_hdf('../h5/X_train_add_supplement.h5', 'table')
train_df.drop(['app', 'channel', 'click_id', 'click_time', 'device', 'ip',
       'is_attributed', 'os', 'day'], axis=1, inplace=True)
df = pd.concat([df, train_df], axis=1)
df.to_hdf("X_train_v2.h5", 'table', complib='blosc', complevel=9)

del df, train_df
gc.collect()

test_supplement_df_click = pd.read_csv("../input/test_supplement.csv", dtype=dtypes, usecols=['click_time', 'click_id'])
test_supplement_df.click_id = test_supplement_df.click_id.values.astype('uint32')
test_supplement_df.drop(['click_time'], axis=1, inplace=True)
test_supplement_df = test_supplement_df.merge(test_supplement_df_click, on=['click_id'], how='left')
test_supplement_df.drop(['click_id','is_attributed', 'day'], axis=1, inplace=True)

df = pd.read_hdf('../h5/X_test_add_supplement.h5', 'table')
df = df.merge(test_supplement_df, on=['ip','app','device','os', 'channel', 'click_time'], how='left')
df.to_hdf("X_test_v2.h5", 'table', complib='blosc', complevel=9)
