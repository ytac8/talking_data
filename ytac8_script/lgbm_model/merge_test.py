import pandas as pd
import pickle


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

test_supplement_df = pd.read_hdf('test_supplement.h5', 'table')
with open('../data/pickle/add_rolling_and_diff_feature.pkl', mode='rb') as f:
    rolling_df = pickle.load(f)

test_supplement_df = pd.merge(test_supplement_df, rolling_df[['ip', 'click_time', 'window_count']], on=[
    'ip', 'click_time'], how='left')


# test_supplement_df_click = pd.read_csv(
#     path + "test_supplement.csv", dtype=dtypes, usecols=['click_time', 'click_id'])
with open('../data/pickle/test_supplement.csv.pkl', mode='rb') as f:
    test_supplement_df_click = pickle.load(f)
test_supplement_df_click = test_supplement_df_click[['click_time', 'click_id']]

# test_df = pd.read_csv(path + "test.csv", dtype=dtypes,
#                       usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
with open('../data/pickle/test.csv.pkl', mode='rb') as f:
    test_df = pickle.load(f)

test_supplement_df.click_id = test_supplement_df.click_id.values.astype(
    'uint32')
test_supplement_df = test_supplement_df.merge(
    test_supplement_df_click, on=['click_id'], how='left')
test_supplement_df.drop(['click_id'], axis=1, inplace=True)

test_df = test_df.merge(test_supplement_df, on=[
                        'ip', 'app', 'device', 'os', 'channel', 'click_time'], how='left')

test_df = test_df.drop_duplicates(subset='click_id')
test_df.to_hdf("X_test_add_supplement.h5", 'table',
               complib='blosc', complevel=9)
