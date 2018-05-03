import pickle
import pandas as pd
import numpy as np


def main():

    with open('../data/pickle/train.csv.pkl', mode='rb') as f:
        train_df = pickle.load(f)
    with open('../data/pickle/test_supplement.csv.pkl', mode='rb') as f:
        test_df = pickle.load(f)

    dataset = pd.concat([train_df, test_df])
    dataset['click_time'] = pd.to_datetime(dataset['click_time'])
    dataset['click_time'] = (dataset['click_time'].astype(
        np.int64) // 10 ** 9).astype(np.int32)

    dataset['diff'] = (dataset.groupby(['ip', 'device']
                                       ).click_time.shift(1) - dataset.click_time).astype(np.float32)

    with open('../data/pickle/diff_time_data.pkl', mode='wb') as f:
        pickle.dump(dataset, f, protocol=-1)


if __name__ == "__main__":
    main()
