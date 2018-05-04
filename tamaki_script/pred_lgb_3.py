import argparse
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    psr.add_argument('-nl', '--num_leaves', default=64, type=int)
    psr.add_argument('-nb', '--num_boost_round', default=1000, type=int)
    psr.add_argument('-d', '--drop_cols', nargs='*', default=None)
    psr.add_argument('-tr', '--train_path', default='../h5/X_train_add_supplement.h5')
    psr.add_argument('-te', '--test_path', default='../h5/X_test_add_supplement.h5')
    args = psr.parse_args()
    print(f'learning_rate:{args.learning_rate}')
    print(f'num_leaves:{args.num_leaves}')
    print(f'num_boost_round:{args.num_boost_round}')
    print(f'drop_cols:{args.drop_cols}')
    print(f'loading {args.train_path}')
    train_df = pd.read_hdf(args.train_path, 'table')
    print(f'loading {args.test_path}')
    test_df = pd.read_hdf(args.test_path, 'table')
    sub = pd.DataFrame()
    sub['click_id'] = test_df.click_id.values.astype('uint32')
    target = 'is_attributed'
    metrics = 'auc'

    drop_cols = ['click_time', 'click_id', 'ip', 'minute', 'second', 'ip_day_hour_minute_second_count']
    if args.drop_cols is not None:
        drop_cols.extend(args.drop_cols)

    categorical_features = ['app','os','channel','device']
    predictors = list(set(train_df.columns)-set([target])-set(drop_cols))
    print(f'predictors: {predictors}')

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': metrics,
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,  # we should let it be smaller than 2^(max_depth) 31
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 32,
        'scale_pos_weight':200,
        'verbose': 0
    }

    num_boost_round = args.num_boost_round
    early_stopping_rounds = 50
    NFOLDS = 3
    val_auc = 0
    kf = KFold(n_splits=NFOLDS, shuffle=False)
    test_skf = np.empty((NFOLDS, len(test_df)))
    with timer("fit and pred"):
        for i, (train_indices, valid_indices) in enumerate(kf.split(train_df)):
            evals_results = {}
            kf_train = train_df.iloc[train_indices]
            kf_valid = train_df.iloc[valid_indices]

            lgbtrain = lgb.Dataset(kf_train[predictors].values, label=kf_train[target].values,
                                  feature_name=predictors,
                                  categorical_feature=categorical_features
                                  )
            lgbvalid = lgb.Dataset(kf_valid[predictors].values, label=kf_valid[target].values,
                                      feature_name=predictors,
                                      categorical_feature=categorical_features
                                      )

            print("Training...")
            start_time = time.time()

            bst1 = lgb.train(params,
                             lgbtrain,
                             valid_sets=[lgbvalid],
                             valid_names=['valid'],
                             evals_result=evals_results,
                             num_boost_round=args.num_boost_round,
                             early_stopping_rounds=early_stopping_rounds,
                             verbose_eval=10)

            print('[{}]: model training time'.format(time.time() - start_time))
            val_auc += evals_results['valid'][metrics][bst1.best_iteration-1]
            print("Predicting...")
            test_skf[i, :] = bst1.predict(test_df[predictors], num_iteration=bst1.best_iteration)
            del bst1, evals_results
            gc.collect()

    sub['is_attributed'] = test_skf.mean(axis=0)
    valid_score = val_auc / NFOLDS
    sub.to_csv(f'auc_{valid_score}_lr_{args.learning_rate}_num_leaves_{args.num_leaves}.csv.gz', index=False, compression='gzip')

    print('done.')

if __name__ == '__main__': main()
