import argparse
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    psr.add_argument('-nl', '--num_leaves', default=64, type=int)
    psr.add_argument('-nb', '--num_boost_round', default=1000, type=int)
    psr.add_argument('-d', '--drop_cols', nargs='*', default=None)
    psr.add_argument('-tr', '--train_path', default='../h5/X_train_add_supplement.h5')
    psr.add_argument('-te', '--test_path', default='../h5/X_test_add_supplement.h5')
    psr.add_argument('--use_all', action='store_true')
    args = psr.parse_args()
    print(f'learning_rate:{args.learning_rate}')
    print(f'num_leaves:{args.num_leaves}')
    print(f'num_boost_round:{args.num_boost_round}')
    print(f'drop_cols:{args.drop_cols}')
    print(f'use_all:{args.use_all}')
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

    if not args.use_all:
        len_train = len(train_df)
        val_size = 2500000
        val_df = train_df[(len_train-val_size):]
        train_df = train_df[:(len_train-val_size)]
        lgbvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                                  feature_name=predictors,
                                  categorical_feature=categorical_features
                                  )

    lgbtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

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

    evals_results = {}
    num_boost_round = args.num_boost_round
    early_stopping_rounds = 50

    print("Training...")
    start_time = time.time()

    if args.use_all:
        bst1 = lgb.train(params, lgbtrain,
                         num_boost_round=args.num_boost_round)

    else:
        bst1 = lgb.train(params,
                         lgbtrain,
                         valid_sets=[lgbvalid],
                         valid_names=['valid'],
                         evals_result=evals_results,
                         num_boost_round=args.num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=10)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    gc.collect()
    print("Predicting...")
    sub['is_attributed'] = bst1.predict(test_df[predictors], num_iteration=bst1.best_iteration)
    if args.use_all:
        sub.to_csv(f'no_cv_it_{args.num_boost_round}_lr_{args.learning_rate}_num_leaves_{args.num_leaves}.csv.gz', index=False, compression='gzip')
    else:
        valid_score = evals_results['valid'][metrics][bst1.best_iteration-1]
        sub.to_csv(f'auc_{valid_score}_it_{bst1.best_iteration}_lr_{args.learning_rate}_num_leaves_{args.num_leaves}.csv.gz', index=False, compression='gzip')

    print('done.')

if __name__ == '__main__': main()
