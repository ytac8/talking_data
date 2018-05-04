import argparse
import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import gc
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('-lr', '--learning_rate', default=0.2, type=float)
    psr.add_argument('--num_leaves', default=31, type=int)
    psr.add_argument('-d', '--drop_cols', nargs='*', default=None)
    psr.add_argument('-tp', '--train_path', default='X_train_add_supplement.h5')
    args = psr.parse_args()
    print(f'loading {args.train_path}')
    train_df = pd.read_hdf(args.train_path, 'table')

    val_df = train_df.loc[train_df.day == 9]
    train_df = train_df.loc[train_df.day < 9]

    if args.drop_cols is not None:
        print(f'drop_cols:{args.drop_cols}')
        train_df = train_df.drop(args.drop_cols, axis=1)

    train_df.info()

    target = 'is_attributed'
    metrics = 'auc'

    categorical_features = ['app','os','channel','device']
    if args.drop_cols is not None:
        for col in args.drop_cols:
            if col in categorical_features:
                categorical_features.remove(col)
    predictors = list(set(train_df.columns)-set([target])-set(['click_time', 'click_id']))

    print(f'predictors: {predictors}')

    lgbtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    lgbvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': args.learning_rate,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
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
    num_boost_round = 10000
    early_stopping_rounds = 20

    print("Training...")
    start_time = time.time()

    bst1 = lgb.train(params,
                     lgbtrain,
                     valid_sets=[lgbvalid],
                     valid_names=['valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()
    print("\nModel Report")
    valid_score = evals_results['valid'][metrics][bst1.best_iteration-1]
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", valid_score)

    mapper = {f: v for f, v in zip(predictors, bst1.feature_importance())}
    x = []
    y = []
    for k, v in sorted(mapper.items(), key=lambda x:x[1]):
        x.append(k)
        y.append(v)

    if args.drop_cols is None:
        drop_cols = 'None'
    else:
        drop_cols = '_'.join(args.drop_cols)
    plt.figure(figsize=(32,18))
    plt.barh(range(len(y)), y, align='center')
    plt.yticks(range(len(x)), x)
    plt.savefig(f'cv_auc_{valid_score}_it_{bst1.best_iteration}_drop_{drop_cols}_data_{args.train_path}.png')
    print('done.')

if __name__ == '__main__': main()
