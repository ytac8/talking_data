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

train_df = pd.read_hdf('./X_train_add_supplement_add_rolling.h5', 'table')
test_df = pd.read_hdf('./X_test_add_supplement.h5', 'table')

sub = pd.DataFrame()
sub['click_id'] = test_df.click_id.values.astype('uint32')

len_train = len(train_df)
val_size = 2500000

val_df = train_df[(len_train - val_size):]
train_df = train_df[:(len_train - val_size)]

target = 'is_attributed'
metrics = 'auc'
lr = 0.15
num_leaves = 64

# categorical_features = ['ip', 'app', 'os', 'channel', 'device']
categorical_features = ['app', 'os', 'channel']
predictors = list(set(train_df.columns) -
                  set([target]) - set(['click_time', 'click_id', 'minute', 'second', 'ip', 'device', 'attributed_time', 'ip_day_hour_minute_count', 'ip_day_hour_minute_second_count', 'day']))

print(f'predictors: {predictors}')

lgbtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                       feature_name=predictors,
                       categorical_feature=categorical_features
                       )
lgbvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                       feature_name=predictors,
                       categorical_feature=categorical_features
                       )

params = {'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': metrics,
          'learning_rate': lr,
          #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
          # we should let it be smaller than 2^(max_depth) 31
          'num_leaves': num_leaves,
          'max_depth': -1,  # -1 means no limit
          # Minimum number of data need in a child(min_data_in_leaf)
          'min_child_samples': 20,
          'max_bin': 255,  # Number of bucketed bin for feature values
          'subsample': 0.6,  # Subsample ratio of the training instance.
          'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
          # Subsample ratio of columns when constructing each tree.
          'colsample_bytree': 0.3,
          # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_child_weight': 5,
          'subsample_for_bin': 200000,  # Number of samples for constructing bin
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'reg_alpha': 0,  # L1 regularization term on weights
          'reg_lambda': 5,  # L2 regularization term on weights
          'nthread': 32,
          'scale_pos_weight': 200, 'verbose': 0
          }

evals_results = {}
num_boost_round = 500
early_stopping_rounds = 50

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
valid_score = evals_results['valid'][metrics][bst1.best_iteration - 1]
print("bst1.best_iteration: ", bst1.best_iteration)
print(metrics + ":", valid_score)

print("Predicting...")
sub['is_attributed'] = bst1.predict(
    test_df[predictors], num_iteration=bst1.best_iteration)
sub.to_csv(f'auc_{valid_score}_it_{bst1.best_iteration}_lr_{lr}_num_leaves_{num_leaves}.csv.gz',
           index=False, compression='gzip')

mapper = {f: v for f, v in zip(predictors, bst1.feature_importance())}
x = []
y = []
for k, v in sorted(mapper.items(), key=lambda x: x[1]):
    x.append(k)
    y.append(v)

plt.figure(figsize=(16, 9))
plt.barh(range(len(y)), y, align='center')
plt.yticks(range(len(x)), x)
plt.savefig(f'auc_{valid_score}_it_{bst1.best_iteration}.png')
print('done.')
