import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from contextlib import contextmanager
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def df_add_means(df_history, df, cols, tag="_mean"):
    print('df_add_means', cols)
    gp = df_history[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].mean().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
    df = df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    gc.collect()
    return df.fillna(-1)

with timer("load training data"):
    train_df = pd.read_hdf('X_train.h5', 'table')

with timer("load test data"):
    test_df = pd.read_hdf('X_test.h5', 'table')

with timer("Adding means to test"):
    test_df = df_add_means(train_df, test_df, ['ip', 'is_attributed'])

kf = KFold(n_splits=3, shuffle=False)
df = pd.DataFrame()
with timer("Adding means to train"):
    for train_indices, valid_indices in kf.split(train_df):
        kf_train = train_df.iloc[train_indices]
        kf_valid = train_df.iloc[valid_indices]
        kf_valid = df_add_means(kf_train, kf_valid, ['ip', 'is_attributed'])
        df = df.append(kf_valid)

train_df = df
del df
gc.collect()

len_train = len(train_df)
val_size=2500000

val_df = train_df[(len_train-val_size):]
train_df = train_df[:(len_train-val_size)]

sub = pd.DataFrame()
sub['click_id'] = test_df.click_id.values.astype('uint32')

target = 'is_attributed'
metrics = 'auc'
lr = 0.2
num_leaves = 31

categorical_features = ['ip','app','os','channel','device']
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
    'learning_rate': lr,
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': num_leaves,  # we should let it be smaller than 2^(max_depth) 31
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
num_boost_round = 500
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

print("Predicting...")
sub['is_attributed'] = bst1.predict(test_df[predictors], num_iteration=bst1.best_iteration)
sub.to_csv(f'auc_{valid_score}_it_{bst1.best_iteration}_lr_{lr}_num_leaves_{num_leaves}.csv.gz', index=False, compression='gzip')

mapper = {f: v for f, v in zip(predictors, bst1.feature_importance())}
x = []
y = []
for k, v in sorted(mapper.items(), key=lambda x:x[1]):
    x.append(k)
    y.append(v)

plt.figure(figsize=(32,18))
plt.barh(range(len(y)), y, align='center')
plt.yticks(range(len(x)), x)
plt.savefig(f'auc_{valid_score}_it_{bst1.best_iteration}.png')
print('done.')
