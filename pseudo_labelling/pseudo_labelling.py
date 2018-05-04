import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
import lightgbm as lgb
from halo import Halo
import click


def train_lightgbm(
    train_df, 
    predictors,
    target,
    categorical_features,
    valid_df=None,
    num_boost_round=1000,
    early_stopping_rounds=20,
    verbose_eval=10,
    **params
):

    print(f'\nTraining on {train_df.shape[0]} samples')
    
    train_lgb = lgb.Dataset(
        data                = train_df[predictors].values, 
        label               = train_df[target].values,
        feature_name        = predictors,
        categorical_feature = categorical_features
    )
    
    if valid_df is not None:
        valid_lgb = lgb.Dataset(
            data                = valid_df[predictors].values, 
            label               = valid_df[target].values,
            feature_name        = predictors,
            categorical_feature = categorical_features
         )
    else:
        valid_lgb = None
        
    evals_result = {}
    lgb_best = lgb.train(
        params,
        train_set             = train_lgb,
        valid_sets            = valid_lgb,
        valid_names           = ['validation'],
        evals_result          = evals_result,
        num_boost_round       = num_boost_round,
        early_stopping_rounds = early_stopping_rounds,
        verbose_eval          = verbose_eval
    )
    
    report_training_result(lgb_best, evals_result)
    
    return lgb_best
       

def predict(model, test_df, predictors, target, id_name, iteration):
    with spin_timer('Predicting'):
        p = model.predict(test_df[predictors], iteration)

        result_df = pd.DataFrame()
        result_df[id_name] = test_df.click_id.values.astype('uint32')
        result_df[target] = p

    return result_df
    
    
def report_training_result(trained_model, evals_result, metrics='auc'):
    print('model report') 
    print(f'best iteration: {trained_model.best_iteration}')
    score = evals_result['validation'][metrics][trained_model.best_iteration - 1]
    print(f'{metrics} score: {score}')
      

def train_lightgbm_with_pseudo_labelling(
    train_df, 
    unlabeled_df,
    predictors, 
    target,
    id_name,
    categorical_features=None,
    valid_df=None,
    num_boost_round=1000,
    early_stopping_rounds=20,
    verbose_eval=10,
    pseudo_label_threshold=0.9,
    **params
):
    trained_lgb = train_lightgbm(
        train_df,
        predictors,
        target,
        categorical_features,
        valid_df,
        num_boost_round,
        early_stopping_rounds,
        verbose_eval,
        **params
    )
        
    pseudo_labels = predict(
        model      = trained_lgb,
        test_df    = unlabeled_df,
        predictors = predictors,
        target     = target,
        id_name    = id_name,
        iteration  = trained_lgb.best_iteration
    )
    pseudo_labels = (pseudo_labels > pseudo_label_threshold).astype(np.uint8)          
    unlabeled_df[target] = pseudo_labels
    train_df.append(unlabeled_df)
    
    trained_lgb_with_pseudo_labels = train_lightgbm(
        train_df,
        predictors,
        target,
        categorical_features,
        valid_df,
        num_boost_round,
        early_stopping_rounds,
        verbose_eval,
        **params
    )
        
    return trained_lgb_with_pseudo_labels
    

@contextmanager
def spin_timer(text):
    start_time = time.time()
    spinner = Halo(text=text, spinner='dots')
    spinner.start()   
    
    yield
    
    spinner.succeed(text + f' - {time.time() -  start_time} sec')
    
          
@click.command()
@click.argument('train',     type=click.Path(exists=True))
@click.argument('test',      type=click.Path(exists=True))
@click.argument('test_supp', type=click.Path(exists=True))
@click.argument('output',    type=click.Path(exists=False))
@click.option('--drop',             '-d',  multiple=True)
@click.option('--learning_rate',    '-lr', default=0.15)
@click.option('--num_leaves',       '-nl', default=128)
@click.option('--num_boost_round',  '-br', default=1000)
def main(train, test, test_supp, output, drop, learning_rate, num_leaves, num_boost_round):

    CATEGORICAL_FEATURES = [
        'ip', 'app', 'os', 'device', 'channel'
    ]
    TARGET  = 'is_attributed'
    ID_NAME = 'click_id' 
          
    with spin_timer('Loading dataset'):
        train_df     = pd.read_hdf(train,     'table')
        test_df      = pd.read_hdf(test,      'table')
        test_supp_df = pd.read_hdf(test_supp, 'table')

    with spin_timer('Splitting validation set'):
        valid_df = train_df[train_df.day == 9]
        train_df = train_df[train_df.day != 9]
    
    if len(drop) > 0:
        train_df.drop(list(drop), axis=1, inplace=True)
    
    CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c not in drop]

    predictors = list(
        set(train_df.columns) -
        set([TARGET]) -
        set(drop)
    )

    print('', '#' * 20 + ' predictors ' + '#' * 20)
    print('\n'.join(predictors))
    print('#' * 10 + ' drop ' + '#' * 10)
    print('\n'.join(drop))
    
    params = {
        'boosting_type':     'gbdt',
        'objective':         'binary',
        'metric':            'auc',
        'learning_rate':     learning_rate,
        'num_leaves':        num_leaves, 
        'max_depth':         -1,  
        'min_child_samples': 20,
        'max_bin':           255, 
        'subsample':         0.6, 
        'subsample_freq':    0,  
        'colsample_bytree':  0.3, 
        'min_child_weight':  5, 
        'subsample_for_bin': 200000, 
        'min_split_gain':    0, 
        'reg_alpha':         0, 
        'reg_lambda':        0, 
        'nthread':           32,
        'scale_pos_weight':  200,
        'verbose':           0
    }
          
    trained_model = train_lightgbm_with_pseudo_labelling(
        train_df, 
        test_supp_df,
        predictors, 
        TARGET,
        ID_NAME,
        categorical_features=CATEGORICAL_FEATURES,
        valid_df=valid_df,
        num_boost_round=num_boost_round,
        early_stopping_rounds=20,
        verbose_eval=10,
        pseudo_label_threshold=0.9,
        **params
    )
        
#     prediction = predict(
#         model      = trained_model, 
#         test_df    = test_df, 
#         predictors = predictors,
#         target     = TARGET, 
#         id_name    = ID_NAME, 
#         iteration  = trained_model.best_iteration
#     )  
    
#    prediction.to_csv(f'{output}.csv.gz', index=False, compression='gzip')    

    
if __name__ == '__main__':
    main()