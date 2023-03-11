import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import graphviz
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold as KFold
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score

def xgb_objective(trial : Trial, train_x, train_y, regressor_name = 'regressor'):
    
    params_xgb = {
        'random_state' : 0,
        'early_stopping_rounds' : 50,
        'n_estimators' : trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'max_leaves': trial.suggest_categorical('max_leaves', [2**i for i in range(4, 10)]),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.0, 1.0),
        'learning_rate':trial.suggest_uniform('learning_rate', 0.01, 0.5)
    }

    if regressor_name == 'regressor':
        kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
        rmse_list = []
        for train_index, valid_index in kf.split(train_x):
            model_reg = xgb.XGBRegressor(**params_xgb)
            train_for_val_x, train_for_val_y = train_x.iloc[train_index], train_y.iloc[train_index]
            valid_x, valid_y = train_x.iloc[valid_index], train_y.iloc[valid_index]
            model_reg.fit(train_for_val_x, train_for_val_y, eval_set = [(train_for_val_x, train_for_val_y), (valid_x, valid_y)], verbose = False)
            xgb_pred = model_reg.predict(valid_x)
            rmse = np.sqrt(mean_squared_error(valid_y, xgb_pred))
            rmse_list.append(rmse) 
        return np.mean(rmse_list)
    
    elif regressor_name == 'classifier':
        stf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        f1_score_list = []
        for train_index, valid_index in stf.split(train_x, train_y):
            model_clf = xgb.XGBClassifier(**params_xgb)
            train_for_val_x, train_for_val_y = train_x.iloc[train_index], train_y.iloc[train_index]
            valid_x, valid_y = train_x.iloc[valid_index], train_y.iloc[valid_index]
            model_clf.fit(train_for_val_x, train_for_val_y, eval_set = [(train_for_val_x, train_for_val_y), (valid_x, valid_y)], verbose = False)
            xgb_pred = model_clf.predict(valid_x)
            f1score = f1_score(valid_y, xgb_pred, average = 'micro') # 다중 분류의 경우 'micro', 'macro', 'samples', 'weighted' 중 하나 선택
            f1_score_list.append(f1score) 
        return np.mean(f1_score_list)
    
def light_objective(trial : Trial, train_x, train_y, regressor_name = 'regressor'):

    params_lgbm = {
        'random_state' : 0,
        'early_stopping_round' : 50,
        'n_estimators' : trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'num_leaves': trial.suggest_categorical('max_leaves', [2**i for i in range(4, 10)]),
        'subsample': trial.suggest_uniform('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)       
    }

    if regressor_name == 'regressor':
        kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
        rmse_list = []
        for train_index, valid_index in kf.split(train_x):
            model_reg = lgbm.LGBMRegressor(**params_lgbm)
            train_for_val_x, train_for_val_y = train_x.iloc[train_index], train_y.iloc[train_index]
            valid_x, valid_y = train_x.iloc[valid_index], train_y.iloc[valid_index]
            model_reg.fit(train_for_val_x, train_for_val_y, eval_set = [(train_for_val_x, train_for_val_y), (valid_x, valid_y)], verbose = False)
            xgb_pred = model_reg.predict(valid_x)
            rmse = np.sqrt(mean_squared_error(valid_y, xgb_pred))
            rmse_list.append(rmse) 
        return np.mean(rmse_list)
    
    elif regressor_name == 'classifier':
        stf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        f1_score_list = []
        for train_index, valid_index in stf.split(train_x, train_y):
            model_clf = lgbm.LGBMClassifier(**params_lgbm)
            train_for_val_x, train_for_val_y = train_x.iloc[train_index], train_y.iloc[train_index]
            valid_x, valid_y = train_x.iloc[valid_index], train_y.iloc[valid_index]
            model_clf.fit(train_for_val_x, train_for_val_y, eval_set = [(train_for_val_x, train_for_val_y), (valid_x, valid_y)], verbose = False)
            xgb_pred = model_clf.predict(valid_x)
            f1score = f1_score(valid_y, xgb_pred, average = 'micro') # 다중 분류의 경우 'micro', 'macro', 'samples', 'weighted' 중 하나 선택
            f1_score_list.append(f1score) 
        return np.mean(f1_score_list)