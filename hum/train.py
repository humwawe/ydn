import xgboost as xgb
import importlib
import util
import constants

importlib.reload(constants)
importlib.reload(util)
from constants import *
from util import *
import pickle
import lightgbm as lgb


def xgb_cv(train_X, train_y):
    param = {'learning_rate': 0.03, 'max_depth': 4,
             'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'silent': 1, 'lambda': 33, 'objective': 'binary:logistic',
             'nthread': 10}
    num_round = 5000
    param['eval_metric'] = "logloss"
    plst = list(param.items())
    plst += [('eval_metric', 'auc')]

    dtrain = xgb.DMatrix(train_X, label=train_y)
    print(plst)

    cv_res = xgb.cv(plst, dtrain, num_round, verbose_eval=50, early_stopping_rounds=100, nfold=5, feval=ks, maximize=False)
    print("---------------------------------------")
    with open(cv_result_path, 'wb') as cv_file:
        pickle.dump(cv_res, cv_file)
    return cv_res.shape[0]


def lgb_cv(train_X, train_y):
    params = {'boosting_type': 'gbdt', 'objective': 'binary',
              'metric': 'auc', 'learning_rate': 0.03, 'sub_feature': 0.8,
              'bagging_fraction': 0.8, 'max_depth': 5, 'lambda_l2': 33,
              'n_jobs': 10, }

    num_round = 5000
    dtrain = lgb.Dataset(train_X, label=train_y)

    print(params)

    lgb_cv_res = lgb.cv(params, dtrain, num_round, verbose_eval=50, early_stopping_rounds=500, nfold=5, feval=lgb_ks)
    print("---------------------------------------")
    return lgb_cv_res


def xgb_model(train_X, train_y, num_round=5000):
    param = {'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 1,
             'lambda': 33, 'objective': 'binary:logistic', 'nthread': 10, 'eval_metric': "logloss"}
    plst = list(param.items())
    plst += [('eval_metric', 'auc')]

    print(plst)
    dtrain = xgb.DMatrix(train_X, label=train_y)
    bst = xgb.train(plst, dtrain, num_boost_round=num_round, feval=ks)
    print("---------------------------------------")
    return bst


def lgb_model(train_X, train_y, num_round=5000):
    params = {'boosting_type': 'gbdt', 'objective': 'binary',
              'metric': 'auc', 'learning_rate': 0.03, 'sub_feature': 0.8,
              'bagging_fraction': 0.8, 'max_depth': 5, 'lambda_l2': 33,
              'n_jobs': 10, }

    print(params)

    dtrain = lgb.Dataset(train_X, label=train_y)
    bst = lgb.train(params, dtrain, num_boost_round=num_round, feval=lgb_ks)
    print("---------------------------------------")
    return bst


train_feature_1 = pd.read_csv(feature_train_profile_path, index_col='user_id')
train_feature_2 = pd.read_csv(feature_train_bank_statement_path, index_col='user_id')
train_feature_3 = pd.read_csv(feature_train_credit_bill_path, index_col='user_id')
train_feature_3_2 = pd.read_csv(feature_test_credit_bill_path_2, index_col='user_id')
train_feature_4 = pd.read_csv(feature_train_behaviors_path, index_col='user_id')
train_label = pd.read_csv(feature_train_label_path, index_col='user_id')
train_merge = pm(train_feature_1, train_feature_2)
train_merge = pm(train_merge, train_feature_3)
train_merge = pm(train_merge, train_feature_3_2)
train_merge = pm(train_merge, train_feature_4)
train_merge = pm(train_merge, train_label)
train_X = train_merge.drop('label', axis=1)
train_y = train_merge['label']

xgb_cv_model = xgb_model(train_X, train_y, 2700)
importance = xgb_cv_model.get_fscore()
temp1 = []
temp2 = []
for k in importance:
    temp1.append(k)
    temp2.append(importance[k])
keep_col = pd.DataFrame({'column': temp1, 'importance': temp2, }).sort_values(by='importance', ascending=False)
