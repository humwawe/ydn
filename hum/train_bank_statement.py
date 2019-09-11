import importlib
import constants
import util

importlib.reload(constants)
importlib.reload(util)
from constants import train_bank_statement_path, feature_train_bank_statement_path, statement_max_time, day_second
from util import *
import pandas as pd

train_bank_statement = pd.read_csv(train_bank_statement_path)
train_bank_statement.columns = ['user_id', 'statement_time', 'transaction_type', 'transaction_amount', 'income_type']

train_bank_statement['statement_time'] = train_bank_statement['statement_time'] // day_second
train_bank_statement.head()


def feature_fun(df):
    df_2 = df[df['statement_time'] > 0]

    feature_bank_statement = pd.DataFrame()

    feature_bank_statement['count_user_id'] = df.groupby('user_id')['user_id'].count()
    feature_bank_statement['count_user_id_2'] = df_2.groupby('user_id')['user_id'].count()

    res = df.groupby('user_id')['statement_time'].min() == 0
    feature_bank_statement['is_statement_time'] = res * 1

    res = df.groupby(['user_id', 'transaction_type']).size().unstack().fillna(0)
    res.columns = ['size_transaction_type_0', 'size_transaction_type_1']
    feature_bank_statement = pm(feature_bank_statement, res)

    res = df_2.groupby('user_id')['statement_time'].agg(['min', 'max', 'median'])
    res.columns = ['min_statement_time', 'max_statement_time', 'median_statement_time']
    res['diff_mm_statement_time'] = res['max_statement_time'] - res['min_statement_time']
    feature_bank_statement = pm(feature_bank_statement, res)

    res = df.groupby('user_id')['transaction_amount'].agg(['min', 'max', 'mean', 'sum', 'var'])
    res.columns = ['min_transaction_amount', 'max_transaction_amount', 'mean_transaction_amount',
                   'sum_transaction_amount', 'var_transaction_amount']
    feature_bank_statement = pm(feature_bank_statement, res)

    res = df.groupby(['user_id', 'transaction_type'])['transaction_amount'].agg(
        ['min', 'max', 'mean', 'sum', 'var']).unstack().fillna(0)
    res.columns = ['min_0_transaction_amount', 'min_1_transaction_amount',
                   'max_0_transaction_amount', 'max_1_transaction_amount',
                   'mean_0_transaction_amount', 'mean_1_transaction_amount',
                   'sum_0_transaction_amount', 'sum_1_transaction_amount',
                   'var_0_transaction_amount', 'var_1_transaction_amount']
    feature_bank_statement = pm(feature_bank_statement, res)

    res = df.groupby('user_id')['income_type'].max() == 1
    feature_bank_statement['is_income_type'] = res * 1

    time_split = [7, 15, 30, 60, 120, 240, 360]
    for i in time_split:
        tmp = df[df['statement_time'] > statement_max_time - i]
        res = tmp.groupby('user_id')['transaction_amount'].agg(['min', 'max', 'mean', 'sum'])
        res.columns = ['min_transaction_amount_t' + str(i), 'max_transaction_amount_t' + str(i),
                       'mean_transaction_amount_t' + str(i), 'sum_transaction_amount_t' + str(i)]
        feature_bank_statement = pm(feature_bank_statement, res)

    for col in feature_bank_statement.columns:
        if tas(feature_bank_statement, col):
            print(col)

    print(feature_bank_statement.shape)

    feature_bank_statement.to_csv(feature_train_bank_statement_path)
