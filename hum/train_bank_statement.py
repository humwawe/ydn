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

    feature_bank_statement['count_user_id_2'] = df_2.groupby('user_id')['user_id'].count()

    res = df.groupby('user_id')['statement_time'].min() == 0
    feature_bank_statement['is_statement_time'] = res * 1

    transaction_type = [-1, 0, 1]
    for i in transaction_type:
        if i == -1:
            tmp = df_2
        else:
            tmp = df_2[df_2['transaction_type'] == i]

        suffix_t = str(i)
        res = tmp.groupby('user_id')['statement_time'].agg(['min', 'max', 'median'])
        res.columns = ['min_statement_time_t' + suffix_t, 'max_statement_time_t' + suffix_t,
                       'median_statement_time_t' + suffix_t]
        res['diff_mm_statement_time_t' + suffix_t] = res['max_statement_time_t' + suffix_t] - res[
            'min_statement_time_t' + suffix_t]
        feature_bank_statement = pm(feature_bank_statement, res)

    res = df.groupby('user_id')['income_type'].max() == 1
    feature_bank_statement['is_income_type'] = res * 1

    time_split = [0, 7, 15, 30, 60, 120, 240, 360]
    for i in time_split:
        if i == 0:
            tmp = df
        else:
            tmp = df[df['statement_time'] > statement_max_time - i]

        suffix_t = str(i)
        res = tmp.groupby(['user_id', 'transaction_type']).size().unstack().fillna(0)
        res.columns = ['size_transaction_type_0_t' + suffix_t, 'size_transaction_type_1_t' + suffix_t]
        feature_bank_statement = pm(feature_bank_statement, res)

        res = tmp.groupby('user_id')['transaction_amount'].agg(['count', 'min', 'max', 'mean', 'sum', 'var'])
        res.columns = ['count_transaction_amount_t' + suffix_t, 'min_transaction_amount_t' + suffix_t,
                       'max_transaction_amount_t' + suffix_t, 'mean_transaction_amount_t' + suffix_t,
                       'sum_transaction_amount_t' + suffix_t, 'var_transaction_amount_t' + suffix_t]
        feature_bank_statement = pm(feature_bank_statement, res)

        res = tmp.groupby(['user_id', 'transaction_type'])['transaction_amount'].agg(
            ['min', 'max', 'mean', 'sum', 'var']).unstack().fillna(0)
        res.columns = ['min_0_transaction_amount_t' + suffix_t, 'min_1_transaction_amount_t' + suffix_t,
                       'max_0_transaction_amount_t' + suffix_t, 'max_1_transaction_amount_t' + suffix_t,
                       'mean_0_transaction_amount_t' + suffix_t, 'mean_1_transaction_amount_t' + suffix_t,
                       'sum_0_transaction_amount_t' + suffix_t, 'sum_1_transaction_amount_t' + suffix_t,
                       'var_0_transaction_amount_t' + suffix_t, 'var_1_transaction_amount_t' + suffix_t]
        feature_bank_statement = pm(feature_bank_statement, res)

    for col in feature_bank_statement.columns:
        if tas(feature_bank_statement, col):
            print(col)

    print(feature_bank_statement.shape)

    feature_bank_statement.to_csv(feature_train_bank_statement_path)
