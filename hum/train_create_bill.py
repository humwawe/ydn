import importlib
import constants
import util

importlib.reload(constants)
importlib.reload(util)
from constants import train_credit_bill_path, feature_train_credit_bill_path, bill_max_time, day_second
from util import *
import pandas as pd

train_credit_bill = pd.read_csv(train_credit_bill_path)
train_credit_bill.columns = ['user_id', 'bank_id', 'bill_time', 'last_bill_amount', 'last_payback_amount',
                             'current_bill_balance', 'credit_limit', 'repayment_status']

train_credit_bill['bill_time'] = train_credit_bill['bill_time'] // day_second
print(train_credit_bill.shape)
train_credit_bill.head()


def feature_fun(df):
    df_2 = df[df['bill_time'] > 0]

    feature_create_bill = pd.DataFrame()
    res = df.groupby('user_id')['repayment_status'].max() == 1
    feature_create_bill['is_repayment_status'] = res * 1

    main_bank = ['bank_id', 'bill_time', 'credit_limit']
    for i in main_bank:
        tmp0 = train_credit_bill.groupby(['user_id', 'bank_id'])[i].count()
        tmp1 = tmp0.groupby('user_id').max()
        tmp = pm(tmp0, tmp1)
        tmp = tmp[tmp[i + '_x'] == tmp[i + '_y']].reset_index().drop([i + '_x', i + '_y'], axis=1)
        feature_create_bill['count_main_bank_id_' + i] = tmp.groupby('user_id')[i].count()
        feature_create_bill['first_main_bank_id_' + i] = tmp.groupby('user_id')[i].first()

    bank_type = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in bank_type:
        if i == -1:
            tmp = df_2
        else:
            tmp = df_2[df_2['bank_id'] == i]

        suffix_b = str(i)
        res = tmp.groupby('user_id')['bill_time'].agg(['min', 'max', 'median'])
        res.columns = ['min_bill_time_b' + suffix_b, 'max_bill_time_b' + suffix_b, 'median_bill_time_b' + suffix_b]
        res['diff_mm_bill_time_b' + suffix_b] = res['max_bill_time_b' + suffix_b] - res['min_bill_time_b' + suffix_b]
        feature_create_bill = pm(feature_create_bill, res)

    time_split = [0, 15, 35, 65, 100, 130, 160, 200, 380, 750, 1200]
    for i in time_split:
        if i == 0:
            tmp = df
        else:
            tmp = df[df['bill_time'] > bill_max_time - i]

        suffix_t = str(i)
        feature_create_bill['count_user_id' + suffix_t] = tmp.groupby('user_id')['user_id'].count()
        feature_create_bill['count_bank_id' + suffix_t] = tmp.groupby(['user_id', 'bank_id'])[
            'bank_id'].count().groupby('user_id').count()

        res = tmp.groupby(['user_id', 'bank_id']).size().unstack().fillna(0)
        res.columns = ['size_bank_id_0_t' + suffix_t, 'size_bank_id_1_t' + suffix_t, 'size_bank_id_2_t' + suffix_t,
                       'size_bank_id_3_t' + suffix_t, 'size_bank_id_4_t' + suffix_t, 'size_bank_id_5_t' + suffix_t,
                       'size_bank_id_6_t' + suffix_t, 'size_bank_id_7_t' + suffix_t, 'size_bank_id_8_t' + suffix_t,
                       'size_bank_id_9_t' + suffix_t, 'size_bank_id_10_t' + suffix_t, 'size_bank_id_11_t' + suffix_t,
                       'size_bank_id_12_t' + suffix_t]
        feature_create_bill = pm(feature_create_bill, res)

        # bank_type = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        bank_type = [-1]
        for j in bank_type:
            if j != -1:
                tmp = tmp[tmp['bank_id'] == i]

            suffix_bt = suffix_t + "_b" + str(j)
            res = tmp.groupby('user_id')['last_bill_amount'].agg(['min', 'max', 'mean', 'sum', 'var'])
            res.columns = ['min_last_bill_amount_t' + suffix_bt, 'max_last_bill_amount_t' + suffix_bt,
                           'mean_last_bill_amount_t' + suffix_bt, 'sum_last_bill_amount_t' + suffix_bt,
                           'var_last_bill_amount_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp[tmp['last_bill_amount'] > 0].groupby('user_id')['last_bill_amount'].agg(['min', 'mean', 'var'])
            res.columns = ['min_last_bill_amount_0_t' + suffix_bt, 'mean_last_bill_amount_0_t' + suffix_bt,
                           'var_last_bill_amount_0_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp.groupby('user_id')['last_payback_amount'].agg(['min', 'max', 'mean', 'sum', 'var'])
            res.columns = ['min_last_payback_amount_t' + suffix_bt, 'max_last_payback_amount_t' + suffix_bt,
                           'mean_last_payback_amount_t' + suffix_bt, 'sum_last_payback_amount_t' + suffix_bt,
                           'var_last_payback_amount_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp[tmp['last_payback_amount'] > 0].groupby('user_id')['last_payback_amount'].agg(
                ['min', 'mean', 'var'])
            res.columns = ['min_last_payback_amount_0_t' + suffix_bt, 'mean_last_payback_amount_0_t' + suffix_bt,
                           'var_last_payback_amount_0_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            tmp['diff_bill_payback_amount'] = tmp['last_bill_amount'] - tmp['last_payback_amount']
            res = tmp.groupby('user_id')['diff_bill_payback_amount'].agg(['min', 'max', 'mean', 'sum', 'var'])
            res.columns = ['min_diff_bill_payback_amount_t' + suffix_bt, 'max_diff_bill_payback_amount_t' + suffix_bt,
                           'mean_diff_bill_payback_amount_t' + suffix_bt, 'sum_diff_bill_payback_amount_t' + suffix_bt,
                           'var_diff_bill_payback_amount_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp.groupby('user_id')['credit_limit'].agg(['min', 'max', 'mean', 'sum', 'var'])
            res.columns = ['min_credit_limit_t' + suffix_bt, 'max_credit_limit_t' + suffix_bt,
                           'mean_credit_limit_t' + suffix_bt, 'sum_credit_limit_t' + suffix_bt,
                           'var_credit_limit_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp[tmp['credit_limit'] > 0].groupby('user_id')['credit_limit'].agg(['min', 'mean', 'var'])
            res.columns = ['min_credit_limit_0_t' + suffix_bt, 'mean_credit_limit_0_t' + suffix_bt,
                           'var_credit_limit_0_t' + suffix_bt]
            feature_create_bill = pm(feature_create_bill, res)

    for col in feature_create_bill.columns:
        if tas(feature_create_bill, col):
            print(col)
            feature_create_bill.drop(col, axis=1, inplace=True)

    print(feature_create_bill.shape)

    feature_create_bill.to_csv(feature_train_credit_bill_path)
