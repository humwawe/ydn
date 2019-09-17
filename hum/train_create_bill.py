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
        tmp0 = df.groupby(['user_id', 'bank_id'])[i].count()
        for j in ['min', 'max']:
            if j == 'max':
                tmp1 = tmp0.groupby('user_id').max()
            else:
                tmp1 = tmp0.groupby('user_id').min()
            tmp = pm(tmp0, tmp1)
            tmp = tmp[tmp[i + '_x'] == tmp[i + '_y']].reset_index().drop([i + '_x', i + '_y'], axis=1)
            suffix_c = i + '-' + j
            feature_create_bill['count_main_bank_id_' + suffix_c] = tmp.groupby('user_id')['bank_id'].count()
            feature_create_bill['first_main_bank_id_' + suffix_c] = tmp.groupby('user_id')['bank_id'].first()

    bank_type = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in bank_type:
        if i == -1:
            tmp = df_2
        else:
            tmp = df_2[df_2['bank_id'] == i]

        suffix_b = str(i)
        res = tmp.groupby('user_id')['bill_time'].agg(['min', 'max', 'median', 'mean', 'count'])
        res.columns = ['min_bill_time_b' + suffix_b, 'max_bill_time_b' + suffix_b, 'median_bill_time_b' + suffix_b,
                       'mean_bill_time_b' + suffix_b, 'count_bill_time_b' + suffix_b]
        res['diff_mm_bill_time_b' + suffix_b] = res['max_bill_time_b' + suffix_b] - res['min_bill_time_b' + suffix_b]
        feature_create_bill = pm(feature_create_bill, res)

        for c in ['last_bill_amount', 'last_payback_amount', 'current_bill_balance', 'credit_limit']:
            suffix_btc = suffix_b + '_c' + c
            res = tmp.groupby('user_id')[c].agg(['min', 'max', 'mean', 'sum', 'var'])
            res.columns = ['min_bank_type_t' + suffix_btc, 'max_bank_type_t' + suffix_btc,
                           'mean_bank_type_t' + suffix_btc, 'sum_bank_type_t' + suffix_btc,
                           'var_bank_type_t' + suffix_btc]
            feature_create_bill = pm(feature_create_bill, res)

            res = tmp[tmp[c] > 0].groupby('user_id')[c].agg(['min', 'mean', 'var'])
            res.columns = ['min_0_bank_type_t' + suffix_btc, 'mean_0_bank_type_t' + suffix_btc,
                           'var_0_bank_type_t' + suffix_btc]
            feature_create_bill = pm(feature_create_bill, res)

    time_split = [0, 15, 35, 65, 100, 130, 160, 200, 380, 750, 1200, 1500]
    for i in time_split:
        if i == 0:
            tmp = df
        else:
            tmp = df[df['bill_time'] > bill_max_time - i]

        suffix_t = str(i)
        feature_create_bill['count_user_id' + suffix_t] = tmp.groupby('user_id')['user_id'].count()
        res = tmp.groupby(['user_id', 'bank_id'])['bank_id'].count().groupby('user_id').agg(
            ['min', 'max', 'count', 'mean', 'sum'])
        res.columns = ['min_bank_id' + suffix_t, 'max_bank_id' + suffix_t, 'count_bank_id' + suffix_t,
                       'mean_bank_id' + suffix_t, 'sum_bank_id' + suffix_t]
        feature_create_bill = pm(feature_create_bill, res)

        # bank_type = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        bank_type = [-1]
        for j in bank_type:
            if j != -1:
                tmp = tmp[tmp['bank_id'] == i]

            suffix_bt = suffix_t + '_b' + str(j)
            for c in ['last_bill_amount', 'last_payback_amount', 'current_bill_balance', 'credit_limit']:
                suffix_btc = suffix_bt + '_c' + c
                res = tmp.groupby('user_id')[c].agg(['min', 'max', 'mean', 'sum', 'var'])
                res.columns = ['min_t' + suffix_btc, 'max_t' + suffix_btc, 'mean_t' + suffix_btc, 'sum_t' + suffix_btc,
                               'var_t' + suffix_btc]
                feature_create_bill = pm(feature_create_bill, res)

                res = tmp[tmp[c] > 0].groupby('user_id')[c].agg(['min', 'mean', 'var'])
                res.columns = ['min_0_t' + suffix_btc, 'mean_0_t' + suffix_btc, 'var_0_t' + suffix_btc]
                feature_create_bill = pm(feature_create_bill, res)

            tmp['diff_current_last_amount'] = tmp['current_bill_balance'] - tmp['last_bill_amount']
            tmp['per_current_last_amount'] = tmp['diff_current_last_amount'] / tmp['last_bill_amount']
            tmp['diff_bill_payback_amount'] = tmp['last_bill_amount'] - tmp['last_payback_amount']
            tmp['per_bill_payback_amount'] = tmp['diff_bill_payback_amount'] / tmp['last_payback_amount']
            for cc in ['diff_current_last_amount', 'diff_bill_payback_amount']:
                suffix_bt_cc = suffix_bt + '_cc' + cc
                res = tmp.groupby('user_id')[cc].agg(['min', 'max', 'mean', 'sum', 'var'])
                res.columns = ['min_t' + suffix_bt_cc, 'max_t' + suffix_bt_cc, 'mean_t' + suffix_bt_cc,
                               'sum_t' + suffix_bt_cc, 'var_t' + suffix_bt_cc]
                feature_create_bill = pm(feature_create_bill, res)
            for cc in ['per_current_last_amount', 'per_bill_payback_amount']:
                suffix_bt_cc = suffix_bt + '_cc' + cc
                res = tmp.groupby('user_id')[cc].agg(['min', 'max', 'mean'])
                res.columns = ['min_t' + suffix_bt_cc, 'max_t' + suffix_bt_cc, 'mean_t' + suffix_bt_cc, ]
                feature_create_bill = pm(feature_create_bill, res)

    for col in feature_create_bill.columns:
        if tas(feature_create_bill, col):
            print(col)
            feature_create_bill.drop(col, axis=1, inplace=True)

    print(feature_create_bill.shape)

    feature_create_bill.to_csv(feature_train_credit_bill_path)
