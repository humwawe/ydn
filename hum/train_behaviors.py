import importlib
import constants
import util

importlib.reload(constants)
importlib.reload(util)
from constants import train_behaviors_path, feature_train_behaviors_path
from util import *
import pandas as pd
import numpy as np

train_behaviors = pd.read_csv(train_behaviors_path)
train_behaviors.columns = ['user_id', 'date', 'weekday', 'behavior_type', 'sub_behavior_type_1', 'sub_behavior_type_2']
train_behaviors.head(100)

train_behaviors['month'] = train_behaviors['date'].apply(month)
train_behaviors['day'] = train_behaviors['date'].apply(day)
train_behaviors['quarter'] = (train_behaviors['month'] - 1) // 6 + 1
# train_behaviors['ten_day'] = train_behaviors['day'].apply(ten_day)
train_behaviors['day_10'] = (train_behaviors['month'] - 1) * 31 + train_behaviors['day']


def feature_fun(df):
    def sub_feature(col, spl):
        feature_behavior = pd.DataFrame()
        for m in spl:
            if m == -1:
                tmp = df
            else:
                tmp = df[df[col] == m]

            suffix_1 = col + str(m)
            feature_behavior['size_user_id_' + suffix_1] = tmp.groupby('user_id').size()

            for i in ['behavior_type', 'weekday', 'month', 'day', 'day_10', 'sub_behavior_type_1', 'sub_behavior_type_2', 'quarter']:
                if (i == col and m != -1) or (i == 'quarter' and col == 'month'):
                    continue
                suffix_2 = suffix_1 + '-' + i
                res = tmp.groupby('user_id')[i].agg(['min', 'max', 'median'])
                res.columns = ['min_' + suffix_2, 'max_' + suffix_2, 'median_' + suffix_2]
                feature_behavior = pm(feature_behavior, res)

                t0 = tmp.groupby(['user_id', i])[i].count()
                res = t0.groupby('user_id').agg(['min', 'max', 'count', 'mean', 'sum', 'median', 'var'])
                res.columns = ['min_user_id' + suffix_2, 'max_user_id' + suffix_2, 'count_user_id' + suffix_2, 'mean_user_id' + suffix_2,
                               'sum_user_id' + suffix_2, 'median_user_id' + suffix_2, 'var_user_id' + suffix_2]
                feature_behavior = pm(feature_behavior, res)
                for j in ['min', 'max']:
                    suffix_3 = suffix_2 + '-' + j
                    if j == 'max':
                        t1 = t0.groupby('user_id').max()
                    else:
                        t1 = t0.groupby('user_id').min()
                    t = pm(t0, t1)
                    t = t[t[i + '_x'] == t[i + '_y']].reset_index().drop([i + '_x', i + '_y'], axis=1)
                    feature_behavior['count_main_' + suffix_3] = t.groupby('user_id')[i].count()
                    feature_behavior['first_main_' + suffix_3] = t.groupby('user_id')[i].first()
        return feature_behavior

    spl1 = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    spl2 = [0, 1, 2, 3, 4, 5, 6]
    spl3 = [0, 1, 2, 3, 4, 5, 6, 7]
    spl4 = [1, 2]
    spl5 = [1, 2, 3]
    f1 = sub_feature('month', spl1)
    f2 = sub_feature('weekday', spl2)
    f3 = sub_feature('behavior_type', spl3)
    f4 = sub_feature('quarter', spl4)
    f5 = sub_feature('ten_day', spl5)
    feature_behaviors = pm(f1, f2)
    feature_behaviors = pm(feature_behaviors, f3)
    feature_behaviors = pm(feature_behaviors, f4)
    feature_behaviors = pm(feature_behaviors, f5)

    def sub_behavior_count(col, sb_type):
        feature_behavior = pd.DataFrame()
        for i in sb_type:
            suffix_sb = '_count_' + str(i)
            feature_behavior[col + suffix_sb] = df[df[col] == i].groupby('user_id').size()
        return feature_behavior

    sb_type_1 = [54, 26, 28, 82, 24, 97, 30, 3, 13, 25, 101, 107, 98, 39, 50, 56, 110, 79, 84, 105, 104, 62, 102, 63, 73, 42, 45, 99, 103, 2, 100,
                 106, 7, 74, 11, 90, 109, 108, 61, 21]
    sb_type_2 = [33, 29, 7, 50, 27, 25, 1, 14, 26, 9, 53, 28, 18, 37, 22, 46, 13, 31, 57, 47, 2, 48, 4, 21, 44, 0, 8, 15, 39, 32, 17, 52, 45, 3, 16,
                 38, 24, 58, 34, 5]
    sb_f1 = sub_behavior_count('sub_behavior_type_1', sb_type_1)
    sb_f2 = sub_behavior_count('sub_behavior_type_2', sb_type_2)
    feature_behaviors = pm(feature_behaviors, sb_f1)
    feature_behaviors = pm(feature_behaviors, sb_f2)

    for col in feature_behaviors.columns:
        if tas(feature_behaviors, col):
            print(col)
            feature_behaviors.drop(col, axis=1, inplace=True)

    print(feature_behaviors.shape)
