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

train_behaviors['mounth'] = train_behaviors['date'].apply(month)
train_behaviors['day'] = train_behaviors['date'].apply(day)
train_behaviors['day_10'] = (train_behaviors['mounth'] - 1) * 31 + train_behaviors['day']


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

            for i in ['behavior_type', 'weekday', 'month', 'day', 'day_10', 'sub_behavior_type_1',
                      'sub_behavior_type_2']:
                if i == col and m != -1:
                    continue
                suffix_2 = suffix_1 + '-' + i
                res = tmp.groupby('user_id')[i].agg(['min', 'max', 'median'])
                res.columns = ['min_' + suffix_2, 'max_' + suffix_2, 'median_' + suffix_2]
                feature_behavior = pm(feature_behavior, res)

                t0 = tmp.groupby(['user_id', i])[i].count()
                res = t0.groupby('user_id').agg(['min', 'max', 'count'])
                res.columns = ['min_user_id' + suffix_2, 'max_user_id' + suffix_2, 'count_user_id' + suffix_2]
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
    f1 = sub_feature('month', spl1)
    f2 = sub_feature('weekday', spl2)
    f3 = sub_feature('behavior_type', spl3)
    feature_behaviors = pm(f1, f2)
    feature_behaviors = pm(feature_behaviors, f3)

    for col in feature_behaviors.columns:
        if tas(feature_behaviors, col):
            print(col)
            feature_behaviors.drop(col, axis=1, inplace=True)

    print(feature_behaviors.shape)

    feature_behaviors.to_csv(feature_train_behaviors_path)
