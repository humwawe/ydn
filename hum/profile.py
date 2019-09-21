import importlib
import constants
import util

importlib.reload(constants)
importlib.reload(util)
from constants import train_profile_path, test_profile_path, feature_train_profile_path, feature_test_profile_path
from util import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_profile = pd.read_csv(train_profile_path)
train_profile.columns = ['user_id', 'sex', 'occupation', 'edu', 'married', 'residence']
train_profile.head()


def feature_fun(df1, df2):
    tmp = pd.concat([df1, df2], axis=0)
    cols = ['sex', 'occupation', 'edu', 'married', 'residence']
    for i in range(0, len(cols) - 1):
        for j in range(i + 1, len(cols)):
            index1 = cols[i]
            index2 = cols[j]
            col = index1 + '_' + index2
            tmp[col] = tmp[index1].astype('str') + '_' + tmp[index2].astype('str')

    for i in range(0, len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            for k in range(j + 1, len(cols)):
                index1 = cols[i]
                index2 = cols[j]
                index3 = cols[k]
                col = index1 + '_' + index2 + '_' + index3
                tmp[col] = tmp[index1].astype('str') + '_' + tmp[index2].astype('str') + '_' + tmp[index3].astype('str')

    category_features = [x for x in tmp.columns if x not in ['user_id']]

    for feature in category_features:
        enc = LabelEncoder()
        tmp[feature] = enc.fit_transform(tmp[feature])
        tmp[feature + '_count'] = tmp.groupby(feature)['user_id'].transform('count')

    feature_profile = tmp

    for col in feature_profile.columns:
        if tas(feature_profile, col):
            print(col)
            feature_profile.drop(col, axis=1, inplace=True)

    print(feature_profile.shape)

    feature_profile.iloc[:df1.shape[0]].to_csv(feature_train_profile_path, index=False)
    feature_profile.iloc[df1.shape[0]:].to_csv(feature_test_profile_path, index=False)
