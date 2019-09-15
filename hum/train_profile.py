import importlib
import constants
import util

importlib.reload(constants)
importlib.reload(util)
from constants import train_profile_path, feature_train_profile_path
from util import *
import pandas as pd

train_profile = pd.read_csv(train_profile_path)
train_profile.columns = ['user_id', 'sex', 'occupation', 'edu', 'married', 'residence']
train_profile.head()


def feature_fun(df):
    cols = ['sex', 'occupation', 'edu', 'married', 'residence']
    feature_profile = df.groupby('user_id').first()
    for i in range(0, len(cols) - 1):
        for j in range(i + 1, len(cols)):
            index1 = cols[i]
            index2 = cols[j]
            col = index1 + '_' + index2
            feature_profile[col] = 10 * df[index1] + df[index2]

    for i in range(0, len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            for k in range(j + 1, len(cols)):
                index1 = cols[i]
                index2 = cols[j]
                index3 = cols[k]
                col = index1 + '_' + index2 + '_' + index3
                feature_profile[col] = 100 * df[index1] + 10 * df[index2] + df[index3]

    for col in feature_profile.columns:
        if tas(feature_profile, col):
            print(col)
            feature_profile.drop(col, axis=1, inplace=True)

    print(feature_profile.shape)

    feature_profile.to_csv(feature_train_profile_path)
