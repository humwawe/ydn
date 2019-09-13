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
    feature_create_bill = pd.DataFrame()
    feature_create_bill['count_user_id'] = df.groupby('user_id')['user_id'].count()

