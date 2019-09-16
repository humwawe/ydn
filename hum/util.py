import pandas as pd
from sklearn.metrics import roc_curve


def pcv(df, col):
    return df[col].value_counts(dropna=False)


def tas(df, col):
    if len(pcv(df, col)) == 1:
        return True
    return False


def pm(df1, df2):
    return pd.merge(df1, df2, how='left', left_index=True, right_index=True)


def ks(preds, real):
    y_true = real.get_label()
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    ks_s = max(tpr - fpr)
    return 'ks', 1 - ks_s


def month(s):
    i = s.index("-")
    return int(s[:i])


def day(s):
    i = s.index("-")
    return int(s[i + 1:])


def ten_day(s):
    if s <= 10:
        return 1
    elif s <= 20:
        return 2
    return 3
