train_path = "../data/problem_1/train/"
train_bank_statement_path = train_path + "train_bankStatement.csv"
train_behaviors_path = train_path + "train_behaviors.csv"
train_credit_bill_path = train_path + "train_creditBill.csv"
train_profile_path = train_path + "train_profile.csv"

test_path = "../data/problem_1/B/"
test_bank_statement_path = test_path + "test_bankStatement_B.csv"
test_behaviors_path = test_path + "test_behaviors_B.csv"
test_credit_bill_path = test_path + "test_creditBill_B.csv"
test_profile_path = test_path + "test_profile_B.csv"

train_label_path = train_path + "train_label.csv"

feature_path = 'feature/'
feature_train_profile_path = feature_path + "feature_train_profile.csv"
feature_train_behaviors_path = feature_path + "feature_train_behaviors.csv"
feature_train_credit_bill_path = feature_path + "feature_train_creditBill.csv"
feature_train_credit_bill_path_2 = feature_path + "feature_train_creditBill_2.csv"
feature_train_bank_statement_path = feature_path + "feature_train_bank_statement.csv"
feature_test_profile_path = feature_path + "feature_test_profile.csv"
feature_test_behaviors_path = feature_path + "feature_test_behaviors.csv"
feature_test_credit_bill_path = feature_path + "feature_test_creditBill.csv"
feature_test_credit_bill_path_2 = feature_path + "feature_test_creditBill_2.csv"
feature_test_bank_statement_path = feature_path + "feature_test_bank_statement.csv"
feature_profile_path = feature_path + "feature_profile.csv"

feature_train_label_path = feature_path + "feature_train_label.csv"

cv_result_path = "tmp/cv_result.pickle"
cv_result_search_path = "tmp/cv_result_search.pickle"

lda_feature_path = "../wangzhen/data/lda_feature.p"
ts_feature_path = "../wangzhen/data/ts_feature.p"
del_col_path = "../wangzhen/data/rel_cols.p"
te_feature_path = "../wangzhen/data/table_exists_feature.p"

xu_feature_path = "../xujunfeng/feature_statement_bill_behaviors.pkl"
# bill_feature_path = "../xujunfeng/FE_creditbill-2.pkl"

# profile_feature_path = '../xujunfeng/df_profile.pkl'

# w2v_feature_train_path = "../xujunfeng/w2v_10/train_w2v.csv"
# w2v_feature_test_path = "../xujunfeng/w2v_10/test_w2v.csv"

w2v_feature_train_path_1 = "../xujunfeng/w2v_20/train_w2v.csv"
w2v_feature_test_path_1 = "../xujunfeng/w2v_20/test_w2v.csv"

w2v_feature_train_path_2 = "../xujunfeng/train_w2v.csv"
w2v_feature_test_path_2 = "../xujunfeng/test_w2v.csv"

w2v_feature_train_path_3 = "../xujunfeng/train_w2v_clear.csv"
w2v_feature_test_path_3 = "../xujunfeng/test_w2v_clear.csv"

w2v_feature_train_path_4 = "../xujunfeng/w2v_delete_subtype1/train_w2v_clear.csv"
w2v_feature_test_path_4 = "../xujunfeng/w2v_delete_subtype1/test_w2v_clear.csv"

w2v_feature_train_path_5 = "../xujunfeng/w2v_64/train_w2v.csv"
w2v_feature_test_path_5 = "../xujunfeng/w2v_64/test_w2v.csv"

day_second = 86400
statement_max_time = 44243
bill_max_time = 44724
# t_bill_max_time = 47100
t_bill_max_time = 44724
bill_time_ratio = 1
