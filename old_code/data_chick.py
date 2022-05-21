import pandas as pd
import numpy as np

train_data = pd.read_csv('../data/MLC/pre_train.csv')
test_data = pd.read_csv('../data/MLC/pre_test.csv')


train_kind_cnt = [0,0,0,0,0]
for idx in range(train_data.shape[0]):
    train_kind_cnt[train_data.iloc[idx,41]] += 1

test_kind_cnt = [0,0,0,0,0]
for idx in range(test_data.shape[0]):
    test_kind_cnt[test_data.iloc[idx,41]] += 1

# test_kind_cnt = np.sum(test_kind_cnt)
print(train_kind_cnt)
print(test_kind_cnt)
