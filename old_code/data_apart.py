import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import os

data = pd.read_csv('../Raw_data/NSL-KDD/KDDTrain+.csv', header=None)
test_data = pd.read_csv('../Raw_data/NSL-KDD/KDDTest+.csv', header=None)


# 标签转异常
# 0-正常 1-DOS  2-PROBE 3-U2R 4-R2L
attack_type_df = pd.read_csv('../Raw_data/label_dict.csv', header=None, names=['name', 'attack_type'])
L = dict(zip(attack_type_df['name'].tolist(), attack_type_df['attack_type'].tolist()))


# 非数值转换
A = dict(zip(list(set(data.iloc[:,1])),[i for i in range(len(set(data.iloc[:,1])))]))
B = dict(zip(list(set(data.iloc[:,2])),[i for i in range(len(set(data.iloc[:,2])))]))
C = dict(zip(list(set(data.iloc[:,3])),[i for i in range(len(set(data.iloc[:,3])))]))

# 识别难度转换为识别率100%-0%
H = dict( zip(range(0,22,1),[i/21 for i in range(0,22,1)] ) )

# data.shape[0]
for idx in range(data.shape[0]):
    data.iloc[idx,1] = A[data.iloc[idx,1]]
    data.iloc[idx,2] = B[data.iloc[idx,2]]
    data.iloc[idx,3] = C[data.iloc[idx,3]]
    if(L[data.iloc[idx,41]] > 0):
        data.iloc[idx,41] = 1
    else:
        data.iloc[idx, 41] = 0
    data.iloc[idx, 42] = H[data.iloc[idx, 42]]
    if idx % 10000 == 0:
        print(idx)

data.to_csv("data/pre_train.csv",index=False)

# test_data.shape[0]
for idx in range(test_data.shape[0]):
    test_data.iloc[idx,1] = A[test_data.iloc[idx,1]]
    test_data.iloc[idx,2] = B[test_data.iloc[idx,2]]
    test_data.iloc[idx,3] = C[test_data.iloc[idx,3]]
    if L[test_data.iloc[idx,41]]>0 :
        test_data.iloc[idx,41] = 1
    else:
        test_data.iloc[idx, 41] = 0
    test_data.iloc[idx, 42] = H[test_data.iloc[idx, 42]]


test_data.to_csv("data/pre_test.csv",index=False)

# # 分类生成
# train_data = pd.read_csv('data/BC/pre_train.csv')
#
# p_d = []
# n_d = []
# for index, row in train_data.iterrows():
#     if row[41]==0:
#         p_d.append(list(row))
#     else:
#         n_d.append(list(row))
#
# p_d = pd.DataFrame(np.array(p_d))
# p_d.to_csv('data/BC/postive_train.csv')
# n_d = pd.DataFrame(np.array(n_d))
# n_d.to_csv('data/BC/negtive_train.csv')
#
# train_data = pd.read_csv('data/BC/pre_test.csv')
#
# p_d = []
# n_d = []
# for index, row in train_data.iterrows():
#     if row[41]==0:
#         p_d.append(list(row))
#     else:
#         n_d.append(list(row))
#
# p_d = pd.DataFrame(np.array(p_d))
# p_d.to_csv('data/BC/postive_test.csv')
# n_d = pd.DataFrame(np.array(n_d))
# n_d.to_csv('data/BC/negtive_test.csv')

