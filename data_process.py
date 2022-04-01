import pandas as pd
from scipy.io import arff

file_name='data/KDDTest+.arff'

data,meta = arff.loadarff(file_name)
traindata = pd.DataFrame(data)

X = traindata.iloc[:,0:42]

A = dict(zip(list(set(X.iloc[:,1])),[i for i in range(len(set(X.iloc[:,1])))]))
B = dict(zip(list(set(X.iloc[:,2])),[i for i in range(len(set(X.iloc[:,2])))]))
C = dict(zip(list(set(X.iloc[:,3])),[i for i in range(len(set(X.iloc[:,3])))]))
L = dict(zip(list(set(X.iloc[:,41])),[i for i in range(len(set(X.iloc[:,41])))]))

for idx in range(traindata.shape[0]):
    traindata.iloc[idx,1] = A[traindata.iloc[idx,1]]
    traindata.iloc[idx, 2] = B[traindata.iloc[idx, 2]]
    traindata.iloc[idx, 3] = C[traindata.iloc[idx, 3]]
    traindata.iloc[idx, 41] = L[traindata.iloc[idx, 41]]

traindata.to_csv("data/pre_test.csv",index=False)