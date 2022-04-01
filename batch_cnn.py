import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


torch.manual_seed(1)

EPOCH = 500
BATCH_SIZE = 491
LR = 0.001
log_interval = 16

train_losses = []
train_counter = []
test_losses = []


train_data = np.array(pd.read_csv('data/pre_train.csv'))
test_data = np.array(pd.read_csv('data/pre_test.csv'))
train_data = np.concatenate((train_data,train_data))
train_data = (train_data[:,:41],train_data[:,41])



train_data = (np.concatenate((train_data[0],np.concatenate((train_data[0],train_data[0]),axis=1)),axis=1),train_data[1])
train_data = (np.concatenate((train_data[0],train_data[0][:,:5]),axis=1),train_data[1])



# norm = 0
# for idx in range(train_data[0].shape[0]):
#     if train_data[1][idx] == 0:
#         norm+=1
#
# print(norm/train_data[0].shape[0])
# train_data = (train_data[0][:125968].reshape(-1,1024),np.sum(train_data[1][:125968].reshape(-1,8),axis=1))

norm_list = []
anorm_list = []
for idx in range(train_data[0].shape[0]):
    if train_data[1][idx] == 0:
        norm_list.append((train_data[0][idx],train_data[1][idx]))
    else:
        anorm_list.append((train_data[0][idx], train_data[1][idx]))


norm_list = np.array(norm_list)
anorm_list = np.array(anorm_list)
anorm_len = int(len(norm_list) / 3)
anorm_list = anorm_list[np.random.choice(anorm_list.shape[0], anorm_len, replace=False)]

mix_list = np.random.choice(norm_list.shape[0], anorm_len, replace=False)
choice_list = np.setdiff1d(np.arange(0,norm_list.shape[0]), mix_list, assume_unique=False)

anorm_list = np.concatenate((norm_list[mix_list],anorm_list),axis=0)
norm_list = norm_list[choice_list]

count = int(norm_list.shape[0] / 8) * 8
data_list = norm_list[:count ,0]
label_list = norm_list[:count ,1]
data_list = np.array([list(i) for i in data_list])
data_list = data_list.reshape(-1,1,32,32)
label_list = np.zeros(data_list.shape[0])

np.random.shuffle(anorm_list)
count = int(anorm_list.shape[0] / 8) * 8
adata_list = anorm_list[:count, 0]
alabel_list = anorm_list[:count, 1]
adata_list = np.array([list(i) for i in adata_list])
adata_list = adata_list.reshape(-1,1,32,32)
alabel_list = np.sum(alabel_list.reshape(-1,8),axis=1)
for idx in range(alabel_list.shape[0]):
    if alabel_list[idx] >= 1:
        alabel_list[idx] = 1
    else:
        alabel_list[idx] = 0

train_data = np.concatenate((adata_list,data_list))
train_label = np.concatenate((alabel_list,label_list))

# print(train_data.shape,train_label.shape)

train_list = np.random.choice(train_data.shape[0], int(train_data.shape[0] * 0.7), replace=False)
test_list = np.setdiff1d(np.arange(0,train_data.shape[0]), mix_list, assume_unique=False)

tem_train = []
for i in train_list:
    tem_train.append((train_data[i],train_label[i]))

train_loader = Data.DataLoader(dataset=tem_train, batch_size=BATCH_SIZE, shuffle=True)

test_data = torch.from_numpy(train_data[test_list]).cuda()
test_label = torch.from_numpy(train_label[test_list].astype(int))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 1x32x32
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=2,
            ),# 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)# 16x16x16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 1, 2),# 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),# 32x8x8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 1, 2),  # 64x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x4x4
        )

        self.out = nn.Sequential(
            nn.Linear(64*4*4,256),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256,128),
            nn.Linear(128,16),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(16, 1),
            nn.Softsign()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 64x2x2)\
        output = self.out(x)
        return output

cnn = CNN()
cnn = cnn.cuda()
print(cnn)
#
optimizer = torch.optim.Adam(cnn.parameters(),LR=LR)
loss_func = nn.BCEWithLogitsLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x.type(torch.FloatTensor).cuda()).reshape(-1)
        # print(output[0])
        loss = loss_func(output, b_y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_interval == 0:
            test_output = cnn(test_data.type(torch.FloatTensor).cuda()).reshape(-1)
            num = 0
            for idx in range(test_output.shape[0]):
                if(test_output[idx] < 0):
                    if (test_label[idx] == 0):
                        num += 1
                else:
                    if (test_label[idx] == 1):
                        num += 1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.2f}%'.format(epoch, step * len(b_x),
                                                                           len(train_loader.dataset),
                                                                           100. * step / len(train_loader),
                                                                           loss.item(),num*100/test_output.shape[0]))
torch.save(cnn,'datch_cnn.pt')
