import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(1)

EPOCH = 30
BATCH_SIZE = 50
LR = 0.0001
log_interval = 1000

train_losses = []
train_counter = []
test_losses = []


train_data = np.array(pd.read_csv('../data/MLC/pre_train.csv'))[:, :41]
train_label = np.array(pd.read_csv('../data/MLC/pre_train.csv'))[:, 41]
test_data = np.array(pd.read_csv('../data/MLC/pre_test.csv'))[:, :41]
test_label = np.array(pd.read_csv('../data/MLC/pre_test.csv'))[:, 41]


train_data = np.concatenate((train_data,np.concatenate((train_data,train_data),axis=1)),axis=1)
train_data = np.concatenate((train_data,train_data),axis=1)
train_data = np.concatenate((train_data,train_data[:,:10]),axis=1)
train_data = train_data.reshape(-1,1,16,16)
tem_train = []
for i in range(train_data.shape[0]):
    tem_train.append((train_data[i],train_label[i]))

train_loader = Data.DataLoader(dataset=tem_train, batch_size=BATCH_SIZE, shuffle=True)


test_data = np.concatenate((test_data, np.concatenate((test_data, test_data),axis=1)),axis=1)
test_data = np.concatenate((test_data, test_data ),axis=1)
test_data = np.concatenate((test_data, test_data[:,:10]),axis=1)

test_data = torch.from_numpy(test_data.reshape(-1,1,16,16)[:2000]).cuda()
test_label = torch.from_numpy(test_label[:2000]).cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 1x16x16
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=2,
            ),# 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)# 16x8x8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 1, 2),# 32x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),# 32x4x4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 1, 2),  # 64x4x4
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x2x2
        )

        self.out = nn.Sequential(
            nn.Linear(64*2*2,256),
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
optimizer = torch.optim.Adadelta(cnn.parameters())
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
                    if (test_label[idx] == 1):
                        num += 1
                else:
                    if (test_label[idx] == 0):
                        num += 1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.2f}%'.format(epoch, step * len(b_x),
                                                                           len(train_loader.dataset),
                                                                           100. * step / len(train_loader),
                                                                           loss.item(),num/20.0))
torch.save(cnn,'cnn.pt')