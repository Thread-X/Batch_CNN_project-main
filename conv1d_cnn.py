import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 1x41
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=1
            ),# 32x41x3
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 128, 9, 1),# 16x20
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 16, 1),  # 256x9
            nn.Tanh(),
        )

        self.out = nn.Sequential(
            nn.Linear(3584,1024),
            nn.Tanh(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 32),
            nn.Tanh(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32, 2)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 64x2x2)
        output = self.out(x)
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

BATCH_SIZE = 128
LR = 0.0001
EPOCH = 100
log_interval = 4096

# train_data = np.array(pd.read_csv('data/BC/01_pre_train.csv'))[:, :40]
# train_label = np.array(pd.read_csv('data/BC/01_pre_train.csv'))[:, 40]
# test_data = np.array(pd.read_csv('data/BC/01_pre_test.csv'))[:, :40]
# test_label = np.array(pd.read_csv('data/BC/01_pre_test.csv'))[:, 40]


# train_data = np.array(pd.read_csv('data/BC/ms_pre_train.csv'))[:, :40]
# train_label = np.array(pd.read_csv('data/BC/ms_pre_train.csv'))[:, 40]
# test_data = np.array(pd.read_csv('data/BC/ms_pre_test.csv'))[:, :40]
# test_label = np.array(pd.read_csv('data/BC/ms_pre_test.csv'))[:, 40]

train_data = np.array(pd.read_csv('data/BC/mix_pre_train.csv'))[:, :40]
train_label = np.array(pd.read_csv('data/BC/mix_pre_train.csv'))[:, 40]
test_data = np.array(pd.read_csv('data/BC/mix_pre_test.csv'))[:, :40]
test_label = np.array(pd.read_csv('data/BC/mix_pre_test.csv'))[:, 40]

train_data = train_data.reshape((-1,1,40))


train_raw = []
for i in range(train_data.shape[0]):
    train_raw.append((train_data[i],train_label[i]))

train_loader = Data.DataLoader(dataset=train_raw, batch_size=BATCH_SIZE, shuffle=True)

test_data = torch.from_numpy(test_data.reshape((-1,1,40))).to(device)
test_label = torch.from_numpy(test_label).to(device)

cnn = CNN()
cnn = cnn.to(device)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x.type(torch.FloatTensor).to(device))
        loss = loss_func(output,b_y.to(device).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(step % log_interval==0):
            pred = output.argmax(dim=1)
            num_correct = torch.eq(pred, b_y.to(device)).sum().float().item()
            print("train:", num_correct / b_y.shape[0])

    test_output = cnn(test_data.type(torch.FloatTensor).to(device))
    pred = test_output.argmax(dim=1)
    num_correct = torch.eq(pred, test_label).sum().float().item()
    print("test:",num_correct / test_data.shape[0])