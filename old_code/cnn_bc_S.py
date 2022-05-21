import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data


# 细检测二分类模型
class CNN_BC_S(nn.Module):
    def __init__(self):
        super(CNN_BC_S, self).__init__()
        self.conv1 = nn.Sequential( # 1x41
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=2,
                stride=1
            ),# 16x41x3
            nn.ReLU(),
            nn.MaxPool1d(2)# 16x20
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 64, 2, 1),# 16x20
            nn.ReLU(),
            nn.MaxPool1d(2),# 64x9
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 256, 2, 1),  # 256x9
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256x4
        )

        self.out = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 64x2x2)
        output = self.out(x)
        return output


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
device = torch.device('cuda:0')

BATCH_SIZE = 256
COPY_TIMES = 1
LR = 0.001
EPOCH = 100
log_interval = 4096

train_data = np.array(pd.read_csv('../data/BC/pre_train.csv'))[:, :41]
train_label = np.array(pd.read_csv('../data/BC/pre_train.csv'))[:, 41]
test_data = np.array(pd.read_csv('../data/BC/pre_test.csv'))[:, :41]
test_label = np.array(pd.read_csv('../data/BC/pre_test.csv'))[:, 41]

train_data = train_data.reshape((-1,1,41))

train_raw = []
for i in range(train_data.shape[0]):
    train_raw.append((train_data[i],train_label[i]))

train_loader = Data.DataLoader(dataset=train_raw, batch_size=BATCH_SIZE, shuffle=True)

test_data = torch.from_numpy(test_data.reshape((-1,1,41))).to(device)
test_label = torch.from_numpy(test_label).to(device)

cnn_bc_s = CNN_BC_S()
cnn_bc_s = cnn_bc_s.to(device)

optimizer = torch.optim.Adam(cnn_bc_s.parameters())
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn_bc_s(b_x.type(torch.FloatTensor).to(device))
        loss = loss_func(output,b_y.to(device).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_interval == 0:
            test_output = cnn_bc_s(test_data.type(torch.FloatTensor).to(device))
            pred = test_output.argmax(dim=1)
            num_correct = torch.eq(pred, test_label).sum().float().item()
            print(num_correct)
            print(num_correct / test_data.shape[0])

# torch.save(cnn_bc_s,'CNN_S.pt')