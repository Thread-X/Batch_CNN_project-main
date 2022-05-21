import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LR = 0.0001
BATCH_SIZE = 128
LOG = 1024
EPOCH = 300

def get_total_pic(p_data,n_data,need_num):

    total_num = p_data.shape[0] + n_data.shape[0]
    # 负图片中含有的负样本数量
    n_num = int(n_data.shape[0] * need_num / total_num)

    # 图片总数pic_num
    pic_num = int(p_data.shape[0] / (2*need_num - n_num))

    # 生成正图片
    postive_part_data = np.array(p_data.iloc[:pic_num*need_num,1:42])
    postive_part_data = postive_part_data.reshape((pic_num,need_num*41))
    add_part = np.zeros((pic_num,(41-need_num)*41))
    postive_part_data = np.concatenate((postive_part_data,add_part),axis=1).reshape((pic_num,41,41))

    # 生成负图片
    negtive_part_p = np.array(p_data.iloc[pic_num*need_num:pic_num*need_num+pic_num*(need_num-n_num),1:42])
    negtive_part_p = negtive_part_p.reshape((pic_num,-1))
    negtive_part_n = np.array(n_data.iloc[:pic_num*n_num,1:42])
    negtive_part_n = negtive_part_n.reshape((pic_num,-1))
    negtive_part_data = np.concatenate((negtive_part_p,negtive_part_n),axis=1).reshape(pic_num,need_num,41)
    for idx in range(pic_num):
        np.random.shuffle(negtive_part_data[idx])
    negtive_part_data = negtive_part_data.reshape((pic_num,-1))
    add_part = np.zeros((pic_num, (41 - need_num) * 41))
    negtive_part_data = np.concatenate((negtive_part_data, add_part), axis=1).reshape((pic_num, 41, 41))

    total_data = np.concatenate((postive_part_data,negtive_part_data),axis=0)
    total_data = total_data.reshape((2*pic_num,1,41,41))
    total_label = np.concatenate((np.zeros(pic_num),np.ones(pic_num)),axis=0)

    return torch.from_numpy(total_data).to(device),torch.from_numpy(total_label).to(device)
# 正样本、负样本、每张图片含有的数据量、batch大小
# 返回随机的batch大小的41×41图片以及标签
def get_pic(p_data,n_data,need_num,batch_size,device):

    total_num = p_data.shape[0] + n_data.shape[0]
    # 负图片中含有的负样本数量
    n_num = int(n_data.shape[0] * need_num / total_num)

    # 图片总数pic_num
    pic_num = int(p_data.shape[0] / (2*need_num - n_num))

    # 生成正图片
    postive_part_data = np.array(p_data.iloc[:pic_num*need_num,1:42])
    postive_part_data = postive_part_data.reshape((pic_num,need_num*41))
    add_part = np.zeros((pic_num,(41-need_num)*41))
    postive_part_data = np.concatenate((postive_part_data,add_part),axis=1).reshape((pic_num,41,41))

    # 生成负图片
    negtive_part_p = np.array(p_data.iloc[pic_num*need_num:pic_num*need_num+pic_num*(need_num-n_num),1:42])
    negtive_part_p = negtive_part_p.reshape((pic_num,-1))
    negtive_part_n = np.array(n_data.iloc[:pic_num*n_num,1:42])
    negtive_part_n = negtive_part_n.reshape((pic_num,-1))
    negtive_part_data = np.concatenate((negtive_part_p,negtive_part_n),axis=1).reshape(pic_num,need_num,41)
    for idx in range(pic_num):
        np.random.shuffle(negtive_part_data[idx])
    negtive_part_data = negtive_part_data.reshape((pic_num,-1))
    add_part = np.zeros((pic_num, (41 - need_num) * 41))
    negtive_part_data = np.concatenate((negtive_part_data, add_part), axis=1).reshape((pic_num, 41, 41))

    total_data = np.concatenate((postive_part_data,negtive_part_data),axis=0)
    total_data = total_data.reshape((2*pic_num,1,41,41))
    total_label = np.concatenate((np.zeros(pic_num),np.ones(pic_num)),axis=0)

    # 总抽样组数
    sampling_num = int(total_data.shape[0]/batch_size)

    # 生成随机抽样序列 n*batch
    sampling = np.random.permutation(np.arange(0,total_data.shape[0]))[:sampling_num*batch_size].reshape(-1,batch_size)
    for idx in range(sampling.shape[0]):
        yield torch.from_numpy(total_data[sampling[idx]]).to(device),torch.from_numpy(total_label[sampling[idx]]).to(device)

# 粗检测
class CNN_M(nn.Module):
    def __init__(self):
        super(CNN_M, self).__init__()
        self.conv1 = nn.Sequential( # 1x41*41
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1,2),
                stride=1
            ),# 16x41x3
            nn.ReLU(),
            nn.MaxPool2d((1,2))# 16x41*20
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, (1,2), 1),# 64x41*9
            nn.ReLU(),
            nn.MaxPool2d((1,2)),# 64x41*9
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, (1,2), 1),  # 256x41*4
            nn.ReLU(),
            nn.MaxPool2d((4,2)),  # 256x10*4
        )
        #
        self.out = nn.Sequential(
            nn.Linear(256*10*4,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            # nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 256*3*3)
        output = self.out(x)
        return output

device = torch.device("cuda:0")
# device = torch.device("cpu")

Postive_train = pd.read_csv('../data/BC/type_out_data/postive_train.csv')
Negtive_train = pd.read_csv('../data/BC/type_out_data/negtive_train.csv')
Postive_test = pd.read_csv('../data/BC/type_out_data/postive_test.csv')
Negtive_test = pd.read_csv('../data/BC/type_out_data/negtive_test.csv')

cnn_m = CNN_M()
cnn_m = cnn_m.to(device)
optimizer = torch.optim.Adam(cnn_m.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

test_data, test_label = get_total_pic(Postive_test, Negtive_test, 41)

step = 0
for epoch in range(EPOCH):
    for x, y in get_pic(Postive_train, Negtive_train, 41, BATCH_SIZE, device):
        out_put = cnn_m(x.float().to(device))
        loss = loss_func(out_put, y.to(device).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += BATCH_SIZE
    test_output = cnn_m(test_data.float().to(device))
    pred = test_output.argmax(dim=1)
    num_correct = torch.eq(pred, test_label).sum().float().item()
    print("EPOCH:",epoch)
    print(num_correct / test_data.shape[0])

torch.save(cnn_m,'CNN_M.pt')