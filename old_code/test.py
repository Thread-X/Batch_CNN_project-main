import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import timeit

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

def get_test_pic(p_data,n_data,need_num):

    total_num = p_data.shape[0] + n_data.shape[0]
    # 负图片中含有的负样本数量
    n_num = int(n_data.shape[0] * need_num / total_num)

    # 图片总数pic_num
    pic_num = int(p_data.shape[0] / (2*need_num - n_num))

    # 生成正图片
    postive_part_data = np.array(p_data.iloc[:pic_num*need_num,1:42])
    postive_part_data = postive_part_data.reshape((pic_num,need_num*41))

    # 补全标签
    postive_part_label = np.concatenate((np.zeros((pic_num,need_num)),-1 * np.ones((pic_num,41-need_num))),axis=1)
    # 补全图片
    add_part = np.zeros((pic_num,(41-need_num)*41))
    postive_part_data = np.concatenate((postive_part_data,add_part),axis=1).reshape((pic_num,41,41))

    # 生成负图片
    negtive_part_p = np.array(p_data.iloc[pic_num*need_num:pic_num*need_num+pic_num*(need_num-n_num),1:43])
    negtive_part_p = negtive_part_p.reshape((pic_num, -1))
    negtive_part_n = np.array(n_data.iloc[:pic_num*n_num,1:43])
    negtive_part_n = negtive_part_n.reshape((pic_num, -1))

    negtive_part_data = np.concatenate((negtive_part_p,negtive_part_n),axis=1).reshape((pic_num, need_num, 42))
    for idx in range(pic_num):
        np.random.shuffle(negtive_part_data[idx])


    negtive_part_label = np.concatenate((negtive_part_data[:,:,-1], -1 * np.ones((pic_num, 41 - need_num))), axis=1)

    # negtive_part_data = (negtive_part_data[:, :, :-1])
    # print(negtive_part_label.shape)
    # print(negtive_part_data.shape)

    negtive_part_data = (negtive_part_data[:,:,:-1]).reshape((pic_num,-1))
    add_part = np.zeros((pic_num, (41 - need_num) * 41))
    negtive_part_data = np.concatenate((negtive_part_data, add_part), axis=1).reshape((pic_num, 41, 41))

    total_data = np.concatenate((postive_part_data,negtive_part_data),axis=0)
    total_data = total_data.reshape((2*pic_num,1,41,41))
    total_label = np.concatenate((postive_part_label,negtive_part_label),axis=0)

    return torch.from_numpy(total_data).to(device),torch.from_numpy(total_label).to(device)

Postive_test = pd.read_csv('../data/BC/type_out_data/postive_test.csv')
Negtive_test = pd.read_csv('../data/BC/type_out_data/negtive_test.csv')

device = torch.device("cpu")
# device = torch.device("cuda:0")

cnn_m = torch.load('./CNN_M.pt').to(device)
cnn_s = torch.load('./CNN_BC_S.pt').to(device)

m_acc_list = []
total_acc_list = []
total_chick_num = []
time_m_list = []
time_s_list = []

for con_num in range(2,42):
    test_data, test_label = get_test_pic(Postive_test, Negtive_test, con_num)
    print(test_data.shape)
    print(test_label.shape)
    test_data = test_data.float().to(device)

    m_out = cnn_m(test_data)
    time_m_list.append(timeit.timeit(stmt='cnn_m(test_data)',setup='from __main__ import cnn_m;from __main__ import test_data',number=1)/1.0)

    pred_label = -1 * np.ones((m_out.shape[0],41))
    m_pred = m_out.argmax(dim=1)
    m_correct = torch.from_numpy(np.concatenate((np.zeros(int(test_data.shape[0] / 2)), np.ones(int(test_data.shape[0] / 2))))).long().to(device)
    m_correct = torch.eq(m_pred,m_correct).to(device).sum().float().item()
    m_acc_list.append(m_correct/m_pred.shape[0])

    print(m_pred.shape)
    print(m_correct/m_pred.shape[0])

    m_pred = m_pred * torch.from_numpy(np.arange(m_pred.shape[0])).to(device)
    choose = []
    for idx in range(m_pred.shape[0]):
        if m_pred[idx]==idx:
            choose.append(idx)
        else:
            pred_label[idx] = np.zeros(41)

    s_test_data = test_data[choose].reshape(-1,1,41).to(device)
    s_test_data = s_test_data.float().to(device)

    print(len(choose))
    print(s_test_data.shape)

    test_output = cnn_s(s_test_data)
    time_s_list.append(timeit.timeit(stmt='cnn_s(s_test_data)',setup='from __main__ import cnn_s;from __main__ import s_test_data',number=10)/10.0)

    pred = test_output.argmax(dim=1)
    pred = pred.reshape((len(choose),41))
    pred_label = torch.from_numpy(pred_label).to(device)

    idx_p = 0
    for idx in choose:
        pred_label[idx] = pred[idx_p]
        idx_p += 1

    pred_label = pred_label.reshape(-1)
    test_label = test_label.reshape(-1)

    total_chick = 0
    num_correct = 0
    for idx in range(pred_label.shape[0]):
        if(test_label[idx]!=-1):
            total_chick += 1
            if(pred_label[idx]==test_label[idx]):
                num_correct += 1

    total_acc_list.append(num_correct / total_chick)
    total_chick_num.append(total_chick)

index_num = np.arange(2,42).reshape(-1,1)
total_acc_list = np.array(total_acc_list).reshape(-1,1)
total_chick_num = np.array(total_chick_num).reshape(-1,1)
m_acc_list = np.array(m_acc_list).reshape(-1,1)
time_m_list = np.array(time_m_list).reshape(-1,1) * 1000
time_s_list = np.array(time_s_list).reshape(-1,1) * 1000
time_total = time_m_list + time_s_list
test_data = np.concatenate((index_num,total_acc_list,total_chick_num ,m_acc_list,time_m_list,time_s_list,time_total),axis=1)

test_data = pd.DataFrame(test_data,columns=['数据打包量','总准确率','测试数量','粗检测准确率','粗检测网络用时(ms)','细检测网络用时(ms)','总用时(ms)'])
test_data.to_csv(str(device) + '_exp_data.csv',index=False)
# print(test_data)

