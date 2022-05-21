import numpy as np
import pandas as pd



train_data = np.array(pd.read_csv('data/BC/pre_train.csv'))
train_max = np.array(pd.read_csv('data/BC/pre_train.csv').max())[:-2]
train_min = np.array(pd.read_csv('data/BC/pre_train.csv').min())[:-2]
train_std = np.array(pd.read_csv('data/BC/pre_train.csv').std())[:-2]
train_mean = np.array(pd.read_csv('data/BC/pre_train.csv').mean())[:-2]

test_data = np.array(pd.read_csv('data/BC/pre_test.csv'))
test_max = np.array(pd.read_csv('data/BC/pre_test.csv').max())[:-2]
test_min = np.array(pd.read_csv('data/BC/pre_test.csv').min())[:-2]
test_std = np.array(pd.read_csv('data/BC/pre_test.csv').std())[:-2]
test_mean = np.array(pd.read_csv('data/BC/pre_test.csv').mean())[:-2]



max_min_list = [1,2,3,6,11,20,21]

for i in range(41):
    if(i!=19):
        if(i in max_min_list):
            train_data[:, i] = (train_data[:, i] - train_min[i]) / (train_max[i] - train_min[i])
            test_data[:, i] = (test_data[:, i] - test_min[i]) / (test_max[i] - test_min[i])
        else:
            train_data[:, i] = (train_data[:, i] - train_mean[i]) / train_std[i]
            test_data[:, i] = (test_data[:, i] - test_mean[i]) / test_std[i]

train_data = np.delete(train_data,19,axis=1)
test_data = np.delete(test_data,19,axis=1)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

train_data.to_csv('data/BC/mix_pre_train.csv',index=False)
test_data.to_csv('data/BC/mix_pre_test.csv',index=False)
# for i in range(41):
#     if(i!=19):
#         train_data[:, i] = (train_data[:, i] - train_min[i]) / (train_max[i] - train_min[i])
#         test_data[:, i] = (test_data[:, i] - test_min[i]) / (test_max[i] - test_min[i])
#
# train_data = np.delete(train_data,19,axis=1)
# test_data = np.delete(test_data,19,axis=1)
# train_data = pd.DataFrame(train_data)
# test_data = pd.DataFrame(test_data)
#
# train_data.to_csv('data/BC/01_pre_train.csv',index=False)
# test_data.to_csv('data/BC/01_pre_test.csv',index=False)


# for i in range(41):
#     if(i!=19):
#         train_data[:, i] = (train_data[:, i] - train_mean[i]) / train_std[i]
#         test_data[:, i] = (test_data[:, i] - test_mean[i]) / test_std[i]
#
# train_data = np.delete(train_data,19,axis=1)
# test_data = np.delete(test_data,19,axis=1)
# train_data = pd.DataFrame(train_data)
# test_data = pd.DataFrame(test_data)
#
# train_data.to_csv('data/BC/ms_pre_train.csv',index=False)
# test_data.to_csv('data/BC/ms_pre_test.csv',index=False)




# 0# @attribute 'duration' real
# 1# @attribute 'protocol_type' {'tcp','udp', 'icmp'}
# 2# @attribute 'service' {'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'}
# 3# @attribute 'flag' { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' }
# 4# @attribute 'src_bytes' real
# 5# @attribute 'dst_bytes' real
# 6# @attribute 'land' {'0', '1'}
# 7# @attribute 'wrong_fragment' real
# 8# @attribute 'urgent' real
# 9# @attribute 'hot' real
# 10# @attribute 'num_failed_logins' real
# 11# @attribute 'logged_in' {'0', '1'}
# 12# @attribute 'num_compromised' real
# 13# @attribute 'root_shell' real
# 14# @attribute 'su_attempted' real
# 15# @attribute 'num_root' real
# 16# @attribute 'num_file_creations' real
# 17# @attribute 'num_shells' real
# 18# @attribute 'num_access_files' real
# 19# @attribute 'num_outbound_cmds' real
# 20# @attribute 'is_host_login' {'0', '1'}
# 21# @attribute 'is_guest_login' {'0', '1'}
# 22# @attribute 'count' real
# 23# @attribute 'srv_count' real
# 24# @attribute 'serror_rate' real
# 25# @attribute 'srv_serror_rate' real
# 26# @attribute 'rerror_rate' real
# 27# @attribute 'srv_rerror_rate' real
# 28# @attribute 'same_srv_rate' real
# 29# @attribute 'diff_srv_rate' real
# 30# @attribute 'srv_diff_host_rate' real
# 31# @attribute 'dst_host_count' real
# 32# @attribute 'dst_host_srv_count' real
# 33# @attribute 'dst_host_same_srv_rate' real
# 34# @attribute 'dst_host_diff_srv_rate' real
# 35# @attribute 'dst_host_same_src_port_rate' real
# 36# @attribute 'dst_host_srv_diff_host_rate' real
# 37# @attribute 'dst_host_serror_rate' real
# 38# @attribute 'dst_host_srv_serror_rate' real
# 39# @attribute 'dst_host_rerror_rate' real
# 40# @attribute 'dst_host_srv_rerror_rate' real
# @attribute 'class' {'normal', 'anomaly'}