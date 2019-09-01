# -*- coding: utf-8 -*-
'''
@Time    : 2019/5/22 19:07
@Author  : Zekun Cai
@File    : Param_DMVST.py.py
@Software: PyCharm
'''
#################################################################
CITY = 'osaka'
START, END = '20170401', '20170709'
MAX_VALUE = 1976.0
freq = '30min'
INTERVAL = 30
HEIGHT = 60
WIDTH = 60
grid_size = 6  # for graph
#################################################################
trainRatio = 0.8  # train/test
SPLIT = 0.2  # train/val
DAYTIMESTEP = int(24 * 60 / INTERVAL)
TIMESTEP = 6
local_image_size = 9
cnn_hidden_dim_first = 32
feature_len = DAYTIMESTEP + 7 + 2
toponet_len = 32
hidden_dim = 512

BATCHSIZE = 1200  # all:(T-TIMESTEP)*60*60, should be a divisor
LOSS = 'mse'
OPTIMIZER = 'adam'
EPOCH = 100
LR = 0.0001

dataPath = '../../bousai_{}_jiang/'.format(CITY)  # used by preprocess
density_path = dataPath + 'densityK_{}_{}_{}_{}min.npy'.format(CITY, START, END, INTERVAL)  # used by preprocess

save_path = dataPath + 'DMVST_density/'
graph_path = save_path + 'graph_embed.txt'  # gene by preprocess, used by line
local_density_path = save_path + 'densityK_{}_{}_{}_30min_local.npy'.format(CITY, START, END)  # gene by preprocess, used by DMVST_Net
temporal_path = save_path + 'day_information_onehot.csv'  # gene by preprocess, used by DMVST_Net
topo_density_path = save_path + 'graph_embed_1and2.txt'  # gene by line, used by DMVST_Net
