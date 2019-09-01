# -*- coding: utf-8 -*-
'''
@Time    : 2019/7/23 23:55
@Author  : Zekun Cai
@File    : Param_DSTN_flow.py
@Software: PyCharm
'''
STARTDATE = '20140401'
ENDDATE = '20140930'
CITY = 'BikeNYC1'
INTERVAL = 60  # 60min
HEIGHT = 21
WIDTH = 12
CHANNEL = 2
DAYTIMESTEP = int(24 * 60 / INTERVAL)
trainRatio = 0.8  # train/test

EPOCH = 200
BATCHSIZE = 4
SPLIT = 0.2  # train/val
LR = 0.0001
LOSS = 'mse'
OPTIMIZER = 'adam'

MAX_FLOWIO = 737.0
dataPath = '../../{}/'.format(CITY)
dataFile = dataPath + 'flowioK_{}_{}_{}_{}min.npy'.format(CITY, STARTDATE, ENDDATE, INTERVAL)

len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
T_closeness, T_period, T_trend = 1, DAYTIMESTEP, DAYTIMESTEP * 7

pre_F = 32  # filter size of conv
conv_F = 32  # input channels of ResPlus
R_N = 2  # nums of ResPlus
is_plus = True
plus = 8  # channels for long range spatial dependence
rate = 1  # pooling size
is_pt = False
drop = 0.1
multi_scale_fusion = True
