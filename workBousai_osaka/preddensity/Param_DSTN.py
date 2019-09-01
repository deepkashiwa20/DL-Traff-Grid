# -*- coding: utf-8 -*-
'''
@Time    : 2019/7/23 17:20
@Author  : Zekun Cai
@File    : Param_DSTN.py
@Software: PyCharm
'''
STARTDATE = '20170401'
ENDDATE = '20170709'
CITY = 'osaka'
INTERVAL = 30
HEIGHT = 60
WIDTH = 60
CHANNEL = 1
MAX_DENSITY = 1976.0

DAYTIMESTEP = int(24 * 60 / INTERVAL)
trainRatio = 0.8  # train/test

EPOCH = 200
BATCHSIZE = 4
SPLIT = 0.2  # train/val
LR = 0.0001
LOSS = 'mse'
OPTIMIZER = 'adam'

dataPath = '../../bousai_{}_jiang/'.format(CITY)
dataFile = dataPath + 'densityK_{}_{}_{}_{}min.npy'.format(CITY, STARTDATE, ENDDATE, INTERVAL)

len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
T_closeness, T_period, T_trend = 1, DAYTIMESTEP, DAYTIMESTEP * 7

pre_F = 32  # filter size of conv
conv_F = 32  # input channels of ResPlus
R_N = 2  # nums of ResPlus
is_plus = True
plus = 8  # channels for long range spatial dependence
rate = 6  # pooling layer size
is_pt = False
drop = 0.1
multi_scale_fusion = True
