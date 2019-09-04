STARTDATE = '20170401'
ENDDATE = '20170709'
CITY = 'tokyo'
INTERVAL = 30
HEIGHT = 80
WIDTH = 80
CHANNEL = 2
DAYTIMESTEP = int(24*60/INTERVAL)
trainRatio = 0.8  # train/test

EPOCH = 200
BATCHSIZE = 4
SPLIT = 0.2  # train/val
LR = 0.0001
LOSS = 'mse'
OPTIMIZER = 'adam'

MAX_FLOWIO = 887.0
dataPath = '../../bousai_{}_jiang/'.format(CITY)
dataFile = dataPath + 'flowioK_{}_{}_{}_{}min.npy'.format(CITY, STARTDATE, ENDDATE, INTERVAL)

len_closeness = 3  # length of closeness dependent sequence
len_period = 2  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
T_closeness, T_period, T_trend = 1, DAYTIMESTEP, DAYTIMESTEP * 7

pre_F = 32  # filter size of conv
conv_F = 32  # input channels of ResPlus
R_N = 2  # nums of ResPlus
is_plus = True
plus = 8  # channels for long range spatial dependence
rate = 8  # pooling size
is_pt = False
drop = 0.1
multi_scale_fusion = True