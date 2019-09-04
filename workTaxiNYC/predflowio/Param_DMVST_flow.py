#################################################################
CITY = 'TaxiNYC'
START, END = '20150101', '20150301'
trainRatio = 0.8  # train/test
SPLIT = 0.2  # train/val
MAX_VALUE = 1283.0
freq = '30min'
INTERVAL = 30
HEIGHT = 10
WIDTH = 20
grid_size = 1  # for graph
#################################################################
DAYTIMESTEP = int(24 * 60 / INTERVAL)
local_image_size = 9
cnn_hidden_dim_first = 32
feature_len = DAYTIMESTEP + 7 + 2
toponet_len = 32
hidden_dim = 512
TIMESTEP = 6

BATCHSIZE = 200  # all:(T-TIMESTEP)*10*20, should be a divisor
LOSS = 'mse'
OPTIMIZER = 'adam'
EPOCH = 100
LR = 0.0001

dataPath = '../../{}/'.format(CITY)  # used by preprocess
flow_path = dataPath + 'flowioK_{}_{}_{}_{}min.npy'.format(CITY, START, END, INTERVAL)  # used by preprocess

save_path = dataPath + 'DMVST_flow/'
graph_in_path = save_path + 'graph_embed_in.txt'  # gene by preprocess, used by line
graph_out_path = save_path + 'graph_embed_out.txt'  # gene by preprocess, used by line

local_flow_in_path = save_path + 'flowioK_{}_{}_{}_{}min_local_in.npy'.format(CITY, START, END, INTERVAL)  # gene by preprocess, used by DMVST_Net
local_flow_out_path = save_path + 'flowioK_{}_{}_{}_{}min_local_out.npy'.format(CITY, START, END, INTERVAL)  # gene by preprocess, used by DMVST_Net
temporal_path = save_path + 'day_information_onehot.csv'  # gene by preprocess, used by DMVST_Net
topo_flow_in_path = save_path + 'graph_embed_1and2_in.txt'  # gene by line, used by DMVST_Net
topo_flow_out_path = save_path + 'graph_embed_1and2_out.txt'  # gene by line, used by DMVST_Net
