from Param_DSTN_flow import *
import numpy as np

data = np.load(dataFile)
train_data = data[:int(len(data) * trainRatio)]

_max = np.max(data)
MAX_FLOWIO = np.max(train_data)
pass
