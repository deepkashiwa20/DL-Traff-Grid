# -*- coding: utf-8 -*-
'''
@Time    : 2019/7/28 20:15
@Author  : Zekun Cai
@File    : find_max.py
@Software: PyCharm
'''
from Param_DSTN_flow import *
import numpy as np

data = np.load(dataFile)
train_data = data[:int(len(data) * trainRatio)]

_max = np.max(data)
MAX_FLOWIO = np.max(train_data)
pass
