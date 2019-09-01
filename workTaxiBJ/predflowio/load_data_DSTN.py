import numpy as np
from Param_DSTN_flow import *

def getXSYS(data):
    interval_p, interval_t = 1, 7  # day interval for period/trend
    depends = [range(1, len_closeness + 1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_period + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_trend + 1)]]

    start = max(len_closeness, interval_p * DAYTIMESTEP * len_period, interval_t * DAYTIMESTEP * len_trend)
    end = data.shape[0]

    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [data[i - j] for j in depends[0]]
        x_p = [data[i - j] for j in depends[1]]
        x_t = [data[i - j] for j in depends[2]]
        XC.append(np.concatenate(x_c, axis=0))
        XP.append(np.concatenate(x_p, axis=0))
        XT.append(np.concatenate(x_t, axis=0))
    XC, XP, XT = np.array(XC), np.array(XP), np.array(XT)
    # XS = [XC, XP, XT, day_feature] if day_feature is not None else [XC, XP, XT]
    YS = data[start:end]
    return XC, XP, XT, YS

def getXSYSFour(mode, data_all):
    testNum = int((4848 + 4368 + 5520 + 6624) * (1 - trainRatio))
    XC_train, XP_train, XT_train, YS_train = [], [], [], []
    XC_test, XP_test, XT_test, YS_test = [], [], [], []

    for i in range(3):
        data = data_all[i]
        XC, XP, XT, YS = getXSYS(data)
        XC_train.append(XC)
        XP_train.append(XP)
        XT_train.append(XT)
        YS_train.append(YS)
    for i in range(3,4):
        data = data_all[i]
        XC, XP, XT, YS = getXSYS(data)
        XC_train.append(XC[:-testNum])
        XP_train.append(XP[:-testNum])
        XT_train.append(XT[:-testNum])
        YS_train.append(YS[:-testNum])
        XC_test.append(XC[-testNum:])
        XP_test.append(XP[-testNum:])
        XT_test.append(XT[-testNum:])
        YS_test.append(YS[-testNum:])
    XC_train = np.vstack(XC_train)
    XP_train = np.vstack(XP_train)
    XT_train = np.vstack(XT_train)
    YS_train = np.vstack(YS_train)
    XC_test = np.vstack(XC_test)
    XP_test = np.vstack(XP_test)
    XT_test = np.vstack(XT_test)
    YS_test = np.vstack(YS_test)
    if mode == 'train':
        return XC_train, XP_train, XT_train, YS_train
    elif mode == 'test':
        return XC_test, XP_test, XT_test, YS_test
    else:
        assert False, 'invalid mode...'
        return None

def preload(dataFile_lst):
    data_all = []
    for item in dataFile_lst:
        data = np.load(item)
        data = data.transpose((0, 3, 1, 2))
        print(item, data.shape)
        data_all.append(data)
    return data_all

def load_data():
    data = preload(dataFile_lst)
    data_norm = [x / MAX_FLOWIO for x in data]
    XC_train, XP_train, XT_train, YS_train = getXSYSFour('train', data_norm)
    print(XC_train.shape, XP_train.shape, XT_train.shape, YS_train.shape)
    XC_test, XP_test, XT_test, YS_test = getXSYSFour('test', data_norm)
    print(XC_test.shape, XP_test.shape, XT_test.shape, YS_test.shape)
    XS_train = np.concatenate((XC_train, XP_train, XT_train), axis=1)
    XS_test = np.concatenate((XC_test, XP_test, XT_test), axis=1)
    return XS_train, YS_train, XS_test, YS_test

def main():
    XS_train, YS_train, XS_test, YS_test = load_data()
    print(XS_train.shape, YS_train.shape, XS_test.shape, YS_test.shape)

if __name__ == '__main__':
    main()
