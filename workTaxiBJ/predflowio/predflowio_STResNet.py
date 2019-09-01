import csv
import numpy as np
import os
import shutil
import sys
import time
import pandas as pd
from datetime import datetime
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from Param import *
from Param_STResNet import *
from ST_ResNet import stresnet

def load_data(dataFile_lst, timeFile):
    data_all = []
    for item in dataFile_lst:
        data = np.load(item)
        print(item, data.shape)
        data_all.append(data)
    timestamps_all = np.load(timeFile)
    timestamps_all = timestamps_all.tolist()
    return data_all, timestamps_all

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def getXSYS(data, timestrings):
    interval_p, interval_t = 1, 7  # day interval for period/trend
    depends = [range(1, len_c + 1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]]

    start = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
    end = data.shape[0]

    day_info = timestamp2vec(timestrings)
    day_feature = day_info[start:end]

    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [data[i - j] for j in depends[0]]
        x_p = [data[i - j] for j in depends[1]]
        x_t = [data[i - j] for j in depends[2]]
        XC.append(np.dstack(x_c))
        XP.append(np.dstack(x_p))
        XT.append(np.dstack(x_t))
    XC, XP, XT = np.array(XC), np.array(XP), np.array(XT)
    # XS = [XC, XP, XT, day_feature] if day_feature is not None else [XC, XP, XT]
    YS = data[start:end]
    return XC, XP, XT, day_feature, YS

def getXSYSFour(mode, data_all, timestamps_all):
    testNum = int((4848 + 4368 + 5520 + 6624) * (1 - trainRatio))
    XC_train, XP_train, XT_train, YD_train, YS_train = [], [], [], [], []
    XC_test, XP_test, XT_test, YD_test, YS_test = [], [], [], [], []

    for i in range(3):
        data, timestamps = data_all[i], timestamps_all[i]
        XC, XP, XT, YD, YS = getXSYS(data, timestamps)
        XC_train.append(XC)
        XP_train.append(XP)
        XT_train.append(XT)
        YD_train.append(YD)
        YS_train.append(YS)
    for i in range(3,4):
        data, timestamps = data_all[i], timestamps_all[i]
        XC, XP, XT, YD, YS = getXSYS(data, timestamps)
        XC_train.append(XC[:-testNum])
        XP_train.append(XP[:-testNum])
        XT_train.append(XT[:-testNum])
        YD_train.append(YD[:-testNum])
        YS_train.append(YS[:-testNum])
        XC_test.append(XC[-testNum:])
        XP_test.append(XP[-testNum:])
        XT_test.append(XT[-testNum:])
        YD_test.append(YD[-testNum:])
        YS_test.append(YS[-testNum:])
    XC_train = np.vstack(XC_train)
    XP_train = np.vstack(XP_train)
    XT_train = np.vstack(XT_train)
    YD_train = np.vstack(YD_train)
    YS_train = np.vstack(YS_train)
    XC_test = np.vstack(XC_test)
    XP_test = np.vstack(XP_test)
    XT_test = np.vstack(XT_test)
    YD_test = np.vstack(YD_test)
    YS_test = np.vstack(YS_test)
    if mode == 'train':
        return XC_train, XP_train, XT_train, YD_train, YS_train
    elif mode == 'test':
        return XC_test, XP_test, XT_test, YD_test, YS_test
    else:
        assert False, 'invalid mode...'
        return None


def getModel(name, nb_res_units, dayInfo_dim):
    if MODELNAME == 'STResNet':
        c_dim = (HEIGHT, WIDTH, nb_channel, len_c)
        p_dim = (HEIGHT, WIDTH, nb_channel, len_p)
        t_dim = (HEIGHT, WIDTH, nb_channel, len_t)

        model = stresnet(c_dim=c_dim, p_dim=p_dim, t_dim=t_dim,
                         residual_units=nb_res_units, dayInfo_dim=dayInfo_dim)
        # model.summary()
        return model
    else:
        return None


def testModel(model, name, testX, testY):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model.load_weights(PATH + '/' + name + '.h5')
    model.summary()

    XS, YS = testX, testY
    # print(XS.shape, YS.shape)
    keras_score = model.evaluate(XS, YS, verbose=1)
    rescale_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    pred = model.predict(XS, verbose=1, batch_size=BATCHSIZE) * MAX_FLOWIO
    groundtruth = YS * MAX_FLOWIO
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', groundtruth)


def trainModel(model, name, trainX, trainY):
    print('Model Training Started ...', time.ctime())
    XS, YS = trainX, trainY
    # print(XS.shape, YS.shape)

    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)

    keras_score = model.evaluate(XS, YS, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on trainData, %f\n" % keras_score)
    f.write("Rescaled MSE on trainData, %f\n" % rescaled_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)
    print('Model Training Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'STResNet'
KEYWORD = 'predflowio_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
################# Parameter Setting #######################

###########################Reproducible#############################
import random

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(100)
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(graph=tf.get_default_graph(), config=config))


###################################################################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_STResNet.py', PATH)

    data, timestamps = load_data(dataFile_lst, timeFile)
    data_norm = [x / MAX_FLOWIO for x in data]

    XC_train, XP_train, XT_train, YD_train, YS_train = getXSYSFour('train', data_norm, timestamps)
    print(XC_train.shape, XP_train.shape, XT_train.shape, YD_train.shape, YS_train.shape)
    XC_test, XP_test, XT_test, YD_test, YS_test = getXSYSFour('test', data_norm, timestamps)
    print(XC_test.shape, XP_test.shape, XT_test.shape, YD_test.shape, YS_test.shape)
    dayInfo_dim = YD_train.shape[1]

    # get model
    model = getModel(MODELNAME, nb_res_units, dayInfo_dim)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()

    print(KEYWORD, 'training started', time.ctime())
    trainModel(model, MODELNAME, [XC_train, XP_train, XT_train, YD_train], YS_train)
    print(KEYWORD, 'testing started', time.ctime())
    testModel(model, MODELNAME, [XC_test, XP_test, XT_test, YD_test], YS_test)

if __name__ == '__main__':
    main()

