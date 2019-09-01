import csv
import numpy as np
import os
import shutil
import sys
import time
import pandas as pd
from datetime import datetime
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Activation, Flatten, Dense,Reshape, Concatenate, Add, Lambda, Layer, add, multiply
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
import keras.backend as K
from Param import *


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
    len_c, len_p, len_t = TIMESTEP, 1, 1
    interval_p, interval_t = 1, 7  # day interval for period/trend
    stepC = list(range(1, len_c + 1))
    periods, trends = [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)], \
                      [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]
    stepP, stepT = [], []
    for p in periods:
        stepP.extend(list(range(p, p + len_c)))
    for t in trends:
        stepT.extend(list(range(t, t + len_c)))
    depends = [stepC, stepP, stepT]

    start = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
    end = data.shape[0]

    day_info = timestamp2vec(timestrings)
    day_feature = day_info[start:end]

    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [data[i - j][np.newaxis, :, :, :] for j in depends[0]]
        x_p = [data[i - j][np.newaxis, :, :, :] for j in depends[1]]
        x_t = [data[i - j][np.newaxis, :, :, :] for j in depends[2]]
        x_c = np.concatenate(x_c, axis=0)
        x_p = np.concatenate(x_p, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        x_c = x_c[::-1, :, :, :]
        x_p = x_p[::-1, :, :, :]
        x_t = x_t[::-1, :, :, :]
        XC.append(x_c)
        XP.append(x_p)
        XT.append(x_t)
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


class Hadamard_fusion(Layer):
    def __init__(self, **kwargs):
        super(Hadamard_fusion, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.Wc = self.add_weight(name='Wc', shape=(input_shape[0][1:]),
                                  initializer='uniform', trainable=True)
        self.Wp = self.add_weight(name='Wp', shape=(input_shape[1][1:]),
                                  initializer='uniform', trainable=True)
        super(Hadamard_fusion, self).build(input_shape)

    def call(self, x, mask=None):
        assert isinstance(x, list)
        hct, hallt = x
        hft = K.relu(hct * self.Wc + hallt * self.Wp)
        return hft

    def get_output_shape(self, input_shape):
        return input_shape


def softmax(ej_lst):
    return K.exp(ej_lst[0]) / (K.exp(ej_lst[0]) + K.exp(ej_lst[1]))


def getModel(x_dim, meta_dim):
    # Input xc, xp, xt --> hct1, hP1, hP2
    XC = Input(shape=x_dim)
    XP = Input(shape=x_dim)
    XT = Input(shape=x_dim)

    shared_model = Sequential()
    shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                padding='same', return_sequences=True, input_shape=x_dim))
    shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                padding='same', return_sequences=True))
    shared_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                padding='same', return_sequences=False))

    hct1 = shared_model(XC)
    hP1 = shared_model(XP)
    hP2 = shared_model(XT)

    # Weighting based fusion
    # daily
    concate1 = Concatenate()([hct1, hP1])
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concate1)
    flat1 = Flatten()(conv1)
    ej1 = Dense(1)(flat1)

    # weekly
    concate2 = Concatenate()([hct1, hP2])
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concate2)
    flat2 = Flatten()(conv2)
    ej2 = Dense(1)(flat2)

    aj1 = Lambda(softmax)([ej1, ej2])
    aj2 = Lambda(softmax)([ej2, ej1])
    hPallt = Add()([multiply([aj1, hP1]), multiply([aj2, hP2])])

    hft = Hadamard_fusion()([hct1, hPallt])

    # transform shape
    hft_reshap = Conv2D(filters=CHANNEL, kernel_size=(1, 1),
                        activation='relu', padding='same')(hft)

    # metadata fusion
    Xmeta = Input(shape=(meta_dim,))
    dens1 = Dense(units=10, activation='relu')(Xmeta)
    dens2 = Dense(units=WIDTH * HEIGHT * CHANNEL, activation='relu')(dens1)
    hmeta = Reshape((HEIGHT, WIDTH, CHANNEL))(dens2)

    add2 = Add()([hft_reshap, hmeta])
    X_hat = Activation('relu')(add2)

    model = Model(inputs=[XC, XP, XT, Xmeta], outputs=X_hat)
    return model

def testModel(name, data_norm, timestamps):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = load_model(PATH + '/'+ name + '.h5', custom_objects={'Hadamard_fusion': Hadamard_fusion, 'softmax': softmax})
    model.summary()

    XC, XP, XT, YD, YS = getXSYSFour('test', data_norm, timestamps)
    print(XC.shape, XP.shape, XT.shape, YD.shape, YS.shape)

    keras_score = model.evaluate(x=[XC, XP, XT, YD], y=YS, verbose=1)
    rescale_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    pred = model.predict([XC, XP, XT, YD], verbose=1, batch_size=BATCHSIZE) * MAX_FLOWIO
    groundtruth = YS * MAX_FLOWIO
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', groundtruth)

def trainModel(name, data_norm, timestamps):
    print('Model Training Started ...', time.ctime())
    XC, XP, XT, YD, YS = getXSYSFour('train', data_norm, timestamps)
    print(XC.shape, XP.shape, XT.shape, YD.shape, YS.shape)

    model = getModel((None, HEIGHT, WIDTH, CHANNEL), YD.shape[1])
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(x=[XC, XP, XT, YD], y=YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)

    keras_score = model.evaluate(x=[XC, XP, XT, YD], y=YS, verbose=1)
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
MODELNAME = 'PCRND'
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
config.gpu_options.visible_device_list = '1'
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

    print(KEYWORD, 'training started', time.ctime())
    trainModel(MODELNAME, data_norm, timestamps)
    print(KEYWORD, 'testing started', time.ctime())
    testModel(MODELNAME, data_norm, timestamps)

if __name__ == '__main__':
    main()

