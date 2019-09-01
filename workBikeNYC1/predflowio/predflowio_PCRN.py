import sys
import shutil
import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import h5py
from copy import copy
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Activation, Flatten, Dense,Reshape, Concatenate, Add, Lambda, Layer, add, multiply
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
import keras.backend as K
from Param import *

def getXSYS_CPT_D(mode, allData, trainData, dayinfo):
    len_c, len_p, len_t = TIMESTEP, 1, 1
    interval_p, interval_t = 1, 7

    stepC = list(range(1, len_c + 1))
    periods, trends = [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)], \
                      [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]
    stepP, stepT = [], []
    for p in periods:
        stepP.extend(list(range(p, p + len_c)))
    for t in trends:
        stepT.extend(list(range(t, t + len_c)))
    depends = [stepC, stepP, stepT]

    if mode == 'train':
        start = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
        end = trainData.shape[0]
    elif mode == 'test':
        start = trainData.shape[0] + len_c
        end = allData.shape[0]
    else:
        assert False, 'invalid mode...'

    XC, XP, XT, YS, YD = [], [], [], [], []
    for i in range(start, end):
        x_c = [allData[i - j][np.newaxis, :, :, :] for j in depends[0]]
        x_p = [allData[i - j][np.newaxis, :, :, :] for j in depends[1]]
        x_t = [allData[i - j][np.newaxis, :, :, :] for j in depends[2]]
        x_c = np.concatenate(x_c, axis=0)
        x_p = np.concatenate(x_p, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        x_c = x_c[::-1, :, :, :]
        x_p = x_p[::-1, :, :, :]
        x_t = x_t[::-1, :, :, :]
        d = dayinfo[i]
        y = allData[i]
        XC.append(x_c)
        XP.append(x_p)
        XT.append(x_t)
        YS.append(y)
        YD.append(d)
    XC, XP, XT, YS, YD = np.array(XC), np.array(XP), np.array(XT), np.array(YS), np.array(YD)

    return XC, XP, XT, YS, YD

##################### PCRN Model ############################

def concat_28(ts):
    lst = []
    for i in range(14):
        lst.append(K.concatenate([ts[:, 2*i,:,:,:], ts[:, 2*i+1,:,:,:]], axis = -1))   # First dimension - sample (batch_size)
    return K.concatenate([i[:,np.newaxis,:,:,:] for i in lst], axis = 1)   # output_shape needs to be specified when it differs from input_shape

def concat_14(ts):
    lst = []
    for i in range(7):
        lst.append(K.concatenate([ts[:, 2*i,:,:,:], ts[:, 2*i+1,:,:,:]], axis = -1))
    return K.concatenate([i[:,np.newaxis,:,:,:] for i in lst], axis = 1)

def ConvLSTMs():
    model = Sequential()
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = True,
                        input_shape = (None, HEIGHT, WIDTH, CHANNEL)))
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = True))
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = False))
    return model

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
        hft = K.relu(hct*self.Wc + hallt*self.Wp)
        return hft

    def get_output_shape(self, input_shape):
        return input_shape

def softmax(ej_lst):
    return K.exp(ej_lst[0])/(K.exp(ej_lst[0]) + K.exp(ej_lst[1]))

def getModel(x_dim, meta_dim):
    # Input xc, xp, xt --> hct1, hP1, hP2
    XC = Input(shape = x_dim)
    XP = Input(shape = x_dim)
    XT = Input(shape = x_dim)

    shared_model = Sequential()
    shared_model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = True, input_shape = x_dim))
    shared_model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = True))
    shared_model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3),
                         padding = 'same', return_sequences = False))

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
    hft_reshap = Conv2D(filters = CHANNEL, kernel_size = (HEIGHT, WIDTH),
                        activation = 'relu', padding = 'same')(hft)
    
    # metadata fusion
    Xmeta = Input(shape = (meta_dim,))
    dens1 = Dense(units = 10, activation = 'relu')(Xmeta)
    dens2 = Dense(units = WIDTH*HEIGHT*CHANNEL, activation = 'relu')(dens1)
    hmeta = Reshape((HEIGHT, WIDTH, CHANNEL))(dens2)
    
    add2 = Add()([hft_reshap, hmeta])
    X_hat = Activation('relu')(add2)
    
    model = Model(inputs = [XC, XP, XT, Xmeta], outputs = X_hat)
    return model

def testModel(name, allData, trainData, dayinfo):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = load_model(PATH + '/'+ name + '.h5', custom_objects={'Hadamard_fusion': Hadamard_fusion, 'softmax': softmax})
    model.summary()

    XC, XP, XT, YS, YD = getXSYS_CPT_D('test', allData, trainData, dayinfo)
    print(XC.shape, XP.shape, XT.shape, YS.shape, YD.shape)

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

def trainModel(name, allData, trainData, dayinfo):
    print('Model Training Started ...', time.ctime())
    XC, XP, XT, YS, YD = getXSYS_CPT_D('train', allData, trainData, dayinfo)
    print(XC.shape, XP.shape, XT.shape, YS.shape, YD.shape)

    model = getModel((None, HEIGHT, WIDTH, CHANNEL), dayinfo.shape[1])
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
MODELNAME = 'PCRN'
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

    data = np.load(dataFile)
    data = data / MAX_FLOWIO
    dayinfo = np.genfromtxt(dataPath + '/day_information_onehot.csv', delimiter=',', skip_header=1)
    print('data.shape, dayinfo.shape', data.shape, dayinfo.shape)
    train_Num = int(data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = data[:train_Num, :, :, :]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainModel(MODELNAME, data, trainvalidateData, dayinfo)

    print(KEYWORD, 'testing started', time.ctime())
    testData = data[train_Num:, :, :, :]
    print('testData.shape', testData.shape)
    testModel(MODELNAME, data, trainvalidateData, dayinfo)


if __name__ == '__main__':
    main()
