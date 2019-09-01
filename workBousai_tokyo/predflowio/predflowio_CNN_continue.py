import csv
import numpy as np
import os
import shutil
import sys
import time
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

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CNN':
        XS = np.expand_dims(XS, axis=-2)
        XS = XS.swapaxes(-2, 1)
        XS = np.squeeze(XS)
        XS = XS.reshape(XS.shape[0], XS.shape[1], XS.shape[2], -1)
    else:
        pass
    return XS, YS

def getModel(name):
    if name == 'CNN':
        vision_model = Sequential()
        vision_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(HEIGHT, WIDTH, TIMESTEP*CHANNEL)))
        vision_model.add(BatchNormalization())
        vision_model.add(Conv2D(32, (3, 3), padding='same'))
        vision_model.add(BatchNormalization())
        vision_model.add(Conv2D(32, (3, 3), padding='same'))
        vision_model.add(BatchNormalization())
        vision_model.add(Conv2D(CHANNEL, (3, 3), activation='relu', padding='same'))
        return vision_model
    else:
        return None

def testModel(name, testData):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = load_model(PATH + '/'+ name + '.h5')
    model.summary()

    XS, YS = getXSYS(testData)
    print(XS.shape, YS.shape)
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

def trainModel(name, trainvalidateData):
    print('Model Training Started ...', time.ctime())
    XS, YS = getXSYS(trainvalidateData)
    print(XS.shape, YS.shape)

    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
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
MODELNAME = 'CNN'
KEYWORD = 'predflowio_CNN_1908041422'
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
config.gpu_options.visible_device_list = '2'
set_session(tf.Session(graph=tf.get_default_graph(), config=config))
###################################################################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    data = np.load(dataFile)
    # print('data.shape', data.shape)
    print('data.shape', data.shape, np.max(data))
    train_Num = int(data.shape[0] * trainRatio)

    print(KEYWORD, 'testing started', time.ctime())
    testData = data[train_Num:, :, :, :]
    print('testData.shape', testData.shape)
    testData = testData / MAX_FLOWIO
    testModel(MODELNAME, testData)


if __name__ == '__main__':
    main()