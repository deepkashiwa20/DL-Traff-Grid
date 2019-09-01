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


def getXSYS(allData, mode, dayInfo=False):
    '''
       Generate features and labels
       allData: input dataset
       mode: {'train', 'test'}
       dayInfo: True/False
    '''
    dates = pd.date_range(STARTDATE, ENDDATE).strftime('%Y%m%d').tolist()
    startIndex, endIndex = dates.index(STARTDATE), dates.index('20170619')
    trainData = allData[startIndex * DAYTIMESTEP : (endIndex + 1) * DAYTIMESTEP]
    interval_p, interval_t = 1, 7    # day interval for period/trend
    depends = [range(1, len_c + 1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]]
    
    # mode
    if mode == 'train':
        start = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
        end = trainData.shape[0]
    elif mode=='test':
        start = trainData.shape[0] # + len_c
        end = allData.shape[0]
    else:
        assert False, 'Invalid mode...'
        
    # day_info
    if dayInfo:
        day_info = np.genfromtxt(dataPath + '/dayInfo_onehot_{}_{}.csv'.format(STARTDATE, ENDDATE),
                                 delimiter = ',', skip_header = 1)
        dayInfo_dim = day_info.shape[1]
        day_feature = day_info[start:end]
    else:
        day_feature = None
        dayInfo_dim = 0

    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [allData[i - j] for j in depends[0]]
        x_p = [allData[i - j] for j in depends[1]]
        x_t = [allData[i - j] for j in depends[2]]
        XC.append(np.dstack(x_c))
        XP.append(np.dstack(x_p))
        XT.append(np.dstack(x_t))
        
    XC, XP, XT = np.array(XC), np.array(XP), np.array(XT)
    print(XC.shape, XP.shape, XT.shape, day_feature.shape)
    XS = [XC, XP, XT, day_feature] if day_feature is not None else [XC, XP, XT]
    YS = allData[start:end]
    print(YS.shape)
    
    return XS, YS, dayInfo_dim


def getModel(name, nb_res_units, dayInfo_dim):
    if MODELNAME == 'STResNet':
        c_dim = (HEIGHT, WIDTH, nb_channel, len_c)
        p_dim = (HEIGHT, WIDTH, nb_channel, len_p)
        t_dim = (HEIGHT, WIDTH, nb_channel, len_t)
        
        model = stresnet(c_dim = c_dim, p_dim = p_dim, t_dim = t_dim,
                         residual_units = nb_res_units, dayInfo_dim = dayInfo_dim)
        #model.summary()
        return model
    else:
        return None

    
def testModel(model, name, testX, testY):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model.load_weights(PATH + '/'+ name + '.h5')
    model.summary()

    XS, YS = testX, testY
    #print(XS.shape, YS.shape)
    keras_score = model.evaluate(XS, YS, verbose=1)
    rescale_MSE = keras_score * MAX_DENSITY * MAX_DENSITY

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    
def trainModel(model, name, trainX, trainY, dayInfo_dim):
    print('Model Training Started ...', time.ctime())
    XS, YS = trainX, trainY
    #print(XS.shape, YS.shape)

    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)

    keras_score = model.evaluate(XS, YS, verbose=1)
    rescaled_MSE = keras_score * MAX_DENSITY * MAX_DENSITY

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
KEYWORD = 'preddensity_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
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

    data = np.load(dataFile)
    print('data.shape', data.shape)
    data_norm = data / MAX_DENSITY

    # get features
    trainX, trainY, dayInfo_dim = getXSYS(data_norm, 'train', dayInfo)
    testX, testY, dayInfo_dim = getXSYS(data_norm, 'test', dayInfo)
    
    # get model
    model = getModel(MODELNAME, nb_res_units, dayInfo_dim)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    
    exit(-1)
    print(KEYWORD, 'training started', time.ctime())
    trainModel(model, MODELNAME, trainX, trainY, dayInfo_dim)
    print(KEYWORD, 'testing started', time.ctime())
    testModel(model, MODELNAME, testX, testY)


if __name__ == '__main__':
    main()