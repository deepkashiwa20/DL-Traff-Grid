# -*- coding: utf-8 -*-
'''
@Time    : 2019/7/23 23:53
@Author  : Zekun Cai
@File    : predflowio_DSTN+.py
@Software: PyCharm
'''
import datetime
import sys
import shutil
import time

from keras.models import load_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard

from DeepSTN_net import DeepSTN
from load_data_DSTN import load_data
from Param_DSTN_flow import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def train_model(X_train, Y_train):
    csv_logger = CSVLogger(PATH + '/' + MODELNAME + '.log')
    checkpointer_path = PATH + '/' + MODELNAME + '.h5'
    checkpointer = ModelCheckpoint(filepath=checkpointer_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    LearnRate = LearningRateScheduler(lambda epoch: LR)

    model = DeepSTN(H=HEIGHT, W=WIDTH, channel=CHANNEL,
                    c=len_closeness, p=len_period, t=len_trend,
                    pre_F=pre_F, conv_F=conv_F, R_N=R_N,
                    is_plus=is_plus, plus=plus, rate=rate,
                    is_pt=is_pt, T=DAYTIMESTEP, drop=drop)

    model.fit(X_train, Y_train, epochs=EPOCH, batch_size=BATCHSIZE,
              validation_split=SPLIT, shuffle=True,
              callbacks=[csv_logger, checkpointer, LearnRate, early_stopping])

    keras_score = model.evaluate(X_train, Y_train, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on trainData, %f\n" % keras_score)
    f.write("Rescaled MSE on trainData, %f\n" % rescaled_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)
    print('Model Training Ended ...', time.ctime())

    return model


def test_model(X_test, Y_test, model):
    print('Model Evaluation Started ...', time.ctime())

    # assert os.path.exists(PATH + '/' + MODELNAME + '.h5'), 'model is not existing'
    # model = load_model(PATH + '/' + MODELNAME + '.h5')
    model.summary()

    keras_score = model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCHSIZE)
    rescale_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    pred = model.predict(X_test, verbose=1, batch_size=BATCHSIZE)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)


################# Path Setting #######################
MODELNAME = 'DeepSTN+'
KEYWORD = 'predflowio_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
###########################Reproducible#############################
import numpy as np
import random
from keras import backend as K
import os
import tensorflow as tf

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3

tf.set_random_seed(100)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.visible_device_list = '2'
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###################################################################

if __name__ == '__main__':
    mkdir(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('DeepSTN_net.py', PATH)
    shutil.copy2('load_data_DSTN.py', PATH)
    shutil.copy2('Param_DSTN_flow.py', PATH)
    StartTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print('#' * 50)
    print('start running at {}'.format(StartTime))
    print('model name: {}'.format(MODELNAME))
    print('#' * 50, '\n')

    X_train, Y_train, X_test, Y_test = load_data(len_closeness, len_period, len_trend,
                                                 T_closeness, T_period, T_trend)
    model = train_model(X_train, Y_train)
    test_model(X_test, Y_test, model)
