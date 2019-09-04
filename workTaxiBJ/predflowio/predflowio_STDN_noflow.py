import pandas as pd
import datetime
import sys
import shutil
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard

import STDN_models_noflow
from STDN_load_data_noflow import *
from Param_STDN import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_data(model_name):
    print('loading data...')
    all_data_len = 0
    for i in range(len(local_flow_in_lst_path)):
        data = np.load(local_flow_in_lst_path[i])
        all_data_len += data.shape[0]
    train_start, valid_start, test_start = \
        0, int(all_data_len * trainRatio * (1 - SPLIT)) - empty_time, int(all_data_len * trainRatio) - empty_time
    print('train len:', valid_start - train_start)
    print('valid len:', test_start - valid_start)
    print('test len:', all_data_len - test_start)
    print('\n')

    trainData, validData, testData = [], [], []
    data_len = 0
    for i in range(len(local_flow_in_lst_path)):
        if model_name == 'density':
            region_window = np.load(local_density_lst_path[i])
        elif model_name == 'flowio':
            region_window_in = np.load(local_flow_in_lst_path[i])
            region_window_out = np.load(local_flow_out_lst_path[i])
            region_window = np.concatenate((region_window_in, region_window_out), axis=-1)
        region_window = region_window / MAX_VALUE
        data_len += region_window.shape[0]
        print('data set:', i)
        print('flow data', region_window.shape)
        print('\n')

        if data_len <= valid_start:
            trainData.append(region_window)
        elif data_len <= test_start:
            trainData.append(region_window[:valid_start - data_len + empty_time])
            validData.append(region_window[valid_start - data_len:])
        else:
            validData.append(region_window[:test_start - data_len + empty_time])
            testData.append(region_window[test_start - data_len:])

    print('train data', sum([len(x) for x in trainData]))
    print('valid data', sum([len(x) for x in validData]))
    print('test data', sum([len(x) for x in testData]))
    print('load finished')
    print('load finished')

    return trainData, validData, testData


def model_train(train_data, valid_data):
    # set callbacks
    csv_logger = CSVLogger(PATH + '/' + MODELNAME + '.log')
    checkpointer_path = PATH + '/' + MODELNAME + '.h5'
    checkpointer = ModelCheckpoint(filepath=checkpointer_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    LearnRate = LearningRateScheduler(lambda epoch: LR)

    # data generator
    train_generator = data_generator(train_data,
                                     att_lstm_num=att_lstm_num,
                                     long_term_lstm_seq_len=long_term_lstm_seq_len,
                                     short_term_lstm_seq_len=short_term_lstm_seq_len,
                                     hist_feature_daynum=hist_feature_daynum,
                                     last_feature_num=last_feature_num,
                                     nbhd_size=nbhd_size,
                                     batchsize=BATCHSIZE)
    val_generator = data_generator(valid_data,
                                   att_lstm_num=att_lstm_num,
                                   long_term_lstm_seq_len=long_term_lstm_seq_len,
                                   short_term_lstm_seq_len=short_term_lstm_seq_len,
                                   hist_feature_daynum=hist_feature_daynum,
                                   last_feature_num=last_feature_num,
                                   nbhd_size=nbhd_size,
                                   batchsize=BATCHSIZE)
    sep = (sum([len(x) for x in train_data]) - empty_time * len(train_data)) * train_data[0].shape[1] // BATCHSIZE
    val_sep = (sum([len(x) for x in valid_data]) - empty_time * len(valid_data)) * valid_data[0].shape[1] // BATCHSIZE

    # train model
    modeler = STDN_models_noflow.models()
    model = modeler.stdn(att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_len,
                         lstm_seq_len=short_term_lstm_seq_len,
                         feature_vec_len=feature_vec_len,
                         cnn_flat_size=cnn_flat_size,
                         nbhd_size=window_size,
                         nbhd_type=DATACHANNEL,
                         output_shape=DATACHANNEL,
                         optimizer=OPTIMIZER, loss=LOSS)
    model.summary()
    model.fit_generator(train_generator, steps_per_epoch=sep, epochs=EPOCH,
                        validation_data=val_generator, validation_steps=val_sep,
                        callbacks=[csv_logger, checkpointer, LearnRate, early_stopping])

    # model.fit_generator(train_generator, steps_per_epoch=sep, epochs=EPOCH)

    # write record
    val_scale_MSE = model.evaluate_generator(val_generator, steps=val_sep)
    val_rescale_MSE = val_scale_MSE * MAX_VALUE * MAX_VALUE
    with open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a') as wf:
        wf.write('train start time: {}\n'.format(StartTime))
        wf.write('train end time:   {}\n'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        wf.write("Keras MSE on trainData, %f\n" % val_scale_MSE)
        wf.write("Rescaled MSE on trainData, %f\n" % val_rescale_MSE)


def model_pred(test_data):
    # test generator
    test_gene = data_generator(test_data,
                               att_lstm_num=att_lstm_num,
                               long_term_lstm_seq_len=long_term_lstm_seq_len,
                               short_term_lstm_seq_len=short_term_lstm_seq_len,
                               hist_feature_daynum=hist_feature_daynum,
                               last_feature_num=last_feature_num,
                               nbhd_size=nbhd_size,
                               batchsize=BATCHSIZE)
    test_sep = (sum([len(x) for x in test_data]) - empty_time * len(test_data)) * test_data[0].shape[1] // BATCHSIZE

    # get predict
    modeler = STDN_models_noflow.models()
    model = modeler.stdn(att_lstm_num=att_lstm_num, att_lstm_seq_len=long_term_lstm_seq_len,
                         lstm_seq_len=short_term_lstm_seq_len,
                         feature_vec_len=feature_vec_len,
                         cnn_flat_size=cnn_flat_size,
                         nbhd_size=window_size,
                         nbhd_type=DATACHANNEL,
                         output_shape=DATACHANNEL,
                         optimizer=OPTIMIZER, loss=LOSS)
    model.load_weights(PATH + '/' + MODELNAME + '.h5')

    # write record
    scale_MSE = model.evaluate_generator(test_gene, steps=test_sep)
    rescale_MSE = scale_MSE * MAX_VALUE * MAX_VALUE
    print("Model scaled MSE", scale_MSE)
    print("Model rescaled MSE", rescale_MSE)
    with open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a') as wf:
        wf.write("Keras MSE on testData, %f\n" % scale_MSE)
        wf.write("Rescaled MSE on testData, %f\n" % rescale_MSE)

    pred_gene = data_generator(test_data,
                               att_lstm_num=att_lstm_num,
                               long_term_lstm_seq_len=long_term_lstm_seq_len,
                               short_term_lstm_seq_len=short_term_lstm_seq_len,
                               hist_feature_daynum=hist_feature_daynum,
                               last_feature_num=last_feature_num,
                               nbhd_size=nbhd_size,
                               batchsize=BATCHSIZE,
                               type='test')
    pred = model.predict_generator(pred_gene, steps=test_sep, verbose=1)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)


################# Path Setting #######################
MODELNAME = 'STDN'
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
session_conf.gpu_options.visible_device_list = '3'
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###################################################################

if __name__ == '__main__':
    mkdir(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param_STDN.py', PATH)
    shutil.copy2('STDN_models_noflow.py', PATH)
    shutil.copy2('STDN_attention.py', PATH)
    StartTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print('#' * 50)
    print('start running at {}'.format(StartTime))
    print('model name: {}'.format(model_name))
    print('#' * 50, '\n')

    volume_train_data, volume_valid_data, volume_test_data = get_data(model_name)

    model_train(volume_train_data, volume_valid_data)
    model_pred(volume_test_data)
