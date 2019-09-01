import numpy as np
from Param_DSTN_flow import *


def load_data(len_closeness, len_period, len_trend, T_closeness, T_period, T_trend):
    all_data = np.load(dataFile)
    all_data = all_data.transpose((0, 3, 1, 2))
    len_total, feature, map_height, map_width = all_data.shape
    len_test = all_data.shape[0] - int(all_data.shape[0] * trainRatio)
    print('all_data shape: ', all_data.shape)

    all_data /= MAX_FLOWIO
    number_of_skip_hours = T_trend * len_trend
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:]

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    X_closeness_train = X_closeness[:-len_test]
    X_period_train = X_period[:-len_test]
    X_trend_train = X_trend[:-len_test]

    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_test = [X_closeness_test, X_period_test, X_trend_test]

    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

    Y_train = Y[:-len_test]
    Y_test = Y[-len_test:]

    len_train = X_closeness_train.shape[0]
    len_test = X_closeness_test.shape[0]
    print('len_train=' + str(len_train))
    print('len_test =' + str(len_test))

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    load_data(len_closeness, len_period, len_trend,
              T_closeness, T_period, T_trend)
