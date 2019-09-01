import numpy as np
import pandas as pd
import math
import os
import datetime as dt
from Param_DMVST_flow import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def build_image(data, size):
    num_time = data.shape[0]
    num_x = data.shape[1]
    num_y = data.shape[2]

    padding = size // 2
    data = np.pad(data, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

    region_image = np.zeros(shape=(num_time, num_x * num_y, size, size, data.shape[-1]))

    for t in range(num_time):
        if t % 500 == 0:
            print('processing %.2f%% data' % (t / num_time * 100))
        for x in range(num_x):
            for y in range(num_y):
                region = data[t, x:x + size, y:y + size]
                region_image[t, x * num_y + y] = region
    print('local image build finish')
    return region_image


def dtw(vec1, vec2):
    d = np.zeros([len(vec1) + 1, len(vec2) + 1])
    d[:] = np.inf
    d[0, 0] = 0
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            cost = abs(vec1[i - 1] - vec2[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[-1][-1]


def build_graph(data, grid_size, type):
    num_time = data.shape[0]
    num_x = data.shape[1]
    num_y = data.shape[2]

    region_merge = np.zeros(shape=(num_time, int(num_x / grid_size), int(num_y / grid_size)))

    for t in range(num_time):
        for x in range(int(num_x / grid_size)):
            for y in range(int(num_y / grid_size)):
                region = data[t, x * grid_size:(x + 1) * grid_size, y * grid_size:(y + 1) * grid_size]
                region_merge[t, x, y] = np.sum(region)

    week_num = 7 * DAYTIMESTEP
    all_fea, fea = [], []
    for x in range(int(num_x / grid_size)):
        for y in range(int(num_y / grid_size)):
            for i in range(int(num_time / week_num)):
                weekly_average = np.mean(region_merge[i * week_num:(i + 1) * week_num, x, y])
                fea.append(weekly_average)
            all_fea.append(fea)
            fea = []
    all_fea = np.array(all_fea)

    a = 0.01
    graph = np.zeros(shape=(all_fea.shape[0], all_fea.shape[0]))
    for i in range(all_fea.shape[0]):
        for j in range(all_fea.shape[0]):
            w = math.exp(-a * dtw(all_fea[i], all_fea[j]))
            graph[i][j] = w

    if type == 'in':
        graph_path = graph_in_path
    elif type == 'out':
        graph_path = graph_out_path
    with open(graph_path, 'w') as f:
        for i in range(all_fea.shape[0]):
            for j in range(all_fea.shape[0]):
                f.writelines([str(i), ' ', str(j), ' ', str(graph[i][j]), '\n'])
    print('graph build finish')


def build_temporal():
    date_info = np.load(date_info_path)
    for i in range(len(date_info)):
        datetime = [str(t[:8], encoding='utf-8') for t in date_info[i]]
        time = [str(t[8:], encoding='utf-8') for t in date_info[i]]

        date_information = pd.DataFrame({'datetime': datetime, 'time': time})
        date_information['datetime'] = pd.to_datetime(date_information['datetime'])
        date_information['day'] = date_information.datetime.dt.dayofweek
        date_information['dayflag'] = 1
        date_information.loc[(date_information.day == 5) | (date_information.day == 6), 'dayflag'] = 0
        date_information.set_index('datetime', inplace=True)

        # date_information.to_csv(dataPath + 'day_information.csv', index=False)

        date_information = date_information.astype('str')
        date_information = pd.get_dummies(date_information)
        date_information.to_csv(temporal_lst_path[i], index=False)
    print('temporal build finish')


##############################################################
if __name__ == '__main__':
    # build 9*9 image
    mkdir(save_path)
    all_data_len = 0
    for idx, flow_path in enumerate(flow_path_lst):
        print('load data from: {}'.format(flow_path))
        data = np.load(flow_path)
        all_data_len += len(data)
        for type in ['in', 'out']:
            if type == 'in':
                flow_data = data[:, :, :, 0:1]
                local_flow_path = local_flow_in_lst_path[idx]
            elif type == 'out':
                flow_data = data[:, :, :, 1:2]
                local_flow_path = local_flow_out_lst_path[idx]

            region_image = build_image(flow_data, local_image_size)
            np.save(local_flow_path, region_image)

            if idx == len(flow_path_lst) - 1:
                # build topo data(used by line)
                print('graph data size: {}*{}'.format(HEIGHT / grid_size, WIDTH / grid_size))
                startIndex, endIndex = 0, flow_data.shape[0] - int(all_data_len * (1 - trainRatio))
                trainData = flow_data[startIndex:endIndex, :, :]
                build_graph(trainData, grid_size, type)

    # build temporal information
    print('generate temporal info')
    build_temporal()
