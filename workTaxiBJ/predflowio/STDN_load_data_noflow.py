import numpy as np
from Param_STDN import *


def data_generator(data_list, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7,
                   hist_feature_daynum=7, last_feature_num=48, nbhd_size=1, batchsize=128, type='train'):
    seed = 0
    np.random.seed(seed)
    while True:
        for n_data in range(len(data_list)):
            data = data_list[n_data]

            cnn_att_features = []
            lstm_att_features = []
            for i in range(att_lstm_num):
                lstm_att_features.append([])
                cnn_att_features.append([])
                for j in range(long_term_lstm_seq_len):
                    cnn_att_features[i].append([])

            cnn_features = []
            for i in range(short_term_lstm_seq_len):
                cnn_features.append([])
            short_term_lstm_features = []
            labels = []

            time_start = empty_time
            time_end = data.shape[0]

            num_region = data.shape[1]
            y_location = int(data.shape[2] // 2)

            time_random = np.arange(time_start, time_end)
            region_random = np.arange(num_region)
            if type != 'test':
                np.random.shuffle(time_random)
                np.random.shuffle(region_random)
            batch_num = 0

            for t in time_random:
                for r in region_random:
                    # sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        ######################### cnn features
                        cnn_feature = data[real_t, r, :, :, :]
                        cnn_features[seqn].append(cnn_feature)

                        ######################### lstm features
                        nbhd_feature = data[real_t, r,
                                       y_location - nbhd_size:y_location + nbhd_size + 1,
                                       y_location - nbhd_size:y_location + nbhd_size + 1, :].flatten()
                        # last feature
                        last_feature = data[real_t - last_feature_num: real_t, r, y_location, y_location, :].flatten()
                        # hist feature
                        hist_feature = data[real_t - hist_feature_daynum * DAYTIMESTEP: real_t: DAYTIMESTEP, r,
                                       y_location, y_location, :].flatten()
                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))
                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    # sample att-lstms
                    for att_lstm_cnt in range(att_lstm_num):

                        # sample lstm at att loc att_lstm_cnt
                        long_term_lstm_samples = []
                        # get time att_t, move forward for (att_lstm_num - att_lstm_cnt) day, then move back for ([long_term_lstm_seq_len / 2] + 1)
                        # notice that att_t-th timeslot will not be sampled in lstm
                        # e.g., **** (att_t - 3) **** (att_t - 2) (yesterday's t) **** (att_t - 1) **** (att_t) (this one will not be sampled)
                        # sample att-lstm with seq_len = 3
                        att_t = t - (att_lstm_num - att_lstm_cnt) * DAYTIMESTEP + \
                                (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        # att-lstm seq len
                        for seqn in range(long_term_lstm_seq_len):
                            # real_t from (att_t - long_term_lstm_seq_len) to (att_t - 1)
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            ######################### att-cnn features
                            cnn_feature = data[real_t, r, :, :, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            ######################### att-lstm features
                            nbhd_feature = data[real_t, r,
                                           y_location - nbhd_size:y_location + nbhd_size + 1,
                                           y_location - nbhd_size:y_location + nbhd_size + 1, :].flatten()
                            # last feature
                            last_feature = data[real_t - last_feature_num: real_t, r, y_location, y_location,
                                           :].flatten()
                            # hist feature
                            hist_feature = data[real_t - hist_feature_daynum * DAYTIMESTEP: real_t: DAYTIMESTEP, r,
                                           y_location, y_location, :].flatten()
                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))
                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    # label
                    labels.append(data[t, r, y_location, y_location, :].flatten())
                    batch_num += 1

                    if batch_num == batchsize:
                        output_cnn_att_features = []
                        for i in range(att_lstm_num):
                            lstm_att_features[i] = np.array(lstm_att_features[i])
                            for j in range(long_term_lstm_seq_len):
                                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                                output_cnn_att_features.append(cnn_att_features[i][j])

                        for i in range(short_term_lstm_seq_len):
                            cnn_features[i] = np.array(cnn_features[i])
                        short_term_lstm_features = np.array(short_term_lstm_features)
                        labels = np.array(labels)

                        if type == 'train':
                            yield output_cnn_att_features + lstm_att_features + cnn_features + [
                                short_term_lstm_features, ], labels
                        elif type == 'test':
                            yield output_cnn_att_features + lstm_att_features + cnn_features + [
                                short_term_lstm_features, ]

                        cnn_att_features = []
                        lstm_att_features = []
                        for i in range(att_lstm_num):
                            lstm_att_features.append([])
                            cnn_att_features.append([])
                            for j in range(long_term_lstm_seq_len):
                                cnn_att_features[i].append([])

                        cnn_features = []
                        for i in range(short_term_lstm_seq_len):
                            cnn_features.append([])
                        short_term_lstm_features = []
                        labels = []
                        batch_num = 0


def get_test_true(data_list):
    labels = []
    for n_data in range(len(data_list)):
        data = data_list[n_data]

        time_start = empty_time
        time_end = data.shape[0]

        num_region = data.shape[1]
        y_location = int(data.shape[2] // 2)

        for t in range(time_start, time_end):
            for r in range(num_region):
                # label
                labels.append(data[t, r, y_location, y_location, :].flatten())
    labels = np.array(labels)
    return labels
