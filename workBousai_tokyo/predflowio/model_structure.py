import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Dense, Input, Flatten, concatenate, Reshape
from keras import activations
from keras.models import Model
from Param_DMVST_flow import *

seq_len = TIMESTEP


class Local_Seq_Conv(Layer):

    def __init__(self, output_dim, seq_len, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(Local_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        batch_size = input_shape[0]
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size,
                                            initializer=self.kernel_initializer,
                                            trainable=True, name='kernel_{0}'.format(eachlen))]

            self.bias += [self.add_weight(shape=(self.kernel_size[-1],),
                                          initializer=self.bias_initializer,
                                          trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):

            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen],
                                      strides=self.strides, padding=self.padding), self.bias[eachlen])

            if self.activation is not None:
                output += [self.activation(tmp)]

        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)


def build_model():
    image_input = Input(shape=(seq_len, local_image_size, local_image_size, None), name='cnn_input')
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, 1, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(image_input)
    spatial = BatchNormalization()(spatial)

    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    spatial = BatchNormalization()(spatial)

    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)

    spatial = Flatten()(spatial)
    spatial = Reshape(target_shape=(seq_len, -1))(spatial)
    spatial_out = Dense(units=64, activation='relu')(spatial)

    lstm_input = Input(shape=(seq_len, feature_len), dtype='float32', name='lstm_input')

    x = concatenate([lstm_input, spatial_out], axis=-1)

    lstm_out = LSTM(units=hidden_dim, return_sequences=False, dropout=0)(x)

    topo_input = Input(shape=(toponet_len,), dtype='float32', name='topo_input')
    topo_emb = Dense(units=6, activation='tanh')(topo_input)
    static_dynamic_concate = concatenate([lstm_out, topo_emb], axis=-1)

    res = Dense(units=1, activation='sigmoid')(static_dynamic_concate)
    model = Model(inputs=[image_input, lstm_input, topo_input], outputs=res)
    return model


def get_model(name):
    if name == 'density':
        return build_model()
