'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2023-11-25 15:13:00
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-09 16:41:17
FilePath: PNC_pred\test_pnc_model\model_test.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import numpy as np
import random
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from keras.models import Sequential
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from keras import layers
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from keras.regularizers import l2
# from keras.preprocessing.sequence import pad_sequences

    
def val_in(x_train, x_test, y_train, y_test, val_size):
    # we resize some data from train as val set, the val size 0.1-0.2
    len_train = int(x_train.shape[0] * (1 - val_size))
    x_train_new = x_train[:len_train, :]
    x_val = x_train[len_train:, :]
    y_train_new = y_train[:len_train]
    y_val = y_train[len_train:]
    x_test = x_test
    y_test = y_test

    x_train_new = np.reshape(x_train_new, (x_train_new.shape[0], 1, 13))
    x_test = np.reshape(x_test,  (x_test.shape[0], 1, 13))
    x_val = np.reshape(x_val,  (x_val.shape[0], 1, 13))
    
    return x_train_new, y_train_new, x_val, y_val, x_test, y_test

def val_in_trans(x_train, x_test, y_train, y_test, val_size):
    # we resize some data from train as val set, the val size 0.1-0.2
    len_test = int(x_test.shape[0] * val_size)
    x_train_new = x_train
    x_val = x_test[:len_test, :]
    y_train_new = y_train
    y_val = y_test[:len_test]
    x_test_new = x_test[len_test:, :]
    y_test_new = y_test[len_test:]

    x_train_new = np.reshape(x_train_new, (x_train_new.shape[0], 1, 13))
    x_test_new = np.reshape(x_test_new,  (x_test_new.shape[0], 1, 13))
    x_val = np.reshape(x_val,  (x_val.shape[0], 1, 13))
    
    return x_train_new, y_train_new, x_val, y_val, x_test_new, y_test_new
        
def val_out(x_train, x_test, y_train, y_test):

    x_train = np.reshape(x_train, (x_train.shape[0], 1, 13))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 13))
    y_train = y_train
    y_test = y_test

    return  x_train, y_train, x_test, y_test
    
def build_rnn_model(input_shape=(None, None)):
    rnn_model = Sequential()
    rnn_model.add(layers.Input(shape=input_shape))
    rnn_model.add(layers.SimpleRNN(units=256, return_sequences=True, activation=None))
    rnn_model.add(Activation('relu'))
    rnn_model.add(Dropout(0.1))
    rnn_model.add(layers.SimpleRNN(units=512, activation=None))
    rnn_model.add(Activation('relu'))
    rnn_model.add(Dropout(0.1))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return rnn_model

def build_advanced_rnn_model(input_shape=(None, None)):
    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(512, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(512, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model


def build_cnn_model(input_shape=(None, None)):
    cnn_model = Sequential()
    cnn_model.add(layers.Input(shape=input_shape))

    cnn_model.add(layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(layers.Flatten())
    cnn_model.add(Dense(1))
    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return cnn_model


def build_optimized_cnn_model(input_shape=(1, 13)):
    cnn_model = Sequential()
    cnn_model.add(layers.Input(shape=input_shape))
    cnn_model.add(layers.Conv1D(filters=256, kernel_size=3, padding='same'))
    cnn_model.add(layers.LeakyReLU(alpha=0.01))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Conv1D(filters=512, kernel_size=3, padding='same'))
    cnn_model.add(layers.LeakyReLU(alpha=0.01))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Dropout(0.2))
    cnn_model.add(layers.Conv1D(filters=256, kernel_size=3, padding='same'))
    cnn_model.add(layers.LeakyReLU(alpha=0.01))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Dropout(0.2))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(128, activation='relu'))
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(1))
    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return cnn_model



def build_lstm_model(input_shape=(None, None)):
    lstm_model = Sequential()
    lstm_model.add(layers.Input(shape=input_shape))
    # LSTM Layer 1 with 256 units
    lstm_model.add(layers.LSTM(units=256, return_sequences=True, activation='tanh'))
    lstm_model.add(Dropout(0.1))
    # LSTM Layer 2 with 512 units
    lstm_model.add(layers.LSTM(units=512, activation='tanh'))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return lstm_model



# mask
def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    if len(mask.shape) == 3:
        mask = tf.reduce_sum(mask, axis=-1)
    return mask[:, tf.newaxis, tf.newaxis, :]

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# define MultiHeadAttention layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights


def build_transformer_model(input_shape=(None, None)):
    d_model = 256
    num_heads = 8
    ffn_units = 1024  #  Add more neurons to the feed forward network
    dropout_rate = 0.2
    num_transformer_layers = 1

    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)

    pos_enc = positional_encoding(input_shape[0], d_model)
    x = layers.Add()([x, pos_enc[:, :input_shape[0], :]])

    # make Multi-Head Attention and Feed-Forward Network stacked multiple times
    for _ in range(num_transformer_layers):
        q = x
        k = x
        v = x

        mask = create_padding_mask(inputs)
        multi_head_attention = MultiHeadAttention(d_model, num_heads)
        attn_output = multi_head_attention(v, k, q, mask=mask)
        attn_output = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn_output]))
        x = layers.Dropout(dropout_rate)(attn_output)

        # more complex feed-forward network
        ffn = Sequential([
            layers.Dense(ffn_units, activation='relu'),
            layers.Dense(ffn_units // 2, activation='relu'),
            layers.Dense(d_model)
        ])
        ffn_output = ffn(x)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ffn_output]))
        x = layers.Dropout(dropout_rate)(x)
        
    outputs = layers.Dense(1)(x)

    transformer_model = keras.Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    return transformer_model