# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: train a LSTM model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import metrics
from MyModule import data
from MyModule import evaluate


def train(df_raw, time_steps=1, train_num=365 * 24):
    # processing the sequence features
    df_raw = data.process_sequence_features(df_raw, time_steps=time_steps)

    # normalization
    y_scaled, y_scaler = data.min_max_scale(np.array(df_raw.pop('PM25')).reshape(-1, 1))
    X_scaled, X_scaler = data.min_max_scale(df_raw)

    # split data to train data and test data
    train_X, train_y, test_X, test_y = data.split_data(X_scaled, y_scaled, train_num=train_num)

    # reshape data
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))

    # build a Sequential model
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(units=1024, activation='linear'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=1024, activation='linear'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(loss='mse', optimizer='RMSprop')
    history = model.fit(train_X, train_y, epochs=100, batch_size=1024, validation_data=(test_X, test_y),
                        verbose=2, shuffle=False)

    # draw the loss curve
    plt.figure(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')

    # draw to compare the original data and the predicted data, and print the evaluation metrics
    pred_y = model.predict(test_X)
    test_y = data.inverse_to_original_data(train_y.reshape(1, -1), test_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    pred_y = data.inverse_to_original_data(train_y.reshape(1, -1), pred_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.print_metrics(test_y, pred_y)
    evaluate.print_curve(test_y, pred_y)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    time_step = 3
    train_num = 365 * 24
    # read
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction']
    df_raw_data = pd.read_csv('../DataSet/Processed/Train/261630033_2016_2017_v1.csv', usecols=cols, dtype=str)

    # 训练
    train(df_raw_data, time_steps=time_step, train_num=train_num)
