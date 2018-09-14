# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: train a multi_LSTM model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os


def build_lstm_models(lstm_config):
    model = Sequential()
    model.add(LSTM(input_shape=(lstm_config['input_shape'][0], lstm_config['input_shape'][1]),
                   output_dim=lstm_config['layers'][0],
                   activation=lstm_config['activation'], recurrent_activation=lstm_config['recurrent_activation'],
                   return_sequences=True))
    model.add(Dropout(lstm_config['dropout']))
    for i in range(1, len(lstm_config['layers'])):
        model.add(LSTM(output_dim=lstm_config['layers'][i],
                       activation=lstm_config['activation'], recurrent_activation=lstm_config['recurrent_activation'],
                       return_sequences=True))
        model.add(Dropout(lstm_config['dropout']))
    return model


def add_multi_dense(model, dense_config):
    for i in range(len(dense_config['layers'])):
        model.add(Dense(dense_config['layers'][i], activation=dense_config['activation']))
        # model.add(BatchNormalization())
        model.add(Dropout(dense_config['dropout']))
    model.add(Dense(1))
    return model


def build_model(model_path, weight_path, lstm_config, dense_config, time_steps=1):
    # 存在模型则直接加载
    if os.path.exists(model_path):
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        # model = load_model('../Models/LSTM2/lstm_model.h5')
    else:
        # 创建多个多层的lstm单元,PM2.5、Press、RH等六个
        lstm_models = []
        for i in range(lstm_config['num']):
            lstm_models.append(build_lstm_models(lstm_config))

        # 为时间特征单独创建一个全连接层
        date_model = Sequential()
        date_model.add(
            Dense(input_shape=(dense_config['date_features_shape'][0], dense_config['date_features_shape'][1]),
                  units=dense_config['date_features_shape'][1], activation=dense_config['activation']))
        lstm_models.append(date_model)

        # 合并六个LSTM和一个时间特征的全链接层
        model = Sequential()
        model.add(Merge(lstm_models, mode='concat'))
        model = add_multi_dense(model, dense_config)
        # save model structure
        open(model_path, 'w').write(model.to_json())
    model.summary()
    model.compile(loss='mse', optimizer='RMSprop')
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    return model


def train(df_raw, model_path, weight_path, lstm_config, dense_config, epochs=100, batch_size=100, time_steps=1,
          test_split=0.3):
    # pop the date features
    df_date = df_raw.pop('Month')
    df_date = pd.concat([df_date, df_raw.pop('Day')], axis=1)
    df_date = pd.concat([df_date, df_raw.pop('Hour')], axis=1)
    df_date = df_date.loc[time_steps:]

    # processing the sequence features
    df_raw = data.process_sequence_features(df_raw, time_steps=time_steps)
    # encoding the date features
    df_date_encode = data.encoding_features(df_date, ['Month', 'Hour', 'Day'])

    # normalization
    y_scaled, y_scaler = data.min_max_scale(np.array(df_raw.pop('PM25')).reshape(-1, 1))
    X_scaled, X_scaler = data.min_max_scale(df_raw)
    date_encode = np.array(df_date_encode)

    # reshape y
    train_y = y_scaled[:int(len(X_scaled) * (1 - test_split))]
    test_y = y_scaled[int(len(X_scaled) * (1 - test_split)):]
    train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
    # reshape X
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    date_encode = date_encode.reshape((date_encode.shape[0], 1, date_encode.shape[1]))
    train_X = []
    test_X = []
    # 分割，将PM2.5,Press等时间序列特征分别作为每个lstm模型的输入
    for i in range(lstm_config['num']):
        train_X.append(X_scaled[:int(len(X_scaled) * (1 - test_split)), :, i * time_steps: (i + 1) * time_steps])
        test_X.append(X_scaled[int(len(X_scaled) * (1 - test_split)):, :, i * time_steps: (i + 1) * time_steps])
    # 日期时间特征
    train_X.append(date_encode[:int(len(X_scaled) * (1 - test_split)), :, :])
    test_X.append(date_encode[int(len(X_scaled) * (1 - test_split)):, :, :])

    # build model
    model = build_model(model_path, weight_path, lstm_config, dense_config, time_steps)

    # checkpoint
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y),
                        verbose=1, callbacks=callbacks_list, shuffle=False)

    # draw the loss curve
    plt.figure(0)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')

    # draw to compare the original data and the predicted data, and print the evaluation metrics
    pred_y = model.predict(test_X)
    test_y = data.inverse_to_original_data(train_y.reshape(1, -1), test_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=int(len(X_scaled) * (1 - test_split)))
    pred_y = data.inverse_to_original_data(train_y.reshape(1, -1), pred_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=int(len(X_scaled) * (1 - test_split)))
    return test_y, pred_y


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # 读取数据
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    metrics = []
    time_steps = [1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 24]
    for time_step in time_steps:
        df_raw_data = pd.read_csv('../DataSet/Processed/Train/261630033_2016_2017_v1.csv', usecols=cols, dtype=str)
        # 参数设置
        epoch = 200
        batch = 1024
        # time_step = 3
        model_path = '../Models/LSTM2/lstm_model_ts' + str(time_step) + '.json'
        weight_path = '../Models/LSTM2/weights_ts' + str(time_step) + '.best.hdf5'
        test_split = 0.4
        lstm_num = 6
        lstm_layers = [50, 100, 100, 50]
        lstm_activation = 'tanh'
        lstm_recurrent_activation = 'hard_sigmoid'
        lstm_input_shape = (1, time_step)
        lstm_dropout = 0.3
        dense_layers = [1024, 1024]
        dense_activation = None
        date_features_shape = (1, 12 + 31 + 24)
        dense_dropout = 0.5

        # 训练
        lstm_conf = {
            'num': lstm_num,
            'input_shape': lstm_input_shape,
            'layers': lstm_layers,
            'activation': lstm_activation,
            'recurrent_activation': lstm_recurrent_activation,
            'dropout': lstm_dropout
        }
        dense_conf = {
            # 时间特征：12个月+31天+24小时
            'date_features_shape': date_features_shape,
            'layers': dense_layers,
            'activation': dense_activation,
            'dropout': dense_dropout
        }
        y_true, y_pred = train(df_raw_data, model_path, weight_path, epochs=epoch, batch_size=batch, lstm_config=lstm_conf,
                               dense_config=dense_conf, time_steps=time_step, test_split=test_split)
        metrics.append(evaluate.print_metrics(y_true, y_pred))
        # evaluate.print_curve(y_true, y_pred, 2)
