# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: train a single LSTM model

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
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os

import tensorflow as tf
from keras import backend as K
import gc


def build_lstm_models(lstm_config):
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(1, lstm_config['max_features'])))
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
        model.add(Dense(dense_config['layers'][i]))
        model.add(Dropout(dense_config['dropout']))
    model.add(Dense(1))
    return model


def build_model(model_path, weight_path, lstm_config, dense_config, time_steps=1):
    # 存在模型则直接加载
    if os.path.exists(model_path['load_model_path']) or os.path.exists(model_path['save_model_path']):
        print('load model...')
        # 优先加载已训练的迁移模型
        if os.path.exists(model_path['save_model_path']):
            json_string = open(model_path['save_model_path']).read()
        else:
            json_string = open(model_path['load_model_path']).read()
        model = model_from_json(json_string)
        # # 迁移学习设置
        if len(lstm_config['frozen_layer_names']) > 0:
            for ln in range(lstm_config['frozen_layer_names']):
                model.get_layer(ln).trainable = False
        if lstm_config['add_layer_num'] > 0:
            for ln in range(lstm_config['frozen_layer_names']):
                model.get_layer(ln).trainable = False
            for j in range(lstm_config['add_layer_num']):
                model.add(LSTM(output_dim=lstm_config['layers'][len(lstm_config['layers'])],
                               activation=lstm_config['activation'],
                               recurrent_activation=lstm_config['recurrent_activation'],
                               return_sequences=True))
                model.add(Dropout(lstm_config['dropout']))
    else:
        print('build model...')
        # 创建一个多层的lstm单元, PM2.5、Press、RH等六个时间序列特征作为输入
        lstm_model = build_lstm_models(lstm_config)
        # 为时间特征单独创建一个全连接层
        date_model = Sequential()
        date_model.add(
            Dense(input_shape=(dense_config['date_features_shape'][0], dense_config['date_features_shape'][1]),
                  units=dense_config['date_features_shape'][1], activation='sigmoid'))

        # 合并一个LSTM和一个时间特征的全链接层
        model = Sequential()
        model.add(Merge([lstm_model, date_model], mode='concat'))
        model = add_multi_dense(model, dense_config)
        # save model structure
        open(model_path['save_model_path'], 'w').write(model.to_json())
    if os.path.exists(weight_path['load_weights_path']):
        print('load weights...')
        model.load_weights(weight_path['load_weights_path'])

    model.summary()
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def process_data(df_raw, time_steps, train_num):
    max_time_step = max(time_steps.values())
    # pop the date features
    df_date = df_raw.pop('Month')
    df_date = pd.concat([df_date, df_raw.pop('Day')], axis=1)
    df_date = pd.concat([df_date, df_raw.pop('Hour')], axis=1)
    df_date = df_date.loc[max_time_step:]

    # processing the sequence features
    df_raw = data.process_sequence_features(df_raw, time_steps=time_steps)
    df_raw = df_raw.loc[max_time_step:]
    # encoding the date features
    df_date_encode = data.encoding_features(df_date, ['Month', 'Hour', 'Day'])

    # normalization
    y_scaled, y_scaler = data.min_max_scale(np.array(df_raw.pop('PM25')).reshape(-1, 1))
    X_scaled, X_scaler = data.min_max_scale(df_raw)
    date_encode = np.array(df_date_encode)

    # reshape y
    train_y = y_scaled[:train_num]
    test_y = y_scaled[train_num:]
    train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
    # reshape X
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    date_encode = date_encode.reshape((date_encode.shape[0], 1, date_encode.shape[1]))
    train_X = []
    test_X = []
    # 分割，将PM2.5,Press等时间序列特征作为一个lstm模型的输入
    train_X.append(X_scaled[:train_num, :, :])
    test_X.append(X_scaled[train_num:, :, :])
    # 日期时间特征
    train_X.append(date_encode[:train_num, :, :])
    test_X.append(date_encode[train_num:, :, :])

    return train_X, test_X, train_y, test_y, y_scaler


def train(df_raw, model_path, weight_path, lstm_config, dense_config, time_steps, epochs=100, batch_size=100,
          test_split=0.3):
    train_num = int(len(df_raw) * (1 - test_split))
    train_X, test_X, train_y, test_y, y_scaler = process_data(df_raw, time_steps, train_num)

    # build model
    model = build_model(model_path, weight_path, lstm_config, dense_config, time_steps)

    # checkpoint
    checkpoint = ModelCheckpoint(weight_path['save_weights_path'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y),
                        verbose=1, callbacks=callbacks_list, shuffle=False)

    # draw the loss curve
    # evaluate.draw_loss_curve(figure_num=0, train_loss=history.history['loss'], val_loss=history.history['val_loss'])

    # draw to compare the original data and the predicted data, and print the evaluation metrics
    pred_y = model.predict(test_X)
    test_y = data.inverse_to_original_data(train_y.reshape(1, -1), test_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    pred_y = data.inverse_to_original_data(train_y.reshape(1, -1), pred_y.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    return test_y, pred_y


def predict(test_split, lstm_layers, lstm_layer_nodes, frozen_layer_names, add_layer_num, mode, draw_fitting_curve=False):
    model_root_path = '../Models/LSTM1/TransferLearning'
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    data_path = '../DataSet/Processed/Train/060376012_2016_2017_v1.csv'
    df_raw = pd.read_csv(data_path, usecols=cols, dtype=str)
    time_steps = {
        'PM25': 24,
        'Press': 24,
        'RH': 8,
        'Temp': 7,
        'Wind Speed': 2,
        'Wind Direction': 4
    }
    if mode == 0:
        model_path = model_root_path + '/lstm_model_layers' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.transfer_frozen' + str(len(frozen_layer_names)) + '_add' + str(
            add_layer_num) + '.json'
        weight_path = model_root_path + '/weights_layer' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.transfer_frozen' + str(len(frozen_layer_names)) + '_add' + str(
            add_layer_num) + '.best.hdf5'
    else:
        model_path = model_root_path + '/lstm_model_layers' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.direct.json'
        weight_path = model_root_path + '/weights_layer' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.direct.best.hdf5'
    train_num = int(len(df_raw) * (1 - test_split))
    X_train, X_test, y_train, y_test, y_scaler = process_data(df_raw, time_steps, train_num)

    # load model and weights
    json_string = open(model_path).read()
    model = model_from_json(json_string)
    model.load_weights(weight_path)
    model.summary()

    # predict
    y_pred = model.predict(X_test)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

    if draw_fitting_curve:
        evaluate.draw_fitting_curve(y_true, y_pred)

    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    return evaluate.all_metrics(y_true, y_pred)


def transfer_learning(use_model_path, use_weights_path, test_split, lstm_layers, lstm_layer_nodes, frozen_layer_names, add_layer_num,
                      mode=0):
    """
    :param use_model_path: 
    :param use_weights_path: 
    :param test_split: 
    :param lstm_layers: 
    :param lstm_layer_nodes: 
    :param model_root_path: 
    :param frozen_layer_names: an array
    :param add_layer_num: 
    :param mode: mode=0使用已有模型迁移学习，mode=1不使用现有模型直接训练
    :return: 
    """""
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    data_path = '../DataSet/Processed/Train/060376012_2016_2017_v1.csv'
    model_root_path = '../Models/LSTM1/TransferLearning'
    transfer_use_model_path = model_root_path + use_model_path
    transfer_use_weights_path = model_root_path + use_weights_path
    model_path = {
        'load_model_path': '',
        'save_model_path': '',

    }
    weight_path = {
        'load_weights_path': '',
        'save_weights_path': ''
    }
    time_steps = {
        'PM25': 24,
        'Press': 24,
        'RH': 8,
        'Temp': 7,
        'Wind Speed': 2,
        'Wind Direction': 4
    }
    if mode == 0:
        model_path['load_model_path'] = transfer_use_model_path
        weight_path['load_weights_path'] = transfer_use_weights_path
        model_path['save_model_path'] = model_root_path + '/lstm_model_layers' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.transfer_frozen' + str(len(frozen_layer_names)) + '_add' + str(
            add_layer_num) + '.json'
        weight_path['save_weights_path'] = model_root_path + '/weights_layer' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.transfer_frozen' + str(len(frozen_layer_names)) + '_add' + str(
            add_layer_num) + '.best.hdf5'
    else:
        model_path['save_model_path'] = model_root_path + '/lstm_model_layers' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.direct.json'
        weight_path['save_weights_path'] = model_root_path + '/weights_layer' + str(lstm_layers) + '_nodes' + str(
            lstm_layer_nodes) + '.direct.best.hdf5'
    df_raw_data = pd.read_csv(data_path, usecols=cols, dtype=str)
    lstm_conf = {
        'max_features': max(time_steps.values()) * len(time_steps),
        'input_shape': (1, 24 * 6),
        'layers': [lstm_layer_nodes] * lstm_layers,
        'activation': 'softsign',
        'recurrent_activation': 'hard_sigmoid',
        'dropout': 0.3,
        'frozen_layer_names': frozen_layer_names,
        'add_layer_num': add_layer_num
    }
    dense_conf = {
        # 时间特征：12个月+31天+24小时
        'date_features_shape': (1, 12 + 31 + 24),
        'layers': [1024, 1024],
        'activation': 'relu',
        'dropout': 0.5
    }
    # 移学习
    y_true, y_pred = train(df_raw_data, model_path, weight_path, epochs=1000, batch_size=512, lstm_config=lstm_conf,
                           dense_config=dense_conf, time_steps=time_steps, test_split=test_split)
    evaluate.all_metrics(y_true, y_pred)


def main(data_path, cols, time_steps, test_split, model_root_path, lstm_layers, lstm_layer_nodes, is_train=True):
    # 参数设置
    epoch = 1000
    # batch = 1024
    batch = 512
    test_split = 0.4
    # lstm_layers = [2, 3, 4, 5]
    # lstm_layers = [3]
    # lstm_layer_nodes = [300, 600, 900, 1200]
    lstm_activation = 'softsign'
    lstm_recurrent_activation = 'hard_sigmoid'
    lstm_input_shape = (1, 24 * 6)
    lstm_dropout = 0.3
    dense_layers = [1024, 1024]
    dense_activation = 'relu'
    date_features_shape = (1, 12 + 31 + 24)
    dense_dropout = 0.5

    if is_train:
        # 训练
        print('训练模型...')
        for l in lstm_layers:
            for n in lstm_layer_nodes:
                df_raw_data = pd.read_csv(data_path, usecols=cols, dtype=str)
                model_path = model_root_path + '/lstm_model_layers' + str(l) + '_nodes' + str(n) + '.json'
                weight_path = model_root_path + '/weights_layer' + str(l) + '_nodes' + str(n) + '.best.hdf5'
                lstm_conf = {
                    'max_features': max(time_steps.values()) * len(time_steps),
                    'input_shape': lstm_input_shape,
                    'layers': [n] * l,
                    'activation': lstm_activation,
                    'recurrent_activation': lstm_recurrent_activation,
                    'dropout': lstm_dropout,
                    'is_transfer': False,
                    'frozen_num': 0,
                    'add_layer_num': 0
                }
                dense_conf = {
                    # 时间特征：12个月+31天+24小时
                    'date_features_shape': date_features_shape,
                    'layers': dense_layers,
                    'activation': dense_activation,
                    'dropout': dense_dropout
                }
                y_true, y_pred = train(df_raw_data, model_path, weight_path, epochs=epoch, batch_size=batch,
                                       lstm_config=lstm_conf,
                                       dense_config=dense_conf, time_steps=time_steps, test_split=test_split)
                evaluate.all_metrics(y_true, y_pred)
    else:
        # 预测
        print('预测...')
        test_split = 1
        df_metrics = pd.DataFrame()
        for l in lstm_layers:
            for n in lstm_layer_nodes:
                df_raw_data = pd.read_csv(data_path, usecols=cols, dtype=str)
                model_path = model_root_path + '/lstm_model_layers' + str(l) + '_nodes' + str(n) + '.json'
                weight_path = model_root_path + '/weights_layer' + str(l) + '_nodes' + str(n) + '.best.hdf5'
                y_true, y_pred = predict(df_raw_data, time_steps, test_split, model_path, weight_path)
                df_temp = evaluate.all_metrics(y_true, y_pred)
                df_temp['layers'] = l
                df_temp['nodes'] = n
                df_metrics = df_metrics.append(df_temp)
                print('Write all metrics to file...')
                df_metrics.to_csv(model_root_path + '/all_metrics_values.csv', index=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    use_model_path = '../Models/LSTM1/lstm_model_layers3_nodes600.json'
    use_weights_path = '../Models/LSTM1/weights_layers3_nodes600.best.hdf5'
    # ################################################################################################################
    # 迁移学习和直接学习对比
    # 使用已有模型迁移,冻结第一层lstm
    # transfer_learning(use_model_path, use_weights_path, 0.8, 3, 600, ['lstm_1'], 0, mode=0)
    # 冻结前两层lstm
    # transfer_learning(use_model_path, use_weights_path, 0.8, 3, 600, ['lstm_1', 'lstm_2'], 0, mode=0)
    # 冻结已有的3层lstm，在增加一层新的lstm
    # transfer_learning(use_model_path, use_weights_path, 0.8, 3, 600, ['lstm_1', 'lstm_2', 'lstm_3'], 1, mode=0)
    # 冻结已有的3层lstm，在增加两层新的lstm
    # transfer_learning(use_model_path, use_weights_path, 0.8, 3, 600, ['lstm_1', 'lstm_2', 'lstm_3'], 2, mode=0)
    # 直接学习
    # transfer_learning(use_model_path, use_weights_path, 0.8, 3, 600, [], 0, mode=1)
    # ################################################################################################################

    # ################################################################################################################
    # 预测
    # 使用冻结第一层lstm 的迁移模型进行预测
    # predict(0.8, 3, 600, ['lstm_1'], 0, 0, draw_fitting_curve=True)
    # 使用冻结前两层lstm 的迁移模型进行预测
    # predict(0.8, 3, 600, ['lstm_1', 'lstm_2'], 0, 0, draw_fitting_curve=True)
    # 使用增加一层新的lstm 的迁移模型进行预测
    predict(0.8, 3, 600, ['lstm_1', 'lstm_2', 'lstm_3'], 1, 0, draw_fitting_curve=True)
    # 使用增加2层新的lstm 的迁移模型进行预测
    predict(0.8, 3, 600, ['lstm_1', 'lstm_2', 'lstm_3'], 2, 0, draw_fitting_curve=True)
    # 使用直接学习的模型进行预测
    # predict(0.8, 3, 600, [], 0, 1, draw_fitting_curve=True)

