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
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os

import tensorflow as tf
from keras import backend as K
import gc


def check_param(origin_params, need_params):
    for np in need_params:
        if not origin_params:
            exit('缺少参数' + np)


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


def load_model_and_weights(model_path, weight_path):
    """
    加载已有模型，有参数则加载相应参数
    :param model_path: 模型路径
    :param weight_path: 参数路径
    :return:
    """
    if os.path.exists(model_path):
        print('load model...')
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        # 有参数则加载
        if os.path.exists(weight_path):
            print('load weights...')
            model.load_weights(weight_path)
        return model
    else:
        exit('找不到模型' + model_path)


def build_model(model_path, weight_path, lstm_config, dense_config):
    # 存在模型则直接加载
    if os.path.exists(model_path):
        print('load model...')
        model = load_model_and_weights(model_path, weight_path)
    else:
        print('build new model...')
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
        open(model_path, 'w').write(model.to_json())

    model.summary()
    model.compile(loss='mse', optimizer='RMSprop')
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
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
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


def predict(df_raw, param_conf, save_result=False, show_fitting_curve=False):
    """
    用某个模型进行预测
    :param df_raw: 数据
    :param param_conf: 配置参数
    :param save_result: 是否保存预测结果
    :param show_fitting_curve: 是否显示拟合曲线
    :return:
    """
    # process data and split data
    train_num = int(len(df_raw) * (1 - param_conf['test_split']))
    X_train, X_test, y_train, y_test, y_scaler = process_data(df_raw, param_conf['time_steps'], train_num)

    # load model and weights
    model = load_model_and_weights(param_conf['model_path'], param_conf['weight_path'])

    # predict
    y_pred = model.predict(X_test)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

    # 保存评估结果
    if save_result:
        df_all_metrics = evaluate.all_metrics(y_true, y_pred)
        df_all_metrics.to_csv(param_conf['model_path'] + '.csv', index=False)

    # 显示拟合曲线
    if show_fitting_curve:
        evaluate.draw_fitting_curve(y_true, y_pred)

    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    return y_true, y_pred


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


def different_lstm_layers_and_nodes(param_conf, lstm_layers, lstm_layer_nodes):
    """
    对比不同lstm层数和每层不同神经元数网络的性能
    :param param_conf:
    :param lstm_layers:
    :param lstm_layer_nodes:
    :return:
    """
    for l in lstm_layers:
        for n in lstm_layer_nodes:
            df_raw_data = pd.read_csv(param_conf['data_path'], usecols=param_conf['cols'], dtype=str)
            model_path = param_conf['model_root_path'] + '/lstm_model_layers' + str(l) + '_nodes' + str(n) + '.json'
            weight_path = param_conf['model_root_path'] + '/weights_layer' + str(l) + '_nodes' + str(n) + '.best.hdf5'


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # ##############################################初始化参数设置######################################################
    # 模型存储的根目录
    model_root_path = '../Models/LSTM1'
    # 数据路径
    data_path = '../DataSet/Processed/Train/261630033_2016_2017_v1.csv'
    # 加载哪些数据
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    # 训练轮数
    epoch = 1000
    # batch size
    batch = 512
    # 测试样本比例
    test_split = 0.4
    # lstm块里lstm层的层数候选值
    lstm_layers = [2]
    # lstm块里lstm层的神经元个数的候选值，如lstm_layers=2，lstm_layer_nodes=300，表示两层lstm层，每层神经元个数均为300
    lstm_layer_nodes = [300]
    # lstm层的激活函数，默认为tanh，这里改为softsign
    lstm_activation = 'softsign'
    lstm_recurrent_activation = 'hard_sigmoid'
    # lstm层之间的doupout比例
    lstm_dropout = 0.3
    # 全连接层个数及对应每层神经元个数
    dense_layers = [1024, 1024]
    # 时间特征层的神经元个数，12个月+31天+24小时
    date_features_shape = (1, 12 + 31 + 24)
    # 全连接层之间的doupout比例
    dense_dropout = 0.5
    # 各个特征的时间不，由各特征的自相关系数算出，系相关系数>0.4
    time_steps = {
        'PM25': 5,
        'Press': 24,
        'RH': 8,
        'Temp': 24,
        'Wind Speed': 11,
        'Wind Direction': 11
    }
    # 全局配置
    config = {
        'model_root_path': model_root_path,
        'data_path': data_path,
        'data_cols': cols,
        'epoch': epoch,
        'batch': batch,
        'test_split': test_split,
        'time_steps': time_steps,
        'lstm_conf': {
            'max_features': max(time_steps.values()) * len(time_steps),
            'input_shape': (1, 24 * len(time_steps)),
            'layers': [lstm_layer_nodes[0]] * lstm_layers[0],
            'activation': lstm_activation,
            'recurrent_activation': lstm_recurrent_activation,
            'dropout': lstm_dropout,
        },
        'dense_conf': {
            'date_features_shape': date_features_shape,
            'layers': dense_layers,
            'dropout': dense_dropout
        }
    }
    # ################################################################################################################

    # 训练
    # lstm_layers = [5]
    # lstm_layer_nodes = [300, 600, 900, 1200]
    # main(data_path, cols, time_steps, 0.4, model_root_path, lstm_layers, lstm_layer_nodes, is_train=True)

    # 预测
    # lstm_layers = [2, 3, 4, 5]
    # lstm_layer_nodes = [300, 600, 900, 1200]
    # main(data_path, cols, time_steps, 1, model_root_path, lstm_layers, lstm_layer_nodes, is_train=False)
    # ################################################################################################################

    # ################################################################################################################
    # 画最优模型的拟合曲线
    # best_model_path = model_root_path + '/lstm_model_layers3_nodes600.json'
    # best_weight_path = model_root_path + '/weights_layer3_nodes600.best.hdf5'
    # draw_one_model_fitting_curve(data_path, cols, time_steps, 0.4, best_model_path, best_weight_path)
    # ###############################################################################################################
    # a = ''
    # if not a:
    #     print(1)