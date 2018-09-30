# Created by Yuexiong Ding
# Date: 2018/9/18
# Description: lstm model
import pandas as pd
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


def get_init_config():
    """
    初始化默认配置信息
    :return:
    """
    # 数据路径
    data_path = '../DataSet/Processed/Train/261630033_2016_2017_v1.csv'
    # 加载哪些数据
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'CO', 'NO2', 'O3', 'SO2', 'PM10', 'Month',
            'Day', 'Hour', 'Date Time']
    # 各个特征的时间不，由各特征的自相关系数算出，系相关系数>0.4
    # time_steps = {
    #     'PM25': 5,
    #     'Press': 24,
    #     'RH': 8,
    #     'Temp': 24,
    #     'Wind Speed': 11,
    #     'Wind Direction': 11,
    #     'CO': 10,
    #     'NO2': 8,
    #     'O3': 7,
    #     'SO2': 4,
    #     'PM10': 4,
    # }
    time_steps = {}
    # 模型存储的根目录
    model_root_path = '../Models/LSTM1'
    # 训练轮数
    epoch = 1000
    # batch size
    batch = 512
    # 测试样本比例
    test_split = 0.4
    # 损失函数
    loss = 'mse'
    # 优化器
    optimizer = 'RMSprop'
    # 是否为lstm加masking层
    lstm_is_masking = False
    # 时间粒度，hour每小时预测，day每天预测，week每周预测，month每月预测
    time_span = 'hour'
    # lstm块的数量
    lstm_block_num = 1
    # lstm输入特征数的最大值, lstm_is_masking=True时才有效
    lstm_max_feature_length = 0
    # lstm块的输入维度
    lstm_input_shape = (1, lstm_max_feature_length)
    # # lstm块里lstm层的层数候选值
    lstm_layers = [3]
    # # lstm块里lstm层的神经元个数的候选值，如lstm_layers=2，lstm_layer_nodes=300，表示两层lstm层，每层神经元个数均为300
    lstm_layer_nodes = [300]
    # lstm层的激活函数，默认为tanh，这里改为softsign
    lstm_activation = 'softsign'
    lstm_recurrent_activation = 'hard_sigmoid'
    # lstm层之间的doupout比例
    lstm_dropout = 0.3
    lstm_recurrent_dropout = 0.0
    # 全连接层个数及对应每层神经元个数
    dense_layers = [1024]
    # 时间特征层的神经元个数，12个月+31天+24小时
    date_features_shape = (1, 12 + 31 + 24)
    # 全连接层之间的doupout比例
    dense_dropout = 0.5

    # 全局配置
    config = {
        'data_conf': {
            'data_path': data_path,
            'use_cols': cols,
        },
        'model_conf': {
            'main_factor': 'PM25',
            'other_factors': ['PM25', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction'],
            'max_time_lag': 72,
            'min_corr_coeff': 0.2,
            'model_root_path': model_root_path,
            'is_transfer': False,
            'epochs': epoch,
            'batch_size': batch,
            'test_split': test_split,
            'time_span': time_span,
            'target_offset': 0,
            'time_steps': time_steps,
            'is_drop_outlier': False,
            'loss': loss,
            'optimizer': optimizer,
            'lstm_conf': {
                'is_masking': lstm_is_masking,
                'padding_value': -1,
                'lstm_block_num': lstm_block_num,
                'max_feature_length': lstm_max_feature_length,
                'input_shape': lstm_input_shape,
                'layers': [lstm_layer_nodes[0]] * lstm_layers[0],
                'activation': lstm_activation,
                'recurrent_activation': lstm_recurrent_activation,
                'dropout': lstm_dropout,
                'recurrent_dropout': lstm_recurrent_dropout
            },
            'dense_conf': {
                'date_features_shape': date_features_shape,
                'layers': dense_layers,
                'dropout': dense_dropout
            },
            'transfer_conf': {
                'frozen_layer_names': [],
                'add_layer_num': 0,
                'add_layer_nodes': []
            }
        }
    }
    return config


def check_param(origin_params, need_params):
    """
    检查参数是否齐全
    :param origin_params: 已有参数
    :param need_params: 必须的参数集
    :return:
    """
    for np in need_params:
        if not origin_params:
            exit('缺少参数' + np)


def load_model_and_weights(model_path, weight_path):
    """
    加载已有模型，有参数则加载相应参数
    :param model_path: 模型路径
    :param weight_path: 参数路径
    :return:
    """
    if os.path.exists(model_path):
        print('load model ' + model_path)
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        # 有参数则加载
        if os.path.exists(weight_path):
            print('load weights ' + weight_path)
            model.load_weights(weight_path)
        return model
    else:
        exit('找不到模型' + model_path)


def build_lstm_models(lstm_config):
    """
    创建一个lstm模型
    :param lstm_config: 配置
    :return: 模型
    """
    model = Sequential()
    if lstm_config['is_masking']:
        model.add(Masking(mask_value=lstm_config['padding_value'], input_shape=(1, lstm_config['max_feature_length'])))
    model.add(LSTM(input_shape=lstm_config['input_shape'], output_dim=lstm_config['layers'][0],
                   activation=lstm_config['activation'], recurrent_activation=lstm_config['recurrent_activation'],
                   dropout=lstm_config['dropout'],
                   recurrent_dropout=lstm_config['recurrent_dropout'],
                   return_sequences=True))
    # model.add(Dropout(lstm_config['dropout']))
    for i in range(1, len(lstm_config['layers'])):
        model.add(LSTM(output_dim=lstm_config['layers'][i],
                       activation=lstm_config['activation'], recurrent_activation=lstm_config['recurrent_activation'],
                       dropout=lstm_config['dropout'],
                       recurrent_dropout=lstm_config['recurrent_dropout'],
                       return_sequences=True))
        # model.add(Dropout(lstm_config['dropout']))
    return model


def build_dense(model, dense_config):
    """
    为模型添加全连接层
    :param model: 模型
    :param dense_config: 配置
    :return: 模型
    """
    for i in range(len(dense_config['layers'])):
        model.add(Dense(dense_config['layers'][i]))
        model.add(Dropout(dense_config['dropout']))
    model.add(Dense(1))
    return model


def transfer_frozen_layer(model, layer_names):
    """
    冻结所给层
    :param model: 模型
    :param layer_names: 要冻结的层的名字
    :return: 模型
    """
    if len(layer_names) > 0:
        for ln in layer_names:
            model.get_layer(ln).trainable = False
    return model


def transfer_add_lstm_layer(model, model_conf):
    """
    添加新的lstm层
    :param model:
    :param model_conf: 模型配置
    :return:
    """
    add_num = model_conf['transfer_conf']['add_layer_num']
    if add_num > 0:
        for j in range(add_num):
            model.add(LSTM(output_dim=model_conf['transfer_conf']['add_layer_nodes'][j],
                           activation=model_conf['lstm_conf']['activation'],
                           recurrent_activation=model_conf['lstm_conf']['recurrent_activation'],
                           dropout=model_conf['lstm_conf']['dropout'],
                           recurrent_dropout=model_conf['lstm_conf']['recurrent_dropout'],
                           return_sequences=True))
            # model.add(Dropout(model_conf['lstm_conf']['dropout']))
    return model


def build_model(model_config):
    """
    建立 model_1, 多元时序特征输入一个lstm块
    :param model_config: 模型配置
    :return: 模型
    """
    # 不是迁移学习
    if not model_config['is_transfer']:
        # 存在模型则直接加载
        if os.path.exists(model_config['model_path']):
            model = load_model_and_weights(model_config['model_path'], model_config['weight_path'])
        else:
            print('build new model...')
            models = []
            # 创建多层的lstm单元, PM2.5、Press、RH等时间序列特征作为输入
            for i in range(model_config['lstm_conf']['lstm_block_num']):
                models.append(build_lstm_models(model_config['lstm_conf']))
            # 为时间特征单独创建一个全连接层
            date_model = Sequential()
            date_model.add(
                Dense(input_shape=model_config['dense_conf']['date_features_shape'],
                      units=model_config['dense_conf']['date_features_shape'][1]))
            models.append(date_model)
            # 合并一个LSTM和一个时间特征的全链接层
            model = Sequential()
            model.add(Merge(models, mode='concat'))
            model = build_dense(model, model_config['dense_conf'])
            # save model structure
            open(model_config['model_path'], 'w').write(model.to_json())
    # 迁移学习
    else:
        # 优先加载已训练的模型
        if os.path.exists(model_config['save_model_path']):
            model = load_model_and_weights(model_config['save_model_path'], model_config['save_weight_path'])
        else:
            model = load_model_and_weights(model_config['model_path'], model_config['weight_path'])
        # 冻结层
        model = transfer_frozen_layer(model, model_config['transfer_conf']['frozen_layer_names'])
        model = transfer_add_lstm_layer(model, model_config)
    model.summary()
    model.compile(loss=model_config['loss'], optimizer=model_config['optimizer'])
    return model


def train(param_conf, draw_loss_curve=False):
    """
    训练
    :param param_conf: 参数配置
    :param draw_loss_curve: 是否画损失曲线
    :return: 训练历史数据
    """
    # get data
    df_raw = data.get_raw_data(param_conf['data_conf']['data_path'], usecol=param_conf['data_conf']['use_cols'])

    # process data and split data
    # train_num = int(len(df_raw) * (1 - param_conf['model_conf']['test_split']))
    X_train, X_test, y_train, y_test, y_scaler = data.process_data(df_raw, param_conf['model_conf'])

    # build model
    model = build_model(param_conf['model_conf'])

    # checkpoint training
    print('建立检查点并训练...')
    if not param_conf['model_conf']['is_transfer']:
        param_conf['model_conf']['save_weight_path'] = param_conf['model_conf']['weight_path']
    checkpoint = ModelCheckpoint(param_conf['model_conf']['save_weight_path'], monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, y_train, epochs=param_conf['model_conf']['epochs'],
                        batch_size=param_conf['model_conf']['batch_size'], validation_data=(X_test, y_test),
                        verbose=1, callbacks=callbacks_list, shuffle=False)

    # draw the loss curve
    if draw_loss_curve:
        print('画损失曲线...')
        evaluate.draw_loss_curve(figure_num=param_conf['model_conf']['model_path'], train_loss=history.history['loss'],
                                 val_loss=history.history['val_loss'])

    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    return history


def predict(param_conf, save_result=False, show_fitting_curve=False):
    """
    用某个模型进行预测
    :param param_conf: 配置参数
    :param save_result: 是否保存预测结果
    :param show_fitting_curve: 是否显示拟合曲线
    :return: a dict result
    """
    print('################### 预测 ######################')
    # get data
    df_raw = data.get_raw_data(param_conf['data_conf']['data_path'], usecol=param_conf['data_conf']['use_cols'])

    # process data and split data
    # train_num = int(len(df_raw) * (1 - param_conf['model_conf']['test_split']))
    X_train, X_test, y_train, y_test, y_scaler = data.process_data(df_raw, param_conf['model_conf'])

    # load model and weights
    model = load_model_and_weights(param_conf['model_conf']['model_path'], param_conf['model_conf']['weight_path'])

    # predict
    y_pred = model.predict(X_test)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

    # 评估
    # df_all_metrics = evaluate.all_metrics(y_true, y_pred)
    df_all_metrics = evaluate.all_metrics(y_true[: len(y_true) - 1], y_pred[1:])

    # 保存评估结果
    if save_result:
        df_all_metrics.to_csv(param_conf['model_conf']['model_path'] + '.all_metrics.csv', index=False)

    # 显示拟合曲线
    if show_fitting_curve:
        # evaluate.draw_fitting_curve(y_true, y_pred)
        evaluate.draw_fitting_curve(y_true[: len(y_true) - 1], y_pred[1:])

    # 结果返回
    result = {
        'all_metrics': df_all_metrics,
        'y_true': y_true,
        'y_pred': y_pred
    }

    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    return result


# def transfer_learning(param_conf):
#     """
#     迁移学习
#     :param param_conf:
#     :return:
#     """
#     # 移学习
#     y_true, y_pred = train(df_raw_data, model_path, weight_path, epochs=1000, batch_size=512, lstm_config=lstm_conf,
#                            dense_config=dense_conf, time_steps=time_steps, test_split=test_split)
#     evaluate.all_metrics(y_true, y_pred)

