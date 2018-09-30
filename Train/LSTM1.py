# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: train a single LSTM model

import pandas as pd
import numpy as np
from MyModule import lstm
from MyModule import data


def find_best_lstm_layers_nodes(is_train=True):
    """
    寻找lstm块的最优层数和每层节点数
    :param is_train:
    :return:
    """
    # layers = [2, 3, 4]
    # nodes = [50, 100, 200, 300, 500]
    layers = [5, 6, 7]
    nodes = [50]
    conf = lstm.get_init_config()
    for l in layers:
        for n in nodes:
            conf['model_conf']['lstm_conf']['layers'] = [n] * l
            conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/FindBestLSTMBlock/model_lstm_block_layers' + str(l) + '_nodes' + str(n) + '.json'
            conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/FindBestLSTMBlock/weights_lstm_block_layers' + str(l) + '_nodes' + str(n) + '.best.hdf5'
            if is_train:
                lstm.train(conf)
            else:
                # conf['model_conf']['test_split'] = 1
                lstm.predict(conf, save_result=True, show_fitting_curve=True)


def find_best_dense_layers_nodes(is_train=True):
    """
    寻找全链接层的最优层数和每层节点数
    :param is_train:
    :return:
    """
    layers = [1, 2]
    nodes = [128, 256, 512]
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [] * 3
    for l in layers:
        for n in nodes:
            conf['model_conf']['dense_conf']['layers'] = [n] * l
            conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/FindBestDense/model_dense_layers' + str(l) + '_nodes' + str(n) + '.json'
            conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/FindBestDense/weights_dense_layers' + str(l) + '_nodes' + str(n) + '.best.hdf5'
            if is_train:
                lstm.train(conf)
            else:
                conf['model_conf']['test_split'] = 1.0
                lstm.predict(conf, save_result=True)


def find_best_epoch_and_batch_size(is_train=True):
    """
    寻找最优的epoch和batch_size
    :param is_train:
    :return:
    """
    epochs = [500, 1000, 1500]
    batch_sizes = [24, 48, 72, 96]
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [] * 3
    conf['model_conf']['dense_conf']['layers'] = []
    for e in epochs:
        for bs in batch_sizes:
            conf['model_conf']['epochs'] = e
            conf['model_conf']['batch_size'] = bs
            conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/FindBestDense/model_epochs' + str(e) + '_batch' + str(bs) + '.json'
            conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/FindBestDense/weights_epochs' + str(e) + '_batch' + str(bs) + '.best.hdf5'
            if is_train:
                lstm.train(conf)
            else:
                conf['model_conf']['test_split'] = 1.0
                lstm.predict(conf, save_result=True)


def effect_of_other_series_feature(is_train=True):
    """
    其他时序特征对预测结果的影响
    :param is_train:
    :return:
    """
    features = {
        'pm25': {
            'cols': ['PM25', 'Month', 'Day', 'Hour'],
            'time_steps': {'PM25': 5}
        },
        'pm25_meteorology': {
            'cols': ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour'],
            'time_steps': {
                'PM25': 5,
                'Press': 24,
                'RH': 8,
                'Temp': 24,
                'Wind Speed': 11,
                'Wind Direction': 11
            }
        },
        'pm25_other_pollutant': {
            'cols': ['PM25', 'CO', 'NO2', 'O3', 'SO2', 'PM10', 'Month', 'Day', 'Hour'],
            'time_steps': {
                'PM25': 5,
                'CO': 10,
                'NO2': 8,
                'O3': 7,
                'SO2': 4,
                'PM10': 4
            }
        }
    }
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [300] * 3
    conf['model_conf']['dense_conf']['layers'] = [1024]
    for f in features:
        conf['data_conf']['use_cols'] = features[f]['cols']
        conf['model_conf']['time_steps'] = features[f]['time_steps']
        conf['model_conf']['lstm_conf']['max_feature_length'] = 24 * len(conf['model_conf']['time_steps'])
        conf['model_conf']['lstm_conf']['input_shape'] = (1, 24 * len(conf['model_conf']['time_steps']))
        conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/EffectOfOtherSeriesFeature/model_' + f + '_.best.json'
        conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/EffectOfOtherSeriesFeature/weights_' + f + '_.best.hdf5'
        if is_train:
            lstm.train(conf)
        else:
            conf['model_conf']['test_split'] = 1.0
            lstm.predict(conf, save_result=True)


def fixed_time_lags(is_train=True):
    """
    对比固定时滞（窗口）大小提取特征
    :param is_train:
    :return:
    """
    fixed_size = 5
    fixed_time_steps = {
        'PM25': fixed_size,
        'Press': fixed_size,
        'RH': fixed_size,
        'Temp': fixed_size,
        'Wind Speed': fixed_size,
        'Wind Direction': fixed_size,
        'CO': fixed_size,
        'NO2': fixed_size,
        'O3': fixed_size,
        'SO2': fixed_size,
        'PM10': fixed_size
    }
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [300] * 3
    conf['model_conf']['dense_conf']['layers'] = [1024]
    conf['model_conf']['time_steps'] = fixed_time_steps
    conf['model_conf']['lstm_conf']['max_feature_length'] = fixed_size * len(conf['model_conf']['time_steps'])
    conf['model_conf']['lstm_conf']['input_shape'] = (1, fixed_size * len(conf['model_conf']['time_steps']))
    conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/FixedTimeLags/model_fixed_time_lags_.best.json'
    conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/FixedTimeLags/weights_fixed_time_lags_.best.hdf5'
    if is_train:
        lstm.train(conf)
    else:
        lstm.predict(conf, save_result=True)


def diff_time_span(is_train=True):
    """
    不同时间粒度的预测能力
    :param is_train:
    :return:
    """
    time_span = {
        'day': (1, 12 + 31),
        'week': (1, 12),
        # 'month': (1, 12)
    }
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [300] * 3
    conf['model_conf']['dense_conf']['layers'] = [1024]
    conf['model_conf']['batch_size'] = 12
    for ts in time_span:
        conf['model_conf']['time_span'] = ts
        conf['model_conf']['dense_conf']['date_features_shape'] = time_span[ts]
        conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/DifferentTimeSpan/model_time_span_' + ts + '.best.json'
        conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/DifferentTimeSpan/weights_time_span_' + ts + '.best.hdf5'
        if is_train:
            lstm.train(conf)
        else:
            conf['model_conf']['test_split'] = 1.0
            lstm.predict(conf, save_result=True, show_fitting_curve=True)


def train_in_other_sites(is_train=True):
    """
    模型在其他站点数据上训练
    :param is_train:
    :return:
    """
    time_steps = {
        'PM25': 5,
        'Press': 24,
        'RH': 8,
        'Temp': 24,
        'Wind Speed': 11,
        'Wind Direction': 11,
        'CO': 10,
        'NO2': 8,
        'O3': 7,
        'SO2': 4,
        'PM10': 4,
    }
    # stations = ['060376012', '060371103', '060374004']
    stations = ['261630001']
    conf = lstm.get_init_config()
    conf['model_conf']['lstm_conf']['layers'] = [300] * 3
    conf['model_conf']['dense_conf']['layers'] = [1024]
    conf['model_conf']['is_drop_outlier'] = True

    # conf['model_conf']['time_steps'] = time_steps
    # conf['model_conf']['lstm_conf']['input_shape'] = (1, np.sum(list(time_steps.values())))
    for s in stations:
        conf['data_conf']['data_path'] = '../DataSet/Processed/Train/' + s + '_2016_2017_v1.csv'
        conf['model_conf']['epochs'] = 500
        conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/TrainInOtherSite/model_in_' + s + '.json'
        conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/TrainInOtherSite/weights_in_' + s + '.best.hdf5'
        if is_train:
            conf['model_conf']['test_split'] = 0.3
            lstm.train(conf)
        else:
            conf['model_conf']['test_split'] = 1
            lstm.predict(conf, save_result=True, show_fitting_curve=True)


def use_best_model_predict_other_sites():
    """
    用最好的模型预测其他站点，查看模型对其他站点的泛化性能
    :return:
    """
    stations = ['261630001', '060376012', '060371103', '060374004']
    years = ['2016', '2017']
    conf = lstm.get_init_config()
    conf['model_conf']['test_split'] = 1.0
    conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/model.best.json'
    conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/weights.best.hdf5'
    metrics = pd.DataFrame()
    for s in stations:
        for y in years:
            conf['data_conf']['data_path'] = '../DataSet/Processed/Train/' + s + '_' + y + '_v1.csv'
            print(conf['data_conf']['data_path'])
            result = lstm.predict(conf)
            df_temp = result['all_metrics']
            df_temp['Station'] = s
            df_temp['Year'] = y
            metrics = metrics.append(df_temp)
    metrics.to_csv(conf['model_conf']['model_root_path'] + '/PredictOtherSites/use_best_model_predict_other_sites.csv', index=False)


def different_future_time_point_predict():
    """
    用过去的数据预测未来不同时间点的值
    :return:
    """
    print('用过去的数据预测未来不同时间点的值....')
    times = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    conf = lstm.get_init_config()
    conf['model_conf']['test_split'] = 1.0
    conf['model_conf']['model_path'] = conf['model_conf']['model_root_path'] + '/model.best.json'
    conf['model_conf']['weight_path'] = conf['model_conf']['model_root_path'] + '/weights.best.hdf5'
    metrics = pd.DataFrame()
    for t in times:
        conf['model_conf']['target_offset'] = t
        result = lstm.predict(conf)
        df_temp = result['all_metrics']
        df_temp['Future Time'] = t
        metrics = metrics.append(df_temp)
    metrics.to_csv(conf['model_conf']['model_root_path'] + '/DifferentFutureTime/use_best_model_predict_diff_future_time_point.csv', index=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # find_best_lstm_layers_nodes()
    find_best_lstm_layers_nodes(is_train=False)

    # find_best_dense_layers_nodes()
    # find_best_dense_layers_nodes(is_train=False)

    # effect_of_other_series_feature()
    # effect_of_other_series_feature(is_train=False)

    # fixed_time_lags()
    # fixed_time_lags(is_train=False)

    # diff_time_span()
    # diff_time_span(is_train=False)

    # train_in_other_sites()
    # train_in_other_sites(is_train=False)

    # use_best_model_predict_other_sites()

    # different_future_time_point_predict()
