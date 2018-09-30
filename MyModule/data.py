# Created by Yuexiong Ding
# Date: 2018/9/4
# Description: 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_raw_data(data_path, usecol=None, dtype=str):
    """
    读取数据
    :param data_path:
    :param usecol:
    :param dtype:
    :return:
    """
    df_raw_data = pd.read_csv(data_path, usecols=usecol, dtype=dtype)
    return df_raw_data


def drop_outlier(df_raw, cols, standard_deviation_times=3):
    """
    删除离异点数据
    :param df_raw:
    :param cols:
    :param standard_deviation_times: 标准差倍数
    :return:
    """
    for c in cols:
        df_raw[c] = df_raw[c].astype(float)
        mean = df_raw[c].mean()
        std = df_raw[c].std()
        # df_raw = df_raw[df_raw[c] <= mean + standard_deviation_times * std]
        # df_raw = df_raw[df_raw[c] <= 80]
        df_raw[c][df_raw[c] > 80] = 80
        df_raw = df_raw.reset_index()
        df_raw.pop('index')
    return df_raw


def get_lag_correlation(df_raw, main_factor, other_factors, max_time_lag):
    """
    获取主要因素与其他因素的滞后时间序列的相关系数
    :param df_raw:
    :param main_factor:
    :param other_factors:
    :param max_time_lag: 最大滞后时间
    :return:
    """
    len_ = len(df_raw[main_factor])
    corr = {}
    for oc in other_factors:
        temp_corr = []
        for tl in range(max_time_lag):
            df_main_col = df_raw[main_factor][tl:]
            df_main_col = df_main_col.reset_index()[main_factor].astype(float)
            df_other_col = df_raw[oc][: len_ - tl].astype(float)
            temp_corr.append(df_main_col.corr(df_other_col))
        corr[oc] = temp_corr
    return corr


def get_time_steps(df_raw, main_factor, other_factors, max_time_lag, min_corr_coeff):
    """
    获取时间步
    :param df_raw:
    :param main_factor:
    :param other_factors:
    :param max_time_lag:
    :param min_corr_coeff: 最小相关系数
    :return:
    """
    print('获取时间步...')
    all_corr = get_lag_correlation(df_raw, main_factor, other_factors, max_time_lag=max_time_lag)
    time_steps = {}
    for oc in all_corr:
        temp = [i for i, x in enumerate(all_corr[oc]) if math.fabs(x) >= min_corr_coeff]
        if len(temp) > 0:
            time_steps[oc] = temp[1:]
    return time_steps


def group_by_diff_time_span(df_raw, time_span):
    """
    按不同时间粒度求分组求均值
    :param df_raw:
    :param time_span:
    :return:
    """
    df_data_time = df_raw.pop('Date Time')
    for c in df_raw.columns:
        df_raw[c] = df_raw[c].astype(float)
    df_raw['Date Time'] = pd.to_datetime(df_data_time)
    if str.lower(time_span) == 'day':
        df_raw.pop('Hour')
        df_raw['Day Of Year'] = [str(x.year) + ('00' if x.dayofyear < 10 else ('0' if x.dayofyear < 100 else '')) + str(x.dayofyear) for x in df_raw['Date Time']]
        df_group_by_day = df_raw.groupby('Day Of Year', as_index=False).mean()
        df_group_by_day['Day'] = df_group_by_day['Day'].astype(int)
        df_group_by_day['Month'] = df_group_by_day['Month'].astype(int)
        df_group_by_day.pop('Day Of Year')
    elif str.lower(time_span) == 'week':
        df_raw.pop('Hour')
        df_raw.pop('Day')
        df_raw['Week Of Year'] = [str(x.year) + ('0' if x.weekofyear < 10 else '') + str(x.weekofyear) for x in df_raw['Date Time']]
        df_group_by_day = df_raw.groupby('Week Of Year', as_index=False).mean()
        df_group_by_day['Month'] = df_group_by_day['Month'].astype(int)
        df_group_by_day.pop('Week Of Year')
    elif str.lower(time_span) == 'month':
        df_raw.pop('Hour')
        df_raw.pop('Day')
        # df_raw.pop('Month')
        df_raw['Month Of Year'] = [str(x.year) + ('0' if x.month < 10 else '') + str(x.month) for x in df_raw['Date Time']]
        df_group_by_day = df_raw.groupby('Month Of Year', as_index=False).mean()
        df_group_by_day['Month'] = df_group_by_day['Month'].astype(int)
        df_group_by_day.pop('Month Of Year')
    else:
        df_group_by_day = df_raw
        df_group_by_day.pop('Date Time')
    df_group_by_day = df_group_by_day.astype(str)
    return df_group_by_day


def encoding_features(df_date_data, cols):
    """
    encoding date features
    :param df_date_data:
    :param cols:
    :return:
    """
    print('One-Hot Encoding...')
    len_set = {
        'Month': 12,
        'Day': 31,
        'Hour': 24
    }
    data_len = len(df_date_data)
    for c in cols:
        num = len(set(df_date_data[c]))
        if num < len_set[c]:
            for i in range(len_set[c]):
                if c == 'Hour':
                    df_date_data = df_date_data.append(pd.DataFrame({c: [str(i) + '.0']}), ignore_index=True)
                else:
                    df_date_data = df_date_data.append(pd.DataFrame({c: [str(i + 1) + '.0']}), ignore_index=True)
    df_date_data = df_date_data.fillna('1.0')
    df_dummies = pd.get_dummies(df_date_data, columns=cols)
    # df_dummies = df_dummies.reindex(index=range(data_len))
    df_dummies = df_dummies.iloc[: data_len]
    return df_dummies


def min_max_scale(data):
    """
    use the min_max_scaler scale data
    :param data: 2D data
    :return: return scaled data and the scaler
    """
    print('最大-最小归一化...')
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data, min_max_scaler


def split_data(X, y, train_num):
    """
    分割训练集和训练集
    :param X:
    :param y:
    :param train_num:
    :return:
    """
    # y = np.array(df_raw.pop('PM25')).reshape(-1, 1)
    # X = np.array(df_raw)
    print('分割样本，%f的测试样本...', train_num)
    return X[: train_num, :], y[: train_num], X[train_num:, :], y[train_num:]


def inverse_to_original_data(scaled_train_data, scaled_test_data, scaler, train_num):
    """
    inverse the scaled data to original data
    :param scaled_train_data:
    :param scaled_test_data:
    :param scaler:
    :param train_num:
    :return:
    """
    print('反归一化...')
    original_data = scaler.inverse_transform(np.append(scaled_train_data, scaled_test_data).reshape(-1, 1))
    return np.array(original_data[train_num:])


def padding(raw_data, padding_value, padding_num):
    """
    填充数组到制定列数
    :param raw_data: 原始数组
    :param padding_value: 填充值
    :param padding_num: 填充列数
    :return:
    """
    new_data = np.array(raw_data)
    padding_array = []
    if padding_num > 0:
        for i in range(padding_num):
            padding_array.append([padding_value] * new_data.shape[0])
        # print(new_data.shape, np.array(padding_array).T.shape)
        return np.append(new_data, np.array(padding_array).T, axis=1)
    else:
        return new_data


def get_sequence_features(df_seq_data, target_offset, time_step, max_time_step, feature_name, padding_value):
    """
    :param df_seq_data: a DataFrame sequence data
    :param target_offset: 预测目标值在序列位置的偏移量，0表示用t+0时刻对应的值作为预测目标值，1则为t+1时刻的值，以此类推
    :param time_step: time step， array [1, 2, 3, 5, 6, 8]
    :param max_time_step: max time step, time_step < t < max_time_step, fill -1
    :param feature_name: name of the feature
    :param padding_value:
    :return: a DataFrane data
    """
    seq_data = np.array(df_seq_data)
    if type(time_step) == int:
        time_step = range(1, time_step + 1)
    new_data = []
    for i in range(len(seq_data) - target_offset):
        if feature_name == 'PM25':
            # 预测目标值
            temp_data = [seq_data[i + target_offset]]
        else:
            temp_data = []
        for j in time_step:
            if i < j:
                temp_data.append(0)
            else:
                temp_data.append(seq_data[i - j])
        new_data.append(temp_data)
    if feature_name == 'PM25':
        header = [feature_name]
    else:
        header = []
    if max_time_step - len(time_step) > 0:
        # 填充-1然维度达到max_time_step
        new_data = padding(new_data, padding_value, max_time_step - len(time_step))
        for i in range(1, max_time_step + 1):
            header.append(feature_name + '(t-' + str(i) + ')')
    else:
        for i in time_step:
            header.append(feature_name + '(t-' + str(i) + ')')

    return pd.DataFrame(new_data, columns=header)


def process_sequence_features(df_raw, target_offset, time_steps, max_time_step, padding_value):
    """
    get the time features from each sequence
    :param df_raw:
    :param target_offset: 预测目标值在序列位置的偏移量，0表示用t+0时刻对应的值作为预测目标值，1则为t+1时刻的值，以此类推
    :param time_steps:
    :param max_time_step:
    :param padding_value:
    :return:
    """
    # 提取特征的时序特征
    print('提取特征的时序特征...')
    df_new = pd.DataFrame()
    # for c in df_raw.columns:
    for c in time_steps:
        df_new = pd.concat([df_new,
                            get_sequence_features(df_raw.pop(c).values, target_offset, time_steps[c], max_time_step,
                                                  feature_name=c, padding_value=padding_value)], axis=1)
    return df_new


# def process_data(df_raw, target_offset, time_steps, max_feature_length, train_num, lstm_block_num, padding_value=-1):
def process_data(df_raw, model_conf):
    """
    处理数据，时序特征提取, 时间特征编码，样本分割
    :param df_raw:
    :param model_conf:
    :param padding_value:
    :return:
    """
    # df_raw = df_raw[df_raw['PM25'].astype(float) < 100]
    if model_conf['is_drop_outlier']:
        df_raw = drop_outlier(df_raw, ['PM25'], 3)
    if len(model_conf['time_steps']) == 0:
        model_conf['time_steps'] = get_time_steps(df_raw, model_conf['main_factor'], model_conf['other_factors'],
                                                  model_conf['max_time_lag'], model_conf['min_corr_coeff'])
        model_conf['lstm_conf']['input_shape'] = (1, np.sum([len(model_conf['time_steps'][x]) for x in model_conf['time_steps']]))
    max_time_step = int(model_conf['lstm_conf']['max_feature_length'] / len(model_conf['time_steps']))
    df_raw = group_by_diff_time_span(df_raw, model_conf['time_span'])
    # pop the date features and encoding
    print('提取日期特征...')
    df_date = pd.DataFrame()
    df_date = pd.concat([df_date, df_raw.pop('Month')], axis=1)
    df_date_cols = ['Month']
    if 'Day' in df_raw.columns:
        df_date = pd.concat([df_date, df_raw.pop('Day')], axis=1)
        df_date_cols = ['Month', 'Day']
    if 'Hour' in df_raw.columns:
        df_date = pd.concat([df_date, df_raw.pop('Hour')], axis=1)
        df_date_cols = ['Month', 'Hour', 'Day']
    df_date = df_date.loc[model_conf['max_time_lag']:]
    df_date_encode = encoding_features(df_date, df_date_cols)

    # processing the sequence features
    df_raw = process_sequence_features(df_raw, model_conf['target_offset'], model_conf['time_steps'], max_time_step,
                                       padding_value=model_conf['lstm_conf']['padding_value'])
    df_raw = df_raw.loc[model_conf['max_time_lag']:]
    df_date_encode = df_date_encode.iloc[: len(df_raw)]

    # train_num
    train_num = int(len(df_raw) * (1 - model_conf['test_split']))

    # normalization
    y_scaled, y_scaler = min_max_scale(np.array(df_raw.pop('PM25')).reshape(-1, 1))
    X_scaled, X_scaler = min_max_scale(df_raw)
    date_encode = np.array(df_date_encode)

    # reshape y
    train_y = y_scaled[:train_num]
    test_y = y_scaled[train_num:]
    train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
    # reshape X
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    date_encode = date_encode.reshape((date_encode.shape[0], 1, date_encode.shape[1]))

    # 分割，根据lstm_block_num将PM2.5,Press等时间序列特征作为lstm模型的输入
    print('分割数据...')
    train_X = []
    test_X = []
    step = int(X_scaled.shape[2] / model_conf['lstm_conf']['lstm_block_num'])
    for i in range(model_conf['lstm_conf']['lstm_block_num']):
        train_X.append(X_scaled[:train_num, :, i * step: (i + 1) * step])
        test_X.append(X_scaled[train_num:, :, i * step: (i + 1) * step])
    # 日期时间特征
    train_X.append(date_encode[:train_num, :, :])
    test_X.append(date_encode[train_num:, :, :])

    return train_X, test_X, train_y, test_y, y_scaler


def get_processed_data(data_path, usecol, dtype, time_steps, train_num, lstm_block_num):
    pass

