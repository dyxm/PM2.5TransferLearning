# Created by Yuexiong Ding
# Date: 2018/9/4
# Description: 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def get_sequence_features(df_seq_data, time_step, max_time_step, feature_name, is_padding=True):
    """
    :param df_seq_data: a DataFrame sequence data
    :param time_step: time step
    :param max_time_step: max time step, time_step < t < max_time_step, fill -1
    :param feature_name: name of the feature
    :return: a DataFrane data
    """
    # features归一化（最大最小归一化）
    # seq_data = minmax_scale(df_seq_data)
    seq_data = np.array(df_seq_data)

    new_data = []
    for i in range(len(seq_data)):
        if feature_name == 'PM25':
            temp_data = [seq_data[i]]
        else:
            temp_data = []
        for j in range(1, time_step + 1):
            if i < j:
                temp_data.append(0)
            else:
                temp_data.append(seq_data[i - j])
        new_data.append(temp_data)
    if feature_name == 'PM25':
        header = [feature_name]
    else:
        header = []
    if is_padding:
        # 填充-1然维度达到max_time_step
        new_data = padding(new_data, -1, max_time_step - time_step)
        for i in range(1, max_time_step + 1):
            header.append(feature_name + '(t-' + str(i) + ')')
    else:
        for i in range(1, time_step + 1):
            header.append(feature_name + '(t-' + str(i) + ')')
    return pd.DataFrame(new_data, columns=header)


def process_sequence_features(df_raw, time_steps, is_padding=True):
    """
    get the time features from each sequence
    :param df_raw:
    :param time_steps:
    :param is_padding:
    :return:
    """
    # 提取特征的时序特征
    print('提取特征的时序特征...')
    max_time_step = max(time_steps.values())
    df_new = get_sequence_features(df_raw.pop('PM25').values, time_steps['PM25'], max_time_step, 'PM25',
                                   is_padding=is_padding)
    if 'Press' in df_raw.columns:
        df_new = pd.concat([df_new,
                            get_sequence_features(df_raw.pop('Press').values, time_steps['Press'], max_time_step,
                                                  'Press', is_padding=is_padding)], axis=1)
    if 'RH' in df_raw.columns:
        df_new = pd.concat([df_new,
                            get_sequence_features(df_raw.pop('RH').values, time_steps['RH'], max_time_step, 'RH',
                                                  is_padding=is_padding)], axis=1)
    if 'Temp' in df_raw.columns:
        df_new = pd.concat([df_new,
                            get_sequence_features(df_raw.pop('Temp').values, time_steps['Temp'], max_time_step, 'Temp',
                                                  is_padding=is_padding)], axis=1)
    if 'Wind Speed' in df_raw.columns:
        df_new = pd.concat([df_new, get_sequence_features(df_raw.pop('Wind Speed').values, time_steps['Wind Speed'],
                                                          max_time_step, 'Wind Speed', is_padding=is_padding)], axis=1)
    if 'Wind Direction' in df_raw.columns:
        df_new = pd.concat([df_new,
                            get_sequence_features(df_raw.pop('Wind Direction').values, time_steps['Wind Direction'],
                                                  max_time_step, 'Wind Direction', is_padding=is_padding)], axis=1)
    return df_new


def encoding_features(df_raw, cols):
    """
    encoding date features
    :param df_raw:
    :return:
    """
    print('One-Hot Encoding...')
    return pd.get_dummies(df_raw, columns=cols)


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

# def process_data():
# # processing the sequence features
# df_raw = process_sequence_features(df_raw, time_steps=time_steps)
# print(df_raw)
# # normalization
# y_scaled, y_scaler = min_max_scale(np.array(df_raw.pop('PM25')).reshape(-1, 1))
# X_scaled, X_scaler = min_max_scale(df_raw)
#
# # train_X, train_y, test_X, test_y = split_data(X_scaled, y_scaled, train_num=train_num)
#
# # split data to train data and test data
# return split_data(X_scaled, y_scaled, train_num=train_num)
