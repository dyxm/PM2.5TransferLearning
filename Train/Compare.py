# Created by Yuexiong Ding
# Date: 2018/9/12
# Description: support vector regression

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os
from MyModule import lstm
from MyModule import draw


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
    if time_span == 'day':
        df_raw.pop('Hour')
        df_raw['Day Of Year'] = [str(x.year) + ('00' if x.dayofyear < 10 else ('0' if x.dayofyear < 100 else '')) + str(x.dayofyear) for x in df_raw['Date Time']]
        df_raw.pop('Date Time')
        df_group_by_day = df_raw.groupby('Day Of Year', as_index=False).mean()
        df_group_by_day.pop('Day Of Year')
    elif time_span == 'week':
        df_raw.pop('Hour')
        df_raw.pop('Day')
        df_raw['Week Of Year'] = [str(x.year) + ('0' if x.weekofyear < 10 else '') + str(x.weekofyear) for x in df_raw['Date Time']]

        df_group_by_day = df_raw.groupby('Week Of Year', as_index=False).mean()
        df_group_by_day.pop('Week Of Year')
    elif time_span == 'month':
        df_raw.pop('Hour')
        df_raw.pop('Day')
        df_raw.pop('Month')
        df_raw['Month Of Year'] = [str(x.year) + ('0' if x.month < 10 else '') + str(x.month) for x in df_raw['Date Time']]
        df_group_by_day = df_raw.groupby('Month Of Year', as_index=False).mean()
        df_group_by_day.pop('Month Of Year')
    else:
        df_group_by_day = df_raw
    df_raw.pop('Date Time')
    return df_group_by_day


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
        # df_raw = df_raw[df_raw[c] <= 80]
        df_raw[c][df_raw[c] > 80] = 80
        # df_raw[c][df_raw[c] > mean + standard_deviation_times * std] = mean + standard_deviation_times * std
        df_raw = df_raw.reset_index()
        df_raw.pop('index')
    return df_raw


def process_data(df_raw_data, time_steps, train_num):
    """
    processing raw data
    :param df_raw_data:
    :param time_steps:
    :param train_num:
    :return:
    """
    # df_raw_data['PM25'].astype(float)
    # df_raw_data = df_raw_data[df_raw_data['PM25'].astype(float) < 100]
    # df_raw_data = drop_outlier(df_raw_data, ['PM25'], 6)
    # draw.draw_time_series(df_raw_data, ['PM25'])
    # time_steps = data.get_time_steps()
    df_raw_data = group_by_diff_time_span(df_raw_data, 'hour')
    max_time_step = max(time_steps.values())
    # pop the date features
    df_date = df_raw_data.pop('Month')
    if 'Day' in df_raw_data.columns:
        df_date = pd.concat([df_date, df_raw_data.pop('Day')], axis=1)
    if 'Hour' in df_raw_data.columns:
        df_date = pd.concat([df_date, df_raw_data.pop('Hour')], axis=1)
    df_date = df_date.loc[max_time_step:]

    # processing the sequence features
    # df_raw_data = df_raw_data[list(time_steps.keys())]
    df_raw_data = data.process_sequence_features(df_raw_data, 0, time_steps, max_time_step, padding_value=-1)
    df_raw_data = df_raw_data.loc[max_time_step:]

    # encoding the date features
    df_date_encoded = data.encoding_features(df_date, ['Month', 'Hour', 'Day'])

    # normalization
    y_scaled, y_scaler = data.min_max_scale(np.array(df_raw_data.pop('PM25')).reshape(-1, 1))
    X_scaled, X_scaler = data.min_max_scale(df_raw_data)
    date_encoded = np.array(df_date_encoded)

    # 分割样本
    X_train = np.append(X_scaled[:train_num, :], date_encoded[:train_num, :], axis=1)
    X_test = np.append(X_scaled[train_num:, :], date_encoded[train_num:, :], axis=1)
    y_train = np.array(y_scaled[:train_num]).reshape(1, -1)[0]
    y_test = np.array(y_scaled[train_num:]).reshape(1, -1)[0]

    return X_train, X_test, y_train, y_test, y_scaler

#
# def predict(model, x, y):
#     y_pred = model.predict(x)
#     y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
#                                                train_num=train_num)


def svr(X_train, X_test, y_train, y_test, y_scaler, train_num):
    """
    support vector regression model
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_scaler:
    :param train_num:
    :return:
    """
    sv = SVR(kernel='rbf', C=1)
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.all_metrics(y_test, y_pred)
    evaluate.draw_fitting_curve(y_test, y_pred, 0)


def ridge(X_train, X_test, y_train, y_test, y_scaler, train_num):
    """
    Ridge regression
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_scaler:
    :return:
    """
    # rig = RidgeCV(alphas=[1, 0.5, 0.1, 0.01, 0.05, 0.001, 0.005])
    rig = RidgeCV(alphas=[5.0, 10.0])
    rig.fit(X_train, y_train)
    y_pred = rig.predict(X_test)
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.all_metrics(y_test, y_pred)
    evaluate.draw_fitting_curve(y_test, y_pred, 0)


def lasso(X_train, X_test, y_train, y_test, y_scaler, train_num):
    """
    LASSO
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_scaler:
    :return:
    """
    # las = LassoCV(alphas=[0.0001, 0.0005, 0.00005, 0.00002, 0.00001, 0.000001])
    las = LassoCV(alphas=[0.0003, 0.0002])
    las.fit(X_train, y_train)
    y_pred = las.predict(X_test)
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.all_metrics(y_test, y_pred)
    evaluate.draw_fitting_curve(y_test, y_pred, 0)


def ann(X_train, X_test, y_train, y_test, y_scaler, train_num):
    """
    artificial neural network
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_scaler:
    :return:
    """
    model_path = '../Models/ANN/ann_model_060376012.json'
    weight_path = '../Models/ANN/ann_weights_060376012.best.hdf5'

    if os.path.exists(model_path):
        print('load model...')
        model = lstm.load_model_and_weights(model_path, weight_path)
    else:
        print('训练模型...')
        model = Sequential()
        model.add(Dense(500, input_dim=(len(X_train[0])), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        open(model_path, 'w').write(model.to_json())
        model.compile(loss='mse', optimizer='RMSprop')
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, epochs=500, batch_size=512, validation_data=(X_test, y_test),
                            verbose=1, callbacks=callbacks_list, shuffle=False)
    y_pred = model.predict(X_test)
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.all_metrics(y_test, y_pred)
    evaluate.draw_fitting_curve(y_test, y_pred, 0)


def rnn(X_train, X_test, y_train, y_test, y_scaler, train_num):
    """
    recurrent neural network
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_scaler:
    :return:
    """
    model_path = '../Models/RNN/rnn_model_060376012.json'
    weight_path = '../Models/RNN/rnn_weights_060376012.best.hdf5'

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    y_test = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))
    if os.path.exists(model_path) and os.path.exists(weight_path):
        print('load model...')
        model = lstm.load_model_and_weights(model_path, weight_path)
    else:
        print('训练模型...')
        model = Sequential()
        model.add(SimpleRNN(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=300, dropout=0.3,
                            return_sequences=True))
        model.add(SimpleRNN(output_dim=300, dropout=0.3, return_sequences=True))
        model.add(SimpleRNN(output_dim=300, dropout=0.3, return_sequences=True))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        open(model_path, 'w').write(model.to_json())
        model.compile(loss='mse', optimizer='RMSprop')
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, epochs=500, batch_size=512, validation_data=(X_test, y_test),
                            verbose=1, callbacks=callbacks_list, shuffle=False)
    y_pred = model.predict(X_test)
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    evaluate.all_metrics(y_test, y_pred)
    evaluate.draw_fitting_curve(y_test, y_pred, 0)


def main(model_name):
    # data_path = '../DataSet/Processed/Train/261630033_2016_2017_v1.csv'
    # data_path = '../DataSet/Processed/Train/261630001_2016_2017_v1.csv'
    data_path = '../DataSet/Processed/Train/060376012_2016_2017_v1.csv'
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'CO', 'NO2', 'O3', 'SO2', 'PM10', 'Month',
            'Day', 'Hour', 'Date Time']
    # cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    # cols = ['PM25', 'CO', 'NO2', 'O3', 'SO2', 'PM10', 'Month', 'Day', 'Hour']
    # cols = ['PM25', 'Month', 'Day', 'Hour']
    df_raw_data = pd.read_csv(data_path, usecols=cols, dtype=str)
    test_split = 0.4
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
    time_steps = {
        'PM25': 24,
        'Press': 24,
        'RH': 8,
        'Temp': 7,
        'Wind Speed': 4,
        'Wind Direction': 1,
        'CO': 24,
        'NO2': 24,
        'O3': 5,
        'SO2': 7,
        'PM10': 4,
    }
    train_num = int(len(df_raw_data) * (1 - test_split))
    X_train, X_test, y_train, y_test, y_scaler = process_data(df_raw_data, time_steps, train_num)
    # X_train, X_test, y_train, y_test, y_scaler = process_data(df_raw_data, train_num)

    if str.lower(model_name) == 'svr':
        print('SVR:')
        svr(X_train, X_test, y_train, y_test, y_scaler, train_num)
    elif str.lower(model_name) == 'lasso':
        print('LASSO:')
        lasso(X_train, X_test, y_train, y_test, y_scaler, train_num)
    elif str.lower(model_name) == 'ridge':
        print('RIDGE:')
        ridge(X_train, X_test, y_train, y_test, y_scaler, train_num)
    elif str.lower(model_name) == 'ann':
        print('ANN:')
        ann(X_train, X_test, y_train, y_test, y_scaler, train_num)
    elif str.lower(model_name) == 'rnn':
        print('RNN:')
        rnn(X_train, X_test, y_train, y_test, y_scaler, train_num)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # main(model_name='svr')
    # main(model_name='lasso')
    # main(model_name='ridge')
    # main(model_name='ann')
    main(model_name='rnn')
