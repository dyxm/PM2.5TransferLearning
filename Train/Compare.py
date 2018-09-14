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
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os

def process_data(df_raw_data, time_steps, train_num):
    max_time_step = max(time_steps.values())
    # pop the date features
    df_date = df_raw_data.pop('Month')
    df_date = pd.concat([df_date, df_raw_data.pop('Day')], axis=1)
    df_date = pd.concat([df_date, df_raw_data.pop('Hour')], axis=1)
    df_date = df_date.loc[max_time_step:]

    # processing the sequence features
    df_raw_data = data.process_sequence_features(df_raw_data, time_steps=time_steps, is_padding=False)
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


def svr(X_train, X_test, y_train):
    # 模型
    sv = SVR(kernel='rbf', C=1)
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    return y_pred


def ridge(X_train, X_test, y_train):
    # rig = RidgeCV(alphas=[0.1, 0.0005, 0.0001, 0.00005, 0.00002])
    rig = RidgeCV(alphas=[5.0, 10.0])
    rig.fit(X_train, y_train)
    y_pred = rig.predict(X_test)
    return y_pred


def lasso(X_train, X_test, y_train):
    # las = LassoCV(alphas=[0.0001, 0.0005, 0.00005, 0.00002, 0.00001, 0.000005])
    las = LassoCV(alphas=[0.0003, 0.0002])
    las.fit(X_train, y_train)
    y_pred = las.predict(X_test)
    return y_pred


def ann(X_train, X_test, y_train, y_test):
    model_path = '../Models/ANN/ann_model_1.json'
    weight_path = '../Models/ANN/ann_weights_1.best.hdf5'

    if os.path.exists(model_path):
        print('load model...')
        json_string = open(model_path).read()
        model = model_from_json(json_string)
    else:
        model = Sequential()
        model.add(Dense(500, input_dim=(len(X_train[0])), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        open(model_path, 'w').write(model.to_json())
    if os.path.exists(weight_path):
        print('load weights...')
        model.load_weights(weight_path)

    model.compile(loss='mse', optimizer='RMSprop')
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, y_train, epochs=1000, batch_size=512, validation_data=(X_test, y_test),
                        verbose=1, callbacks=callbacks_list, shuffle=False)
    y_pred = model.predict(X_test)
    return y_pred


def main():
    data_path = '../DataSet/Processed/Train/261630033_2016_2017_v1.csv'
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    df_raw_data = pd.read_csv(data_path, usecols=cols, dtype=str)
    test_split = 0.4
    cv = 5
    time_steps = {
        'PM25': 5,
        'Press': 24,
        'RH': 8,
        'Temp': 24,
        'Wind Speed': 11,
        'Wind Direction': 11
    }
    train_num = int(len(df_raw_data) * (1 - test_split))
    X_train, X_test, y_train, y_test, y_scaler = process_data(df_raw_data, time_steps, train_num)

    # SVR
    y_pred_svr = svr(X_train, X_test, y_train)

    # lasso
    y_pred_las = lasso(X_train, X_test, y_train)

    # ridge
    y_pred_ridge = ridge(X_train, X_test, y_train)

    # ANN
    # y_pred_ann = ann( X_train, X_test, y_train, y_test)

    # 反归一化
    y_test = data.inverse_to_original_data(y_train.reshape(1, -1), y_test.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)

    y_pred_svr = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred_svr.reshape(1, -1), scaler=y_scaler,
                                           train_num=train_num)
    y_pred_las = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred_las.reshape(1, -1), scaler=y_scaler,
                                                 train_num=train_num)
    y_pred_ridge = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred_ridge.reshape(1, -1), scaler=y_scaler,
                                               train_num=train_num)
    # y_pred_ann = data.inverse_to_original_data(y_train.reshape(1, -1), y_pred_ann.reshape(1, -1), scaler=y_scaler,
    #                                              train_num=train_num)

    #
    print('SVR:')
    evaluate.all_metrics(y_test, y_pred_svr)
    evaluate.draw_fitting_curve(y_test, y_pred_svr, 0)
    print('LASSO:')
    evaluate.all_metrics(y_test, y_pred_las)
    evaluate.draw_fitting_curve(y_test, y_pred_las, 1)
    print('Ridge:')
    evaluate.all_metrics(y_test, y_pred_ridge)
    evaluate.draw_fitting_curve(y_test, y_pred_ridge, 1)
    print('ANN:')
    # evaluate.all_metrics(y_test, y_pred_ann)
    # evaluate.draw_fitting_curve(y_test, y_pred_ann, 1)


if __name__ == '__main__':
    main()
