# Created by Yuexiong Ding
# Date: 2018/9/29
# Description:
import numpy as np
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


def load_model(model_path, weight_path):
    if os.path.exists(model_path):
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        # 有参数则加载
        if os.path.exists(weight_path):
            print('load weights ' + weight_path)
            model.load_weights(weight_path)
        return model
    else:
        exit('找不到模型' + model_path)


def main(is_train=True):
    data_path = '../DataSet/Processed/Train/261630033_2016_2017_v1.csv'
    # model_path = '../Models/Test/model.best.json'
    # weight_path = '../Models/Test/weights.best.hdf5'
    model_path = '../Models/Test/model_epochs10_batch24.best.json'
    weight_path = '../Models/Test/weights_epochs10_batch24.best.hdf5'
    df_raw = data.get_raw_data(data_path, ['PM25'], dtype=float)
    seq_data = np.array(df_raw).reshape(1, -1)[0]
    test_split = 0.4
    time_steps = 4
    new_data = []
    for i in range(len(df_raw) - time_steps):
        new_data.append(list(seq_data[i: i + time_steps + 1]))
    new_data = np.array(new_data)
    train_num = int(len(new_data) * (1 - test_split))

    y_scaled, y_scaler = data.min_max_scale(new_data[:, -1].reshape(-1, 1))
    X_scaled, X_scaler = data.min_max_scale(new_data[:, 0: time_steps])

    y_train = y_scaled[:train_num, :].reshape(1, -1)[0]
    y_test = y_scaled[train_num:, :].reshape(1, -1)[0]
    X_train = X_scaled[:train_num, :]
    X_test = X_scaled[train_num:, :]
    X_train = X_train.reshape(X_train.shape[0], time_steps, 1)
    X_test = X_test.reshape(X_test.shape[0], time_steps, 1)

    if is_train:
        if os.path.exists(model_path):
            json_string = open(model_path).read()
            model = model_from_json(json_string)
            # 有参数则加载
            if os.path.exists(weight_path):
                print('load weights ' + weight_path)
                model.load_weights(weight_path)
        else:
            model = Sequential()
            model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(units=64, activation='linear'))
            model.add(Dense(units=1))
            open(model_path, 'w').write(model.to_json())
        model.compile(loss='mse', optimizer='RMSprop')
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, epochs=20, batch_size=24, validation_data=(X_test, y_test), verbose=1,
                            callbacks=callbacks_list, shuffle=False)

        evaluate.draw_loss_curve(figure_num='PM2.5', train_loss=history.history['loss'],
                                 val_loss=history.history['val_loss'])
    else:
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        model.load_weights(weight_path)
        y_pred = model.predict(X_test)
        y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        df_all_metrics = evaluate.all_metrics(y_true[: len(y_true) - 1], y_pred[1:])
        evaluate.draw_fitting_curve(y_true[: len(y_true) - 1], y_pred[1:])


if __name__ == '__main__':
    main(is_train=True)
    main(is_train=False)

