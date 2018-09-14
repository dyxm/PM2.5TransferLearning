# Created by Yuexiong Ding
# Date: 2018/9/6
# Description: use the best model to pridect the concentration of PM2.5 in other station
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os


def load_data(path, time_steps, lstm_num, cols=None, dtype=str):
    """
    load data
    :param path: data file path
    :param cols: which features
    :return: X, X_scaler, y, y_scaler
    """
    df_raw = pd.read_csv(path, usecols=cols, dtype=dtype)
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
    y = y_scaled.reshape((y_scaled.shape[0], 1, y_scaled.shape[1]))
    # reshape X
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    date_encode = date_encode.reshape((date_encode.shape[0], 1, date_encode.shape[1]))
    X = []
    # 分割，将PM2.5,Press等时间序列特征分别作为每个lstm模型的输入
    split_size = int(X_scaled.shape[2] / lstm_num)
    for i in range(lstm_num):
        X.append(X_scaled[:, :, i * split_size: (i + 1) * split_size])
    # 日期时间特征
    X.append(date_encode)

    return X, X_scaler, y, y_scaler


def my_load_model(model_path, weight_path):
    """
    load model and weights
    :param model_path:
    :param weight_path:
    :return:
    """
    if os.path.exists(model_path) and os.path.exists(weight_path):
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        model.load_weights(weight_path)
        model.summary()
        model.compile(loss='mse', optimizer='RMSprop')
        return model
    else:
        exit('The model or weight file does not exist.')


def process_data(df_raw, time_steps):
    pass


def predict(data_path, cols, lstm_model, time_step, lstm_num):
    model_root_path = '../Models/LSTM1'
    l = 2
    n = 900
    model_path = model_root_path + '/lstm_model_layers' + str(l) + '_nodes' + str(n) + '.json'
    weight_path = model_root_path + '/weights_layer' + str(l) + '_nodes' + str(n) + '.best.hdf5'

    X, X_scaler, y, y_scaler = load_data(data_path, time_step, lstm_num, cols=cols)
    model = my_load_model(model_path, weight_path)
    y_pred = model.predict(X)

    y_true = y_scaler.inverse_transform(y.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    # metrics = evaluate.print_metrics(y_true, y_pred)
    return evaluate.print_metrics(y_true, y_pred)


if __name__ == '__main__':
    # lstm_model = 2
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'Month', 'Day', 'Hour']
    data_path = '../DataSet/Processed/Train/261630033_2017_v1.csv'
    time_steps = [1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 24]
    # time_steps = [1]
    metrics_str = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    metrics = []
    model = {
        'Single LSTM': {
            'lstm_num': 1
        },
        'Multi LSTM': {
            'lstm_num': 6
        }
    }
    pred = predict(data_path, cols, m, ts, model[m]['lstm_num'])

    # 所有
    # for m in model:
    #     temp = []
    #     for ts in time_steps:
    #         pred = predict(data_path, cols, m, ts, model[m]['lstm_num'])
    #         temp.append([pred[x] for x in pred])
    #     metrics.append(temp)
    #     print(metrics)
    # metrics = [[[20.179334526234747, 4.492141418770645, 2.990918948189574, 43.990023939288356, 0.43949996971393357], [19.29648802103844, 4.3927768007307675, 2.8113086999067267, 38.875435236115706, 0.46408407687280206], [21.078197155384103, 4.591099776239251, 3.077419776860034, 48.0216908711037, 0.4146691292693532], [20.30879597785852, 4.506528151233333, 2.8988543994883935, 39.94266916301561, 0.4360987529655913], [21.345145984327022, 4.620080733529125, 2.9201418984849656, 41.53129840644216, 0.4073899113974391], [20.672534748909275, 4.5467059228533, 2.9416815377105543, 42.931070939958246, 0.426011287364092], [22.322744241470872, 4.72469514799324, 3.1275801141568906, 45.23662642748159, 0.3794162794573467], [21.993641854052758, 4.689737930210254, 3.074929143182454, 43.425069328471736, 0.3885382557789676], [24.6282102922202, 4.962681764149319, 3.3178497933144073, 50.93226757801074, 0.31551775295451345], [24.84235966369666, 4.984211037235147, 3.3274862816733632, 49.40710728777195, 0.3098663099620522], [24.501352842111892, 4.949884124109563, 3.2571211253991783, 47.282690344924355, 0.3193369133535995]], [[16.045090183782637, 4.005632307611701, 2.280478260023125, 29.262930758742144, 0.5543325017848899], [16.129024332565297, 4.0160956577956775, 2.3020464713076083, 30.354496490282145, 0.5520531531487146], [16.315267224445552, 4.039216164609855, 2.4015513458318942, 33.946949874858376, 0.5469332837012324], [16.098251402027476, 4.012262628745466, 2.3211485664345086, 32.435405427335176, 0.5530102301203023], [16.092826410238217, 4.011586520348055, 2.324008663430777, 31.555239745219488, 0.5532112410081707], [16.68124938684675, 4.084268525311081, 2.4899515108364385, 37.4906873134909, 0.5368323731457322], [16.18635113997948, 4.023226458948026, 2.3409070769207183, 32.297723572484486, 0.5500111498927245], [16.13740327577987, 4.017138692624375, 2.307401192235359, 31.04184654891672, 0.5513519398158109], [15.98367960808658, 3.9979594305203476, 2.315647913997491, 31.05039240745721, 0.5557718240836107], [16.14376891717093, 4.017930924887949, 2.3168441813415734, 29.979339423441164, 0.5515176913645403], [16.0198331311329, 4.00247837359965, 2.3228563911522593, 30.239158834819648, 0.5549588981121212]]]
    # metrics = np.array(metrics)
    # for mts in range(len(metrics_str)):
    #     plt.figure(mts)
    #     for mod in range(len(model)):
    #         plt.title(metrics_str[mts])
    #         plt.plot(time_steps, metrics[mod][:, mts], label=list(model.keys())[mod])
    #         plt.ylabel(metrics_str[mts] + 'Value')
    #         plt.xlabel('Time Lags(Hour)')
    #         plt.legend()
    # plt.show()
