# Created by Yuexiong Ding
# Date: 2018/9/5
# Description: evaluation functions

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def reshape(y_true, y_pred):
    """
    reshape y_true、y_pred to 1D array
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = np.array(y_true).reshape(1, -1)[0], np.array(y_pred).reshape(1, -1)[0]
    return y_true, y_pred


def mse(y_true, y_pred):
    """
    mean square error
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = reshape(y_true, y_pred)
    return metrics.mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """
    root mean square error
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    mean absolute error
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = reshape(y_true, y_pred)
    return metrics.mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """
    mean absolute percentage error
    :param y_true:
    :param y_pred:
    :return:
    """
    # y_true, y_pred = np.array(y_true).reshape(1, -1)[0], np.array(y_pred).reshape(1, -1)[0]
    y_true, y_pred = reshape(y_true, y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def r2(y_true, y_pred):
    """
    R square
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = reshape(y_true, y_pred)
    return metrics.r2_score(y_true, y_pred)


def all_metrics(y_true, y_pred):
    """
    return all the evaluation metrics
    :param y_true:
    :param y_pred:
    :return: a DataFrame data
    """
    # y_true, y_pred = np.array(y_true).reshape(1, -1)[0], np.array(y_pred).reshape(1, -1)[0]
    MSE = mse(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mae(y_true, y_pred)
    MAPE = mape(y_true, y_pred)
    R2 = metrics.r2_score(y_true, y_pred)
    print('Test MSE: %f' % MSE)
    print('Test RMSE: %f' % RMSE)
    print('Test MAE: %f' % MAE)
    print('Test MAPE: %f' % MAPE)
    print('Test R2: %f' % R2)

    return pd.DataFrame({'MSE': [MSE], 'RMSE': [RMSE], 'MAE': [MAE], 'MAPE': [MAPE], 'R2': [R2]})


def draw_loss_curve(figure_num, train_loss=None, val_loss=None):
    """
    draw loss curve
    :param figure_num:
    :param train_loss:
    :param val_loss:
    :return:
    """
    plt.figure(figure_num)
    if len(train_loss) > 0:
        plt.plot(train_loss, label='train')
    if len(val_loss) > 0:
        plt.plot(val_loss, label='test')
    if len(train_loss) > 0 or len(val_loss) > 0:
        plt.show()


def draw_fitting_curve(y_true, y_pred, figure_num=None):
    """
    draw the fitting curve
    :param y_true:
    :param y_pred:
    :param figure_num:
    :return:
    """
    y_true, y_pred = reshape(y_true, y_pred)
    if figure_num:
        plt.figure(figure_num)
    plt.plot(y_true, c='b')
    plt.plot(y_pred, c='r')
    plt.show()
