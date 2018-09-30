# Created by Yuexiong Ding
# Date: 2018/9/26
# Description: 画图工具
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from MyModule import data


def draw_time_series(df_raw, cols):
    """
    画时序图
    :param df_raw:
    :param cols:
    :return:
    """
    for c in cols:
        plt.figure(c)
        plt.plot(df_raw[c])
        plt.title(c + ' time series')
        plt.ylabel(c + ' value')
        plt.xlabel('Time(Hour)')
    plt.show()


def draw_lag_correlation(df_raw, main_col, other_cols, max_time_lag, min_corr_coeff, separate_draw=False):
    """
    画主要因素与其他因素的滞后时间序列的关系
    :param df_raw:
    :param main_col:
    :param other_cols:
    :param max_time_lag: 最大滞后小时
    :param min_corr_coeff:
    :param separate_draw: 是否分开画图，默认否
    :return:
    """
    all_corr = data.get_lag_correlation(df_raw, main_col, other_cols, max_time_lag=max_time_lag)
    for oc in all_corr:
        if separate_draw:
            plt.figure(oc)
        plt.plot(range(max_time_lag), all_corr[oc], label=oc)
        plt.plot(range(max_time_lag + 1), [min_corr_coeff] * (max_time_lag + 1), ls='--', c='r')
        plt.title('The correlation between ' + main_col + ' time series and lagged time series')
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Time Lags(Hour)')
        plt.legend()
    plt.show()


def draw_auto_correlation(df_raw, cols, max_time_lag):
    """
    画自相关图
    :param df_raw:
    :param cols:
    :param max_time_lag:
    :return:
    """
    for c in cols:
        # plt.figure(c)
        plot_acf(df_raw[c], lags=max_time_lag)
        plt.title('Autocorrelation Coefficient of ' + c)
        # plt.plot(range(max_time_lag + 1), [0.4] * max_time_lag, '--', c='r')
        plt.ylabel('Autocorrelation Coefficient')
        plt.xlabel('Time Lags(Hour)')
        plt.xticks(np.arange(0, max_time_lag + 2, step=2))
        plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.show()


