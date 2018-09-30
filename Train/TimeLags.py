# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: draw time lags of each time series feature

import pandas as pd

from MyModule import draw
from MyModule import data


if __name__ == '__main__':
    # df_raw = pd.read_csv('../DataSet/Processed/Train/261630033_2016_2017_v1.csv')
    # df_raw = pd.read_csv('../DataSet/Processed/Train/261630033_2016_v1.csv')
    df_raw = pd.read_csv('../DataSet/Processed/Train/060376012_2016_2017_v1.csv')
    # df_raw = pd.read_csv('../DataSet/Processed/Train/060374004_2016_2017_v1.csv')
    # df_raw = pd.read_csv('../DataSet/Processed/Train/060374004_2016_v1.csv')

    # #############################画出各个特征的时序图############################
    # cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'CO', 'NO2', 'O3', 'PM10', 'SO2']
    # draw.draw_time_series(df_raw, cols)
    # #################################################################################

    # #####################################自相关系数####################
    cols = ['PM25', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction', 'CO', 'NO2', 'O3', 'PM10', 'SO2']
    draw.draw_auto_correlation(df_raw, cols, max_time_lag=48)
    # #################################################################################

    # #####################################PM2.5与其他因素的time lags ####################
    # main_col = 'PM25'
    # # other_cols = ['PM25', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'Press', 'RH', 'Temp', 'Wind Speed', 'Wind Direction']
    # other_cols = ['PM25', 'PM10', 'O3']
    # draw.draw_lag_correlation(df_raw, main_col, other_cols, max_time_lag=48, min_corr_coeff=0.2)
    # # print(draw.get_lag_corelation(df_raw, main_col, other_cols, max_time_lag=72))
    # print(data.get_time_steps(df_raw, main_col, other_cols, max_time_lag=48, min_corr_coeff=0.2))

