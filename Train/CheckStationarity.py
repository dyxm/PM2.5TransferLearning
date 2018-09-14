# Created by Yuexiong Ding
# Date: 2018/8/31
# Description: checking the stationarity of the training data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# df_raw = pd.read_csv('../DataSet/Processed/Train/261630033_2016_2017_v1.csv')
df_raw = pd.read_csv('../DataSet/Processed/Train/060376012_2016_2017_v1.csv')
df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'])
# #############################画出各个特征的时序图############################
# 单个
# plt.plot(np.log(df_raw['PM25']))
# plt.show()

# 全部
# i = 1
# plt.figure(0)
# for c in range(2, 8):
#     plt.subplot(7, 1, c)
#     plt.plot(df_raw.iloc[:, c])
#     plt.title(df_raw.columns[c], y=0.5, loc='right')
#     i += 1
# plt.show()
# #################################################################################

# #####################################自相关系数、偏置相关系数####################
# plot_acf(df_raw['PM25'], lags=31)
# plt.show()
# 全部
for c in range(2, 8):
    plt.figure(c)
    plot_acf(df_raw.iloc[:, c], lags=40)
    plt.title('Autocorrelation Coefficient of '+df_raw.columns[c])
    plt.plot(range(41), [0.4] * 41, '--', c='r')
    plt.ylabel('Autocorrelation Coefficient')
    plt.xlabel('Time Lags(Hour)')
    plt.xticks(np.arange(0, 42, step=2))
    plt.yticks(np.arange(0, 1.1, step=0.1))
plt.show()
# plot_pacf(df_raw['PM25'])
# plt.show()
# #################################################################################

# #####################################PM2.5与其他因素的相关系数####################
# print(df_raw.corr()['PM25'])

