# Created by Yuexiong Ding
# Date: 2018/8/29
# Description: preview data

import pandas as pd

# 查看原始数据
# df_pm25_88101 = pd.read_csv('../DataSet/Raw/PM25/hourly_88101_2016.csv', dtype=str)
# df_pm25_88101['State County Site Code'] = df_pm25_88101['State Code'] + df_pm25_88101['County Code'] + df_pm25_88101['Site Num']
# df_pm25_88101_groupby = df_pm25_88101.groupby('State County Site Code')
# print(set(df_pm25_88101['Parameter Code']))
#
# df_pm25 = pd.read_csv('../DataSet/Raw/PM25/hourly_88101_2016.csv', dtype=str)
# df_pm25['State County Site Code'] = df_pm25['State Code'] + df_pm25['County Code'] + df_pm25['Site Num']
# df_pm25_groupby = df_pm25.groupby('State County Site Code')
# print(set(df_pm25['Parameter Code']))
#
# df_press = pd.read_csv('../DataSet/Raw/Meteorology/hourly_PRESS_2016.csv', dtype=str)
# df_press['State County Site Code'] = df_press['State Code'] + df_press['County Code'] + df_press['Site Num']
# df_press_groupby = df_press.groupby('State County Site Code')
# print(set(df_press['Parameter Code']))
#
# df_rh_dp = pd.read_csv('../DataSet/Raw/Meteorology/hourly_RH_DP_2016.csv', dtype=str)
# df_rh_dp['State County Site Code'] = df_rh_dp['State Code'] + df_rh_dp['County Code'] + df_rh_dp['Site Num']
# df_rh_dp_groupby = df_rh_dp.groupby('State County Site Code')
# print(set(df_rh_dp['Parameter Code']))
#
# df_temp = pd.read_csv('../DataSet/Raw/Meteorology/hourly_TEMP_2016.csv', dtype=str)
# df_temp['State County Site Code'] = df_temp['State Code'] + df_temp['County Code'] + df_temp['Site Num']
# df_temp_groupby = df_temp.groupby('State County Site Code')
# print(set(df_temp['Parameter Code']))

# df_wind = pd.read_csv('../DataSet/Raw/Meteorology/hourly_WIND_2016.csv', dtype=str)
# df_wind_61103 = df_wind[df_wind['Parameter Code'] == '61103']
# df_wind_61104 = df_wind[df_wind['Parameter Code'] == '61104']
# df_wind['State County Site Code'] = df_wind['State Code'] + df_wind['County Code'] + df_wind['Site Num']
# df_wind_groupby = df_wind.groupby('State County Site Code')
# print(set(df_wind['Parameter Code']))

# 查看合并的数据
df_raw_88101 = pd.read_csv('../DataSet/Processed/MergedData/all_counties_88101_2016.csv', dtype=str)
df_raw_88101['State County Site Code'] = df_raw_88101['State Code'] + df_raw_88101['County Code'] + df_raw_88101['Site Num']
df_raw_88101_groupby = df_raw_88101.groupby('State County Site Code')
print(df_raw_88101_groupby.count())

df_raw_88502 = pd.read_csv('../DataSet/Processed/MergedData/all_counties_88502_2016.csv', dtype=str)
df_raw_88502['State County Site Code'] = df_raw_88502['State Code'] + df_raw_88502['County Code'] +df_raw_88502['Site Num']
df_raw_88502_groupby = df_raw_88502.groupby('State County Site Code')
print(df_raw_88502_groupby.count())
