# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: Merged pm2.5 and meteorology data

import pandas as pd

year = ['2016', '2017']
classes = ['88101', '88502']
meteorology = ['PRESS_64101', 'RH_DP_62201', 'TEMP_62101', 'WIND_61103', 'WIND_61104']
other_pollutant = ['CO_42101', 'NO2_42602', 'O3_44201', 'PM10_81102', 'SO2_42401']
# 将全部数据合并
for c in classes:
    for y in year:
        df_pm25 = pd.read_csv('../DataSet/Processed/PM25/hourly_' + c + '_' + y + '.csv', dtype=str)
        # 合并天气状况
        for m in meteorology:
            df_meteo = pd.read_csv('../DataSet/Processed/Meteorology/hourly_' + m + '_' + y + '.csv', dtype=str)
            df_meteo.pop('POC')
            df_meteo.pop('Site Num')
            df_pm25 = pd.merge(df_pm25, df_meteo, how='inner',
                               on=['State Code', 'County Code', 'Latitude', 'Longitude', 'Date Local', 'Time Local'])
            print(c, y, m)
        df_pm25['Index'] = df_pm25['State Code'] + '#' + df_pm25['Date Local'] + '#' + df_pm25['Time Local']
        for p in other_pollutant:
            df_pollutant = pd.read_csv('../DataSet/Processed/Pollutant/hourly_' + p + '_' + y + '.csv', dtype=str)
            df_pm25 = pd.merge(df_pm25, df_pollutant, how='inner', on=['Index'])
            print(c, y, p)
        df_pm25.pop('Index')
        df_pm25['Date Time'] = df_pm25.pop('Date Local') + ' ' + df_pm25.pop('Time Local')
        df_pm25 = df_pm25.sort_values(by="Date Time")
        df_pm25.to_csv('../DataSet/Processed/MergedData/all_counties_' + c + '_' + y + '.csv', index=False)

# 将合并好的数据按县划分
for c in classes:
    for y in year:
        df_all = pd.read_csv('../DataSet/Processed/MergedData/all_counties_' + c + '_' + y + '.csv', dtype=str)
        # df_all['Date Time'] = df_all.pop('Date Local') + ' ' + df_all.pop('Time Local')
        df_all['site'] = df_all['State Code'] + df_all['County Code'] + df_all['Site Num']
        df_all.pop('State Code')
        df_all.pop('County Code')
        df_all.pop('Site Num')
        df_all.pop('POC')
        site = set(df_all['site'])
        for s in site:
            df_site = df_all[df_all['site'] == s]
            df_site.pop('site')
            print(len(df_site))
            df_site.to_csv('../DataSet/Processed/MergedData/' + c + '/' + s + '_' + y + '.csv', index=False)
