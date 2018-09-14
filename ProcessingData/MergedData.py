# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: Merged pm2.5 and meteorology data

import pandas as pd

year = ['2016', '2017']
classes = ['88101', '88502']
meteorology = ['PRESS_64101', 'RH_DP_62201', 'TEMP_62101', 'WIND_61103', 'WIND_61104']
# 将全部数据合并
# for c in classes:
#     for y in year:
#         df_pm25 = pd.read_csv('../DataSet/Processed/PM25/hourly_' + c + '_' + y + '.csv', dtype=str)
#         for m in meteorology:
#             df_meteo = pd.read_csv('../DataSet/Processed/Meteorology/hourly_' + m + '_' + y + '.csv', dtype=str)
#             # df_meteo.pop('Latitude')
#             # df_meteo.pop('Longitude')
#             df_meteo.pop('POC')
#             df_meteo.pop('Site Num')
#             df_pm25 = pd.merge(df_pm25, df_meteo, how='inner',
#                                on=['State Code', 'County Code', 'Latitude', 'Longitude', 'Date Local', 'Time Local'])
#             print(c, y, m)
#         df_pm25.to_csv('../DataSet/Processed/MergedData/all_counties_' + c + '_' + y + '.csv', index=False)

# 将合并好的数据按县划分
for c in classes:
    for y in year:
        df_all = pd.read_csv('../DataSet/Processed/MergedData/all_counties_' + c + '_' + y + '.csv', dtype=str)
        df_all['Month'] = [x[-5: -3] for x in df_all['Date Local']]
        df_all['Day'] = [x[-2:] for x in df_all['Date Local']]
        df_all['Hour'] = [x[: 2] for x in df_all.pop('Time Local')]
        # df_all['Hour'] = df_all.pop('Time Local')
        df_all['site'] = df_all['State Code'] + df_all['County Code'] + df_all['Site Num']
        df_all.pop('State Code')
        df_all.pop('County Code')
        df_all.pop('Site Num')
        df_all.pop('POC')
        df_all.pop('Date Local')
        site = set(df_all['site'])
        for s in site:
            df_site = df_all[df_all['site'] == s]
            df_site.pop('site')
            print(len(df_site))
            df_site.to_csv('../DataSet/Processed/MergedData/' + c + '/' + s + '_' + y + '.csv', index=False)
