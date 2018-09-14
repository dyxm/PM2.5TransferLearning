# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: preprocessing meteorology data

import pandas as pd

classes = ['PRESS', 'RH_DP', 'TEMP', 'WIND']
year = ['2016', '2017']
columns = ["State Code", "County Code", "Site Num", "Parameter Code", "POC", "Latitude", "Longitude", "Date Local",
           "Time Local", "Sample Measurement"]
state_county_code = ['06037', '06075', '17031', '25025', '26163', '36061', '42003', '42101', '42057', '47083']
for c in classes:
    for y in year:
        df_pm25 = pd.read_csv('../DataSet/Raw/Meteorology/hourly_' + c + '_' + y + '.csv', usecols=columns, dtype=str)
        df_pm25['State County Code'] = df_pm25['State Code'] + df_pm25['County Code']
        df_pm25_new = df_pm25[df_pm25['State County Code'].isin(state_county_code)]
        df_pm25_new.pop('State County Code')
        param_code = set(df_pm25_new['Parameter Code'])
        for pc in param_code:
            df_pm25_new_pc = df_pm25_new[df_pm25_new['Parameter Code'] == pc]
            df_pm25_new_pc.pop('Parameter Code')
            df_pm25_new_pc.to_csv('../DataSet/Processed/Meteorology/hourly_' + c + '_' + pc + '_' + y + '.csv', index=False)
            print(c, pc, y)
