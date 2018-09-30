# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: preprocessing PM2.5 data

import pandas as pd

classes = ['88101', '88502']
year = ['2016', '2017']
columns = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude", "Date Local", "Time Local",
           "Sample Measurement"]
state_county_code = ['06037', '06075', '17031', '25025', '26163', '36061', '42003', '42101', '42057', '47083']
state = ['06', '17', '25', '26', '36', '42', '47']
for c in classes:
    for y in year:
        df_pm25 = pd.read_csv('../DataSet/Raw/PM25/hourly_' + c + '_' + y + '.csv', usecols=columns, dtype=str)
        df_pm25['State County Code'] = df_pm25['State Code'] + df_pm25['County Code']
        # df_pm25_new = df_pm25[df_pm25['State County Code'].isin(state_county_code)]
        df_pm25_new = df_pm25[df_pm25['State Code'].isin(state)]
        df_pm25_new.pop('State County Code')
        # df_pm25_new.to_csv('../DataSet/Processed/PM25/hourly_' + c + '_' + y + '.csv', index=False)
        df_pm25_new.to_csv('../DataSet/Processed/PM25/state_hourly_' + c + '_' + y + '.csv', index=False)
        print(c, y)
