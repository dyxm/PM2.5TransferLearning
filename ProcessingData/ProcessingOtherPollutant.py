# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: preprocessing other pollutant data

import pandas as pd
import numpy as np

classes = ['CO', 'NO2', 'O3', 'PM10', 'SO2']
year = ['2016', '2017']
columns = ["State Code", "County Code", "Site Num", "Parameter Code", "POC", "Latitude", "Longitude", "Date Local",
           "Time Local", "Sample Measurement"]
state_county_code = ['06037', '06075', '17031', '25025', '26163', '36061', '42003', '42101', '42057', '47083']
state_code = ['06', '06', '17', '25', '26', '36', '42', '42', '42', '47']
for c in classes:
    for y in year:
        df_raw = pd.read_csv('../DataSet/Raw/Others/hourly_' + c + '_' + y + '.csv', usecols=columns, dtype=str)
        df_new = df_raw[df_raw['State Code'].isin(state_code)]
        param_code = set(df_new['Parameter Code'])
        for pc in param_code:
            df_new_pc = df_new[df_new['Parameter Code'] == pc]
            df_new_pc['Sample Measurement'] = np.array(df_new_pc['Sample Measurement']).astype(float)
            df_new_pc['groupby_col'] = df_new_pc['State Code'] + '#' + df_new_pc['Date Local'] + '#' + df_new_pc['Time Local']
            df_gb_pc = df_new_pc.groupby('groupby_col')['Sample Measurement'].mean()
            df_final = pd.DataFrame({
                'Index': df_gb_pc.index,
                c: df_gb_pc.values
            })
            df_final.to_csv('../DataSet/Processed/Pollutant/hourly_' + c + '_' + pc + '_' + y + '.csv', index=False)
            print(c, pc, y)
