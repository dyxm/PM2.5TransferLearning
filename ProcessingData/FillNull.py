# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: find the breakpoint and fill it

import pandas as pd
import math
import datetime

# 用前一天的同个时间段填补缺失值(只填补缺失间隔小于等于24小时的缺失段)
# 将pm2.5只小于等于0的记录用前一条记录替换
sites = ['261630033', '261630001', '060376012', '060371103', '060374004', '060376012']
year = ['2016', '2017']
for s in sites:
    for y in year:
        df_all = pd.read_csv('../DataSet/Processed/MergedData/88502/' + s + '_' + y + '.csv', dtype=str)
        df_all['Date Time'] = pd.to_datetime(df_all['Date Time'])
        print(s, y, 'v1')
        for i in range(1, len(df_all)):
            delta_seconds = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).seconds)
            # print(delta_h)
            if delta_seconds > 3600:
                # print(i, len(df_all), int(delta_seconds / 3600) - 1)
                df_prior = df_all.loc[:i-1]
                df_last = df_all.loc[i:]
                for h in range(int(delta_seconds / 3600) - 1):
                    df_prior_time = df_prior.loc[len(df_prior) - 1]['Date Time']
                    loc = i - 24 + h
                    if loc < 0:
                        loc = 0
                    df_temp = df_prior.loc[loc]
                    df_temp['Date Time'] = df_prior_time + datetime.timedelta(hours=1)
                    df_prior = df_prior.append(df_temp)
                    df_prior.index = [x for x in range(len(df_prior))]
                df_all = df_prior.append(df_last)
                df_all.index = [x for x in range(len(df_all))]
            elif float(df_all.loc[i]['PM25']) <= 0:
                # 用前一条记录替换
                if i > 0:
                    df_all['PM25'].loc[i] = df_all.loc[i - 1]['PM25']
                else:
                    df_all['PM25'].loc[i] = math.fabs(df_all.loc[i]['PM25'])
        df_all['Month'] = [x.month for x in df_all['Date Time']]
        df_all['Day'] = [x.day for x in df_all['Date Time']]
        df_all['Hour'] = [x.hour for x in df_all['Date Time']]
        df_all.to_csv('../DataSet/Processed/Train/' + s + '_' + y + '_v1.csv', index=False)

# 填补全部缺失段
for s in sites:
    for y in year:
        df_all = pd.read_csv('../DataSet/Processed/Train/' + s + '_' + y + '_v1.csv', dtype=str)
        df_all['Date Time'] = pd.to_datetime(df_all['Date Time'])
        print(s, y, 'v2')
        for i in range(1, len(df_all)):
            delta_hours = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).seconds / 3600)
            delta_days = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).days)
            if delta_days > 0:
                delta_hours += delta_days * 24
            if delta_hours > 1:
                df_prior = df_all.loc[:i-1]
                df_last = df_all.loc[i:]
                for h in range(delta_hours - 1):
                    df_prior_time = df_prior.loc[len(df_prior) - 1]['Date Time']
                    loc = i - delta_hours + h + 1
                    if loc < 0:
                        loc = 0
                    df_temp = df_prior.loc[loc]
                    # print(i, delta_hours, df_temp['Date Time'])
                    df_temp['Date Time'] = df_prior_time + datetime.timedelta(hours=1)
                    # print(df_temp['Date Time'])
                    df_prior = df_prior.append(df_temp)
                    df_prior.index = [x for x in range(len(df_prior))]
                df_all = df_prior.append(df_last)
                df_all.index = [x for x in range(len(df_all))]
                print(len(df_all))
        # df_all.pop('Date Time')
        df_all['Month'] = [x.month for x in df_all['Date Time']]
        df_all['Day'] = [x.day for x in df_all['Date Time']]
        df_all['Hour'] = [x.hour for x in df_all['Date Time']]
        df_all.to_csv('../DataSet/Processed/Train/' + s + '_' + y + '_v2.csv', index=False)


# 测试
# df_all = pd.read_csv('../DataSet/Processed/Train/060376012_2016_v2.csv')
# print(df_all[df_all['PM25'] <= 0])
# month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
# index = 0
# for m in month:
#     for d in range(1, month[m] + 1):
#         for h in range(24):
#             if not (df_all.loc[index]['Month'] == m and df_all.loc[index]['Day'] == d and df_all.loc[index]['Hour'] == h):
#                 print(index)
#             index += 1
# print(index)

# df_all['Date Time'] = pd.to_datetime(df_all['Date Time'])
# # # print(df_all.loc[3683]['Date Time'])
# # # print(df_all.loc[3682]['Date Time'])
# # print(int((df_all.loc[3683]['Date Time'] - df_all.loc[3682]['Date Time']).days))
# for i in range(1, len(df_all)):
#     delta_hours = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).seconds / 3600)
#     delta_days = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).days)
#     # print(delta_hours)
#     if delta_days > 0:
#         delta_hours += delta_days * 24
#     if delta_hours > 1:
#         print(i, len(df_all), delta_hours - 1)