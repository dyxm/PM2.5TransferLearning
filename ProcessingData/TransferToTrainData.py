# Created by Yuexiong Ding
# Date: 2018/8/30
# Description: transfer all data to training data

# import pandas as pd
# pd.set_option('display.max_columns', None)
# df_all = pd.read_csv('../DataSet/Processed/Train/261630033_2016_v1.csv', dtype=str)
# # df_all['Day'] = [x[-2:] for x in df_all['Date']]
# # df_all['Month'] = [x[-5: -3] for x in df_all['Date']]
# # print(df_all['Month'])
# df_all = pd.get_dummies(df_all, columns=['Hour', 'Day', 'Month'])
# print(df_all)
