# Created by Yuexiong Ding
# Date: 2018/9/30
# Description: 处理原始数据
import pandas as pd
import numpy as np
from MyModule import data


def extract_data_by_state(extract_conf):
    """
    提取出相应州的所有信息
    :param extract_conf:
    :return:
    """
    # print('提取相应州的信息...')
    names = []
    for n in extract_conf['names']:
        for y in extract_conf['years']:
            df_raw = pd.read_csv(extract_conf['data_read_root_path'] + 'hourly_' + n + '_' + y + '.csv',
                                 usecols=extract_conf['columns'], dtype=str)
            df_new = df_raw[df_raw['State Code'].isin(extract_conf['states'])]
            param_code = set(df_new['Parameter Code'])
            for pc in param_code:
                df_new_pc = df_new[df_new['Parameter Code'] == pc]
                if extract_conf['is_group_by']:
                    df_new_pc['Sample Measurement'] = np.array(df_new_pc['Sample Measurement']).astype(float)
                    df_new_pc['groupby_col'] = df_new_pc['State Code'] + '#' + df_new_pc['Date Local'] + '#' + df_new_pc[
                        'Time Local']
                    df_gb_pc = df_new_pc.groupby('groupby_col')['Sample Measurement'].mean()
                    df_final = pd.DataFrame({
                        'Index': df_gb_pc.index,
                        n: df_gb_pc.values
                    })
                else:
                    df_new_pc.pop('Parameter Code')
                    df_final = df_new_pc
                file_name = 'state_level_hourly_' + n + '_' + pc + '_' + y + '.csv'
                df_final.to_csv(extract_conf['data_save_root_path'] + file_name, index=False)
                names.append(file_name)
    return names


def extract_data_by_county(extract_conf):
    """
    提取相应县的所有信息
    :param extract_conf:
    :return:
    """
    names = []
    for n in extract_conf['names']:
        for y in extract_conf['years']:
            df_raw = pd.read_csv(extract_conf['data_read_root_path'] + 'hourly_' + n + '_' + y + '.csv',
                                 usecols=extract_conf['columns'], dtype=str)
            df_raw['State County Code'] = df_raw['State Code'] + df_raw['County Code']
            df_new = df_raw[df_raw['State County Code'].isin(extract_conf['state_county_code'])]
            df_new.pop('State County Code')
            param_code = set(df_new['Parameter Code'])
            for pc in param_code:
                df_new_pc = df_new[df_new['Parameter Code'] == pc]
                df_new_pc.pop('Parameter Code')
                file_name = 'county_level_hourly_' + n + '_' + pc + '_' + y + '.csv'
                df_new.to_csv(extract_conf['data_save_root_path'] + file_name, index=False)
                names.append(file_name)


# def merge(merge_conf):
#     df_data = pd.read_csv(merge_conf['data_read_root_path'] + merge_conf['file_names'][0], dtype=str)
#     for i in range(1, len(merge_conf['file_names'])):
#         df_temp = pd.read_csv(merge_conf['data_read_root_path'] + merge_conf['file_names'][i], dtype=str)
#         # df_temp.pop('POC')
#         # df_temp.pop('Site Num')
#         df_merge = pd.merge(df_data, df_temp, how=merge_conf['how'], on=merge_conf['on'])
