#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd
from highpackage.dataloader import DataLoader


class ReLoader(object):
    def __init__(self, begin_of_year, end_of_year, load_data=True,
                 data_path='./data/',
                 time_style='year', whether_plt=True,
                 re_path='./data/report of defaulted.xlsx',
                 list_remain=['发生日期', '名称'],):
        self.load_data = load_data
        self.data_path = data_path
        self.boy = begin_of_year
        self.eoy = end_of_year
        self.time_style = time_style
        self.whether_plt = whether_plt
        self.re_path = re_path
        self.list_remain = list_remain

    def loader(self,):

        re_data = pd.read_excel(self.re_path)[:-2].loc[:, self.list_remain]
        print('违约记录数:', len(re_data['名称']))
        re_data = re_data.groupby(['名称'], as_index=False).apply(lambda x:
                                                                x.sort_values(['发生日期']).iloc[0])
        print('违约公司数:', len(re_data['名称']))
        if self.load_data:
            data = DataLoader(self.boy, self.eoy, self.data_path,
                              self.time_style, self.whether_plt).loader(ss=False)
            data['whetherin'] = data['名称'].isin(re_data['名称'])  # 总表中的违约公司
            loss_com_number = len(set(data.loc[data['whetherin'], '名称']))
            print('总表里的违约公司数:', loss_com_number)
            del loss_com_number
            re_data['whetherin'] = re_data['名称'].isin(data['名称'])*1  # 在总表中有记录的违约公司
            print('缺失数:', len(re_data['名称']) - sum(re_data['whetherin']))

            loss_re_name = re_data.loc[re_data['whetherin'] == 0, '名称']
            print('缺失的公司名如下:\n---------------------------\n', set(loss_re_name))
            del loss_re_name
            re_data = re_data.loc[re_data['whetherin'] == 1, :]

            re_data = re_data.iloc[:, :-1]
            data = data.iloc[:, :-1]
            return data, re_data
        else:
            return re_data

