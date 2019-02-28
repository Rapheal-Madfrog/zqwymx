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
        '''
        :param int begin_of_year: 开始的年份, 定义在外层函数了
        :param int end_of_year: 结束的年份, 定义在外层函数了
        :param bool load_data: 是否读取财报数据
        :param str data_path: 财务数据的路径, 不用改
        :param str time_style: 'year', 'half_year', 'season', 定义在外层函数了
        :param bool whether_plt: 是否画一些关于财务数据的分布图
        :param str re_path: 违约数据的路径
        :param list(str) list_remain: 违约数据的指标精简, 只要发生日期和公司名就够了
        '''
        self.load_data = load_data
        self.data_path = data_path
        self.boy = begin_of_year
        self.eoy = end_of_year
        self.time_style = time_style
        self.whether_plt = whether_plt
        self.re_path = re_path
        self.list_remain = list_remain

    def loader(self,):
        # 读取违约数据并精简指标
        # :-2 是因为数据末端会有 数据来源: 万德
        re_data = pd.read_excel(self.re_path)[:-2].loc[:, self.list_remain]
        print('违约记录数:', len(re_data['名称']))
        # 挑选出每家公司的第一次违约数据
        re_data = re_data.groupby(['名称'], as_index=False).apply(lambda x:
                                                                x.sort_values(['发生日期']).iloc[0])
        print('违约公司数:', len(re_data['名称']))
        # 读取财务数据
        if self.load_data:
            # 读取财报数据
            data = DataLoader(self.boy, self.eoy, self.data_path,
                              self.time_style, self.whether_plt).loader(ss=False)
            # 财报数据中的违约公司标记为1
            data['whetherin'] = data['名称'].isin(re_data['名称'])
            # 违约公司数
            loss_com_number = len(set(data.loc[data['whetherin'], '名称']))
            print('总表里的违约公司数:', loss_com_number)
            del loss_com_number
            # 违约公司有财报数据的标记为1
            re_data['whetherin'] = re_data['名称'].isin(data['名称'])*1
            print('缺失数:', len(re_data['名称']) - sum(re_data['whetherin']))

            loss_re_name = re_data.loc[re_data['whetherin'] == 0, '名称']
            print('缺失的公司名如下:\n---------------------------\n', set(loss_re_name))
            del loss_re_name
            # 只取有财报数据的违约公司, 最后一列whetherin不要了
            re_data = re_data.loc[re_data['whetherin'] == 1].iloc[:, :-1]

            data = data.iloc[:, :-1]
            return data, re_data
        else:
            return re_data

