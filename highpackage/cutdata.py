#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd
from package.timetrim import Time_Trim
import datetime as d


class CutData(object):

    def __init__(self, time_style, time_delta, constant):
        self.time_style = time_style
        self.time_delta = time_delta
        self.constant = constant

    def cutdata(self, data, re_data, com_data):

        gettb = Time_Trim(self.time_style, delta=1)
        re_data['发生日期'] = re_data['发生日期'].map(lambda x: gettb.get_time(x))
        data = data.loc[data['名称'].isin(com_data['名称']), :]
        re_data = re_data.merge(data, on=['名称', ], )  # 留下违约公司所有财报数据re_of_de
        re_data = re_data.loc[re_data['发生日期'] >= re_data['报告期'], :]  # 只留下发生日期之前的

        gettb = Time_Trim(self.time_style, delta=self.time_delta - 1)
        re_data['发生日期'] = re_data['发生日期'].map(lambda x: gettb.get_time(x))
        re_data.insert(0, 'target', (re_data['发生日期'] -
                                     re_data['报告期']).map(lambda x: x.days))
        if self.constant:
            re_data.loc[:, 'target'] = re_data.loc[:, 'target'].map(lambda x:
                                                                    1
                                                                    if x <= 0
                                                                    else 0)
        else:
            re_data.loc[:, 'target'] = re_data.loc[:, 'target'].map(lambda x:
                                                                    1
                                                                    if x == 0
                                                                    else 0)
        re_data = re_data.loc[re_data['target'] >= 0, :]

        # fin_col = list(re_data)
        data = data.loc[~data['名称'].isin(re_data['名称']), :]

        # s_time = pd.to_datetime(d.date.today())  # 当前时间
        s_time = max(pd.to_datetime(data['报告期']))
        s_time = Time_Trim(self.time_style, self.time_delta-1).get_time(s_time)
        pre_data = data.loc[data['报告期'] >= s_time, :]
        data = data.loc[data['报告期'] < s_time, :]

        pre_data.insert(0, 'target', -2)
        data.insert(0, 'target', -1)
        data.dropna(inplace=True)
        return re_data, pre_data, data
