#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd
from package.timetrim import Time_Trim
import datetime as d


class CutData(object):

    def __init__(self, time_style, time_delta, constant):
        '''
        :param str time_style: 'year', 'half_year', 'season', 外层函数定义了
        :param int time_delta: 预测间隔, 外层函数定义了
        :param bool constant: 预测是否连续, 外层函数定义了
        '''
        self.time_style = time_style
        self.time_delta = time_delta
        self.constant = constant

    def cutdata(self, data, re_data, com_data):

        # 定义时间跨度函数
        gettb = Time_Trim(self.time_style, delta=1)
        # 截断违约时间到上一个单位时间节点
        re_data['发生日期'] = re_data['发生日期'].map(lambda x: gettb.get_time(x))
        # 筛选有公司信息的财报数据
        data = data.loc[data['名称'].isin(com_data['名称'])]
        # 违约公司拼接财报数据
        re_data = re_data.merge(data, on=['名称', ], )
        # 违约公司只留下违约发生日期之前的
        re_data = re_data.loc[re_data['发生日期'] >= re_data['报告期'], :]

        # 定义时间跨度函数
        # 如果time_delta是1, 则保持时间不变
        # 如果time_delta大于1, 则往前截止(time_delta-1)个单位时间
        gettb = Time_Trim(self.time_style, delta=self.time_delta - 1)
        # 截断时间
        re_data['发生日期'] = re_data['发生日期'].map(lambda x: gettb.get_time(x))
        # 插入一列, (发生日期-报告期)的天数
        re_data.insert(0, 'target', (re_data['发生日期'] -
                                     re_data['报告期']).map(lambda x: x.days))
        if self.constant:
            # 如果时间连续, 则(发生日期-报告期)的天数<=0的都作为违约样本
            re_data.loc[:, 'target'] = re_data.loc[:, 'target'].map(lambda x:
                                                                    1
                                                                    if x <= 0
                                                                    else 0)
        else:
            # 如果时间不连续, 则只有(发生日期-报告期)的天数==0的作为违约样本
            re_data.loc[:, 'target'] = re_data.loc[:, 'target'].map(lambda x:
                                                                    1
                                                                    if x == 0
                                                                    else 0)
        ### 举个例子 ###
        # 2018-2-2违约, time_delta是2, 所以最终阶段日期是2017-09-30
        # 如果是连续的, 三季报四季报都作为违约样本.
        # 如果是不连续的, 只有三季报是违约样本.
        re_data = re_data.loc[re_data['target'] >= 0]

        # fin_col = list(re_data)
        # 没有违约公司的财报数据
        data = data.loc[~data['名称'].isin(re_data['名称']), :]

        # s_time = pd.to_datetime(d.date.today())  # 当前时间
        # 报告期里面最新的日期
        s_time = max(pd.to_datetime(data['报告期']))
        # 把最新日期截止, 比如2018年三季报, 截止到二季报(self.time_delta==2)
        s_time = Time_Trim(self.time_style, self.time_delta-1).get_time(s_time)
        # 拆分未违约财报数据
        pre_data = data.loc[data['报告期'] >= s_time]
        data = data.loc[data['报告期'] < s_time]

        pre_data.insert(0, 'target', -2)
        data.insert(0, 'target', -1)
        data.dropna(inplace=True)
        return re_data, pre_data, data
