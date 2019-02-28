#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd


class Time_Trim(object):
    def __init__(self, time_style, delta):
        '''
        :param str time_style: 时间类型
        :param int delta: 时间单位位移间隔
        demo:
        d = pd.to_datetime('2018-02-02')
        c = Time_Trim('year', 0)
        c.get_time(d)
        output: Timestamp('2018-12-31 00:00:00')
        '''
        self.time_style = time_style
        self.delta = delta

    def get_time(self, date,):
        if self.time_style == 'year':
            aaa = date - pd.tseries.offsets.DateOffset(years=1 * self.delta)
            return aaa + pd.tseries.offsets.DateOffset(years=1, months=1 - aaa.month, days=- aaa.day)

        elif self.time_style == 'season':
            aaa = date - pd.tseries.offsets.DateOffset(months=3 * self.delta)
            return aaa + pd.tseries.offsets.DateOffset(months=3 - ((aaa.month - 1) % 3), days=-aaa.day)

        elif self.time_style == 'half_year':
            aaa = date - pd.tseries.offsets.DateOffset(months=6 * self.delta)
            return aaa + pd.tseries.offsets.DateOffset(months=6 - ((aaa.month - 1) % 6), days=-aaa.day)

