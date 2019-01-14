#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd


class Time_Trim(object):
    def __init__(self, time_style, delta):
        ''' time_type: year, half_year, season'''
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

