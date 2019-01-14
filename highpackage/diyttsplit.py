#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DiyttSplit(object):
    def __init__(self, re_data_1, simple, time_style,
                 simple_list='default',):
        self.re_1 = re_data_1
        self.simple = simple
        self.time_style = time_style
        self.simple_list = ['货币资金/短期债务', '净利润(亿元)', '主营业务利润率(%)',
                            '主营业务收入增长率(%)', '净资产回报率(%)', '总资产报酬率(%)',
                            '流动比率', ] if simple_list == 'default' else simple_list

    def diyttsplit(self, re_0, per_re_0, data, num_data, test_size=0.3, random_state=None):

        if self.simple:
            self.re_1, re_0, data = self.re_1[self.simple_list], \
                                    re_0[self.simple_list], \
                                    data[self.simple_list]
        else:
            pass

        if self.time_style != 'year':
            col_all = list(re_0)
            col_all = [i for i in col_all if i.find('现金流(亿元)') == -1]
            self.re_1, re_0, data = self.re_1[col_all], re_0[col_all], data[col_all]
        else:
            pass

        index_0 = np.random.permutation(len(re_0))[:int(len(re_0)*per_re_0)]
        index_a = np.random.permutation(len(data))[:num_data]
        X = pd.concat([self.re_1, re_0.iloc[index_0, :], data.iloc[index_a, :]], sort=False)
        y = np.r_[[1]*len(self.re_1), [0]*(len(index_0)), [0]*(len(index_a))]
        if test_size == 0:
            x1, y1 = shuffle(X, y, random_state=random_state)
            x2, y2 = None, None
        else:
            x1, x2, y1, y2 = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return x1, x2, y1, y2
