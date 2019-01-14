#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


def tend_scatter(df, n_picture, n_r, n_c):
    for i in range(n_picture):
        ax = plt.subplot(n_r, n_c, i + 1)
        ax.scatter(range(len(df)), df.iloc[:, i].sort_values(), s=3)
        plt.title(df.columns[i])
    # plt.suptitle('散点趋势图')


class DataLoader(object):
    def __init__(self, begin_of_year, end_of_year,
                 path='../data/',
                 time_style='year', whether_plt=True):
        self.path = path
        self.boy = begin_of_year
        self.eoy = end_of_year
        self.time_style = time_style
        self.whether_plt = whether_plt

    def loader(self, ss=True):
        data = pd.DataFrame()
        for year in range(self.boy, self.eoy+1, ):
            if year % 3 == 0 or year == self.boy:
                print(f'is concating {year} year, {year - self.boy + 1}/{self.eoy - self.boy + 1}')
            try:
                if self.time_style == 'year':
                    path = self.path + f'y/{year}y.xlsx'
                elif self.time_style != 'year':
                    path = self.path + f'a/{year}a.xlsx'
                data_ = pd.read_excel(path)[:-2]

                if self.time_style == 'year':
                    data_.drop(['是否经过审计', '审计意见', ] +
                               [i for i in data_.columns if i.find('E') != -1],
                               axis=1, inplace=True)
                    # data_.drop(['是否经过审计', '审计意见', ],
                    #            axis=1, inplace=True)
                    if year == 2018:
                        data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']] = \
                            data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']].apply(
                                lambda x: x * 12 / data_['报告期'].dt.month)
                elif self.time_style != 'year':
                    data_.drop(['是否经过审计', '审计意见', '获息倍数'] +
                               [i for i in data_.columns if i.find('E') != -1],
                               axis=1, inplace=True)
                    if self.time_style == 'half_year':
                        data_ = data_.loc[data_['报告期'].dt.month % 6 == 0, :]
                    else:
                        pass
                    if ss:
                        data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']] = \
                            data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']].apply(
                                lambda x: x / data_['报告期'].dt.month)
                    else:
                        pass
                data = pd.concat([data, data_])
            except:
                print(f'no {year}')
                pass
        del data_
        print(f'finish concat data_{self.time_style}')

        print(sorted(list(zip(list(data), data.isnull().sum(0))), key=lambda x: x[1], reverse=False))
        data.dropna(thresh=data.shape[1] - 6, inplace=True)
        data['n_loss'] = data.isnull().sum(1)
        data.index = range(len(data))

        if self.whether_plt:
            tend_scatter(data.iloc[:, 2:], data.shape[1]-3, 5, 5)
        return data
