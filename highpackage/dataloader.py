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
        '''
        :param int begin_of_year: 开始年份, 外层函数定义了
        :param int end_of_year: 结束年份, 外层函数定义了
        :param str path: 只需要提供财报数据***根目录***
        :param str time_style: 'year', 'half_year', 'season', 外层函数定义了
        :param bool whether_plt: 是否画分布图
        '''
        self.path = path
        self.boy = begin_of_year
        self.eoy = end_of_year
        self.time_style = time_style
        self.whether_plt = whether_plt

    def loader(self, ss=True):
        '''
        :param bool ss: 在季报和半年报的情况下, 人为指定一些指标, 除以相应的月份, 得到一个平均值
        :return: 返回的是财报数据的dataframe
        '''
        # 定义空表
        data = pd.DataFrame()
        for year in range(self.boy, self.eoy+1, ):
            # 打印一下进程
            if year % 3 == 0 or year == self.boy:
                print(f'is concating {year} year, {year - self.boy + 1}/{self.eoy - self.boy + 1}')
            try:
                # 根据time_style指定详细目录
                if self.time_style == 'year':
                    path = self.path + f'y/{year}y.xlsx'
                elif self.time_style != 'year':
                    path = self.path + f'a/{year}a.xlsx'
                # 末端有万德
                data_ = pd.read_excel(path)[:-2]

                # 假如是年报数据
                if self.time_style == 'year':
                    # 删掉一些指标(删掉所有含有Ebitda)
                    data_.drop(['是否经过审计', '审计意见', ] +
                               [i for i in data_.columns if i.find('E') != -1],
                               axis=1, inplace=True)
                    # data_.drop(['是否经过审计', '审计意见', ],
                    #            axis=1, inplace=True)
                    # 这个时候的18年还没有年报, 用季报代替, 做点处理
                    # ps: 个人认为这个处理有待商榷
                    if year == 2018:
                        data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']] = \
                            data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']].apply(
                                lambda x: x * 12 / data_['报告期'].dt.month)
                # 如果数据用的不是年报,而是季报和半年报
                elif self.time_style != 'year':
                    # 删掉一些指标(删掉所有含有Ebitda)
                    # 还额外删掉了获悉倍数, 这个指标其实非常有用, 但是无奈, 缺的太多了
                    data_.drop(['是否经过审计', '审计意见', '获息倍数'] +
                               [i for i in data_.columns if i.find('E') != -1],
                               axis=1, inplace=True)
                    # 如果是半年报, 只要6月和12月
                    if self.time_style == 'half_year':
                        data_ = data_.loc[data_['报告期'].dt.month % 6 == 0, :]
                    else:
                        pass
                    # 某些指标取月度平均
                    if ss:
                        data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']] = \
                            data_.loc[:, ['主营业务收入(亿元)', '主营业务利润(亿元)', '净利润(亿元)']].apply(
                                lambda x: x / data_['报告期'].dt.month)
                    else:
                        pass
                # 拼接表
                data = pd.concat([data, data_])
            # 如果该路径的年份没有指定文件, 打印该年份
            except:
                print(f'no {year}')
                pass
        del data_
        print(f'finish concat data_{self.time_style}')

        # 打印指标和缺失的个数, 按照缺失值个数从小到大排列
        print(sorted(list(zip(list(data), data.isnull().sum(0))), key=lambda x: x[1], reverse=False))
        # 每个样本, 缺少6个及以上特征的, 就不要了
        data.dropna(thresh=data.shape[1] - 6, inplace=True)
        # 每行的缺失数量
        data['n_loss'] = data.isnull().sum(1)
        data.reset_index(drop=True, inplace=True)
        # data.index = range(len(data))

        if self.whether_plt:
            tend_scatter(data.iloc[:, 2:], data.shape[1]-3, 5, 5)
        return data
