#!/user/bin/env python
#!-*-coding:utf-8 -*-

from package.onehot import OneHot


class ComOneHot(object):
    def __init__(self, com_data, com_list=['名称', '是否交通', '企业性质', '是否上市', '二级分类',]):
        self.com_data = com_data
        # self.com_list = com_list    # '名称', '是否交通', '企业性质', '是否上市', '二级分类', '一级分类',
        self.com_data = self.com_data[com_list]
        self.com_data = OneHot('str', ).get_onehot(self.com_data, '企业性质')
        self.com_data = OneHot('str', ).get_onehot(self.com_data, '是否上市')
        self.com_data = OneHot('str', ).get_onehot(self.com_data, '二级分类')

    def coh(self, df):
        # self.com_data = OneHot(self.com_data,'所属省市').one_hot_str(replace=False)
        # self.com_data = OneHot(self.com_data,'注册资本(万元)').one_hot_int(number=3)

        df = df.merge(self.com_data, on='名称', )
        return df
