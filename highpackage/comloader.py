#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd


class ComLoader(object):
    def __init__(self, industry_ss=False,
                 sheet_name=['产业类企业', '交通运输'],
                 path='./data/comp_feature/产业类发债企业行业分类0910.xlsx'):
        '''
        :param bool industry_ss: 是否人为挑选一些行业做精简
        :param list(str) sheet_name: 这个不用改, 原表里面的 sheet_name
        :param str path: 原表的相对路径
        '''
        self.industry_ss = industry_ss
        self.path = path
        self.sheet = sheet_name

    def loader(self):
        # 读取数据
        indestry = pd.read_excel(self.path, sheet_name=self.sheet[0])
        indestry.insert(1, '是否交通', 0)
        transport = pd.read_excel(self.path, sheet_name=self.sheet[1])
        transport.insert(1, '是否交通', 1)
        # 人为筛选指标
        list_keep = ['名称', '是否交通', '最新评级', '企业性质', '是否上市', '所属省市',
                     '一级分类', '二级分类']
        all_com = pd.concat([indestry, transport], ignore_index=True, sort=False)[list_keep]
        del indestry, transport

        # 人为精简企业性质
        list_company_property = ['民营企业', '地方国有企业', '中央国有企业']
        all_com['企业性质'] = all_com['企业性质'].map(lambda x: x if x in list_company_property else '其他')

        # 人为精简行业
        if self.industry_ss:
            list_industry_property = ['物流快递', '石油天然气', '零售', '节能服务',
                                      '建筑业', '文化传媒', '电气', '信息技术',
                                      '日用品']
            all_com['二级分类'] = all_com['二级分类'].map(lambda x: x if x in list_industry_property else '其他')
        else:
            pass

        return all_com

