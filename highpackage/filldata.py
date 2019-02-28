#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
import pandas as pd
from package.dropout import DROPOUT
from package.diyss import DiySs
from package.onehot import OneHot
from package.fillna import FillNa
from highpackage.cutdata import CutData
from sklearn.externals import joblib
import gc


def rechange(dataframe, col, weight):
    if col in weight.keys():
        if weight[col] == 'log1p':
            return np.exp(dataframe[col]) - 1
        else:
            return dataframe[col] * weight[col][1] + weight[col][0]
    else:
        return dataframe


class FillData(object):
    def __init__(self, time_style, time_delta,
                 a_save_path=None, w_save_path=None, if_save=True, constant=True):
        '''
        :param str time_style: 'year', 'half_year', 'season', 外层函数定义了
        :param int time_delta: 预测间隔, 外层函数定义了
        :param str a_save_path: 文件存放路径
        :param str w_save_path: 参数存放路径
        :param bool if_save: 是否保存文件和参数
        :param bool constant: 预测是否连续, 和time_delta参数一起使用
        '''
        self.time_style = time_style
        self.time_delta = time_delta
        self.asp = a_save_path
        self.wsp = w_save_path
        self.if_save = if_save
        self.constant = constant

    def filldata(self, data, re_data, com_data, ):
        # 数据处理, 得到的是违约财报数据, 要预测的财报数据, 确定为尚未违约的财报数据
        re_data, pre_data, data = CutData(self.time_style, self.time_delta,
                                          self.constant
                                          ).cutdata(data, re_data, com_data)

        fin_col = list(re_data)
        # 去除异常值
        # 未违约财报数据拼接公司信息
        data = data.merge(com_data.loc[:, ['名称', '二级分类']], on='名称')   # type: pd.DataFrame
        drop_out = DROPOUT(model='gauss', g_alpha=2.5)
        drop_out_list = data.columns[3:-2]
        drop_out_list = [i for i in drop_out_list if '亿元' not in i]
        data = data.sort_values(['名称', '二级分类'])
        for k in range(1):
            col_ = np.random.choice(drop_out_list, len(drop_out_list), replace=False,)
            for j in col_:
                # data = drop_out.drop_out(data, j)
                data = data.groupby(['报告期', '二级分类'], as_index=False).\
                    apply(lambda x: drop_out.drop_out(x, j))
        data = data.iloc[:, :-1]

        a = pd.concat([re_data, pre_data, data], sort=False)[fin_col]
        del data, re_data, pre_data, fin_col
        # 加特征
        a['chouzi_less_0'] = [1 if i < 0 else 0 for i in a['筹资活动现金流(亿元)']]
        a['jinglirun_less_0'] = [1 if i < 0 else 0 for i in a['净利润(亿元)']]
        a['liudongbilv_less_2'] = [1 if i < 2 else 0 for i in a['流动比率']]
        a['sudongbilv_less_1'] = [1 if i < 1 else 0 for i in a['速动比率']]
        a['dz/zz_greater_0.5'] = [1 if i > 0.5 else 0 for i in a['短期债务/总债务']]
        a['hb/dz_less_0.2'] = [1 if i < 0.2 else 0 for i in a['货币资金/短期债务']]
        gc.collect()
        # 自定义标准化
        weight_dict = {}
        a.iloc[:, 4:-7], weight_dict = DiySs(frame=a.iloc[:, 4:-7], weight_dict=weight_dict)
        # 公司属性哑变量
        drop_length = 1  # 一级分类不用onehot
        a = pd.concat([a, pd.get_dummies(a['报告期'].dt.year)], axis=1)
        drop_length += len(set(a['报告期'].dt.year))
        if self.time_style != 'year':
            a = pd.concat([a, pd.get_dummies(a['报告期'].dt.month)], axis=1)
            drop_length += len(set(a['报告期'].dt.month))
        com_flist = ['名称', '是否交通', '一级分类', '企业性质', '是否上市', '最新评级']
        a = a.merge(com_data[com_flist], on='名称')
        com_flist_ = com_flist.copy()

        for o in ['名称', '一级分类', '最新评级']:
            com_flist_.remove(o)
        for i in com_flist_:
            drop_length += len(set(a[i]))
            a = OneHot('str', replace=True).get_onehot(a, i)
        cols_a = list(a)
        cols_a.insert(4, cols_a.pop(cols_a.index('最新评级')))
        cols_a.insert(len(cols_a), cols_a.pop(cols_a.index('一级分类')))
        a = a[cols_a]
        # 填充空白值
        a.iloc[:, 5:-1] = FillNa(frame=a.iloc[:, 5:-1], group_col=a['一级分类'], n_epoch=4)
        # 补充变量
        a = a.iloc[:, :drop_length*-1]
        a.insert(len(list(a)), '净利润/带息债务',
                 rechange(a, '净利润(亿元)', weight_dict) /
                 ((rechange(a, '带息债务(亿元)', weight_dict)) + 0.01)
                 )
        a.insert(len(list(a)), '总现金流(亿元)',
                 rechange(a, '经营活动现金流(亿元)', weight_dict) +
                 rechange(a, '投资活动现金流(亿元)', weight_dict) +
                 rechange(a, '筹资活动现金流(亿元)', weight_dict)
                 )
        a.insert(len(list(a)), '非筹资活动现金流(亿元)',
                 rechange(a, '经营活动现金流(亿元)', weight_dict) +
                 rechange(a, '投资活动现金流(亿元)', weight_dict)
                 )
        a.insert(len(list(a)), '可供投资活动现金流(亿元)',
                 rechange(a, '经营活动现金流(亿元)', weight_dict) +
                 rechange(a, '筹资活动现金流(亿元)', weight_dict)
                 )
        a.insert(len(list(a)), '经营活动现金流/短期债务',
                 (rechange(a, '经营活动现金流(亿元)', weight_dict) /
                 rechange(a, '总债务(亿元)', weight_dict)) /
                 rechange(a, '短期债务/总债务', weight_dict)
                 )

        diyss_col = ['净利润/带息债务',
                     '总现金流(亿元)',
                     '非筹资活动现金流(亿元)',
                     '可供投资活动现金流(亿元)',
                     '经营活动现金流/短期债务'
                     ]
        a.loc[:, diyss_col], weight_dict = DiySs(a[diyss_col], weight_dict)

        # 加区域排名
        area_data = com_data[['名称', '所属省市']]
        a = a.merge(area_data, on='名称')
        # a = a.sort(['报告期', '所属省市'], ascending=[1, 1])
        a['rank_by_p'] = a.groupby(['报告期', '所属省市'])['主营业务收入(亿元)'].rank(method='dense').\
            transform(lambda x: pd.qcut(x, 4,
                                        labels=['rank_by_p' + str(i) for i in [1, 2, 3, 4]]))
        a = pd.concat([a, pd.get_dummies(a['rank_by_p'])], axis=1)
        a.drop(['所属省市', 'rank_by_p'], axis=1, inplace=True)

        # 离散化规模值
        # a = OneHot('int', n_number=6).get_onehot(a, '总资产(亿元)')
        a_ = rechange(a, '总资产(亿元)', weight_dict).copy()
        a_ = pd.cut(a_, [0, 100, 400, 700, 1000, 1500], labels=False)
        a_ = [-1 if i != i else i for i in a_]
        a_ = pd.get_dummies(a_, prefix='总资产(亿元)')
        a = pd.concat([a, a_], axis=1)

        # 保存
        if self.if_save:
            a.to_csv(self.asp, index=False, encoding='utf-8')
            joblib.dump(weight_dict, self.wsp)

        return a, weight_dict
