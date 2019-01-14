#!/user/bin/env python
#!-*-coding:utf-8 -*-

from functools import reduce
import pandas as pd
from sklearn.preprocessing import StandardScaler
from highpackage.dataloader import DataLoader
from highpackage.comloader import ComLoader
from package.onehot import OneHot
from package.dropout import DROPOUT
from package.fillna import FillNa
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.externals import joblib
import zipfile
import os


def plt_season(com_name, keep_yjfl=True, mohu=True):
    #读取数据
    com_data = pd.read_excel('../../data/comp_feature/产业类发债企业行业分类0910.xlsx', sheet_name='金融类企业')
    list_keep = ['名称', '最新评级', '企业性质', '是否上市',
                 '一级分类', '二级分类']
    com_data = com_data[list_keep]
    list_company_property = ['民营企业', '地方国有企业', '中央国有企业']
    com_data['企业性质'] = com_data['企业性质'].map(lambda x: x if x in list_company_property else '其他')

    data = DataLoader(2013, 2018, path='../../data/', year_or_season='s', whether_plt=False).loader(ss=False)
    data.drop('n_loss', axis=1, inplace=True)
    index_remain = pd.Series([False] * data.shape[0])
    if mohu:
        index_remain = reduce(lambda x, y: x | y,
                              [data['名称'].map(lambda z: z.find(i) >= 0)
                               for i in com_name], index_remain)
    else:
        index_remain = reduce(lambda x, y: x | y,
                              [data['名称'].map(lambda z: z == i)
                               for i in com_name], index_remain)
    data_remain = data.loc[index_remain, :]
    data = data.loc[~index_remain, :]
    if True:
        list_remain = ['名称', '报告期',
                       '货币资金/短期债务', '净利润(亿元)', '筹资活动现金流(亿元)',
                       '主营业务利润率(%)', '主营业务收入增长率(%)', '投资活动现金流(亿元)',
                       '净资产回报率(%)', '总资产报酬率(%)', '流动比率']
        cols_a = list(data)
        for index_, col in enumerate(list_remain):
            cols_a.insert(index_, cols_a.pop(cols_a.index(col)))
        excel_path = f'./result/{com_name}.xlsx'
        excelWriter = pd.ExcelWriter(excel_path)
        data_remain.sort_values(['名称', '报告期'])[cols_a]\
            .to_excel(excelWriter, '最原始的数据', index=False)
    #drop异常值
    diy_drop = DROPOUT(model='gauss', g_alpha=3)
    for k in range(1):
        col_ = np.random.choice(data.columns[2:], len(data.columns[2:]), replace=False, )
        for j in col_:
            data = diy_drop.drop_out(data, j)
    #drop缺失值
    data.dropna(inplace=True)
    #拼接
    data = pd.concat([data_remain, data],)
    del data_remain
    data.index = range(len(data))
    #join并对变量onehot
    data = pd.concat([data, pd.get_dummies(data['报告期'].dt.year, drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['报告期'].dt.month, drop_first=True)], axis=1)
    com_flist = ['名称', '最新评级', '一级分类', '企业性质', '是否上市', '二级分类']
    data = data.merge(com_data[com_flist], on='名称')
    com_flist_ = com_flist.copy()
    for o in ['名称', '最新评级', '一级分类', '二级分类']:
        com_flist_.remove(o)
    for i in com_flist_:
        data = OneHot('str', True).get_onehot(data, i)
    keep_yjfl = keep_yjfl                # 保留一级分类
    drop_length = 1-keep_yjfl
    drop_length += len(set(data['报告期'].dt.year))-1
    drop_length += len(set(data['报告期'].dt.month))-1
    for i in com_flist_:
        drop_length += len(set(com_data[i]))-1
    cols_a = list(data)
    cols_a.insert(2, cols_a.pop(cols_a.index('最新评级')))
    cols_a.insert(len(cols_a) - 1, cols_a.pop(cols_a.index('一级分类')))
    cols_a.insert(len(cols_a) - 1, cols_a.pop(cols_a.index('二级分类')))
    data = data[cols_a]
    data.iloc[:, 3:-2] = FillNa(frame=data.iloc[:, 3:-2], group_col=data['一级分类'], n_epoch=4)
    cols_a.insert(len(cols_a) - drop_length - keep_yjfl, cols_a.pop(cols_a.index('二级分类')))
    data = data[cols_a]
    data = data.iloc[:, :-1*drop_length]
    data.sort_values(['名称', '报告期'])
    sec_clf = data.iloc[0, :].loc['二级分类']

    print(f'目标公司:{sec_clf}',
          set(data.iloc[:sum(index_remain,), :].loc[:, '二级分类']),
          '总共有', len(set(com_data.loc[com_data['二级分类'] == sec_clf, '名称'])), '家',
          '有数据的有', len(set(data.loc[data['二级分类'] == sec_clf, '名称'])), '家',
          )
    del com_data, com_flist, com_flist_
    #标准化处理
    data_origin = data.iloc[:sum(index_remain,), :].copy()
    ss = StandardScaler()
    if keep_yjfl:
        for i in set(data['二级分类']):
            index_i = data['二级分类'] == i
            data.loc[index_i, list(data)[3:-1]] = ss.fit_transform(data.loc[index_i, list(data)[3:-1]])
    else:
        data.iloc[:, 3:] = ss.fit_transform(data.iloc[:, 3:])
    data = data.iloc[:sum(index_remain,), :]
    #保存数据
    if True:
        change_index = True
        if change_index:
            list_remain = ['名称', '报告期', '最新评级',
                           '货币资金/短期债务', '净利润(亿元)', '筹资活动现金流(亿元)',
                           '主营业务利润率(%)', '主营业务收入增长率(%)', '投资活动现金流(亿元)',
                           '净资产回报率(%)', '总资产报酬率(%)', '流动比率']
            cols_a = list(data)
            for index_, col in enumerate(list_remain):
                cols_a.insert(index_, cols_a.pop(cols_a.index(col)))
            data[cols_a].to_excel(excelWriter, '标准化数据', index=False)
            data[cols_a].stack().unstack(0).to_excel(excelWriter, sheet_name='竖向标准化数据', )
            data_origin[cols_a].to_excel(excelWriter, '原始数据', index=False)
            data_origin[cols_a].stack().unstack(0).to_excel(excelWriter, sheet_name='竖向原始数据', )
    else:
        pass
    #挑选画图指标
    list_remain = ['名称', '报告期', '最新评级',
                   '货币资金/短期债务', '净利润(亿元)', '筹资活动现金流(亿元)',
                   '主营业务利润率(%)', '主营业务收入增长率(%)', '投资活动现金流(亿元)',
                   '净资产回报率(%)', '总资产报酬率(%)', '流动比率']
    data, data_origin = data[list_remain], data_origin[list_remain]

    weight = [('货币资金/短期债务', 0.08466747934544402), ('净债务(亿元)', 0.015673135947545064),
              ('净利润(亿元)', 0.08119844386846073), ('筹资活动现金流(亿元)', 0.07392047950839306),
              ('主营业务利润率(%)', 0.07373536447871205), ('主营业务收入增长率(%)', 0.0681957815907612),
              ('货币资金/总债务', 0.059822816991053754), ('投资活动现金流(亿元)', 0.056517651957992715),
              ('净资产回报率(%)', 0.05488686308572023), ('总资产报酬率(%)', 0.036039995560997476),
              ('短期债务/总债务', 0.0350832996152875), ('经营活动现金流(亿元)', 0.0243060183484769),
              ('流动比率', 0.020910667354810814), ('货币资产(亿元)', 0.018929246296673925),
              ('带息债务/总投入资本', 0.013195889611687595), ('主营业务利润(亿元)', 0.012345481772175503),
              ('总资产(亿元)', 0.01071488982344452), ('净资产(亿元)', 0.009601482353160072),
              ('总债务(亿元)', 0.009112083719364693), ('主营业务收入(亿元)', 0.006851438201812423),
              ('速动比率', 0.005393658415696681), ('存货周转率', 0.00521645305170003),
              ('资产负债率', 0.0034618581489861853), ('带息债务(亿元)', 0.002858337636445684),
              ]
    weight = dict(weight)
    weight = [weight[i] for i in data.columns[3:]]
    weight = [j/sum(weight) for j in weight]
    data.loc[:, '加权平均'] = data.iloc[:, 3:].values.dot(np.array(weight).reshape(-1, 1))
    weight.append(1)
    #plt
    data.set_index('报告期', drop=False, inplace=True)

    name2index = {name: index+1 for index, name in enumerate(set(data['名称']))}

    plt.figure(figsize=(20, 10), dpi=100)
    for k in range(3, len(list(data))):
        if k == len(list(data))-1:
            i = k+2
        else:
            i = k+1
        ax = plt.subplot(4, 3, i-3)
        ax.plot(list(set(data.index)), [0] * len(list(set(data.index))),
                color=cm.get_cmap('Set1')(0), linewidth=1)
        for j in set(data['名称']):
            data.loc[data['名称'].isin([j]), :].iloc[:, k].plot(label=j,
                                                              color=cm.get_cmap('Set1')(name2index[j]))
        ax.legend(loc='best', prop={'size': 10})
        ax.set_title(data.columns[k]+':'+str(round(weight[k-3], 2)))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlabel('')
        # plt.suptitle(j)
    plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 400 #分辨率
    plt.tight_layout()
    plt_z_path = f'./result/plt_z_{com_name}.png'
    plt.savefig(plt_z_path)
    plt.show()

    plt_title = max(data.index)
    plt_title = f'{plt_title.year}年 1月-{plt_title.month}月'
    data = data.loc[data.index == max(data.index), :]
    data.index = range(len(data))
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(np.linspace(-0.5, len(list(data)) - 3.5, 10), [0] * 10,
             color=cm.get_cmap('Set1')(0), linewidth=1)
    for row_number in range(len(data)):
        plt.plot(range(len(list(data))-3), data.iloc[row_number, 3:].tolist(),
                 label=data['名称'][row_number],
                 color=cm.get_cmap('Set1')(name2index[data['名称'][row_number]]))
    plt.legend(loc='best', prop={'size': 15})
    plt.xticks(range(len(list(data))-3),
               list(data)[3:],
               rotation=20, fontsize=14)
    plt.xlim(-0.5, len(list(data))-3.5)
    plt.title(plt_title)
    plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 400 #分辨率
    plt.tight_layout()
    plt_h_path = f'./result/plt_h_{com_name}.png'
    plt.savefig(plt_h_path)
    plt.show()

    if True:
        data_origin.set_index('报告期', drop=False, inplace=True)
        data_origin = data_origin.loc[data_origin.index == max(data_origin.index), :]
        data.to_excel(excelWriter, '最后一期标准化数据', index=False)
        data_origin.to_excel(excelWriter, '最后一期原始数据', index=False)
        excelWriter.save()

    def zip_files(files, zip_name):
        new_zip = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for file in files:
            print('compressing', file)
            new_zip.write(file)
        new_zip.close()
        print('compressing finished')

    zip_path = f'./result/{com_name}.zip'
    zip_files([excel_path, plt_z_path, plt_h_path], zip_path)

    for file in [excel_path, plt_z_path, plt_h_path]:
        os.remove(file)

    return None


if __name__ == '__main__':
    com_name = ['河北省金融租赁', '江苏金融', ]
    plt_season(com_name, True, mohu=True)
