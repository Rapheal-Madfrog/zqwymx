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
# from sklearn.externals import joblib
import zipfile
import os


def plt_season(com_name, keep_yjfl=True, mohu=True):

    # 读取公司行业数据
    com_data = ComLoader(path='../../data/comp_feature/产业类发债企业行业分类0910.xlsx').loader()
    # 读取公司财务数据
    data = DataLoader(2015, 2018, path='../../data/', time_style='season',
                      whether_plt=False).loader(ss=False)
    # 去除不要的列
    data.drop('n_loss', axis=1, inplace=True)
    # 构建一个bool列, 用来做索引
    index_remain = pd.Series([False] * data.shape[0])
    if mohu:
        # 匹配关键字
        index_remain = reduce(lambda x, y: x | y,
                              [data['名称'].map(lambda z: z.find(i) >= 0)
                               for i in com_name], index_remain)
    else:
        # 必须一样才行
        index_remain = reduce(lambda x, y: x | y,
                              [data['名称'].map(lambda z: z == i)
                               for i in com_name], index_remain)
    # 符合要求的样本数量
    all_select_length = sum(index_remain)
    # 用bool来索引要留下来的部分
    data_remain = data.loc[index_remain, :]
    # 用bool反向索引
    data = data.loc[~index_remain, :]
    if True:
        # 定义人为特征列表
        list_remain = ['名称', '报告期',
                       '资产负债率', '短期债务/总债务',
                       '总债务(亿元)', '流动比率', '货币资金/短期债务',

                       '投资活动现金流(亿元)', '筹资活动现金流(亿元)',

                       '净利润(亿元)', '净资产回报率(%)'
                       ]
        # 得到特征总和
        cols_a = list(data)
        # 改变列名里的债务改成负债
        list_remain = [i.replace('债务', '负债') for i in list_remain]
        cols_a = [i.replace('债务', '负债') for i in cols_a]
        # 更新列名
        data_remain.columns = cols_a
        data.columns = cols_a
        # 改变顺序, 把人为挑选的顺序放到前面来
        for index_, col in enumerate(list_remain):
            cols_a.insert(index_, cols_a.pop(cols_a.index(col)))

        # 定义输出表的路径
        excel_path = f'./result/{com_name}.xlsx'
        # 初始化一个excelWriter
        excelWriter = pd.ExcelWriter(excel_path)

        # 输出第一波
        data_remain.sort_values(['名称', '报告期'])[cols_a]\
            .to_excel(excelWriter, '最原始的数据', index=False)

    # drop那个反向选择的data中的异常值, 让他们正常点
    diy_drop = DROPOUT(model='gauss', g_alpha=3)
    for k in range(1):
        col_ = np.random.choice(data.columns[2:], len(data.columns[2:]), replace=False, )
        # 对每个列执行这个函数 # (这里是否需要优化, 只对百分比类的特征drop, 或者限定一个阀值)
        for j in col_:
            data = diy_drop.drop_out(data, j)

    # drop缺失值
    data.dropna(inplace=True)

    # 拼接
    data = pd.concat([data_remain, data],)
    del data_remain
    # 重置index
    data.index = range(len(data))

    # join并对变量onehot
    # 如果是True, drop_length = 1, 即保留
    # 如果是False, drop_length = 2, 即删除
    keep_yjfl = keep_yjfl  # 保留二级分类
    drop_length = 2 - keep_yjfl

    # 表中添加年份的独热码, 增加对应的drop_length
    data = pd.concat([data, pd.get_dummies(data['报告期'].dt.year, drop_first=True)], axis=1)
    drop_length += len(set(data['报告期'].dt.year)) - 1

    # 表中添加月份的独热码, 增加对应的drop_length
    data = pd.concat([data, pd.get_dummies(data['报告期'].dt.month, drop_first=True)], axis=1)
    drop_length += len(set(data['报告期'].dt.month)) - 1

    # 挑选公司行业信息表的特征
    com_flist = ['名称', '最新评级', '是否交通', '一级分类', '企业性质', '是否上市', '二级分类']
    # merge两张表
    data = data.merge(com_data[com_flist], on='名称')
    # 拷贝一列备用
    com_flist_ = com_flist.copy()
    # 备用表去除这些特征
    for o in ['名称', '最新评级', '一级分类', '二级分类']:
        com_flist_.remove(o)

    # 对备用表用剩下的特征做独热, 增加对应的drop_length
    for i in com_flist_:
        data = OneHot('str', True).get_onehot(data, i)
        drop_length += len(set(com_data[i])) - 1

    # 取得表的列名
    cols_a = list(data)
    # 改变列名顺序到指定位置
    cols_a.insert(2, cols_a.pop(cols_a.index('最新评级')))
    cols_a.insert(len(cols_a) - 1, cols_a.pop(cols_a.index('一级分类')))
    cols_a.insert(len(cols_a) - 1, cols_a.pop(cols_a.index('二级分类')))
    data = data[cols_a]

    # 根据一级分类拆分样本, 分别填充空白值
    data.iloc[:, 3:-2] = FillNa(frame=data.iloc[:, 3:-2], group_col=data['一级分类'], n_epoch=4)

    # 把二级分类放到一个固定的位置
    cols_a.insert(len(cols_a) - drop_length - keep_yjfl, cols_a.pop(cols_a.index('二级分类')))
    # 改变顺序
    data = data[cols_a]
    # 截断独热特征, 根据keep_yjfl来判断是否保留二级分类
    data = data.iloc[:, :-1*drop_length]   # type: pd.DataFrame

    # 所选择的公司的二级分类去重结果
    all_sec_clf = list(set(data.iloc[:sum(index_remain,), :].loc[:, '二级分类']))
    # 从data中挑选出和 所选择公司的二级分类一样的公司
    index_remain = pd.Series([False] * data.shape[0])
    index_remain = reduce(lambda x, y: x | y,
                          [data['二级分类'].map(lambda z: z == i)
                           for i in all_sec_clf], index_remain)
    # 有点搞不懂这个if是啥意思了??????
    if keep_yjfl:
        # bool索引
        data = data.loc[index_remain, :]

    # 自定义一些特征
    # data['经营活动现金流/短期债务'] = data['经营活动现金流(亿元)']/data['总债务(亿元)']/data['短期债务/总债务']
    data.insert(4, '经营活动现金流/短期负债',
                (data['经营活动现金流(亿元)'] / data['总负债(亿元)'] / (data['短期负债/总负债'] / 100)).tolist())
    # data['可供投资现金流(亿元)'] = data['筹资活动现金流(亿元)'] + data['经营活动现金流(亿元)']
    data.insert(4, '可供投资现金流(亿元)',
                (data['筹资活动现金流(亿元)'] + data['经营活动现金流(亿元)']).tolist())
    # lkj说不要这些指标
    data.drop(['存货周转率', '速动比率', '带息负债/总投入资本'], axis=1, inplace=True)
    # 优哉游哉排个序
    data.sort_values(['名称', '报告期'])
    # 得到目标公司的二级分类
    sec_clf = data.iloc[0, :].loc['二级分类']

    print(f'目标公司:{sec_clf}',
          all_sec_clf,
          '总共有', len(set(com_data.loc[com_data['二级分类'] == sec_clf, '名称'])), '家',
          '有数据的有', len(set(data.loc[data['二级分类'] == sec_clf, '名称'])), '家',
          )
    del com_data, com_flist, com_flist_

    # 标准化处理
    data_origin = data.iloc[:all_select_length, :].copy()

    # 如果保留行业标签
    if keep_yjfl:
        # 每一个行业分开来标准化
        for i in set(all_sec_clf):
            ss = StandardScaler()
            index_i = data['二级分类'] == i
            data.loc[index_i, list(data)[3:-1]] = ss.fit_transform(data.loc[index_i, list(data)[3:-1]])
    else:
        # 不分行业, 一锅子标准化
        ss = StandardScaler()
        data.iloc[:, 3:] = ss.fit_transform(data.iloc[:, 3:])
    # data截取到需要的公司的位置
    data = data.iloc[:all_select_length, :]

    # xgb跑出来的feature_importance_
    # 为什么这里要是元祖, 说出来都是泪 TuT
    weight = [('短期债务/总债务', 0.0569), ('主营业务收入(亿元)', 0.0483), ('流动比率', 0.0456),
              ('筹资活动现金流(亿元)', 0.0451), ('主营业务收入增长率(%)', 0.0451), ('存货周转率', 0.0424),
              ('非筹资活动现金流(亿元)', 0.0392), ('净资产回报率(%)', 0.037), ('资产负债率', 0.0365),
              ('经营活动现金流/短期债务', 0.0359), ('货币资金/短期债务', 0.0354), ('速动比率', 0.0343),
              ('净资产(亿元)', 0.0338), ('投资活动现金流(亿元)', 0.0338), ('带息债务/总投入资本', 0.0322),
              ('货币资金/总债务', 0.0317), ('货币资产(亿元)', 0.0311), ('总现金流(亿元)', 0.03),
              ('净利润/带息债务', 0.0295), ('可供投资现金流(亿元)', 0.0284), ('总资产报酬率(%)', 0.0274),
              ('净利润(亿元)', 0.0252), ('经营活动现金流(亿元)', 0.0231), ('主营业务利润率(%)', 0.022),
              ('净债务(亿元)', 0.0209), ('主营业务利润(亿元)', 0.0204), ('总债务(亿元)', 0.0193),
              ('企业性质_民营企业', 0.0193), ('总资产(亿元)', 0.0156), ('带息债务(亿元)', 0.0139),
              ('二级分类_其他', 0.0059), ('企业性质_地方国有企业', 0.0054), ('rank_by_p2', 0.0048),
              ('企业性质_其他', 0.0032), ('企业性质_中央国有企业', 0.0027), ('whether_h/d', 0.0021),
              ('rank_by_p3', 0.0021), ('是否上市_否', 0.0021), ('二级分类_零售', 0.0021), ('rank_by_p4', 0.0016),
              ('总资产(亿元)_1.0', 0.0016), ('是否上市_是', 0.0016), ('whether_chouzi', 0.0011),
              ('二级分类_节能服务', 0.0011), ('whether_su', 0.0005), ('总资产(亿元)_4.0', 0.0005),
              ('二级分类_信息技术', 0.0005), ('二级分类_文化传媒', 0.0005), ('二级分类_日用品', 0.0005),
              ('二级分类_物流快递', 0.0005), ('n_loss', 0.0), ('whether_jin', 0.0), ('whether_liu', 0.0),
              ('whether_d/a', 0.0), ('rank_by_p1', 0.0), ('总资产(亿元)_-1.0', 0.0), ('总资产(亿元)_0.0', 0.0),
              ('总资产(亿元)_2.0', 0.0), ('总资产(亿元)_3.0', 0.0), ('是否交通', 0.0), ('二级分类_建筑业', 0.0),
              ('二级分类_电气', 0.0), ('二级分类_石油天然气', 0.0)]
    # 改成嵌套列表
    weight = [list(i) for i in weight]
    # 改负债
    for sl in weight:
        temp = sl[0]
        temp = temp.replace('债务', '负债')
        sl[0] = temp

    # 保存数据
    if True:
        # 这是个啥玩意儿
        change_index = True
        # 好像就是这个玩意儿
        if change_index:
            list_remain = ['名称', '报告期', '最新评级',
                           '资产负债率', '短期债务/总债务',
                           '总债务(亿元)', '流动比率', '货币资金/短期债务',
                           '经营活动现金流/短期债务',
                           '投资活动现金流(亿元)', '筹资活动现金流(亿元)',
                           '可供投资现金流(亿元)',
                           '净利润(亿元)', '净资产回报率(%)'
                           ]
            # 改负债
            list_remain = [i.replace('债务', '负债') for i in list_remain]

            # sorted_remain = [i[0] for i in weight if i[0] in list_remain]
            # list_remain = ['名称', '报告期', '最新评级'] + sorted_remain
            # del sorted_remain

            # 获取当前列名
            cols_a = list(data)
            # 为什么还要改顺序
            # 因为好像有自定义的新特征
            for index_, col in enumerate(list_remain):
                cols_a.insert(index_, cols_a.pop(cols_a.index(col)))
            # 第二波输出
            data[cols_a].to_excel(excelWriter, '标准化数据', index=False)
            data[cols_a].stack().unstack(0).to_excel(excelWriter, sheet_name='竖向标准化数据', )
            data_origin[cols_a].to_excel(excelWriter, '原始数据', index=False)
            data_origin[cols_a].stack().unstack(0).to_excel(excelWriter, sheet_name='竖向原始数据', )
    else:
        pass

    # 挑选画图指标
    # list_remain = ['名称', '报告期', '最新评级',
    #                '货币资金/短期债务', '净利润(亿元)', '筹资活动现金流(亿元)',
    #                '主营业务利润率(%)', '主营业务收入增长率(%)', '投资活动现金流(亿元)',
    #                '净资产回报率(%)', '总资产报酬率(%)', '流动比率',
    #                '经营活动现金流/短期债务', '可供投资现金流(亿元)']
    data, data_origin = data[list_remain], data_origin[list_remain]

    # 构建字典
    weight = dict(weight)
    # 这两个的逻辑好像是越小越好, 所以加权的时候加符号
    weight = [-weight[i] if (i == '短期债务/总债务' or i == '资产负债率')
              else weight[i]
              for i in list_remain[3:]]
    # 加权平均计算
    weight = [j/sum(np.abs(weight)) for j in weight]
    # 把加权平均插入到表里
    data.loc[:, '加权平均'] = data.iloc[:, 3:].values.dot(np.array(weight).reshape(-1, 1))
    # 给加权平均末端加个1, 为了画最后一张加权平均的图
    weight.append(1)
    # 同上, 为了画图
    list_remain.append('加权平均')

    # plt
    # 把报告期设置成index
    data.set_index('报告期', drop=False, inplace=True)
    # 这里应该是为了固定颜色, 空出0是为了给中线
    name2index = {name: index+1 for index, name in enumerate(set(data['名称']))}

    # 定义一个画布
    plt.figure(figsize=(20, 10), dpi=250)
    # 因为这是一个四行三列的图, 所以r=4, n=3
    # 因为第一个指标在表里的index是3, 所以c=3
    n = 3
    r = 4
    c = 3
    # 行列转换函数
    f_transpose = lambda x: n + (x + (c - n)) % c * r + (x - n) // c
    # 对列循环
    for k in range(n, len(list(data))):
        if k == len(list(data))-1:
            i = k+1
        else:
            i = k+1
        # 新增一个小画布
        ax = plt.subplot(r, c, i-n)
        # 画 x 轴 (y=0) 中线
        ax.plot(list(set(data.index)), [0] * len(list(set(data.index))),
                color=cm.get_cmap('Set1')(0), linewidth=1)
        # 得到转换的位置
        k = f_transpose(k)
        # 对公司循环
        for j in set(data['名称']):
            data.loc[data['名称'].isin([j]), :].iloc[:, k]\
                .plot(label=j, color=cm.get_cmap('Set1')(name2index[j]))
        # 标签位置
        ax.legend(loc='best', prop={'size': 10}, framealpha=0.5)
        # 设置title
        ax.set_title(list_remain[k]+': '+str(round(abs(weight[k-n]), 2)))
        # 隐藏右边框
        ax.spines['right'].set_color('none')
        # 隐藏上边框
        ax.spines['top'].set_color('none')
        # 去除x方向的标签
        ax.set_xlabel('')
        # plt.suptitle(j)
    plt.rcParams['savefig.dpi'] = 250   # 图片像素
    # plt.rcParams['figure.dpi'] = 400 #分辨率
    plt.tight_layout()   # 画面平铺
    plt_z_path = f'./result/plt_z_{com_name}.png'   # 图片路径
    plt.savefig(plt_z_path)   # 保存图片
    plt.show()   # show

    # 最新的时间
    last_time = max(data.index)
    # 如果最新时间的数量少于公司数量, 则存在公司没有最新数据, 迭代的方向是当前时间的次最新时间
    while sum(data.index == last_time) < len(com_name):
        data.drop(last_time, axis=0, inplace=True)
        last_time = max(data.index)
    # 标题
    plt_title = f'{last_time.year}年 1月-{last_time.month}月'
    # 截取最新时间的数据
    data = data.loc[data.index == max(data.index), :]
    # 重置index
    data.index = range(len(data))

    # bar
    bar = False
    if bar:
        # 定义一组直方图的总宽度, 和数量
        total_width, n = 0.8, len(com_name)
        # 一组直方图的平均长度
        every_width = total_width/n

    # 定义一个画布
    plt.figure(figsize=(20, 10), dpi=100)
    # 画中线
    plt.plot(np.linspace(-0.5,
                         len(list(data)) - 3.5 + (total_width if bar else 0),
                         10),
             [0] * 10,
             color=cm.get_cmap('Set1')(0), linewidth=1)

    # 对行循环
    for row_number in range(len(data)):
        if bar:
            # 因为前三列不是特征, 所以-3
            x = list(range(len(list(data))-3))
            # x = [0,1,2,3,4]
            # x = [0.2, ...]
            x = [i + every_width/2 + row_number*every_width for i in x]
            # 画图, 每一个的x坐标不同
            # 每个柱子都是以x为中心
            plt.bar(x, data.iloc[row_number, 3:].tolist(),
                    width=every_width,
                    label=data['名称'][row_number],
                    fc=cm.get_cmap('Set1')(name2index[data['名称'][row_number]]))
        else:
            # 简单粗暴折线图
            plt.plot(range(len(list(data))-3), data.iloc[row_number, 3:].tolist(),
                     label=data['名称'][row_number],
                     color=cm.get_cmap('Set1')(name2index[data['名称'][row_number]]))
    # 标签
    plt.legend(loc='best', prop={'size': 15}, framealpha=0.5)
    # x轴坐标
    if bar:
        plt.xticks([i+total_width/2 for i in range(len(list(data)) - 3)],
                   list(data)[3:],
                   rotation=20, fontsize=14)
    else:
        plt.xticks(range(len(list(data))-3),
                   list(data)[3:],
                   rotation=20, fontsize=14)
    # x轴范围
    plt.xlim(-0.5, len(list(data))-3.5+(total_width if bar else 0))
    # title
    plt.title(plt_title)
    plt.rcParams['savefig.dpi'] = 200   # 图片像素
    # plt.rcParams['figure.dpi'] = 400 #分辨率
    plt.tight_layout()
    plt_h_path = f'./result/plt_h_{com_name}.png'
    plt.savefig(plt_h_path)
    plt.show()

    if True:
        # 保存最新一期原始数据
        data_origin.set_index('报告期', drop=False, inplace=True)
        data_origin = data_origin.loc[data_origin.index == last_time, :]
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
    # 打包
    zip_path = f'./result/{com_name}.zip'
    zip_files([excel_path, plt_z_path, plt_h_path], zip_path)
    # 删除
    for file in [excel_path, plt_z_path, plt_h_path]:
        os.remove(file)

    return None


if __name__ == '__main__':
    com_name = ['步步高', '华地国际', ]
    plt_season(com_name, True, mohu=True)
