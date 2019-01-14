# -*- Coding:utf-8 -*-

from scipy import stats
from highpackage.dataloader import DataLoader
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def drop_out(frame, col, ):
    old = 'old' + col
    new = 'new' + col
    frame.index = range(len(frame))
    frame[old] = len(frame)
    if len(frame) > 2:
        u_ = frame[col].mean()
        v_ = frame[col].std()
        cond_ = np.abs((frame[col] - u_) / v_) < 3
        index_ = np.where(frame[col] != frame[col], True,
                          np.where(cond_, True, False))
        frame = frame.loc[index_, :]
        frame[new] = len(frame)
        return frame
    else:
        frame[new] = len(frame)
        return frame


def abc(frame):
    frame['123'] = [len(frame)]*len(frame)
    return frame


def diy_mean(df: pd.DataFrame, weight):
    weight = weight.map(lambda x: x / sum(weight)).values.reshape(1, -1)

    # print('\n+++', df.iloc[0, :5], '---\n')
    df_new = df.fillna(0).copy()

    df_dict = {}

    df_dict['二级分类'] = [df_new['二级分类'].iloc[0]]
    df_dict['报告期'] = [df_new['报告期'].iloc[0]]
    df_dict['number'] = [df_new.iloc[0, -1]]

    end_col_index = list(df).index('获息倍数')
    for col in list(df_new.iloc[:, 3:end_col_index]):
        df_dict[col] = weight.dot(df_new[col].values.reshape(-1, 1)).reshape(1).tolist()

    df_new = pd.DataFrame(df_dict)
    return df_new


# path = r'../data/'
com_data = pd.read_excel('../../data/comp_feature/产业类发债企业行业分类0910.xlsx',
                         sheet_name=None)
com_data_new = pd.DataFrame()
for step, (key, value) in enumerate(com_data.items()):
    com_data_new = pd.concat([com_data_new, value.loc[:, ['名称', '二级分类']]],
                             ignore_index=True)
del com_data
com_data = com_data_new.groupby('名称').apply(lambda x: x.iloc[0, :])
del com_data_new

data = DataLoader(2003, 2017, path='../../data/',
                  whether_plt=False).loader(ss=False)
data = data.iloc[:, :-1]
data = com_data.merge(data, on='名称', how='inner')
data.index = range(len(data))
# data = data.groupby(['二级分类', '报告期']).apply(lambda x: abc(x))
# data.groupby(['二级分类', '报告期']).mean().to_csv('./年度行业统计.csv', encoding='utf_8_sig')

# drop_out_list = data.columns[3:-1]
# drop_out_list = [i for i in drop_out_list if '亿元' not in i]
drop_out_list = ['短期债务/总债务', '资产负债率', '存货周转率',
                 '流动比率', '总资产报酬率(%)', '主营业务收入增长率(%)',
                 '主营业务利润率(%)']
# for k in range(1):
col_ = np.random.choice(drop_out_list, len(drop_out_list), replace=False,)
for j in col_:
    data = data.groupby(['二级分类', '报告期']).apply(lambda x: drop_out(x, j))
data.index = range(len(data))
data = data    # type: pd.DataFrame

# data.groupby(['二级分类', '报告期'], as_index=False).mean().to_csv('./去异年度行业统计-.csv', encoding='utf_8_sig')

data_new = data.groupby(['二级分类', '报告期'], as_index=False).\
    apply(lambda x: diy_mean(x, x['总资产(亿元)']))

data_new.to_csv('./去异年度行业统计-.csv', encoding='utf_8_sig', index=False)
