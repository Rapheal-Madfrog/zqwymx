# -*- Coding:utf-8 -*-

from package.dropout import DROPOUT
from highpackage.dataloader import DataLoader
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# path = r'../data/'
data = DataLoader(2003, 2017, path='../../data/',
                  whether_plt=False).loader(ss=False)
com_data = pd.read_excel('../../data/comp_feature/产业类发债企业行业分类0910.xlsx',
                         sheet_name='正确的行业分类')
com_data = com_data.loc[:, ['名称', '二级分类']]
data = com_data.merge(data, on='名称', how='inner')
data.index = range(len(data))
drop_out = DROPOUT(model='gauss', g_alpha=3)
drop_out_list = data.columns[3:-1]
drop_out_list = [i for i in drop_out_list if '亿元' not in i]
for k in range(1):
    col_ = np.random.choice(drop_out_list, len(drop_out_list), replace=False,)
    for j in col_:
        data = drop_out.drop_out(data, j)
excelname = '去异年度统计'
excel_writer = pd.ExcelWriter(f'./{excelname}.xlsx')
data.groupby('报告期').median().to_excel(excel_writer, 'median')
data.groupby('报告期').mean().to_excel(excel_writer, 'mean')
excel_writer.save()
