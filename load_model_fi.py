#!/user/bin/env python
#!-*-coding:utf-8 -*-

from sklearn.externals import joblib
import pandas as pd

time_style = 'season'
time_delta = 2
begin_year = 2013
end_year = 2018
col_number = 20

col_name = list(pd.read_excel(r'.\project\year\1_2013-2017_end.xlsx'))
if time_style != 'year':
    col_name.remove('获息倍数')


def get_fp(model):
    try:
        return model.feature_importances_
    except:
        try:
            return model.coef_.ravel()
        except:
            return None


model_list = joblib.load(f'./project/{time_style}/{time_delta}_model_{begin_year}-{end_year}.m')
model_list = [model_tuple[0] for model_tuple in model_list if model_tuple != '']

length_fi = len(get_fp(model_list[0]))
col_name = col_name[::-1][:length_fi][::-1]

for step, model in enumerate(model_list):
    fp = get_fp(model)
    if fp is None:
        pass
    else:
        fp = map(lambda x: round(x, 4), fp)
        fp = dict(zip(col_name, fp))
        fp_sorted = sorted(fp.items(),
                           key=lambda x: abs(x[1]),
                           reverse=True)
        print(fp_sorted[:col_number])
        print('\n \n')
