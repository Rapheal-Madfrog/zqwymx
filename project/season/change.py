#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd

df = pd.read_excel('./2_2013-2018_end.xlsx')
df.to_csv('./2_2013-2018_end.csv', encoding='utf-8', index=False)
