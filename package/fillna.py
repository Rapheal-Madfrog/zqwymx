#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import gc


def FillNa(frame, group_col, n_epoch=3):
    def return_index(aa, bb):
        j = 0
        cc = []
        for i in range(len(aa)):
            if aa[i] == False:
                cc.append(aa[i])
            else:
                cc.append(bb[j])
                j += 1
        return cc

    frame_col = frame.columns

    for l, comp in enumerate(set(group_col)):

        index_y = list(group_col == comp)
        full_col = []
        loss_col = {}

        for col in frame.columns:
            if frame.loc[index_y, col].isnull().sum() == 0:
                full_col.append(col)
            else:
                loss_col[col] = frame.loc[index_y, col].isnull().sum()

        loss_col = sorted(loss_col.items(), key=lambda x: x[1])
        loss_col = [i[0] for i in loss_col]

        index_dict = {}
        if len(full_col) == 0:
            index_dict[loss_col[0]] = frame.loc[index_y, loss_col[0]].isnull()
            index_dict[loss_col[0]].fillna(index_dict[loss_col[0]].median(), inplace=True)
            full_col.append(loss_col[0])
            loss_col = loss_col[1:]

        for epoch in range(n_epoch):

            if epoch == 0:
                for _, col in enumerate(loss_col):
                    if np.random.rand() > 0.8:
                        print(comp, f'{l}/{len(set(group_col))}', col, f'{_}/{len(loss_col)}')
                    index_l = list(frame.loc[index_y, col].isnull())
                    index_f = list(frame.loc[index_y, col].notnull())
                    index_l_ = return_index(index_y, index_l)
                    index_f_ = return_index(index_y, index_f)
                    index_dict[col] = (index_l_, index_f_)
                    rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1, max_features=0.9)
                    rfr.fit(frame.loc[index_f_, full_col], frame.loc[index_f_, col])
                    pre = rfr.predict(frame.loc[index_l_, full_col])
                    frame.loc[index_l_, col] = pre
                    full_col.append(col)

            else:
                for col in index_dict:
                    index_l_ = index_dict[col][0]
                    index_f_ = index_dict[col][1]
                    rfr = RandomForestRegressor(n_estimators=20, n_jobs=-1, max_features=0.6)
                    rfr.fit(frame.loc[index_f_, full_col], frame.loc[index_f_, col])
                    pre = rfr.predict(frame.loc[index_l_, full_col])
                    frame.loc[index_l_, col] = pre

    gc.collect()
    return frame[frame_col]