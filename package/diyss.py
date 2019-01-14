#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np


def DiySs(frame, weight_dict, quantile=0.2):
    for i in frame.columns:
        if frame[i].min() >= 0:
            frame.loc[:, i] = np.log1p(frame[i])
            weight_dict[i] = 'log1p'
        else:
            mean_ = frame[i][(frame[i] >= frame[i].quantile(quantile)) &
                             (frame[i] <= frame[i].quantile(1-quantile))].mean()
            std_ = frame[i][(frame[i] >= frame[i].quantile(quantile)) &
                            (frame[i] <= frame[i].quantile(1-quantile))].std()
            weight_dict[i] = [mean_, std_]
            frame.loc[:, i] = (frame[i]-mean_)/std_
    return frame, weight_dict
