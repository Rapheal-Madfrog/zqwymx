#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd


class OneHot(object):
    def __init__(self, model, replace=False, n_number=3):
        self.model = model
        self.replace = replace
        self.number = n_number

    def get_onehot(self, frame, col):
        if self.model == 'str':
            return self.one_hot_str(frame=frame, col=col)
        else:
            return self.one_hot_int(frame, col)

    def one_hot_str(self, frame, col):
        if self.replace:
            a_ = frame.pop(col)
        else:
            a_ = frame[col]
        a_.fillna('miss', inplace=True)
        a_ = pd.get_dummies(a_, prefix=a_.name)
        frame = pd.concat([frame, a_], axis=1)
        del a_
        return frame

    def one_hot_int(self, frame, col):
        if self.replace:
            a_ = frame.pop(col)
        else:
            a_ = frame[col]
        a_ = pd.qcut(a_, self.number)
        col_name_ = [a_.name + '_' + str(i + 1) for i in range(self.number)]
        a_ = pd.get_dummies(a_)
        a_.columns = col_name_
        frame = pd.concat([frame, a_], axis=1)
        del a_, col_name_
        return frame
