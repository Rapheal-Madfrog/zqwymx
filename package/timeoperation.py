#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np


class TimeOp(object):
    def __init__(self, frame, gb, col, time_col,):
        assert isinstance(col, str)
        self.frame = frame
        self.gb = gb
        self.col = col
        self.time_col = time_col

    def mms(self, ):
        ''' 年份维度上 0-1 归一化 '''
        max_ = dict(self.frame[[self.col]].groupby(by=self.gb, )[self.col].max())
        min_ = dict(self.frame[[self.col]].groupby(by=self.gb, )[self.col].min())
        max_ = self.frame[self.gb].map(lambda x: max_[x])
        min_ = self.frame[self.gb].map(lambda x: min_[x])
        result = (self.frame[self.col].values - min_) / (max_ - min_)
        return [round(i, 2) for i in result]

    def moving(self, wins=3, weight=[0.55, 0.3, 0.15]):
        '''wins是窗口数,weight是权重,这两个必须对应'''
        frame = self.frame[[self.col]+[self.gb]+[self.time_col]]\
            .sort_values([self.gb, self.time_col])
        for i in range(wins):
            if i == 0:
                list_ = weight[i] * frame[self.col].values
            else:
                list_ += weight[i] * frame.groupby(self.gb)[self.col].rolling(i+1).mean().values
        return list_

    def bodonglv(self, wins):
        frame = self.frame[[self.col]+[self.groupby]+[self.time_col]]\
            .sort_values([self.groupby, self.time_col])
        std_ = frame.groupby(self.groupby)[self.col].rolling(wins).std().values
        mean_ = frame.groupby(self.groupby)[self.col].rolling(wins).mean().values
        return [round(std_[i]/mean_[i], 2) for i in range(len(std_))]

    def tongbi(self, diff):
        frame = self.frame[[self.col]+[self.groupby]+[self.time_col]]\
            .sort_values([self.groupby, self.time_col])
        diff_ = frame.groupby(self.groupby)[self.col].diff(diff).values
        ori_ = frame[self.col].values
        return [np.nan]*diff+[round(diff_[i]/abs(ori_[i-diff]), 2) for i in range(diff, frame.shape[0])]
