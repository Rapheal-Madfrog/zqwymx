#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from scipy import stats


class DROPOUT(object):
    def __init__(self, model='Confidence interval', t_alpha=0.95, g_alpha=2, IQR_rate=1.5, head_tail=0.95):
        ''' modle: 'Confidence interval','gauss','box','head_tail',
            default = 'Confidence interval'
        '''
        assert isinstance(model, str)
        self.model = model
        self.t_alpha = t_alpha
        self.g_alpha = g_alpha
        self.IQR_rate = IQR_rate
        self.head_tail = head_tail

    def drop_out(self, frame, col, ):
        if len(frame) <= 2:
            return frame
        else:
            if self.model == 'Confidence interval':
                cond_ = self.Confidence(frame, col)

            elif self.model == 'gauss':
                cond_ = self.Gauss(frame, col)

            elif self.model == 'box':
                cond_ = self.Box(frame, col)

            elif self.model == 'head_tail':
                cond_ = self.Head_tail(frame, col)

            else:
                print('please try again')
                return frame

            index_ = np.where(frame[col] != frame[col], True,
                              np.where(cond_, True, False))
            frame = frame.loc[index_, :]
            return frame

    def Confidence(self, frame, col):
        u_ = frame[col].mean()
        v_ = frame[col].std()
        interval_ = stats.t.interval(self.t_alpha, frame[col].count() - 1, u_, v_)
        cond_ = (frame[col] < interval_[1]) & (frame[col] > interval_[0])
        return cond_

    def Gauss(self, frame, col):
        u_ = frame[col].mean()
        v_ = frame[col].std()
        cond_ = np.abs((frame[col] - u_) / v_) < self.g_alpha
        return cond_

    def Box(self, frame, col):
        q1 = frame[col].quantile(0.25)
        q3 = frame[col].quantile(0.75)
        IQR = (q3 - q1) * self.IQR_rate
        q1 -= IQR
        q3 += IQR
        cond_ = (frame[col] < q3) & (frame[col] > q1)
        return cond_

    def Head_tail(self, frame, col):
        top_ = frame[col].quantile(self.head_tail)
        bottom_ = frame[col].quantile(1 - self.head_tail)
        cond_ = (frame[col] < top_) & (frame[col] > bottom_)
        return cond_
