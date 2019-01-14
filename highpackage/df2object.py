#!/user/bin/env python
#!-*-coding:utf-8 -*-

import pandas as pd


class Df2Object(object):
    def __init__(self, object_list_drop, object_list_remain):
        self.object_list_drop = object_list_drop
        self.object_list_remain = object_list_remain

    def dftoobject(self, df):
        df_object = pd.DataFrame()
        for i in self.object_list_drop:
            df_object = pd.concat([df_object, df.pop(i)], axis=1, sort=False)
        for j in self.object_list_remain:
            df_object = pd.concat([df_object, df[[j]]], axis=1, sort=False)
        return df_object, df
