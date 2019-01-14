#!/user/bin/env python
#!-*-coding:utf-8 -*-

from sklearn.externals import joblib
import numpy as np


cv_file = joblib.load('project/season/2_model_2013-2018.m')
model_name = ['lgb', 'xgb', 'rf', 'lr', 'gdbt', 'svc']

for i, m_name in enumerate(model_name):
    if cv_file[i][-1] != '':
        print(m_name)
        model = cv_file[i][0][0]
        try:
            print(sorted(dict(zip(cv_file[-1],
                                  map(lambda x: round(x, 4), model.feature_importances_)
                                  ),
                              ).items(),
                         key=lambda x: np.abs(x[1]), reverse=True
                         )[:15]
                  )
        except:
            try:
                print(sorted(dict(zip(cv_file[-1],
                                      map(lambda x: round(x, 4), model.coef_.ravel())
                                      ),
                                  ).items(),
                             key=lambda x: np.abs(x[1]), reverse=True
                             )[:15]
                      )
            except:
                pass
