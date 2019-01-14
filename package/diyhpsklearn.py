#!/user/bin/env python
#!-*-coding:utf-8 -*-

import hpsklearn
import sklearn.metrics as metrics
from hpsklearn import HyperoptEstimator, components
from hyperopt import tpe
from highpackage.diyttsplit import DiyttSplit
import numpy as np


class DiyHpSklearn(object):
    def __init__(self, simple, c_or_r, preprocess='auto', model='auto',):
        self.pp = preprocess
        self.corr = c_or_r
        if self.corr == 'c':
            if model == 'rf':
                self.model = hpsklearn.random_forest('MyRfc')
            elif model == 'svc':
                self.model = hpsklearn.svc_linear('MySvc')
            elif model == 'gdbt':
                self.model = hpsklearn.gradient_boosting('MyGdbtc')
            elif model == 'xgb':
                self.model = hpsklearn.xgboost_classification('Myxgbc')
            else:
                self.model = 'auto'
        elif self.corr == 'r':
            if model == 'rf':
                self.model = hpsklearn.random_forest_regression('MyRfr')
            elif model == 'svc':
                self.model = hpsklearn.svr_linear('MySvr')
            elif model == 'gdbt':
                self.model = hpsklearn.gradient_boosting_regression('MyGdbtr')
            elif model == 'xgb':
                self.model = hpsklearn.xgboost_regression('Myxgbr')
            else:
                self.model = 'auto'
        self.simple = simple

    def diyhpsklearn(self, re, data, data_length, max_evals, seed_number=None):
        seed_number = seed_number if seed_number else np.random.choice(range(2018), 1)[0]
        x1, x2, y1, y2 = DiyttSplit(re, self.simple).diyttsplit(data, data_length,
                                                                random_state=seed_number)
        x1, x2 = x1.values, x2.values
        if self.corr == 'c':
            estimator = HyperoptEstimator(preprocessing=self.pp if self.pp != 'auto' else
                                          components.any_preprocessing('pp'),
                                          classifier=self.model if self.model != 'auto' else
                                          components.any_classifier('clf'),
                                          algo=tpe.suggest,
                                          max_evals=max_evals if self.model != 'auto' else 200,
                                          seed=None)
        elif self.corr == 'r':
            estimator = HyperoptEstimator(preprocessing=self.pp if self.pp != 'auto' else
                                          components.any_preprocessing('pp'),
                                          regressor=self.model if self.model != 'auto' else
                                          components.any_regressor('reg'),
                                          algo=tpe.suggest,
                                          max_evals=max_evals if self.model != 'auto' else 200,
                                          seed=None)
        iterator = estimator.fit_iter(x1, y1,)
        next(iterator)
        for _ in range(max_evals):
            iterator.send(1)
        estimator.retrain_best_model_on_full_data(x1, y1)
        print(f'Test result:\n{metrics.confusion_matrix(y2, estimator.predict(x2))}')
        return estimator, seed_number
