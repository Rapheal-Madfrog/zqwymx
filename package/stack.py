#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold
# from package.diygsearch import DiyGridSearch
import sklearn.metrics as metrics


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_model):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_model = base_model

    def fit_predict(self, X, y, T):
        print('start stacking')
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = [KFold(n_splits=self.n_folds, shuffle=True, random_state=1 + i)
                 for i in range(len(self.base_model))]

        s_train = np.zeros((X.shape[0], len(self.base_model)))
        s_test = np.zeros((T.shape[0], len(self.base_model)))

        for i, clf in enumerate(self.base_model):
            s_test_i = np.zeros((T.shape[0], self.n_folds))
            print(f'{i+1}/{len(self.base_model)}')
            # try:
            #     clf = clf.best_model()['learner']
            # except:
            #     clf = clf
            for j, (train_idx, test_idx) in enumerate(folds[i].split(X, y)):
                x_train = X[train_idx]
                y_train = y[train_idx]
                x_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(x_train, y_train)
                s_train[test_idx, i] = clf.predict(x_holdout)
                s_test_i[:, j] = clf.predict(T)

            s_test[:, i] = s_test_i.mean(1)

        self.stacker.fit(s_train, y)
        print('stack result on train:\n')
        print(f'{metrics.confusion_matrix(y, self.stacker.predict(s_train))}')
        y_pre_pro = self.stacker.predict_proba(s_test)[:, 1]
        y_pre = self.stacker.predict(s_test)
        return y_pre, y_pre_pro
