# -*- Coding:utf-8 -*-

import numpy as np
import pandas as pd
from highpackage.diyttsplit import DiyttSplit
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import gc


class DIYXGBCV(object):
    def __init__(self, simple,
                 n_epoch=2, cv_folds=3,
                 early_stopping_rounds=50,
                 ):
        self.n_epoch = n_epoch
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.simple = simple
        self.alg = XGBClassifier(learning_rate=0.001,
                                 n_estimators=1000,
                                 max_depth=4,
                                 min_child_weight=2,
                                 gamma=0.1,
                                 reg_lambda=1,
                                 subsample=0.8,
                                 colsample_bytree=0.7,
                                 objective='binary:logistic',
                                 )

    def xgbcv(self, re, data, data_length, seed_number=None):
        seed_number = seed_number if seed_number else np.random.choice(range(2018), 1)[0]
        x1, x2, y1, y2 = DiyttSplit(re, self.simple).diyttsplit(data, data_length,
                                                                random_state=seed_number)
        for epoch in range(self.n_epoch):
            xgb_param = self.alg.get_xgb_params()
            xgtrain = xgb.DMatrix(x1, label=y1)
            cvresult = xgb.cv(xgb_param, xgtrain,
                              num_boost_round=self.alg.get_params()['n_estimators'],
                              nfold=self.cv_folds, metrics='auc',
                              early_stopping_rounds=self.early_stopping_rounds,)
            self.alg.set_params(n_estimators=cvresult.shape[0])
            del xgtrain
            gc.collect()

            if epoch == 0:      # init
                param_test = [{'learning_rate': [i for i in np.geomspace(0.001, 0.5, 20)]},
                              {'max_depth': [int(i) for i in np.linspace(3, 10, 7)],
                               'min_child_weight':[int(i) for i in np.linspace(2, 10, 4)]},
                              {'subsample': [i for i in np.linspace(0.35, 0.8, 8)],
                               'colsample_bytree':[i for i in np.linspace(0.3, 0.7, 8)]},
                              {'gamma': [i for i in np.geomspace(0.05, 2, 8)],
                               'reg_alpha':[i for i in np.geomspace(0, 2, 8)]},
                              {'scale_pos_weight': [i for i in np.linspace(1, 11, 4)]}]
                for k in range(len(param_test)):
                    gsearch = GridSearchCV(estimator=self.alg, param_grid=param_test[k],
                                           scoring='roc_auc', iid=False, cv=self.cv_folds)
                    gsearch.fit(x1, y1)
                    self.alg = gsearch.best_estimator_

            param_test_little_list = [['learning_rate'],
                                      ['max_depth', 'subsample', 'colsample_bytree'],
                                      ['min_child_weight', 'scale_pos_weight'],
                                      ['reg_alpha', 'gamma'],
                                      ]
            int_para = ['max_depth', 'min_child_weight', 'n_estimators']
            geom_para = ['learning_rate', 'reg_alpha', 'gamma']
            for little in range(len(param_test_little_list)):
                param_test_little_dict = {}
                for para in param_test_little_list[little]:
                    para_now = self.alg.get_xgb_params()[para]
                    delta_now = self.n_epoch - epoch
                    per_delta_now = 1 + 1.5 * delta_now / 10
                    if para in int_para:
                        param_test_little_dict[para] = [int(i) for i in
                                                        np.linspace(para_now - (delta_now),
                                                                    para_now + (delta_now),
                                                                    3)]
                    elif para in geom_para:
                        param_test_little_dict[para] = [i for i in
                                                        np.geomspace(para_now / per_delta_now,
                                                                     para_now * per_delta_now,
                                                                     4)]
                    else:
                        param_test_little_dict[para] = [i for i in
                                                        np.linspace(para_now / per_delta_now,
                                                                    para_now * per_delta_now,
                                                                    4)]
                gsearch = GridSearchCV(estimator=self.alg, param_grid=param_test_little_dict,
                                       scoring='roc_auc', iid=False, cv=self.cv_folds)
                gsearch.fit(x1, y1)
                self.alg = gsearch.best_estimator_

        #Fit the algorithm on the data
        self.alg.fit(x1, y1, eval_metric='auc')

        #Print model report:
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y1, self.alg.predict_proba(x1)[:, 1]))
        print("AUC Score (Test): %f" % metrics.roc_auc_score(y2, self.alg.predict_proba(x2)[:, 1]))
        print('Recall Score (Test): %f' % metrics.recall_score(y2, self.alg.predict(x2)))

        print(sorted(list(zip(list(x1), self.alg.feature_importances_, )), key=lambda x: x[1], reverse=True))
        return self.alg

    def modelfit(self, alg, x1, y1, n_estimators=None, seed_number=None):
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x1.values, label=y1)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'] if n_estimators is None else n_estimators,
                          nfold=self.cv_folds, metrics='auc',
                          early_stopping_rounds=self.early_stopping_rounds,
                          show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        return alg

