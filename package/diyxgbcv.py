# -*- Coding:utf-8 -*-

from functools import reduce
import numpy as np
from highpackage.diyttsplit import DiyttSplit
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import gc


class DiyXgbCv(object):
    def __init__(self, cv_folds=3,
                 early_stopping_rounds=50):

        self.cv_folds = cv_folds
        param_tests = [[{'max_depth': [range(3, 10, 2),
                                       'small_int']},
                        {'min_child_weight': [range(1, 8, 2),
                                              'small_int']},
                        {'scale_pos_weight': [[i for i in np.linspace(1, 9, 7)],
                                              'float']}
                        ],
                       [{'gamma': [[0] + [i for i in np.geomspace(0.1, 10, 10)],
                                   'geom']}
                        ],
                       [{'subsample': [[i for i in np.linspace(0.3, 1, 8)],
                                       'no_zero_percentage']},
                        {'min_child_weight': [[i for i in np.linspace(0.3, 1, 8)],
                                              'no_zero_percentage']}
                        ],
                       [{'subsample': ['last',
                                       'no_zero_percentage']},
                        {'min_child_weight': ['last',
                                              'no_zero_percentage']},
                        {'colsample_bytree': [[i for i in np.linspace(0.35, 0.8, 8)],
                                              'no_zero_percentage']}
                        ],
                       [{'reg_alpha': [[0] + [i for i in np.geomspace(0.1, 1.5, 7)],
                                       'geom']},
                        {'reg_lambda': [[0] + [i for i in np.geomspace(0.1, 1.5, 7)],
                                        'geom']}
                        ]
                       ]
        self.space_dict = param_tests
        self.early_stopping_rounds = early_stopping_rounds
        self.alg = XGBClassifier(learning_rate=0.1,
                                 n_estimators=1000,
                                 max_depth=5,
                                 min_child_weight=1,
                                 gamma=0.1,
                                 subsample=0.7,
                                 colsample_bytree=0.7,
                                 objective='binary:logistic',
                                 scale_pos_weight=1,
                                 )

    def diy_gsearch(self, x1, x2, y1, y2,
                    seed_number=None):

        seed_number = seed_number if seed_number else np.random.choice(range(2018), 1)[0]
        self.alg.set_params(random_state=seed_number)

        # x1, x2, y1, y2 = DiyttSplit(re_1, self.simple).diyttsplit(re_0=re_0, per_re_0=per_re_0,
        #                                                           data=data, num_data=num_data,
        #                                                           test_size=0,
        #                                                           random_state=seed_number)

        self.alg = self.xgbcv(self.alg, x1, y1,)
        final_para_dict = self.alg.get_xgb_params()
        for step, part in enumerate(self.space_dict):

            print(f'step {step+1}')

            space_dict, space_type, str_para = self.get_param(part, final_para_dict)
            print({key: final_para_dict[key] for key in space_dict.keys()})

            for param in space_dict.keys():
                del final_para_dict[param]
            self.alg = XGBClassifier(**final_para_dict)

            gsearch = GridSearchCV(estimator=self.alg,
                                   param_grid=space_dict, scoring='roc_auc',
                                   n_jobs=-1, iid=False, cv=self.cv_folds)
            gsearch.fit(x1, y1)
            result_dict = gsearch.best_params_
            result_dict, bool_dict = self.little_change(x1, y1,
                                                        final_para_dict,
                                                        str_para, result_dict,
                                                        space_type)
            if 1 - reduce(lambda x, y: x and y, [i for i in bool_dict.values()]):
                while 1 - reduce(lambda x, y: x and y, [i for i in bool_dict.values()]):
                    result_dict, bool_dict = self.little_change(x1, y1,
                                                                final_para_dict,
                                                                str_para, result_dict,
                                                                space_type,
                                                                whether=bool_dict)
            for key, value in result_dict.items():
                final_para_dict[key] = value

            print({key: result_dict[key] for key in space_dict.keys()})

        self.alg = XGBClassifier(**final_para_dict)
        self.alg = self.xgbcv(self.alg, x1, y1, last=True, num_boost_round=5000)

        self.alg.fit(x1, y1, eval_metric='auc')

        print(f'Train result:\n{metrics.confusion_matrix(y1, self.alg.predict(x1))}')
        print(sorted(list(zip(list(x1),
                              self.alg.feature_importances_, )
                          ),
                     key=lambda x: x[1], reverse=True)[:10])
        return self.alg, '', seed_number

    def xgbcv(self, alg, x1, y1, last=False, num_boost_round=None, ):
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x1.values, label=y1)
        if last:
            xgb_param['learning_rate'] /= 4
        num_boost_round = xgb_param['n_estimators'] if num_boost_round is None else num_boost_round
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=num_boost_round,
                          nfold=self.cv_folds, metrics='auc',
                          early_stopping_rounds=self.early_stopping_rounds,
                          show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        del xgtrain
        gc.collect()
        return alg

    def little_change(self, x1, y1,
                      fin_para_dict,
                      str_para,
                      result_dict, space_type,
                      whether=None):
        whether = {para: False for para in space_type.keys()} if whether is None else whether
        param_dict_old = result_dict
        param_test_now = {}
        old_param_str = []
        for param in space_type.keys():
            param_value = param_dict_old[param]
            if whether[param]:
                param_test_now[param] = [param_value]
            else:
                if space_type[param] is 'str':
                    old_param_str.append(param_value)
                else:
                    pass
                if space_type[param] is 'str':
                    param_test_now[param] = str_para[param]
                elif 'int' in space_type[param]:
                    if space_type[param] is 'large_int':
                        param_test_now[param] = range(param_value - 3,
                                                      param_value + 4)
                    elif space_type[param] is 'small_int':
                        param_test_now[param] = [param_value - 1,
                                                 param_value,
                                                 param_value + 1]
                        param_test_now[param] = [i if i >= 0 else 0 for i in param_test_now[param]]
                elif space_type[param] is 'geom':
                    if param_value == 0:
                        param_test_now[param] = [i for i in np.linspace(0, 0.1, 5, endpoint=False)]
                    else:
                        param_test_now[param] = [i for i in np.geomspace(param_value / 1.3,
                                                                         param_value * 1.3,
                                                                         5)]
                        param_test_now[param] = [i if i > 0.0001 else 0 for i in param_test_now[param]]
                elif 'float' in space_type[param]:
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.75,
                                                                    param_value + 0.75,
                                                                    5)]
                    if 'no_zero' in space_type[param]:
                        param_test_now[param] = [i if i > 0 else 0.001 for i in param_test_now[param]]
                    else:
                        param_test_now[param] = [i if i > 0 else 0 for i in param_test_now[param]]
                elif 'half' in space_type[param]:
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.075,
                                                                    param_value + 0.075,
                                                                    5)]
                    param_test_now[param] = [i if i <= 0.5 else 0.5 for i in param_test_now[param]]
                    if 'no_zero' in space_type[param]:
                        param_test_now[param] = [i if i > 0 else 0.001 for i in param_test_now[param]]
                    else:
                        param_test_now[param] = [i if i > 0 else 0 for i in param_test_now[param]]
                elif 'percentage' in space_type[param]:
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.075,
                                                                    param_value + 0.075,
                                                                    5)]
                    param_test_now[param] = [i if i <= 1 else 1. for i in param_test_now[param]]
                    if 'no_zero' in space_type[param]:
                        param_test_now[param] = [i if i > 0 else 0.001 for i in param_test_now[param]]
                    else:
                        param_test_now[param] = [i if i > 0 else 0 for i in param_test_now[param]]
                else:
                    pass
        self.alg = XGBClassifier(**fin_para_dict)
        gsearch = GridSearchCV(self.alg, param_test_now, scoring='roc_auc', cv=self.cv_folds,
                               n_jobs=-1, pre_dispatch=4, iid=False)
        gsearch.fit(x1, y1)
        result_dict_new = gsearch.best_params_

        bool_dict = {}
        for param in space_type.keys():
            if whether[param]:
                bool_dict[param] = True
            else:
                if space_type[param] is 'str':
                    new_para = result_dict_new[param]
                    bool_dict[param] = new_para in old_param_str
                else:
                    param_now = result_dict_new[param]
                    max_ = max(param_test_now[param])
                    min_ = min(param_test_now[param])
                    if param_now == min_ == 0:
                        bool_dict[param] = True
                    elif param_now == min_ == 0.001:
                        bool_dict[param] = True
                    elif param_now == max_ == 0.5:
                        bool_dict[param] = True
                    elif param_now == max_ == 1:
                        bool_dict[param] = True
                    else:
                        bool_dict[param] = min_ < param_now < max_
        return result_dict_new, bool_dict

    def get_param(self, part, fin_param_dict):
        space_dict = {list(i.keys())[0]: list(i.values())[0][0] for i in part}
        space_type = {list(i.keys())[0]: list(i.values())[0][1] for i in part}
        str_para = {list(i.keys())[0]: list(i.values())[0][0] for i in part
                    if list(i.values())[0][1] == 'str'}
        for param, value in space_dict.items():
            if value == 'last':
                param_value = fin_param_dict[param]
                if param == 'learning_rate':
                    space_dict[param] = [param_value / 4]
                elif param == 'n_estimators':
                    space_dict[param] = range(int(param_value * 2.5) - 6,
                                              int(param_value * 2.5) + 7)
                elif 'int' in space_type[param]:
                    if space_type[param] is 'large_int':
                        space_dict[param] = range(param_value - 3,
                                                  param_value + 4)
                    elif space_type[param] is 'small_int':
                        space_dict[param] = [param_value - 1,
                                             param_value,
                                             param_value + 1]
                        space_dict[param] = [i if i >= 0 else 0 for i in space_dict[param]]
                elif space_type[param] is 'geom':
                    if param_value == 0:
                        space_dict[param] = [i for i in np.linspace(0, 0.1, 5, endpoint=False)]
                    else:
                        space_dict[param] = [i for i in np.geomspace(param_value / 1.3,
                                                                     param_value * 1.3,
                                                                     5)]
                        space_dict[param] = [i if i > 0.0001 else 0 for i in space_dict[param]]
                elif 'float' in space_type[param]:
                    space_dict[param] = [i for i in np.linspace(param_value - 0.75,
                                                                param_value + 0.75,
                                                                5)]
                    if 'no_zero' in space_type[param]:
                        space_dict[param] = [i if i > 0 else 0.001 for i in space_dict[param]]
                    else:
                        space_dict[param] = [i if i > 0 else 0 for i in space_dict[param]]
                elif 'half' in space_type[param]:
                    space_dict[param] = [i for i in np.linspace(param_value - 0.075,
                                                                param_value + 0.075,
                                                                5)]
                    space_dict[param] = [i if i <= 0.5 else 0.5 for i in space_dict[param]]
                    if 'no_zero' in space_type[param]:
                        space_dict[param] = [i if i > 0 else 0.001 for i in space_dict[param]]
                    else:
                        space_dict[param] = [i if i > 0 else 0 for i in space_dict[param]]
                elif 'percentage' in space_type[param]:
                    space_dict[param] = [i for i in np.linspace(param_value - 0.075,
                                                                param_value + 0.075,
                                                                5)]
                    space_dict[param] = [i if i <= 1 else 1. for i in space_dict[param]]
                    if 'no_zero' in space_type[param]:
                        space_dict[param] = [i if i > 0 else 0.001 for i in space_dict[param]]
                    else:
                        space_dict[param] = [i if i > 0 else 0 for i in space_dict[param]]
                else:
                    pass
            else:
                pass
        return space_dict, space_type, str_para
