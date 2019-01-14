#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from functools import reduce
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from highpackage.diyttsplit import DiyttSplit
from lightgbm import LGBMClassifier
import lightgbm as lgb
import gc


class DiyLgbCv(object):
    def __init__(self, n_fold=3,
                 early_stopping_rounds=50,):

        self.n_fold = n_fold
        param_tests = [[{'num_leaves': [[2 ** i - 1 for i in range(4, 11, 1)],
                                        'small_int']},
                        {'class_weight': [['balanced', None],
                                          'str']}
                        ],
                       [{'min_child_samples': [range(1, 19, 2),
                                               'small_int']},
                        {'min_split_gain': [[i for i in np.linspace(0, 9, 7)],
                                            'float']}
                        ],
                       [{'min_child_samples': ['last',
                                               'small_int']},
                        {'min_split_gain': ['last',
                                            'float']},
                        {'min_child_weight': [[i for i in np.linspace(0.3, 1, 8)],
                                              'no_zero_percentage']}
                        ],
                       [{'subsample_freq': [range(1, 9, 2),
                                            'small_int']},
                        {'subsample': [[i for i in np.linspace(0.3, 1, 8)],
                                       'no_zero_percentage']},
                        {'colsample_bytree': [[i for i in np.linspace(0.3, 1, 8)],
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
        self.alg = LGBMClassifier(learning_rate=0.08,
                                  boosting_type='gbdt',
                                  n_estimators=1000,
                                  objective='binary',
                                  metric=['auc'],
                                  verbose=-1,
                                  n_jobs=-1,
                                  )

    def diy_gsearch(self, x1, x2, y1, y2,
                    seed_number=None, ):

        seed_number = seed_number if seed_number else np.random.choice(range(2018), 1)[0]
        self.alg.set_params(random_state=seed_number)

        # x1, x2, y1, y2 = DiyttSplit(re_1, self.simple).diyttsplit(re_0=re_0, per_re_0=per_re_0,
        #                                                           data=data, num_data=num_data,
        #                                                           test_size=0,
        #                                                           random_state=seed_number)
        self.alg = self.lgbcv(self.alg, x1, y1, )
        final_para_dict = self.alg.get_params()
        for step, part in enumerate(self.space_dict):
            print(step+1)

            space_dict, space_type, str_para = self.get_param(part, final_para_dict)
            print({key: final_para_dict[key] for key in space_dict.keys()})

            for param in space_dict.keys():
                del final_para_dict[param]
            self.alg = LGBMClassifier(**final_para_dict)

            gsearch = GridSearchCV(self.alg, space_dict, scoring='roc_auc', cv=self.n_fold,
                                   n_jobs=-1, pre_dispatch=4, iid=False)
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

        self.alg = LGBMClassifier(**final_para_dict)
        self.alg = self.lgbcv(self.alg, x1, y1, last=True, num_boost_round=5000)
        self.alg.fit(x1, y1)
        print(f'Train result:\n{metrics.confusion_matrix(y1, self.alg.predict(x1))}')
        print(sorted(dict(zip(list(x1),
                              map(lambda x: round(x, 4), self.alg.feature_importances_)
                              ),
                          ).items(),
                     key=lambda x: np.abs(x[1]), reverse=True
                     )[:10]
              )
        return self.alg, '', seed_number

    def lgbcv(self, alg, x1, y1, last=False, num_boost_round=None, ):
        lgb_param = alg.get_params()
        lgtrain = lgb.Dataset(x1.values, label=y1)
        del lgb_param['silent']
        del lgb_param['importance_type']
        if last:
            lgb_param['learning_rate'] /= 4
        num_boost_round = lgb_param['n_estimators'] if num_boost_round is None else num_boost_round
        cvresult = lgb.cv(lgb_param, lgtrain,
                          num_boost_round=num_boost_round,
                          nfold=self.n_fold, metrics=['auc'],
                          early_stopping_rounds=self.early_stopping_rounds,
                          show_stdv=False)
        alg.set_params(n_estimators=len(cvresult['auc-mean']))
        del lgtrain
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
        self.alg = LGBMClassifier(**fin_para_dict)
        gsearch = GridSearchCV(self.alg, param_test_now, scoring='roc_auc', cv=self.n_fold,
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
