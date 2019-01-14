#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from functools import reduce
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from highpackage.diyttsplit import DiyttSplit
from hyperopt import tpe, fmin, hp


# [ [[{},s], [{},s], [{},s]], [[{},s], [{},s], [{},s]],]
class DiyHyperopt(object):
    def __init__(self, model, space_dict, weight_list, simple, ):
        self.model = model
        self.space_dict = space_dict
        # self.space = [hp.choice(key, value) for key, value in self.space_dict.items()]
        self.weight_list = weight_list
        self.simple = simple
        # self.default_index_number = reduce(lambda x, y: x * y,
        #                                    [len(value) for value in self.space_dict.values()]
        #                                    )

    def diy_hyporept(self, re, data, data_length, seed_number=None, ):
        weight_list = self.weight_list
        seed_number = seed_number if seed_number else np.random.choice(range(2018), 1)[0]
        result_dict = {}
        x1, x2, y1, y2 = DiyttSplit(re, self.simple).diyttsplit(data, data_length,
                                                                random_state=seed_number)
        # print(f'x1,{sum(y1)/len(y1)};\nx2,{sum(y2)/len(y2)}')
        ######################
        def model_score(args):
            parameter_dict = dict(zip([i for i in space_dict.keys()], args
                                      )
                                  )
            # print('out', parameter_dict)
            try:
                model = self.model(random_state=seed_number+1, **parameter_dict)
            except:
                model = self.model(**parameter_dict)
            model.fit(x1, y1)
            pre = model.predict(x2)
            pre_pro = model.predict_proba(x2)[:, 1]
            recall_ = metrics.recall_score(pre, y2) if weight_list[0] else 0
            accu_ = metrics.accuracy_score(pre, y2) if weight_list[1] else 0
            auc_ = metrics.roc_auc_score(y2, pre_pro) if weight_list[2] else 0
            score_ = (recall_ * weight_list[0] +
                      accu_ * weight_list[1] +
                      auc_ * weight_list[2]
                      )
            cv_score = (cross_val_score(model, x1, y1, scoring='roc_auc', cv=3,)).mean()
            return -(0.2*score_+0.8*cv_score)
        ######################
        # [ [{:[,]}, {:[,]}, {:[,]} ], [{:[,]}, {:[,]}, {:[,]} ],]
        final_para_dict = {}
        for step, part in enumerate(self.space_dict):
            print(step+1)
            # part  [{:[,]}, {:[,]}, {:[,]} ]
            space_dict = {list(i.keys())[0]: list(i.values())[0][0] for i in part}
            space_dict.update(final_para_dict)
            # space_dict {:, :, :,}
            space = [hp.choice(key, value) for key, value in space_dict.items()]
            # space hp.choice
            space_type = {list(i.keys())[0]: list(i.values())[0][1] for i in part}
            str_para = {list(i.keys())[0]: list(i.values())[0][0] for i in part
                        if list(i.values())[0][1] == 's'}
            index = fmin(model_score, space, tpe.suggest,
                         reduce(lambda x, y: x * y,
                                [len(value) for value in space_dict.values()]
                                )
                         )
            for key in index.keys():
                result_dict[key] = space_dict[key][index[key]]
            result_dict, bool_dict = self.little_change(weight_list, x1, x2, y1, y2,
                                                        final_para_dict, seed_number,
                                                        str_para, result_dict, space_type)
            if 1 - reduce(lambda x, y: x and y, [i for i in bool_dict.values()]):
                while 1 - reduce(lambda x, y: x and y, [i for i in bool_dict.values()]):
                    result_dict, bool_dict = self.little_change(weight_list, x1, x2, y1, y2,
                                                                final_para_dict, seed_number,
                                                                str_para, result_dict, space_type,
                                                                whether=bool_dict)
            for key, value in result_dict.items():
                final_para_dict[key] = [value]
            print(result_dict)
        # try:
        #     model = self.model(random_state=seed_number, **result_dict)
        # except:
        final_para_dict = {key: value[0] for key, value in final_para_dict.items()}
        model = self.model(**final_para_dict)
        model.fit(x1, y1)
        print(self.weight_list)
        print(f'Test result:\n{metrics.confusion_matrix(y2, model.predict(x2))}')
        try:
            print(sorted(dict(zip(list(x1),
                                  map(lambda x: round(x, 4), model.feature_importances_)
                                  ),
                              ).items(),
                         key=lambda x: np.abs(x[1]), reverse=True
                         )[:10]
                  )
        except:
            try:
                print(sorted(dict(zip(list(x1),
                                      map(lambda x: round(x, 4), model.coef_.ravel())
                                      ),
                                  ).items(),
                             key=lambda x: np.abs(x[1]), reverse=True
                             )[:10]
                      )
            except:
                pass
        return model, result_dict, seed_number

    def little_change(self, weight_list, x1, x2, y1, y2,
                      fin_para_dict,
                      seed_number, str_para,
                      result_dict, space_type,
                      whether=None):
        ################
        def model_score(args):
            parameter_dict = dict(zip([i for i in param_test_now.keys()], args
                                      )
                                  )
            # print('in', parameter_dict)
            try:
                model = self.model(random_state=seed_number+1, **parameter_dict)
            except:
                model = self.model(**parameter_dict)
            model.fit(x1, y1)
            pre = model.predict(x2)
            pre_pro = model.predict_proba(x2)[:, 1]
            recall_ = metrics.recall_score(pre, y2) if weight_list[0] else 0
            accu_ = metrics.accuracy_score(pre, y2) if weight_list[1] else 0
            auc_ = metrics.roc_auc_score(y2, pre_pro) if weight_list[2] else 0
            score_ = (recall_ * weight_list[0] +
                      accu_ * weight_list[1] +
                      auc_ * weight_list[2]
                      )
            cv_score = (cross_val_score(model, x1, y1, scoring='roc_auc', cv=3,)).mean()
            return -(0.2*score_+0.8*cv_score)
        ################
        whether = {para: False for para in space_type.keys()} if whether is None else whether
        param_dict_old = result_dict
        param_test_now = {}
        old_param_str = []
        for param in space_type.keys():
            param_value = param_dict_old[param]
            if whether[param]:
                param_test_now[param] = [param_value]
            else:
                if space_type[param] is 's':
                    old_param_str.append(param_value)
                else:
                    pass
                if space_type[param] is 's':
                    param_test_now[param] = str_para[param]
                elif space_type[param] is 'li':
                    param_test_now[param] = range(param_value - 3,
                                                  param_value + 4)
                elif space_type[param] is 'si':
                    if param_value == 1:
                        param_test_now[param] = [param_value,
                                                 param_value + 1]
                    else:
                        param_test_now[param] = [param_value - 1,
                                                 param_value,
                                                 param_value + 1]
                elif space_type[param] is 'g':
                    if param_value == 0:
                        param_test_now[param] = [i for i in np.linspace(0, 0.1, 5, endpoint=False)]
                    else:
                        param_test_now[param] = [i for i in np.geomspace(param_value / 1.3,
                                                                         param_value * 1.3,
                                                                         5)]
                elif space_type[param] is 'f':
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.75,
                                                                    param_value + 0.75,
                                                                    5)]
                    param_test_now[param] = [i if i >= 0 else 0. for i in param_test_now[param]]
                elif space_type[param] is 'h':
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.075,
                                                                    param_value + 0.075,
                                                                    5)]
                    param_test_now[param] = [i if i >= 0 else 0. for i in param_test_now[param]]
                    param_test_now[param] = [i if i <= 0.5 else 0.5 for i in param_test_now[param]]
                elif space_type[param] is 'p':
                    param_test_now[param] = [i for i in np.linspace(param_value - 0.075,
                                                                    param_value + 0.075,
                                                                    5)]
                    param_test_now[param] = [i if i >= 0 else 0. for i in param_test_now[param]]
                    param_test_now[param] = [i if i <= 1 else 1. for i in param_test_now[param]]
                else:
                    pass
        param_test_now.update(fin_para_dict)
        space = [hp.choice(key, value) for key, value in param_test_now.items()]
        index = fmin(model_score, space, tpe.suggest,
                     reduce(lambda x, y: x * y,
                            [len(value) for value in param_test_now.values()]
                            )
                     )
        result_dict_new = {}
        for key in index.keys():
            result_dict_new[key] = param_test_now[key][index[key]]

        bool_dict = {}
        for param in space_type.keys():
            if whether[param]:
                bool_dict[param] = True
            else:
                if space_type[param] is 's':
                    new_para = result_dict_new[param]
                    bool_dict[param] = new_para in old_param_str
                else:
                    param_now = result_dict_new[param]
                    max_ = max(param_test_now[param])
                    min_ = min(param_test_now[param])
                    if param_now == min_ == 0:
                        bool_dict[param] = True
                    elif space_type[param] is 'si':
                        if min_ == param_now == 1:
                            bool_dict[param] = True
                        else:
                            bool_dict[param] = min_ < param_now < max_
                    else:
                        bool_dict[param] = min_ < param_now < max_
        return result_dict_new, bool_dict
