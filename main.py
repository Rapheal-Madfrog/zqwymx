#!/user/bin/env python
#!-*-coding:utf-8 -*-

import numpy as np
from highpackage.reloader import ReLoader
from highpackage.comloader import ComLoader
from highpackage.filldata import FillData, rechange
from highpackage.comonehot import ComOneHot
from highpackage.df2object import Df2Object
from package.diyxgbcv import DiyXgbCv
from package.diylgbcv import DiyLgbCv
from package.diygsearch import DiyGridSearch
from highpackage.diyttsplit import DiyttSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from mlxtend.classifier import EnsembleVoteClassifier, StackingCVClassifier
import pandas as pd
from sklearn.externals import joblib
import gc
import warnings
warnings.filterwarnings('ignore')


def predict(begin_of_year, end_of_year,
            time_style, time_delta,
            simple, constant=True,
            per_re_0=0.75, num_data=200,
            fillna=False, cv='auto'):

    '''
    :param int begin_of_year: 开始的年份, 默认为2013年
    :param int end_of_year: 结束的年份, 一般为数据最新的年度
    :param str time_style: 三种可选: 默认为season
    1. year - 年度模型
    2. half_year - 半年模型
    3. season - 季度模型
    :param int time_delta: 预测的时间差. 默认为2. 如果为1,即去年预测今年, 或者上半年预测下半年, 二季度预测三季度.
    :param bool simple: 指标是否人为挑选. 默认为False. 如果为True, 筛选的指标在 DiyttSplit 函数中
    :param bool constant: 是否连续. 该指标只有在 time_delta >= 2 时有用. 默认为 True.
    如果为True, 且time_delta=2, 以season为例, 三季度违约, 一季度和二季度同时判定为违约样本, 以二季度为界, 截
    断该公司样本.
    如果为False, 则三季度违约, 则一季度判定为违约样本, 并以一季度为界, 截断该公司样本.
    :param float per_re_0: 这是一个百分数, 控制违约公司违约前的样本的比例, 以平衡样本.
    :param int num_data: 这是一个整数, 控制未违约公司的样本的数量, 以平衡样本.
    :param bool fillna: 如果已经填充过空白值, 则选择False, 直接读取即可.
    如果没有填充过, 则选择True, 填充完成后会保存, 下次读取即可.
    :param list(int)|str cv: 接受一个列表, 列表里面是整数. 默认为字符串'auto'.
    *** 0 删除模型  1 弹性训练 预测不用  2 弹性训练 预测用  3 刚性训练 预测用
    *** 弹性训练: 如果检测到有对应的老模型, 则不训练
    *** 刚性训练: 无论如何, 重新训练
    *** 0 删除老模型, 不训练并且不训练, 也不在stack里
    *** 1 弹性训练, 不在stack里
    *** 2 弹性训练, 在stack里
    *** 3 刚性训练, 在stack里
    *** 'auto' 即 老模型有几个, 就几个是2, 剩余的都是3
    *** 推荐用自定义列表
    :return:
    '''

    # 读取公司信息
    com_data = ComLoader(industry_ss=True).loader()
    # 填充完的路径
    a_path = f'./project/{time_style}/{time_delta}_{begin_of_year}-{end_of_year}.csv'
    # 模型路径
    w_path = f'./project/{time_style}/{time_delta}_weight_{begin_of_year}-{end_of_year}.m'

    if fillna:
        # 读取财报data, 违约数据re_data
        # 违约数据取第一次违约记录(在ReLoader中)
        # 返回财报数据和违约数据
        data, re_data = ReLoader(begin_of_year, end_of_year,
                                 time_style=time_style,
                                 whether_plt=False).loader()
        # 填充空白值
        filldata = FillData(time_style=time_style,
                            time_delta=time_delta,
                            a_save_path=a_path, w_save_path=w_path,
                            constant=constant)
        a, weight_dict = filldata.filldata(data=data, re_data=re_data, com_data=com_data)
        del data, re_data
        gc.collect()
    else:
        print('loading')
        a = pd.read_csv(a_path)
        weight_dict = joblib.load(w_path)
        print('finish loading')

    a.drop('发生日期', axis=1, inplace=True)
    a.loc[:, '报告期'] = pd.to_datetime(a['报告期'])
    re_data_0 = a.loc[a['target'] == 0, :]
    re_data_1 = a.loc[a['target'] == 1, :]
    predict_data = a.loc[(a['target'] == -2), :]
    data = a.loc[(a['报告期'].dt.year >= (2015 if time_style == 'year' else 2016)) &
                 (a['target'] == -1), :]
    data.loc[:, 'target'] = 0
    del a
    gc.collect()
    # return re_data_1, re_data_0, predict_data, data, com_data
    # company feature onehot
    coh = ComOneHot(com_data)
    re_data_1 = coh.coh(re_data_1)
    re_data_0 = coh.coh(re_data_0)
    predict_data = coh.coh(predict_data)
    data = coh.coh(data)
    del com_data

    object_list_drop = ['名称', '报告期', '企业性质', '是否上市', '二级分类', '最新评级']
    object_list_remain = ['是否交通', ]
    df2ob = Df2Object(object_list_drop, object_list_remain)
    re_object_1, re_data_1 = df2ob.dftoobject(re_data_1)
    re_object_0, re_data_0 = df2ob.dftoobject(re_data_0)
    predict_object, predict_data = df2ob.dftoobject(predict_data)
    data_object, data = df2ob.dftoobject(data)

    re_data_1.drop('target', axis=1, inplace=True)
    re_data_0.drop('target', axis=1, inplace=True)
    predict_data.drop('target', axis=1, inplace=True)
    data.drop('target', axis=1, inplace=True)

    m_path = f'./project/{time_style}/{time_delta}_model_{begin_of_year}-{end_of_year}.m'

    print('cv_begin')
    space_dict_rf = [[{'class_weight': [['balanced'],
                                        'str']},
                      {'n_jobs': [[-1],
                                  'str']},
                      {'n_estimators': [[int(i) for i in np.linspace(50, 100, 10)],
                                        'large_int']},
                      {'max_depth': [[int(i) for i in np.linspace(3, 9, 4)],
                                     'small_int']}
                      ],
                     [{'min_samples_split': [[i for i in np.linspace(0.1, 0.7, 5)],
                                             'no_zero_percentage']},
                      {'min_samples_leaf': [[i for i in np.linspace(0.1, 0.5, 4)],
                                            'no_zero_half']},
                      {'min_weight_fraction_leaf': [[i for i in np.linspace(0, 0.5, 3)],
                                                    'half']}
                      ],
                     [{'max_features': [[i for i in np.linspace(0.4, 0.8, 5)],
                                        'percentage']}
                      ]
                     ]

    space_dict_lr = [[{'penalty': [['l1', 'l2'],
                                   'str']},
                      {'class_weight': [['balanced'],
                                        'str']},
                      {'n_jobs': [[1],
                                  'str']},
                      {'solver': [['liblinear'],
                                  'str']}
                      ],
                     [{'tol': [[i for i in np.geomspace(0.01, 0.5, 10)],
                               'geom']},
                      {'C': [[i for i in np.geomspace(0.1, 6, 10)],
                             'geom']},
                      ]
                     ]

    space_dict_gd = [[{'learning_rate': [[i for i in np.geomspace(0.05, 0.5, 5)],
                                         'geom']},
                      {'n_estimators': [[int(i) for i in np.linspace(50, 100, 5)],
                                        'large_int']}
                      ],
                     [{'min_samples_split': [[i for i in np.linspace(0.2, 0.8, 7)],
                                             'no_zero_percentage']},
                      {'max_depth': [[int(i) for i in np.linspace(3, 9, 4)],
                                     'small_int']}
                      ],
                     [{'min_samples_split': ['last',
                                             'no_zero_percentage']},
                      {'min_samples_leaf': [[i for i in np.linspace(0.01, 0.5, 6, )],
                                            'no_zero_half']}
                      ],
                     [{'max_leaf_nodes': [[None],
                                          'str']},
                      {'min_weight_fraction_leaf': [[0, 0.25, 0.5],
                                                    'half']},
                      ],
                     [{'max_features': [[i for i in np.linspace(0.4, 0.8, 5)],
                                        'percentage']},
                      {'subsample': [[i / 10 for i in np.linspace(5, 8, 3)],
                                     'no_zero_percentage']
                       }],
                     [{'learning_rate': ['last',
                                         'geom']},
                      {'n_estimators': ['last',
                                        'large_int']}
                      ]
                     ]

    space_dict_sv = [[{'probability': [[True],
                                       'str']},
                      {'kernel': [['linear', 'poly', 'rbf'],
                                  'str']},
                      {'class_weight': [['balanced'],
                                        'str']},
                      {'gamma': [['auto'],
                                 'str']}
                      ],
                     [{'C': [[i for i in np.linspace(0.1, 12, 12)],
                             'no_zero_float']},
                      {'degree': [[3, 4, ],
                                  'small_int']},
                      ]
                     ]

    diy_cv_rf = DiyGridSearch(RandomForestClassifier,
                              space_dict_rf,
                              )
    diy_cv_lr = DiyGridSearch(LogisticRegression,
                              space_dict_lr,
                              )
    diy_cv_gd = DiyGridSearch(GradientBoostingClassifier,
                              space_dict_gd,
                              )
    diy_cv_sv = DiyGridSearch(SVC,
                              space_dict_sv,
                              )

    try:
        cv_file = joblib.load(m_path)
    except:
        cv_file = []
    length_cv = len(cv_file)
    # auto
    if cv == 'auto':
        bool_cv = [1]*length_cv + [2]*(6-length_cv)
    else:
        bool_cv = cv

    cv_file_new = []
    model_name = ['lgb', 'xgb', 'rf', 'lr', 'gdbt', 'svc']
    model_init_class = [DiyLgbCv(), DiyXgbCv(),
                        diy_cv_rf, diy_cv_lr,
                        diy_cv_gd, diy_cv_sv
                        ]

    Diy_SS = DiyttSplit(re_data_1, simple, time_style=time_style)
    for step, style_bool in enumerate(bool_cv):
        print(f'{model_name[step]} begin')
        if style_bool == 0:
            cv_file_new.append(model_name[step])
        elif (style_bool == 1) or (style_bool == 2):
            try:
                g_value = cv_file[step]
                if g_value == '':
                    x1, x2, y1, y2 = Diy_SS.diyttsplit(re_0=re_data_0, per_re_0=per_re_0,
                                                       data=data, num_data=num_data,
                                                       test_size=0,
                                                       random_state=step+1)
                    cv_file_new.append([tuple(model_init_class[step].diy_gsearch(x1, x2, y1, y2,
                                                                                 seed_number=step + 2)),
                                        style_bool])
                else:
                    cv_file_new.append(g_value)
            except:
                x1, x2, y1, y2 = Diy_SS.diyttsplit(re_0=re_data_0, per_re_0=per_re_0,
                                                   data=data, num_data=num_data,
                                                   test_size=0,
                                                   random_state=step+1)
                cv_file_new.append([tuple(model_init_class[step].diy_gsearch(x1, x2, y1, y2,
                                                                             seed_number=step + 2)),
                                   style_bool])
        else:
            x1, x2, y1, y2 = Diy_SS.diyttsplit(re_0=re_data_0, per_re_0=per_re_0,
                                               data=data, num_data=num_data,
                                               test_size=0,
                                               random_state=step+1)
            cv_file_new.append([tuple(model_init_class[step].diy_gsearch(x1, x2, y1, y2,
                                                                         seed_number=step + 2)),
                               style_bool])
        joblib.dump(cv_file_new, m_path)
        print(f'{model_name[step]} end')

    try:
        del x1, x2, y1, y2
    except:
        pass

    print('cv_end')

    # cv_file_new.append(list(re_data_1))
    # joblib.dump(cv_file_new, m_path)

    # model_list = [model_tuple[0] for step, model_tuple in enumerate(cv_file_new[:-1])
    #               if (bool_cv[step] == 2) or (bool_cv[step] == 3)]
    # [[( x, x,), x], ]
    # print(cv_file_new)

    model_list = [model_list[0][0] for model_list in cv_file_new
                  if ((model_list[-1] == 2)
                      or (model_list[-1] == 3))
                  and not isinstance(model_list, str)
                  ]

    voting_ensemble = EnsembleVoteClassifier(model_list, voting='soft', weights=[1]*len(model_list),)
    stacking_ensemble = StackingCVClassifier(model_list, LogisticRegression(),
                                             use_probas=True, cv=3,
                                             use_features_in_secondary=True)

    x, x_, y, y_ = Diy_SS.diyttsplit(re_0=re_data_0, per_re_0=per_re_0,
                                     data=data, num_data=num_data,
                                     test_size=0, random_state=0)

    predict_col = list(x)
    cv_file_new.append(list(predict_col))
    joblib.dump(cv_file_new, m_path)

    # LogisticRegression().fit(x, y)
    voting_ensemble.fit(x, y)
    stacking_ensemble.fit(x.values, y)

    print('voting: ', metrics.confusion_matrix(y, voting_ensemble.predict(x)))
    print('stacking ', metrics.confusion_matrix(y, stacking_ensemble.predict(x.values)))

    voting_y = voting_ensemble.predict(predict_data[predict_col])
    voting_y_pro = voting_ensemble.predict_proba(predict_data[predict_col])[:, 1]

    stacking_y = stacking_ensemble.predict(predict_data[predict_col].values)
    stacking_y_pro = stacking_ensemble.predict_proba(predict_data[predict_col].values)[:, 1]

    predict_object.insert(0, 'voting_y', voting_y)
    predict_object.insert(0, 'voting_y_pro', voting_y_pro)

    predict_object.insert(0, 'stack_y', stacking_y)
    predict_object.insert(0, 'stack_y_pro', stacking_y_pro)

    # print(sum([1 if i >= 0.5 else 0 for i in end_y]), len(end_y),
    #       f'{round(100*sum([1 if i >= 0.5 else 0 for i in end_y])/len(end_y), 2)}%')
    for col in list(predict_data):
        predict_data.loc[:, col] = rechange(predict_data, col, weight_dict)
    pd.concat([predict_object, predict_data], axis=1, sort=False). \
        sort_values(by=['voting_y_pro'], ascending=False).\
        to_csv(f'./project/{time_style}/{time_delta}_{begin_of_year}-{end_of_year}_end.csv',
               index=False, encoding='utf-8')

    return None


if __name__ == '__main__':
    # cv    0 删除模型  1 弹性训练 预测不用  2 弹性训练 预测用  3 刚性训练 预测用
    predict(begin_of_year=2013, end_of_year=2018,
            time_style='season', time_delta=2,
            per_re_0=0.6, num_data=280,
            simple=False, fillna=False,
            constant=True,
            cv=[2, 2, 2, 2, 0, 0]
            )
