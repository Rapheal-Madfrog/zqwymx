# -*- Coding:utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from highpackage.diyttsplit import DiyttSplit
from sklearn.model_selection import train_test_split

class DIYLGBCV(object):
    def __init__(self, simple,
                 n_fold=3, early_stop=40,
                 seed_number=None):
        self.simple = simple
        self.n_fold = n_fold
        self.early_stop = early_stop
        self.seed_number = np.random.choice(range(2018), 1)[0] if seed_number is None else seed_number

    def modelfit(self, re, data, data_length):
        x1, x2, y1, y2 = DiyttSplit(re, simple=self.simple).diyttsplit(data, data_length, test_size=0,
                                                                       random_state=self.seed_number)

        ### 数据转换
        print('数据转换')
        lgb_train = lgb.Dataset(x1, y1, free_raw_data=False)
        lgb_eval = lgb.Dataset(x2, y2, reference=lgb_train, free_raw_data=False)

        ### 设置初始参数--不含交叉验证参数
        print('设置参数')
        params = {'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'binary_logloss',
                  }

        ### 交叉验证(调参)
        print('交叉验证')
        min_merror = float('Inf')
        best_params = {}

        # 准确率
        print("调参1：提高准确率")
        for num_leaves in range(20, 200, 5):
            for max_depth in range(3, 8, 1):
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth

                cv_results = lgb.cv(params,
                                    lgb_train,
                                    seed=self.seed_number,
                                    nfold=self.n_fold,
                                    metrics=['binary_error'],
                                    early_stopping_rounds=self.early_stop,
                                    verbose_eval=True,
                                    show_stdv=False
                                    )

                mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth

        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']



        # 过拟合
        print("调参2：降低过拟合")
        for max_bin in range(2, 255, 5):
            for min_data_in_leaf in range(10, 200, 5):
                    params['max_bin'] = max_bin
                    params['min_data_in_leaf'] = min_data_in_leaf

                    cv_results = lgb.cv(params,
                                        lgb_train,
                                        seed=self.seed_number,
                                        nfold=self.n_fold,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=self.early_stop,
                                        verbose_eval=True
                                        )

                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['max_bin'] = max_bin
                        best_params['min_data_in_leaf'] = min_data_in_leaf

        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']

        print("调参3：降低过拟合")
        for feature_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for bagging_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for bagging_freq in range(0,50,5):
                    params['feature_fraction'] = feature_fraction
                    params['bagging_fraction'] = bagging_fraction
                    params['bagging_freq'] = bagging_freq

                    cv_results = lgb.cv(params,
                                        lgb_train,
                                        seed=self.seed_number,
                                        nfold=self.n_fold,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=self.early_stop,
                                        verbose_eval=True
                                        )

                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['feature_fraction'] = feature_fraction
                        best_params['bagging_fraction'] = bagging_fraction
                        best_params['bagging_freq'] = bagging_freq

        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']

        print("调参4：降低过拟合")
        for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                    params['lambda_l1'] = lambda_l1
                    params['lambda_l2'] = lambda_l2
                    params['min_split_gain'] = min_split_gain

                    cv_results = lgb.cv(params,
                                        lgb_train,
                                        seed=self.seed_number,
                                        nfold=self.n_fold,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=self.early_stop,
                                        verbose_eval=True
                                        )

                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain

        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']
        params['min_split_gain'] = best_params['min_split_gain']


        print(best_params)

        ### 训练
        params['learning_rate']=0.01
        lgb.train(
                  params,                     # 参数字典
                  lgb_train,                  # 训练集
                  valid_sets=lgb_eval,        # 验证集
                  num_boost_round=2000,       # 迭代次数
                  early_stopping_rounds=50    # 早停次数
                  )

        ### 特征选择
        df = pd.DataFrame(x1.columns.tolist(), columns=['feature'])
        df['importance']=list(lgb.feature_importance())                           # 特征分数
        df = df.sort_values(by='importance',ascending=False)                      # 特征排序
        df.to_excel("./feature_score_20180331.xlsx",index=None,encoding='gbk')    # 保存分数

        print("AUC Score (Train): %f" % metrics.roc_auc_score(y1, lgb.predict_proba(x1)[:, 1]))
        print("AUC Score (Test): %f" % metrics.roc_auc_score(y2, lgb.predict_proba(x2)[:, 1]))
        print('Recall Score (Test): %f' % metrics.recall_score(y2, lgb.predict(x2)))
