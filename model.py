import os
import pickle

import numpy as np
import pandas as pd
import pyecharts
import time
import xgboost as xgb
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.externals import joblib # 用joblib来存储模型

from data_analysis import load
from data_analysis import excel_parse

#file_in = ['./output/opinfo.txt', './output/churn_user.txt']
file_in = ['./output/sc_train.txt', './output/sc_label.pkl']
#X, Y, op = load(file_in, method='tfidf', sample=True)
X, Y, op = load(file_in)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
X_train, X_validate, Y_train, Y_validate = train_test_split(
    X_train, Y_train, test_size=0.2)
op_verbose = excel_parse()


def writeTo(parent_dir, path, file_name, pd_file, output_format='xlsx'):
    full_path = os.path.join(parent_dir, path)
    os.makedirs(full_path, exist_ok=True)

    timeFlag = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_stamp = '_'.join(timeFlag.split())
    file_name = file_name + "_" + time_stamp
    full_path = os.path.join(full_path, file_name)

    if output_format.lower() == 'csv':
        full_path = full_path + "." + output_format.lower()
        print(full_path)
        pd_file.to_csv(full_path)
    else:
        full_path = full_path + ".xlsx"
        pd_file.to_excel(full_path, "sheet1", index=False, engine='xlsxwriter')


def classify(method):
    '''
    Use multiple model to classify churn uesr 
    '''
    if method.lower() == 'svm':
        '''
        '''
        model = SVC()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_train)  # Y_pred -> np.ndarray Y_train -> list
        print('training score {}'.format(accuracy_score(Y_train, Y_pred)))
        Y_pred = model.predict(X_validate)
        print('validate score {}'.format(accuracy_score(Y_validate, Y_pred)))
        Y_pred = model.predict(X_test)
        print('test score {}'.format(accuracy_score(Y_test, Y_pred)))
        print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
        pass
    elif method.lower() == 'xgboost':
        '''
        TODO：需要先得到动作信息的重要性，观察动作是否是vip的居多 
            暂时得到了一些动作信息，发现这些信息与vip无明显关系，还需要在模型调优之后做更多的测试
        '''

        model = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=100, max_depth=10, subsample=1)
        eval_set = [(X_validate, Y_validate)]
        model.fit(X_train, Y_train, early_stopping_rounds=20,
                  eval_metric="logloss", eval_set=eval_set, verbose=True)

        Y_pred = model.predict(X_train)  # Y_pred -> np.ndarray Y_train -> list
        print('training score {}'.format(accuracy_score(Y_train, Y_pred)))
        Y_pred = model.predict(X_validate)
        print('validate score {}'.format(accuracy_score(Y_validate, Y_pred)))
        Y_pred = model.predict(X_test)
        print('test score {}'.format(accuracy_score(Y_test, Y_pred)))

        print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
        # grid search and k-fold cross validation
        # model.feature_importances_ -> np.ndarray
        importance_index = list(np.argsort(model.feature_importances_))
        importance_feature = list(model.feature_importances_)
        op_importance = []
        ops = []
        importances = []
        for index in importance_index[-1::-1]:
            if importance_feature[index] == 0:
                break

            verbose = ''
            if op[index] in op_verbose:
                verbose = op_verbose[op[index]]

            op_importance.append(
                [op[index], verbose, importance_feature[index]])
            ops.append(op[index])
            importances.append(importance_feature[index])

        # pd_dist = pd.DataFrame(data=op_importance, columns=[
        #                        'op', 'verbose', 'importance'])
        # writeTo(parent_dir='output', path='op_importance_table',
        #         file_name='xgb动作重要性信息', pd_file=pd_dist, output_format='csv')

        xgb_op_importance = {}
        for o, imp in zip(ops, importances):
            xgb_op_importance[o] = imp

        return xgb_op_importance
        # from pyecharts import Bar
        # bar = Bar('动作重要性表')
        # bar.add('', ops, importances)
        # bar.show_config()
        # bar.render()

        # model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100)
        # learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        # n_estimators = [10, 20, 50, 100, 200]
        # max_depth = [10, 20]
        # param_grid = dict(max_depth=max_depth)
        # kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
        # gscv = GridSearchCV(estimator=model,
        #                     param_grid=param_grid,
        #                     scoring=None,
        #                     cv=kfold)

        # params_res = gscv.fit(X_train, Y_train)
        # print('best score is {} best params is {}'.format(
        #     params_res.best_score_, params_res.best_params_))
        # 'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 10

    elif method.lower() == 'random_forest':
        '''
        TODO：需要先得到动作信息的重要性，观察动作是否是vip的居多
        '''
        model = RandomForestClassifier(n_estimators=100, oob_score=True)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_train)  # Y_pred -> np.ndarray Y_train -> list
        print('training score {}'.format(accuracy_score(Y_train, Y_pred)))
        Y_pred = model.predict(X_validate)
        print('validate score {}'.format(accuracy_score(Y_validate, Y_pred)))
        Y_pred = model.predict(X_test)
        print('test score {}'.format(accuracy_score(Y_test, Y_pred)))
        print(precision_recall_fscore_support(Y_test, Y_pred, average=None))

        print(model.max_features)
        print(model.n_features_)  # 4867 # 动作的总数
        print(model.feature_importances_)

        importance_index = list(np.argsort(model.feature_importances_))
        importance_feature = list(model.feature_importances_)
        op_importance = []
        ops = []
        importances = []
        for index in importance_index[-1::-1]:
            if importance_feature[index] == 0:
                break

            verbose = ''
            if op[index] in op_verbose:
                verbose = op_verbose[op[index]]

            op_importance.append(
                [op[index], verbose, importance_feature[index]])
            ops.append(op[index])
            importances.append(importance_feature[index])
        pass

        pd_dist = pd.DataFrame(data=op_importance, columns=[
                               'op', 'verbose', 'importance'])
        # writeTo(parent_dir='output', path='op_importance_table',
        #         file_name='RF动作重要性信息', pd_file=pd_dist, output_format='csv')

        rf_op_importance = {}
        for o, importance in zip(ops, importances):
            #print('{} \t {}'.format(o, importance))
            rf_op_importance[o] = importance

        return rf_op_importance

    elif method.lower() == 'naive_bayes':
        '''
        '''
        model = BernoulliNB()
        pass
    elif method.lower() == 'logistic_regression':
        '''
        '''
        model = LogisticRegression()
        pass
    else:
        raise ValueError("Method must be 'svm', 'xgboost', 'random_forest', 'naive_bayes', 'logistic_regression'."
                         "Got %s instead"
                         % method)
        pass

    #model.fit(X_train, Y_train)
    '''
    TODO: 判断模型是否真正在做预测而不是所有的都预测成1或者0
        预测没有出现这种问题
    '''


def main():
    xgb_op_importance = classify('xgboost')
    rf_op_importance = classify('random_forest')
    #classify('svm')
    # with open('./output/res_op_importance.pkl', 'wb') as f:
    #     pickle.dump(xgb_op_importance, f)
    #     pickle.dump(rf_op_importance, f)
    #     pass


def func():
    with open('./output/res_op_importance.pkl', 'rb') as f:
        xgb_op_importance = pickle.load(f) # -> xgb_op_importance == {}
        rf_op_importance = pickle.load(f)

    '''
    再针对这两个文件进行操作
    '''
    xgb_op = sorted(xgb_op_importance.items(),
                    key=lambda d: d[1], reverse=True)
    rf_op = sorted(rf_op_importance.items(), key=lambda d: d[1], reverse=True)
    set_xgb_op = set()
    set_rf_op = set()

    for op_num in [20, 50, 100, 150, 200, 250, 300, 350, 400]:
        for i in range(op_num):
            set_rf_op.add(rf_op[i][0])
        for i in range(op_num):
            set_xgb_op.add(xgb_op[i][0])

        # 选取两个set中相同的动作
        same_op = set_xgb_op & set_rf_op
        #print(same_op)
        print(op_num)
        print(len(same_op))

    # op_info_importance = []
    # for op in same_op:
    #     if op not in xgb_op_importance:
    #         temp_xgb = 'not found'
    #     if op not in rf_op_importance:
    #         temp_rf = 'not found'
    #     else:
    #         temp_xgb = xgb_op_importance[op]
    #         temp_rf = rf_op_importance[op]
    #     op_info_importance.append([op, temp_xgb, temp_rf])

    # pd_dist = pd.DataFrame(data=op_info_importance, columns=[
    #                        'op', 'xgb重要性', 'rf重要性'])
    # pd_dist.to_csv('./output/res_op_xgb_rf.csv')
    pass


if __name__ == '__main__':
    main()
    #func()
    pass
