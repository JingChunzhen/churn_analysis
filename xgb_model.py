from data_analysis import Additional_Features_Extractor
from data_analysis import Action_Feature_Extractor
# from data_analysis import features_merge
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import sklearn
from sklearn import metrics
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# X, Y = Additional_Features_Extractor(file_in='./data/kzby.db').load(file_in=['./output/add_fc_train.pkl', './output/add_fc_label.pkl'])
# print(np.shape(X)) # (7142)
# X, Y, _ = Action_Feature_Extractor(file_in='./data/kzby.db').load(file_in=['./output/op_feature/sc_train.txt', './output/op_feature/sc_label.pkl'])
# print(np.shape(X)) # (7142)
# print(np.shape(X))
# print(X[:, 0].shape)
# #print(np.shape(X[:, 0].T))
# temp_x_0 = X[:, 0]
# temp_x_1 = X[:, 1]
# temp_x_6 = X[:, 6]
# temp_x_0.shape = (17964, 1)
# temp_x_1.shape = (17964, 1)
# temp_x_6.shape = (17964, 1)
# X = np.concatenate((temp_x_0, temp_x_1, temp_x_6), axis=1)
# print(X.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
# X_train, X_validate, Y_train, Y_validate = train_test_split(
#     X_train, Y_train, test_size=0.2)

# Additional_Features_Extractor(file_in='./data/kbzy.db').check(file_in=['./output/fc_train.txt', './output/fc_label.pkl'])
# X, Y = Additional_Features_Extractor(file_in='./data/kbzy.db').load(file_in=['./output/fc_train.txt', './output/fc_label.pkl'])
# #X, Y, _ = Action_Feature_Extractor(file_in='./data/kbzy.db').load(file_in=['./output/op_feature/fc_train.txt', './output/op_feature/fc_label.pkl'])
# print(np.shape(X))  # -> 17965 2069
# print(len(Y)) # -> 17965

# 在其余特征中抽取关键特征
# 模型调参很重要 模型训练误差很小，但是验证误差较大，应该是有过拟合现象，先调参再上大点的数据集
# 将代码调至可以一次性跑通，并得到关键性指标


def features_merge():
    '''
    动作信息由XGBoost提取出来之后
    提取出来的动作进行TF-IDF处理，结合其余的特征，得到新的特征
    作为数据的标签,其实两个文件都进行了数据的标签的处理，这里可以之选取其中的一个，来进行
    '''
    '''
    act_fe = Action_Feature_Extractor(file_in='./data/kbzy.db')
    add_fe = Additional_Features_Extractor(file_in='./data/kbzy.db')

    act_fe.features_preprocess()
    add_fe.features_preprocess()
    
    # 以下仅在测试时使用
    act_fe.write(
        file_out=[
            './output/act_fc_train.pkl',
            './output/act_sc_train.pkl',
            './output/act_tc_train.pkl',
            './output/act_fc_label.pkl',
            './output/act_sc_label.pkl',
            './output/act_tc_label.pkl',
            './output/action_id.pkl'
        ]
    ) 
    add_fe.write(
        file_out=[
            './output/add_fc_train.pkl',
            './output/add_sc_train.pkl',
            './output/add_tc_train.pkl',
            './output/add_fc_label.pkl',
            './output/add_sc_label.pkl',
            './output/add_tc_label.pkl'
        ]
    )
    '''
    file_in = [
        './output/act_fc_train.pkl',
        './output/act_sc_train.pkl',
        './output/act_tc_train.pkl',
        './output/act_fc_label.pkl',
        './output/act_sc_label.pkl',
        './output/act_tc_label.pkl',
        './output/action_id.pkl'
    ]

    with open(file_in[0], 'rb') as f_fc_train, open(file_in[1], 'rb') as f_sc_train, open(file_in[2], 'rb') as f_tc_train, \
            open(file_in[3], 'rb') as f_fc_label, open(file_in[4], 'rb') as f_sc_label, open(file_in[5], 'rb') as f_tc_label,\
            open(file_in[6], 'rb') as f_action_id:
        fc_user_ops = pickle.load(f_fc_train)
        sc_user_ops = pickle.load(f_sc_train)
        tc_user_ops = pickle.load(f_tc_train)
        fc_user_label = pickle.load(f_fc_label)
        sc_user_label = pickle.load(f_sc_label)
        tc_user_label = pickle.load(f_tc_label)
        action_id = pickle.load(f_action_id)

    file_in = [
        './output/add_fc_train.pkl',
        './output/add_sc_train.pkl',
        './output/add_tc_train.pkl',
        './output/add_fc_label.pkl',
        './output/add_sc_label.pkl',
        './output/add_tc_label.pkl'
    ]

    with open(file_in[0], 'rb') as f_fc_train, open(file_in[1], 'rb') as f_sc_train, open(file_in[2], 'rb') as f_tc_train:
        fc_user_features = pickle.load(f_fc_train)
        sc_user_features = pickle.load(f_sc_train)
        tc_user_features = pickle.load(f_tc_train)

    for i in range(3):
        act_corpus = []
        add_corpus = []
        label = []
        if i == 0:
            for user in fc_user_features:
                act_corpus.append(' '.join(fc_user_ops[user]))
                temp = [fc_user_features[user][i] for i in [
                    2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]]  # 只增加了动作种类，动作最大时间间隔，和钻石平均增长速度进行判断
                add_corpus.append(temp)
                label.append(fc_user_label[user])
        elif i == 1:
            for user in sc_user_features:
                act_corpus.append(' '.join(sc_user_ops[user]))
                temp = [sc_user_features[user][i]
                        for i in [2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]]
                add_corpus.append(temp)
                label.append(sc_user_label[user])
        else:
            for user in tc_user_features:
                act_corpus.append(' '.join(tc_user_ops[user]))
                temp = [tc_user_features[user][i]
                        for i in [2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]]
                add_corpus.append(temp)
                label.append(tc_user_label[user])

        vectorizer = CountVectorizer(analyzer=str.split)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(act_corpus))  # 这里应该是str 但是却是list的形式
        X_1 = tfidf.toarray()
        X_2 = np.array(add_corpus)
        X = np.concatenate((X_1, X_2), axis=1)
        Y = label
        op = vectorizer.get_feature_names()
        yield X, Y, op, action_id
    '''
    # act_fe add_fe 
    for i in range(3):
        act_corpus = []
        add_corpus = []
        label = []
        if i == 0:
            for user in add_fe.fc_user_features:
                act_corpus.append(' '.join(act_fe.fc_user_ops[user]))
                temp = [add_fe.fc_user_features[user][i] for i in [2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]] # 只增加了动作种类，动作最大时间间隔，和钻石平均增长速度进行判断 
                add_corpus.append(temp)                
                label.append(add_fe.fc_user_label[user])
        elif i == 1:
            for user in add_fe.sc_user_features:
                act_corpus.append(' '.join(act_fe.sc_user_ops[user]))
                temp = [add_fe.sc_user_features[user][i] for i in [2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]]
                add_corpus.append(temp)                      
                label.append(add_fe.sc_user_label[user])
        else:
            for user in add_fe.tc_user_features:
                act_corpus.append(' '.join(act_fe.tc_user_ops[user]))
                temp = [add_fe.tc_user_features[user][i] for i in [2, 3, 4, 8, 7, 5, 1, 11, 6, 27, 0, 28]]
                add_corpus.append(temp)                
                label.append(add_fe.tc_user_label[user])

        vectorizer = CountVectorizer(analyzer=str.split)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(act_corpus))  # 这里应该是str 但是却是list的形式
        X_1 = tfidf.toarray()
        X_2 = np.array(add_corpus)
        X = np.concatenate((X_1, X_2), axis=1)
        Y = label
        op = vectorizer.get_feature_names()
        yield X, Y, op, action_id
        '''


def excel_parse():
    '''
    提取相关的动作信息和动作的详细说明信息
    '''
    import xlrd
    data = xlrd.open_workbook('./data/狂暴之翼.xlsx')
    table = data.sheet_by_name('actions_index')

    nrows = table.nrows
    op_verbose = {}

    for i in range(nrows):
        verbose = table.row_values(i)
        if verbose[0] not in op_verbose:
            op_verbose[verbose[0]] = '/'.join(verbose[1:])
    return op_verbose


def xgb_model(op, action_id, op_verbose, op_clicks):
    '''
    超参数调试 得到的结果 max_depth 3 n_estimator 20 learning_rate 0.1
    '''
    model = xgb.XGBClassifier(
        learning_rate=0.1, n_estimators=20, max_depth=3, subsample=1)
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
    id_action = {v: k for k, v in action_id.items()}
    op_importances_indexes = list(np.argsort(
        list(model.feature_importances_)[0: len(op)]))[::-1]

    # c = 0
    # for index in op_importances_indexes:  # 需要在这个地方加入action_id这个参数进来
    #     op_name = id_action[op[index]]
    #     verbose = op_verbose[op_name] if op_name in op_verbose else ' '
    #     print('{}|{}|{:.2f}|{:.2f}|{:.5f}'.format(id_action[op[index]], verbose, op_clicks[id_action[op[index]]]
    #                                   [0], op_clicks[id_action[op[index]]][1], model.feature_importances_[index]))
    #     c += 1
    #     if c >= 300:
    #         break
    # pass

    
    #for index in [len(op)]

    print(type(model.feature_importances_))    
    #print(model.feature_importances_.shape)
    print(list(model.feature_importances_)[-1:-20:-1])
    #print(len(list(model.feature_importances_)))
    #for importance in list(model.feature_importances_)[len(op), model.feature_importances_.shape[0]]:
    #    print(importance)


def test_xgb():
    model = xgb.XGBClassifier()
    learning_rate = [0.01, 0.1, 0.2, 0.3]
    n_estimators = [20, 50, 100]
    max_depth = [3, 10, 20]
    param_grid = dict(max_depth=max_depth,
                      n_estimators=n_estimators, learning_rate=learning_rate)
    print(param_grid)
    kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
    gscv = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring=None,
                        cv=kfold)

    params_res = gscv.fit(X_train, Y_train)
    print('best score is {} best params is {}'.format(
        params_res.best_score_, params_res.best_params_))
    #'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 10
    pass


def op_statictis(file_in, action_id):
    '''
    此时还需要一个action_id这样的一个字典
    '''
    op_clicks = {}
    sum_users = [0] * 2
    id_action = {v: k for k, v in action_id.items()}
    with open(file_in[0], 'rb') as f_op, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_op)
        user_label = pickle.load(f_label)

        for user, ops in user_ops.items():
            sum_users[user_label[user]] += 1
            for op in ops:
                if op not in op_clicks:
                    op_clicks[op] = [0] * 2
                op_clicks[op][user_label[user]] += 1                      

    new_op_clicks = {id_action[k]: [v[0] * 1.0 / sum_users[0],
                                v[1] * 1.0 / sum_users[1]] for k, v in op_clicks.items()}       
    return new_op_clicks

def op_check(file_in):
    ops_click = [0] * 2
    user_num = [0] * 2
    average_click = [0] * 2
    with open(file_in[0], 'rb') as f_op, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_op)
        user_label = pickle.load(f_label)        
        for user, ops in user_ops.items():
            ops_click[user_label[user]] += len(ops)
            user_num[user_label[user]] += 1 

        average_click = [ops_click[i] * 1.0 / user_num[i] for i in [0, 1]]
    
    return average_click

if __name__ == '__main__':
    op_verbose = excel_parse()

    day = 0
    for X, Y, op, action_id in features_merge():
        if day == 0:
            op_clicks = op_statictis(['./output/act_fc_train.pkl', './output/act_fc_label.pkl'], action_id)
            pass
        elif day == 1:
            op_clicks = op_statictis(['./output/act_sc_train.pkl', './output/act_sc_label.pkl'], action_id)
            pass
        else:
            op_clicks = op_statictis(['./output/act_tc_train.pkl', './output/act_tc_label.pkl'], action_id)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.33)
        X_train, X_validate, Y_train, Y_validate = train_test_split(
            X_train, Y_train, test_size=0.2)
        xgb_model(op, action_id, op_verbose, op_clicks)
        day += 1

# if __name__ == '__main__':
#     average_click = op_check(['./output/act_fc_train.pkl', './output/act_fc_label.pkl'])
#     print(average_click)
#     average_click = op_check(['./output/act_sc_train.pkl', './output/act_sc_label.pkl'])
#     print(average_click)
#     '''
#     [1240.2091851022124, 277.8913425113185] -> 4.476
#     [1499.3808710503843, 463.64921465968587] -> 3.23
#     '''
#     pass
