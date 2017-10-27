from data_analysis import Additional_Features_Extractor
from data_analysis import Action_Feature_Extractor
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
import matplotlib


def get_op_features_labels_from_pickle():
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

    return fc_user_ops, fc_user_label, sc_user_ops, sc_user_label, tc_user_ops, tc_user_label, action_id


def get_add_features_labels_from_pickle():
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

    return fc_user_features, sc_user_features, tc_user_features


def statistics_for_op_length():
    '''
    本函数用以确定下面函数中的support的值
    fc: 非流失玩家最大动作序列长度20219 最小序列长度1， 流失玩家最大序列长度7323 ，最小序列长度 1 
    选取的最小支持度是16，此时剔除掉了非流失玩家用户193 流失玩家用户1497
    '''
    from pyecharts import Line
    fc_user_ops, fc_user_label, sc_user_ops, sc_user_label, tc_user_ops, tc_user_label, action_id = get_op_features_labels_from_pickle()
    ops_length = [[], []]
    [ops_length[fc_user_label[user]].append(
        len(ops)) for user, ops in fc_user_ops.items()]

    user_length = [[], []]
    c = [0, 0]
    temp = [list(np.sort(ops_length[0])), list(np.sort(ops_length[1]))]
    for support in range(1, max(ops_length[1])):
        # 在1 和 max ops_length[1] 之间集中了大部分的游戏玩家
        for i in [0, 1]:
            for t in temp[i][c[i]:]:
                if t <= support:
                    c[i] += 1
            user_length[i].append(c[i])

    line = Line()
    index = [i for i in range(0, 100)]
    line.add("非流失玩家", index, user_length[0][:100])
    line.add("流失玩家", index, user_length[1][:100])
    line.show_config()
    line.render()
    # 由图观察得到的结论是在8到16长度的动作序列之间，用户就是量较大，观察这部分的玩家的动作信息，看操作量最大的动作
    # 也有可能是单纯的由于玩家对于游戏的不喜欢导致的流失


def features_merge(support=16):
    '''
    由上一个函数获得最小的支持度16
    '''
    fc_user_ops, fc_user_label, sc_user_ops, sc_user_label, tc_user_ops, tc_user_label, action_id = get_op_features_labels_from_pickle()
    fc_user_features, sc_user_features, tc_user_features = get_add_features_labels_from_pickle()
    for i in range(3):
        act_corpus = []
        add_corpus = []
        label = []
        if i == 0:
            for user in fc_user_features:
                if len(fc_user_ops[user]) > support:  # 应该只针对第一天的玩家
                    # 可以在预测时去掉动作的种类和玩家的总时长
                    act_corpus.append(' '.join(fc_user_ops[user]))
                    # temp = [fc_user_features[user][i] for i in [
                    #     1, 2, 3, 4, 8, 9, 11, 14, 21, 22, 23, 26]]
                    temp = []
                    for i in [1, 2, 3, 4, 8, 9, 11, 14, 21, 22, 23, 26]:
                        if i in [9, 11, 14, 21, 22, 23, 26]:
                            temp.append(fc_user_features[user][i] * 1.0 / fc_user_features[user][27])
                        else:
                            temp.append(fc_user_features[user][i])                        
                    add_corpus.append(temp)
                    label.append(fc_user_label[user])
        elif i == 1:
            for user in sc_user_features:
                act_corpus.append(' '.join(sc_user_ops[user]))
                temp = [sc_user_features[user][i]
                        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26]]
                add_corpus.append(temp)
                label.append(sc_user_label[user])
        else:
            for user in tc_user_features:
                act_corpus.append(' '.join(tc_user_ops[user]))
                temp = [tc_user_features[user][i]
                        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26]]
                add_corpus.append(temp)
                label.append(tc_user_label[user])

        vectorizer = CountVectorizer(analyzer=str.split)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(act_corpus))
        X_1 = tfidf.toarray()
        X_2 = np.array(add_corpus)
        print(len(add_corpus))
        print(np.shape(X_1))
        print(np.shape(X_2))
        X = np.concatenate((X_1, X_2), axis=1)
        Y = label
        op = vectorizer.get_feature_names()
        yield X, Y, op, action_id


def xgb_model(X, Y, op, action_id):
    '''
    超参数调试 得到的结果 max_depth 3 n_estimator 20 learning_rate 0.1
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33)
    X_train, X_validate, Y_train, Y_validate = train_test_split(
        X_train, Y_train, test_size=0.2)
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

    # 对动作信息的抽取
    id_action = {v: k for k, v in action_id.items()}
    op_importances_indexes = list(np.argsort(
        list(model.feature_importances_)[0: len(op)]))[::-1]
    # 新的挑取出来的动作信息
    # 新的挑取出来的未经抽取的其余信息

    print('add importance:')
    add_importances_indexs = list(np.argsort(
        list(model.feature_importances_)[len(op):]))
    # print(add_importances_indexs)
    for index in add_importances_indexs:
        print('{} {}'.format(
            index, model.feature_importances_[index + len(op)]))


def dig_deeper_for_ops(file_in, important_ops):
    '''
    是否需要在重要特征中进行抽取
    Args:
        file_in (list): [0] -> user:ops [1]-> user:label        
        important_ops (list): 存储由xgb提取出的重要的动作
    '''
    op_churn = {}
    with open(file_in[0], 'rb') as f_train, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_train)
        user_label = pickle.load(f_label)
        for user, ops in user_ops.items():
            if len(ops) > support:
                if user_label[user] == 0:
                    for op in ops:
                        # 对每一个动作记录非流失玩家的动作数量
                        if op not in op_churn:
                            op_churn[op] = [0, 0]
                            op_churn[op][0] += 1
                else:
                    # 对流失玩家的最后一个动作进行统计
                    if ops[-1] not in op_churn:
                        op_churn[ops[-1]] = [0, 0]
                    op_churn[ops[-1]][1] += 1


# def dig_deeper_for_ops(file_in, ops_list):
#     '''
#     对xgb抽取出的动作进行深入的挖掘
#     Args:
#         file_in (string): database file path
#         ops_list (list):
#     '''
#     conn = sqlite3.connect(self.file_in)
#     c = conn.cursor()
#     query_sql = "SELECT user_id, action, num_days_played, current_day FROM maidian ORDER BY \
#                 user_id, relative_timestamp ASC"

#     op_intervals = {}
#     op_relativetimestamp = {}
#     previous_userid = None
#     previous_relative_timestamp = None
#     previous_day = None
#     previous_num_days_played = None
#     previous_action = None

#     for row in c.execute(query_sql):
#         user_id = row[0]
#         action = row[1]
#         num_days_played = row[2]
#         current_day = row[3]
#         if current_day == day:   # day
#             if previous_userid is None:
#                 pass
#             elif previous_userid == user_id and current_day == previous_day:
#                 if num_days_played != current_day:  # 这里需要指定是首流用户还是次流用户
#                     if previous_action in important_actions:
#                         if previous_action not in op_intervals:
#                             op_intervals[previous_action] = []
#                         op_intervals[previous_action].append(
#                             relative_timestamp - previous_relative_timestamp)  # 只针对非流失用户
#                 # update the features
#             else:
#                 # previous_userid != user_id 如果不等于的话
#                 if previous_day == previous_num_days_played:
#                     # 同样需要指定玩家是否是首流或者次流玩家
#                     if previous_action in important_actions:
#                         # 该操作的进行只针对第一天流失的用户
#                         op_relativetimestamp[previous_action] = previous_relative_timestamp

#                 if action in important_actions:
#                     if day == 1:
#                         fc_op_df[op][0] += 1
#                         pass
#                     elif day == 2:
#                         pass
#                     elif day == 3:
#                         pass

#                 pass
#                 # update the features
#             pass
#     pass


def classify():
    for X, Y, op, action_id in features_merge():
        xgb_model(X, Y, op, action_id)
        break


if __name__ == '__main__':
    classify()
    pass
