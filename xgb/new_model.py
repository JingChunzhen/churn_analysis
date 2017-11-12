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
                            temp.append(
                                fc_user_features[user][i] * 1.0 / fc_user_features[user][27])
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

    file_in = ['./output/act_fc_train.pkl', './output/act_fc_label.pkl']
    new_op_clicks = op_statictis(file_in, action_id)
    op_churn = dig_deeper_for_ops(file_in, action_id)
    op_verbose = excel_parse()
    print('op feature importance:')
    for op_index in op_importances_indexes[:300]:
        verbose = op_verbose[id_action[op[op_index]]
            ] if id_action[op[op_index]] in op_verbose else ' '

        temp = [id_action[op[op_index]], verbose, op_churn[op[op_index]][0], op_churn[op[op_index]][1],
            op_churn[op[op_index]][1] * 1.0 / op_churn[op[op_index]][0], new_op_clicks[id_action[op[op_index]]], model.feature_importances_[op_index]]

        # new_op_clicks[id_action[op[op_index]]],

        stemp='|'.join([str(t) for t in temp])
        #table_temp = '|'.join(str_temp)
        print(stemp)
    # 对挑选出来的动作进行深入挖掘
    # 新的挑取出来的动作信息
    # 新的挑取出来的未经抽取的其余信息

    print('add importance:')
    add_importances_indexs = list(np.argsort(
        list(model.feature_importances_)[len(op):]))
    # print(add_importances_indexs)
    for index in add_importances_indexs:
        print('{} {}'.format(
            index, model.feature_importances_[index + len(op)]))


def dig_deeper_for_ops(file_in, action_id):
    '''
    是否需要在重要特征中进行抽取 # 对挑选出来的动作进行深入挖掘，以下是动作的留存比
    Args:
        file_in (list): [0] -> user:ops [1]-> user:label
        important_ops (list): 存储由xgb提取出的重要的动作
    '''
    op_churn = {}
    with open(file_in[0], 'rb') as f_train, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_train)
        user_label = pickle.load(f_label)
        op_categories = set()

        for user, ops in user_ops.items():
            [op_categories.add(op) for op in ops]
        for user, ops in user_ops.items():
            if len(ops) > 16:
                if user_label[user] == 0:
                    for op in op_categories:
                        if op not in op_churn:
                            op_churn[op] = [0, 0]
                        op_churn[op][0] += 1
                else:
                    if ops[-1] not in op_churn:
                        op_churn[ops[-1]] = [0, 0]
                    op_churn[ops[-1]][1] += 1
    return op_churn


def statistics_for_add_features():
    with open('./output/act_fc_train.pkl', 'rb') as f_ops, open('./output/add_fc_train.pkl', 'rb') as f_add,\
            open('./output/add_fc_label.pkl', 'rb') as f_label:
        user_ops = pickle.load(f_ops)
        user_add = pickle.load(f_add)
        user_label = pickle.load(f_label)
        churn_user = []
        unchurn_user = []
        for user, ad in user_add.items():
            if len(user_ops[user]) > 16:
                if user_label[user] == 0:
                    if ad[27] == 0 or ad[24] == 0 or ad[25] == 0 or ad[26] == 0:
                        continue
                    temp = [ad[2], ad[9] * 1.0 / ad[27], ad[22] * 1.0 / ad[27], ad[11] * 1.0 /
                            ad[23], ad[12] * 1.0 / ad[24], ad[13] * 1.0 / ad[25], ad[14] * 1.0 / ad[26]]
                    unchurn_user.append(temp)
                elif user_label[user] == 1:
                    if ad[27] == 0 or ad[24] == 0 or ad[25] == 0 or ad[26] == 0:
                        continue
                    temp = [ad[2], ad[9] * 1.0 / ad[27], ad[22] * 1.0 / ad[27], ad[11] * 1.0 /
                            ad[23], ad[12] * 1.0 / ad[24], ad[13] * 1.0 / ad[25], ad[14] * 1.0 / ad[26]]
                    churn_user.append(temp)
        res_0 = np.mean(a=unchurn_user, axis=0)
        res_1 = np.mean(a=churn_user, axis=0)
        print(res_0)
        print(res_1)


'''
[  6.55735457e+00   1.62013953e+00   4.05850511e-03   1.87753706e+02
   1.40680699e-01   1.45887817e-01   1.25812784e-01]
[  7.34459216e+00   1.42090304e+00   4.50303578e-03   2.39307167e+02
   7.93049880e-02   1.05893425e-01   1.94140189e-01]

[  5.94552488e+00   1.76161117e+00   3.78918888e-03   4.41496689e+03
   8.18580681e+01   4.94309270e+01   9.65293098e+00]
[  6.44433753e+00   1.65508654e+00   4.74055134e-03   6.37536205e+03
   7.06899831e+01   4.28144158e+01   1.69668257e+01]
'''


def classify():
    for X, Y, op, action_id in features_merge():
        xgb_model(X, Y, op, action_id)
        break


def op_statictis(file_in, action_id):
    '''
    此时还需要一个action_id这样的一个字典
    对于重要的动作每一个动作在所有点击该动作的玩家中的比值
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
    new_op_clicks = {}
    for k, v in op_clicks.items():
        a = v[0] * 1.0 / sum_users[0]
        b = v[1] * 1.0 / sum_users[1]
        if b == 0 or a == 0:
            value = str(a) + ' ' + str(b)
        else:
            value = str(a / b)
        new_op_clicks[id_action[k]] = value
    return new_op_clicks


def op_check(file_in):
    '''
    流失玩家和非流失玩家的平均点击量
    '''
    ops_click = [0] * 2
    user_num = [0] * 2
    average_click = [0] * 2
    with open(file_in[0], 'rb') as f_op, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_op)
        user_label = pickle.load(f_label)
        for user, ops in user_ops.items():
            if len(ops) > 16:
                continue
            ops_click[user_label[user]] += len(ops)
            user_num[user_label[user]] += 1
        average_click = [ops_click[i] * 1.0 / user_num[i] for i in [0, 1]]
    return average_click


# def dig_for_add_features():
#     conn = sqlite3.connect(self.file_in)
#     c = conn.cursor()
#     query_sql = "SELECT user_id, action, zhanli, dengji, jinbi, zuanshi, heizuan, tili, \
#             num_days_played, relative_timestamp, vip FROM maidian WHERE current_day=1 ORDER BY \
#                 user_id, relative_timestamp ASC"
#     previous_userid = None

#     previous_features = [None] * 7

#     for row in c.execute(query_sql):
#         user_id = row[]
#         action = row[]
#         zhanli = row[]
#         dengji = row[]
#         jinbi = row[]
#         zuanshi = row[]
#         heizuan = row[]
#         tili = row[]
#         num_days_played = row[]


if __name__ == '__main__':
    # classify()
    for x, y, op, action_id in features_merge():
        xgb_model(x, y, op, action_id)
        break
    
    # op_verbose = excel_parse()
    # for op, verbose in op_verbose.items():
    #     print('{}, {}'.format(op, verbose))
    # pass
