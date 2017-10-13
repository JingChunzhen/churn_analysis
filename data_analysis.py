import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def raw_data_op_preprocessing(file_in, file_out):
    '''
    TODO: 1.VIP用户流失较少 由得到的动作信息重要性来看并无明显关联
    TODO: 2.动作序列的长度会是一个影响较大的因素 目前已解决使用TFIDF
    TODO: 3.动作的种类即流失玩家的动作种类和非流失玩家的动作种类差别较大，目前以自然天为单位将动作序列分割，实际效果待定
    该函数以时间为单位
    将数据分为第一天流失和非流失，第二天，，，一直到第三天
    可反复调用load函数来获取上述自然天的流失训练数据tfidf和标签
    Args:
        file_in (string): database path 
        file_out (list): pathes head for training data and labels 
    '''
    conn = sqlite3.connect(file_in)
    c = conn.cursor()
    query_sql = "SELECT user_id, action, num_days_played, current_day FROM maidian ORDER BY \
                user_id, relative_timestamp ASC"

    # 存放训练数据
    # fc -> first churn -> 只用来判断第一天是否会流失
    # sc -> secondary churn -> 只用来判断第二天是否会流失
    # tc -> third churn -> 只用来判断第三天是否会流失
    fc_user_op = {}
    sc_user_op = {}
    tc_user_op = {}
    # 存放标签
    fc_user = {}
    sc_user = {}
    tc_user = {}

    for row in c.execute(query_sql):

        user_id = row[0]
        action = row[1]
        num_days_played = row[2]
        current_day = row[3]

        if current_day == 1:
            fc_user[user_id] = 1 if num_days_played == 1 else 0
            if user_id not in fc_user_op:
                fc_user_op[user_id] = []
            fc_user_op[user_id].append(action)
        elif current_day == 2:
            sc_user[user_id] = 1 if num_days_played == 2 else 0
            if user_id not in sc_user_op:
                sc_user_op[user_id] = []
            sc_user_op[user_id].append(action)
        elif current_day == 3:
            tc_user[user_id] = 1 if num_days_played == 3 else 0
            if user_id not in tc_user_op:
                tc_user_op[user_id] = []
            tc_user_op[user_id].append(action)
        else:
            pass

    with open(file_out[0], 'a') as f_fc_train, open(file_out[1], 'a') as f_sc_train, open(file_out[2], 'a') as f_tc_train, \
            open(file_out[3], 'wb') as f_fc_label, open(file_out[4], 'wb') as f_sc_label, open(file_out[5], 'wb') as f_tc_label:
        fc_labels = []
        sc_labels = []
        tc_labels = []
        for user in fc_user_op:
            s = ' '.join(fc_user_op[user])
            f_fc_train.write(s + '\n')
            fc_labels.append(fc_user[user])
        for user in sc_user_op:
            s = ' '.join(sc_user_op[user])
            f_sc_train.write(s + '\n')
            sc_labels.append(sc_user[user])
        for user in tc_user_op:
            s = ' '.join(tc_user_op[user])
            f_tc_train.write(s + '\n')
            tc_labels.append(tc_user[user])

        pickle.dump(fc_labels, f_fc_label)
        pickle.dump(sc_labels, f_sc_label)yingwen
        pickle.dump(tc_labels, f_tc_label)
    pass


def raw_data_additional_features_preprocessing(file_in):
    '''
    统计除动作信息之外的其余信息
    包括动作之间的时间间隔（衡量动作之间的连贯性），等级，金币，黑钻，钻石，体力的增长速度，
    同时也应该包括动作的种类（即玩家在这一个自然天玩了多少种动作）
    上述的统计同样也是以天为单位进行    
    如果上述进行的较好， 或者达到了一定的效果，可以结合之前的动作信息作为特征继续进行分类，以观察最终的效果
    '''
    conn = sqlite3.connect(file_in)
    c = conn.cursor()
    '''
    CREATE TABLE maidian (ObjectID INTEGER PRIMARY KEY, riqi INTEGER, user_id INTEGER,\
     action TEXT, zhanli INTEGER, dengji INTEGER, jinbi INTEGER, zuanshi INTEGER, heizuan INTEGER,\
      tili INTEGER, ip TEXT, vip TEXT, xitong TEXT, qudao INTEGER, num_days_played INTEGER, current_day INTEGER, \
      relative_timestamp REAL);
    '''
    query_sql = "SELECT user_id, action, zhanli, dengji, jinbi, zuanshi, heizuan, tili, \
        num_days_played, current_day, relative_timestamp FROM maidian ORDER BY user_id, relative_timestamp"
    

    fc_user = []
    sc_user = []
    tc_user = []

    previous_userid = None
    previous_times = [None] * 7
    previous_features = [None] * 7
    times = [None] * 7
    features = [None] * 7 # TODO
    feature_matrix = [[], [], [], [], [], []], []] 
    op_categories = set()
    for row in c.execute(query_sql):
        user_id = row[0]                
        features = [row[i + 1] for i in range(7)]
        # 0 action 1 zhanli 2 dengji 3 jinbi 4 zuanshi 5 heizuan 6 tili         
        num_days_played = row[8]
        current_day = row[9]
        relative_timestamp = row[10]

        # 是否需要更换user_id 
        if current_day == 1:
            # 存储标签
            fc_user[user_id] = 1 if num_days_played == 1 else 0
            # 获得特征数据 
            for i in range(7): #只从除op以外的动作开始进行循环
                if i == 0:
                    # 处理动作特征
                    if previous_features[0] != features[0]:
                        op_categories.add(features[0]) # 记录该用户的动作种类个数
                        previous_features[0] = features[0]                
                        times_interval = relative_timestamp - previous_times[0]
                        # TODO 将interval append入一个矩阵列表当中
                        feature_matrix[0].append(times_interval)
                        previous_times[0] = relative_timestamp
                    else:
                        pass                    
                elif features[i] != previous_features[i]:
                    # 处理其余特征
                    features_diff = features[i] - previous_features[i]
                    times_diff = relative_timestamp - previous_times[i]         
                    derivative = features_diff * 1.0 / times_diff
                    # TODO 将derivative append入一个矩阵列表当中
                    feature_matrix[i].append(derivative)
                    previous_features[i] = features[i]           
                    previous_times[i] = relative_timestamp
                    pass
                else:
                    pass
                pass
            if user_id not in fc_user_op:
                fc_user_op[user_id] = []
            fc_user_op[user_id].append(action)


        elif current_day == 2:
            sc_user[user_id] = 1 if num_days_played == 2 else 0
            if user_id not in sc_user_op:
                sc_user_op[user_id] = []
            sc_user_op[user_id].append(action)
        elif current_day == 3:
            tc_user[user_id] = 1 if num_days_played == 3 else 0
            if user_id not in tc_user_op:
                tc_user_op[user_id] = []
            tc_user_op[user_id].append(action)
        else:
            pass         


        pass

    pass


def load(file_in, method='tfidf', sample_rate=0, support=5):
    '''
    本函数需要对操作数量较少的动作进行剔除
    Args:
        file_in (list): []
        method (string): 目前只会用到tfidf, 默认为tfidf
        sample_rate (float): 只针对流失用户进行采样, 如果是0表示不采样，默认不采样
        support (integer): 动作的操作数量的阈值，小于这个阈值的动作将会被剔除， 默认为5
    '''
    corpus = []
    new_corpus = []
    new_labels = []
    op_counts = {}

    # , open(file_in[1], 'rb') as f_label:
    with open(file_in[0], 'rb') as f_train:

        for line in f_train:
            line = line.decode('utf-8').strip()
            ops = set(line.split(' '))
            for op in ops:
                if op not in op_counts:
                    op_counts[op] = 1
                else:
                    op_counts[op] += 1

    with open(file_in[0], 'rb') as f_train, open(file_in[1], 'rb') as f_label:
        for line in f_train:
            line = line.decode('utf-8').strip()
            ops = line.split(' ')
            [ops.remove(op) for op in ops if op_counts[op] <= support]
            line = ' '.join(ops)
            corpus.append(line)

        labels = pickle.load(f_label)

    if sample_rate != 0:

        sampled_corpus = []
        sampled_labels = []
        sample_index = []

        for i in range(len(labels)):
            if labels[i] == 0:
                sampled_corpus.append(corpus[i])
                sampled_labels.append(labels[i])
            else:
                # labels[i] == 1
                if random.randint(0, 100) > sample_rate * 100:
                    sampled_corpus.append(corpus[i])
                    sampled_labels.append(labels[i])
                else:
                    pass

        new_corpus = sampled_corpus
        new_labels = sampled_labels
    else:
        new_corpus = corpus
        new_labels = labels
        pass

    vectorizer = CountVectorizer(analyzer=str.split)

    if method.lower() == 'count':
        X = vectorizer.fit_transform(
            new_corpus).toarray()  # 该步骤仅仅取得了动作的计数而不是比值的大小
    if method.lower() == 'tf':
        X = vectorizer.fit_transform(new_corpus).toarray()
        for i in range(np.shape(X)[0]):
            s = sum(list(X[i]))
            for j in range(np.shape(X)[1]):
                X[i][j] = X[i][j] * 1.0 / s
                if X[i][j] != 0:
                    print(X[i][j])
        print(X)

    elif method.lower() == 'tfidf':
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(new_corpus))
        X = tfidf.toarray()

    Y = new_labels
    op = vectorizer.get_feature_names()

    return X, Y, op


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
            op_verbose[verbose[0]] = ' '.join(verbose[1:])

    return op_verbose

# 上述为对数据的处理， 下面的代码是对处理之后的数据进行分析


def function_name(file_in):
    '''
    分析数据的种类，观察以自然天进行分割的方法是否降低了流失和非流失用户的动作种类差别
    '''
    x, y, op = load(file_in)
    churn_counts = [[], []]  # index 0 for unchurn user 1 for churn user
    [churn_counts[y[i]].append(
        sum(list(map(lambda n:0 if n == 0 else 1, x[i])))) for i in range(np.shape(x)[0])]  # 验证本行代码

    print(np.shape(x))  # (17965, 2069)
    print(len(churn_counts[0]))  # -> 7142
    print(len(churn_counts[1]))  # -> 10823
    for i in [0, 1]:
        # -> 194.96 -> 60
        print('mean {:.2f}'.format(np.mean(churn_counts[i])))
        print('median {:.2f}'.format(
            np.median(churn_counts[i])))  # -> 152 -> 30
        print('max {:.2f}'.format(max(churn_counts[i])))  # -> 874 -> 679
        print('min {:.2f}'.format(min(churn_counts[i])))  # -> 1 -> 0

    '''
    fc
    (17965, 2069)
    7142
    10823
    mean 194.96
    median 152.00
    max 874.00
    min 1.00
    mean 60.80
    median 30.00
    max 679.00
    min 0.00
    sc
    (7142, 2346)
    3513
    3629
    mean 268.96
    median 258.00
    max 981.00
    min 1.00
    mean 112.53
    median 74.00
    max 698.00
    min 1.00
    tc # 三流用户可以暂时忽略不计，因为第四天的服务器只持续了很短的时间，用户是否真的流失了无法确定
    (3513, 2356)
    12
    3501
    mean 290.08
    median 322.00
    max 565.00
    min 64.00
    mean 240.72
    median 218.00
    max 955.00
    min 1.00
    '''


if __name__ == '__main__':
    # raw_data_preprocessing(
    #     file_in='./data/kbzy.db',
    #     file_out=[
    #         './output/fc_train.txt',
    #         './output/sc_train.txt',
    #         './output/tc_train.txt',
    #         './output/fc_label.pkl',
    #         './output/sc_label.pkl',
    #         './output/tc_label.pkl'
    #     ]
    # )

    function_name(file_in=['./output/tc_train.txt', './output/tc_label.pkl'])
    pass
