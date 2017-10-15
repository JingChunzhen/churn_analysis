import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Base_Feature_Extractor(object):
    '''
    '''

    def write(file_in, file_out):
        '''
        '''        
        
        pass
    pass

class Action_Feature_Extractor(object):
    '''
    考虑如果数据量太大，可以给动作进行编码，已节省内存
    但相应的需要存储动作编码和动作字符串之间的映射
    '''

    def __init__(file_in, file_out):
        self.fc_user_ops = {}
        self.sc_user_ops = {}
        self.tc_user_ops = {}
        self.fc_user_label = {}
        self.sc_user_label = {}
        self.tc_user_label = {}        
        pass
    
    def action_features_preprocess():
        '''
        '''
        conn = sqlite3.connect(file_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, num_days_played, current_day FROM maidian ORDER BY \
                    user_id, relative_timestamp ASC"
        for row in c.execute(query_sql):
            user_id = row[0]
            action = row[1]
            num_days_played = row[2]
            current_day = row[3]
            if current_day == 1:
                self.fc_user_label[user_id] = 1 if num_days_played == 1 else 0
                if user_id not in self.fc_user_ops:
                    self.fc_user_ops[user_id] = []
                self.fc_user_ops[user_id].append(action)
            elif current_day == 2:
                self.sc_user_label[user_id] = 1 if num_days_played == 2 else 0
                if user_id not in self.sc_user_ops:
                    self.sc_user_ops[user_id] = []
                self.sc_user_ops[user_id].append(action)
            elif current_day == 3:
                self.tc_user_label[user_id] = 1 if num_days_played == 3 else 0
                if user_id not in self.tc_user_ops:
                    self.tc_user_ops[user_id] = []
                self.tc_user_ops[user_id].append(action)
            else:
                pass
            pass

    pass
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
        pickle.dump(sc_labels, f_sc_label)
        pickle.dump(tc_labels, f_tc_label)
    pass


'''
CREATE TABLE maidian (ObjectID INTEGER PRIMARY KEY, riqi INTEGER, user_id INTEGER,\
    action TEXT, zhanli INTEGER, dengji INTEGER, jinbi INTEGER, zuanshi INTEGER, heizuan INTEGER,\
    tili INTEGER, ip TEXT, vip TEXT, xitong TEXT, qudao INTEGER, num_days_played INTEGER, current_day INTEGER, \
    relative_timestamp REAL);
'''


class Additional_Features_Extractor(object):
    '''
    处理埋点数据中除动作特征之外的其余特征，如金币，钻石等
    TODO: 暂未实现记录玩家在一个自然天玩的总时长
    '''

    def __init__(self, file_in, file_out):
        '''
        只存放全局变量
        '''
        self.previous_times = [None] * 7
        self.previous_features = [None] * 7
        self.features = [None] * 7  # TODO
        self.feature_matrix = [[], [], [], [], [], [], []]
        self.op_categories = set()
        self.relative_timestamp = None
        self.fc_user_features = {}
        self.sc_user_features = {}
        self.tc_user_features = {}
        self.fc_user_label = {}
        self.sc_user_label = {}
        self.tc_user_label = {}
        pass

    def features_update():
        for i in range(7):  # 只从除op以外的动作开始进行循环
            if i == 0:
                # 处理动作特征 index 0
                if self.previous_features[0] != self.features[0]:
                    self.op_categories.add(self.features[0])
                    # 记录该用户的动作种类个数 op_categories 做为一个单独的特征不在features_matrix中记录
                    times_interval = self.relative_timestamp - \
                        self.previous_times[0]
                    self.feature_matrix[0].append(times_interval)
                    # update
                    self.previous_features[0] = self.features[0]
                    self.previous_times[0] = self.relative_timestamp
                else:
                    pass
            elif self.features[i] != self.previous_features[i]:
                # 处理其余特征 index 1 -> 6
                features_diff = self.features[i] - self.previous_features[i]
                times_diff = self.relative_timestamp - self.previous_times[i]
                derivative = features_diff * 1.0 / times_diff
                # TODO 将derivative append入一个矩阵列表当中，应该注意的问题是，可能求到的数字过小
                self.feature_matrix[i].append(derivative)
                # update
                self.previous_features[i] = self.features[i]
                self.previous_times[i] = self.relative_timestamp
            else:
                pass
        pass

    def additional_features_preprocess():
        conn = sqlite3.connect(file_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, zhanli, dengji, jinbi, zuanshi, heizuan, tili, \
        num_days_played, current_day, relative_timestamp FROM maidian ORDER BY user_id, relative_timestamp"
        for row in c.execute(query_sql):
            user_id = row[0]
            self.features = [row[i + 1] for i in range(7)]
            # 0 action 1 zhanli 2 dengji 3 jinbi 4 zuanshi 5 heizuan 6 tili
            num_days_played = row[8]
            current_day = row[9]
            self.relative_timestamp = row[10]
            
            if user_id == previous_userid and previous_day == current_day:
                if current_day == 1:
                    # 存储标签
                    fc_user_label[user_id] = 1 if num_days_played == 1 else 0
                    self.features_update()
                elif current_day == 2:
                    sc_user_label[user_id] = 1 if num_days_played == 2 else 0
                    self.features_update()
                elif current_day == 3:
                    tc_user_label[user_id] = 1 if num_days_played == 3 else 0
                    self.features_update()
                else:
                    pass                
            else:
                user_features = [len(self.op_categories)]
                user_features.extend(
                    [np.mean(self.feature_matrix[i]) for i in range(1, 7)])
                if previous_day == 1:
                    if previous_userid not in self.fc_user_features:
                        # if 条件的加入是为了增强程序的鲁棒性
                        self.fc_user_features[previous_userid] = user_features
                elif previous_day == 2:
                    if previous_userid not in self.sc_user_features:
                        self.sc_user_features[previous_userid] = user_features
                elif previous_day == 3:
                    if previous_userid not in self.tc_user_features:
                        self.tc_user_features[previous_userid] = user_features
                else:
                    pass
                
                self.feature_matrix = [[], [], [], [], [], []], []]
                [self.previous_features[i] = self.features[i] for i in range(7)]
                [self.previous_times[i] = self.relative_timestamp for i in range(7)]
                previous_userid = user_id # TODO：bug here 有两个全部发生了变化，也有可能是只有一个发生了变化
                previous_day = current_day
                self.op_categories.clear()
                pass
            pass
    pass



def op_features_extract():
    '''
    剔除不常用的动作信息，将XGBoost的效果调至最优之后，提取关键动作信息
    同时这里还可以使用随机森林来共同提取，并使用关联规则进行验证
    '''
    pass

def features_merge(file_in, file_out):
    '''
    动作信息由XGBoost提取出来之后
    提取出来的动作进行TF-IDF处理，结合其余的特征，得到新的特征
    作为数据的标签,其实两个文件都进行了数据的标签的处理，这里可以之选取其中的一个，来进行
    '''
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
        X=vectorizer.fit_transform(new_corpus).toarray()
        for i in range(np.shape(X)[0]):
            s=sum(list(X[i]))
            for j in range(np.shape(X)[1]):
                X[i][j]=X[i][j] * 1.0 / s
                if X[i][j] != 0:
                    print(X[i][j])
        print(X)

    elif method.lower() == 'tfidf':
        transformer=TfidfTransformer()
        tfidf=transformer.fit_transform(vectorizer.fit_transform(new_corpus))
        X=tfidf.toarray()

    Y=new_labels
    op=vectorizer.get_feature_names()

    return X, Y, op


def excel_parse():
    '''
    提取相关的动作信息和动作的详细说明信息
    '''
    import xlrd
    data=xlrd.open_workbook('./data/狂暴之翼.xlsx')
    table=data.sheet_by_name('actions_index')

    nrows=table.nrows
    op_verbose={}

    for i in range(nrows):
        verbose=table.row_values(i)
        if verbose[0] not in op_verbose:
            op_verbose[verbose[0]]=' '.join(verbose[1:])

    return op_verbose

# 上述为对数据的处理， 下面的代码是对处理之后的数据进行分析


def function_name(file_in):
    '''
    分析数据的种类，观察以自然天进行分割的方法是否降低了流失和非流失用户的动作种类差别
    '''
    x, y, op=load(file_in)
    churn_counts=[[], []]  # index 0 for unchurn user 1 for churn user
    [churn_counts[y[i]].append(
        sum(list(map(lambda n: 0 if n == 0 else 1, x[i])))) for i in range(np.shape(x)[0])]

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
