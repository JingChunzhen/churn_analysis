import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Base_Feature_Extractor(object):
    '''
    提取原始数据库文件中的数据特征或者其（派生属性）作为训练数据，并获得数据标签
    '''
    def features_preprocess():
        '''
        处理数据，将数据按照天来进行分割，分别得到fc，sc，tc用户数据特征和三种用户的标签 即流失:1, 非流失:0
        fc表示first churn, sc表示secondary churn, tc表示third churn
        -> 数据库中的字段如下

            CREATE TABLE maidian (ObjectID INTEGER PRIMARY KEY, riqi INTEGER, user_id INTEGER,\
                action TEXT, zhanli INTEGER, dengji INTEGER, jinbi INTEGER, zuanshi INTEGER, heizuan INTEGER,\
                tili INTEGER, ip TEXT, vip TEXT, xitong TEXT, qudao INTEGER, num_days_played INTEGER, current_day INTEGER, \
                relative_timestamp REAL);

        -> 在动作特征的处理中，考虑到数据量太大，可以给动作进行编码，以节省内存，但需要相应的存储动作编码和动作字符串之间的映射
            在载入特征时，需要剔除不常用动作
        -> 在其余特征（如金币，钻石等）的处理中
            目前只考虑到了玩家动作的种类，玩家动作的连贯性（玩家动作之间的时间间隔），战力，等级，金币，钻石，黑钻，体力等的成长速度
            以他们的均值作为该玩家的特征
        -> 还应该将玩家在一个自然天中最多攒了多少金币，钻石，黑钻，战力等信息纳入考虑范围，由XGBoost进行提取关键的指标
        -> 以及上述指标的最小值，由此一并进入XGBoost进行重要特征的提取
        -> 以及上述指标的变化次数，变化次数反映了玩家在游戏中的活跃程度
        -> 在其余特征的处理中，只遍历一次可将用户特征以天分割，并得到原始数据的派生属性
        '''
        pass

    def write():
        '''将处理完毕的数据写入文件'''
        pass

    def load():
        '''载入处理完成的数据，针对动作, 包含冷门动作的剔除，动作特征可以使用TF-IDF来进行载入，
        所有的载入操作应只针对某一天的数据
        '''
        pass


class Action_Feature_Extractor(Base_Feature_Extractor):

    def __init__(self, file_in):
        self.fc_user_ops = {}
        self.sc_user_ops = {}
        self.tc_user_ops = {}
        self.fc_user_label = {}
        self.sc_user_label = {}
        self.tc_user_label = {}
        self.action_id = {}
        # 保存动作信息和标号之间的映射关系，以节省内存
        self.file_in = file_in
        pass

    def features_preprocess(self):
        conn = sqlite3.connect(self.file_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, num_days_played, current_day FROM maidian ORDER BY \
                    user_id, relative_timestamp ASC"
        actionid = 0
        for row in c.execute(query_sql):
            user_id = row[0]
            action = row[1]

            if action not in self.action_id:
                self.action_id[action] = str(actionid)  # 将这个地方进行字符转换
                actionid += 1

            num_days_played = row[2]
            current_day = row[3]
            if current_day == 1:
                self.fc_user_label[user_id] = 1 if num_days_played == 1 else 0
                if user_id not in self.fc_user_ops:
                    self.fc_user_ops[user_id] = []
                self.fc_user_ops[user_id].append(self.action_id[action])
            elif current_day == 2:
                self.sc_user_label[user_id] = 1 if num_days_played == 2 else 0
                if user_id not in self.sc_user_ops:
                    self.sc_user_ops[user_id] = []
                self.sc_user_ops[user_id].append(self.action_id[action])
            elif current_day == 3:
                self.tc_user_label[user_id] = 1 if num_days_played == 3 else 0
                if user_id not in self.tc_user_ops:
                    self.tc_user_ops[user_id] = []
                self.tc_user_ops[user_id].append(self.action_id[action])
            else:
                pass
            pass
    pass

    def write_in(self, file_out):
        with open(file_out[0], 'a') as f_fc_train, open(file_out[1], 'a') as f_sc_train, open(file_out[2], 'a') as f_tc_train, \
                open(file_out[3], 'wb') as f_fc_label, open(file_out[4], 'wb') as f_sc_label, open(file_out[5], 'wb') as f_tc_label:
            fc_labels = []
            sc_labels = []
            tc_labels = []
            for user in self.fc_user_ops:
                s = ' '.join(self.fc_user_ops[user])
                f_fc_train.write(s + '\n')
                fc_labels.append(self.fc_user_label[user])
            for user in self.sc_user_ops:
                s = ' '.join(self.sc_user_ops[user])
                f_sc_train.write(s + '\n')
                sc_labels.append(self.sc_user_label[user])
            for user in self.tc_user_ops:
                s = ' '.join(self.tc_user_ops[user])
                f_tc_train.write(s + '\n')
                tc_labels.append(self.tc_user_label[user])

            pickle.dump(fc_labels, f_fc_label)
            pickle.dump(sc_labels, f_sc_label)
            pickle.dump(tc_labels, f_tc_label)

    def write(self, file_out):
        with open(file_out[0], 'wb') as f_fc_train, open(file_out[1], 'wb') as f_sc_train, open(file_out[2], 'wb') as f_tc_train, \
                open(file_out[3], 'wb') as f_fc_label, open(file_out[4], 'wb') as f_sc_label, open(file_out[5], 'wb') as f_tc_label, \
                open(file_out[6], 'wb') as f_action_id:
            pickle.dump(self.fc_user_ops, f_fc_train)
            pickle.dump(self.sc_user_ops, f_sc_train)
            pickle.dump(self.tc_user_ops, f_tc_train)
            pickle.dump(self.fc_user_label, f_fc_label)
            pickle.dump(self.sc_user_label, f_sc_label)
            pickle.dump(self.tc_user_label, f_tc_label)
            pickle.dump(self.action_id, f_action_id)
        pass

    def load(self, file_in, method='tfidf', sample_rate=0, minimum_support=5):
        '''
        Args:
            file_in (list): []
            method (string): 默认为tfidf
            sample_rate (float): 只针对流失用户进行采样, 如果是0表示不采样，默认不采样
            minimum_support (integer): 动作的操作数量的阈值，小于这个阈值的动作将会被剔除， 默认为5
        Returns:
            X (np.array): 存放np数组，如果使用tfidf会存放动作的tfidf矩阵
            Y (list): 用户流失与否标签，1表示流失，0表示非流失
            op (dict): 存放tfidf矩阵列索引与相应动作信息之间的映射关系
        '''
        corpus = []
        new_corpus = []
        new_labels = []
        op_counts = {}

        with open(file_in[0], 'rb') as f_train:
            for line in f_train:
                ops = set(line.decode('utf-8').strip().split(' '))
                for op in ops:
                    if op not in op_counts:
                        op_counts[op] = 1
                    else:
                        op_counts[op] += 1

        with open(file_in[0], 'rb') as f_train, open(file_in[1], 'rb') as f_label:
            for line in f_train:
                ops = line.decode('utf-8').strip().split(' ')
                [ops.remove(op)
                 for op in ops if op_counts[op] <= minimum_support]
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
                    if random.randint(0, 100) > sample_rate * 100:
                        sampled_corpus.append(corpus[i])
                        sampled_labels.append(labels[i])
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
        elif method.lower() == 'tfidf':
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(
                vectorizer.fit_transform(new_corpus))
            X = tfidf.toarray()

        Y = new_labels
        op = vectorizer.get_feature_names()

        return X, Y, op


class Additional_Features_Extractor(Base_Feature_Extractor):

    def __init__(self, file_in):
        self.previous_times = [None] * 8
        # 最后一个previous_times用来求最终的时间
        self.previous_features = [None] * 7
        self.features = [None] * 7  # TODO
        self.feature_matrix = [[], [], [], [], [], [], [], []]
        # 最后一个应该是最重要的，表示的是玩家在某一天玩的总时长
        self.op_categories = set()
        self.relative_timestamp = None
        self.fc_user_features = {}
        self.sc_user_features = {}
        self.tc_user_features = {}
        self.fc_user_label = {}
        self.sc_user_label = {}
        self.tc_user_label = {}
        self.max_features = [0] * 6  # 除动作之外的其余指标
        # 1 zhanli 2 dengji 3 jinbi 4 zuanshi 5 heizuan 6 tili 下同
        self.min_features = [sys.maxsize] * 6
        self.features_change_times = [0] * 6
        self.vip = 0
        self.file_in = file_in
        pass

    def features_update(self):
        if self.previous_features[0] != self.features[0]:
            self.op_categories.add(self.features[0])
            # 记录该用户的动作种类个数 op_categories 做为一个单独的特征不在features_matrix中记录
            time_interval = self.relative_timestamp - self.previous_times[0]
            self.feature_matrix[0].append(time_interval)
            # update
            self.previous_features[0] = self.features[0]
            self.previous_times[0] = self.relative_timestamp
        else:
            pass

        for i in range(1, 7):  # 只从除op以外的动作开始进行循环
            if self.features[i] != self.previous_features[i]:
                # 处理其余特征 index 1 -> 6
                features_diff = self.features[i] - self.previous_features[i]
                times_diff = self.relative_timestamp - self.previous_times[i]
                if times_diff != 0:
                    derivative = np.exp(
                        np.log(abs(features_diff)) - np.log(abs(times_diff)))
                    if features_diff < 0:
                        self.feature_matrix[i].append(-1 * derivative)
                    else:
                        self.feature_matrix[i].append(derivative)

                # 记录最大，最小，以及变化次数
                if self.features[i] > self.max_features[i - 1]:
                    self.max_features[i - 1] = self.features[i]
                if self.features[i] < self.min_features[i - 1]:
                    self.min_features[i - 1] = self.features[i]
                self.features_change_times[i - 1] += 1
                # update
                self.previous_features[i] = self.features[i]
                self.previous_times[i] = self.relative_timestamp
        self.previous_times[7] = self.relative_timestamp
        pass

    def features_preprocess(self):
        conn = sqlite3.connect(self.file_in)
        c = conn.cursor()
        c.execute("PRAGMA temp_store = 2")
        query_sql = "SELECT user_id, action, zhanli, dengji, jinbi, zuanshi, heizuan, tili, \
            num_days_played, current_day, relative_timestamp, vip FROM maidian ORDER BY user_id, relative_timestamp"

        previous_day = None
        previous_userid = None
        test_c = 0
        for row in c.execute(query_sql):
            user_id = row[0]
            self.features = [row[i + 1] for i in range(7)]
            # 0 action 1 zhanli 2 dengji 3 jinbi 4 zuanshi 5 heizuan 6 tili
            num_days_played = row[8]
            current_day = row[9]
            self.relative_timestamp = row[10]
            # self.vip = int(float(row[11])) # TODO # 这个特征可以不进行更新
            self.vip = row[11]

            if previous_userid is None:
                for i in range(7):
                    self.previous_features[i] = self.features[i]
                    self.previous_times[i] = self.relative_timestamp
                previous_day = current_day
                previous_userid = user_id

            elif user_id == previous_userid and previous_day == current_day:
                self.features_update()
            else:

                self.feature_matrix[7].append(self.previous_times[7])
                # 增加用户在该自然天的动作的种类 index: 0
                user_features = [len(self.op_categories)]
                # 增加用户的动作上的最大时间间隔  index: 1
                user_features.append(np.max(self.feature_matrix[0]) if len(
                    self.feature_matrix[0]) != 0 else 0)
                # 在用户特征中增加 动作的平均时间间隔， 战力， 等级， 金币， 钻石， 黑钻， 体力特征随时间的比值 index: 2 3 4 5 6 7 8
                [user_features.append(np.nanmean(self.feature_matrix[i]) if len(
                    self.feature_matrix[i]) != 0 else 0) for i in range(7)]
                # 在用户特征中增加 战力， 等级， 金币， 钻石， 黑钻， 体力特征的最大值 index: 9 10 11 12 13 14
                [user_features.append(value) for value in self.max_features]
                # 在用户特征中增加 战力， 等级， 金币， 钻石， 黑钻， 体力特征的最小值 index: 15 16 17 18 19 20
                [user_features.append(value) for value in self.min_features]
                # 在用户特征中增加 战力， 等级， 金币， 钻石， 黑钻， 体力特征的变化次数 index: 21 22 23 24 25 26
                [user_features.append(value)
                 for value in self.features_change_times]

                if previous_day == 1:
                    self.fc_user_label[previous_userid] = 1 if num_days_played == 1 else 0
                    # 在用户特征中增加了用户在该自然天玩的总时长 index: 27
                    user_features.append(self.feature_matrix[7][0])
                    # 在用户特征中增加了用户的vip属性 index: 28
                    user_features.append(0 if self.vip == 0 else 1)
                    self.fc_user_features[previous_userid] = user_features
                    print(len(self.fc_user_features[previous_userid]))
                elif previous_day == 2:
                    self.sc_user_label[previous_userid] = 1 if num_days_played == 2 else 0
                    user_features.append(
                        self.feature_matrix[7][-1] - self.feature_matrix[7][-2])
                    user_features.append(0 if self.vip == 0 else 1)
                    self.sc_user_features[previous_userid] = user_features
                elif previous_day == 3:
                    self.tc_user_label[previous_userid] = 1 if num_days_played == 3 else 0
                    user_features.append(
                        self.feature_matrix[7][-1] - self.feature_matrix[7][-2])
                    user_features.append(0 if self.vip == 0 else 1)
                    self.tc_user_features[previous_userid] = user_features
                else:
                    pass

                if user_id == previous_userid:
                    for i in range(7):
                        self.feature_matrix[i] = []
                else:
                    self.feature_matrix = [[], [], [], [], [], [], [], []]

                for i in range(7):
                    self.previous_features[i] = self.features[i]
                    self.previous_times[i] = self.relative_timestamp
                self.previous_times[7] = self.relative_timestamp  # 更新操作
                previous_userid = user_id  # TODO
                previous_day = current_day
                self.op_categories.clear()
                self.max_features = [0] * 6  # 除动作之外的其余指标
                # 1 zhanli 2 dengji 3 jinbi 4 zuanshi 5 heizuan 6 tili 下同
                self.min_features = [sys.maxsize] * 6
                self.features_change_times = [0] * 6
                self.vip = 0
            # test_c += 1
            # if test_c >= 1500:
            #     break

    def write(self, file_out):
        with open(file_out[0], 'wb') as f_fc_train, open(file_out[1], 'wb') as f_sc_train, open(file_out[2], 'wb') as f_tc_train, \
                open(file_out[3], 'wb') as f_fc_label, open(file_out[4], 'wb') as f_sc_label, open(file_out[5], 'wb') as f_tc_label:
            pickle.dump(self.fc_user_features, f_fc_train)
            pickle.dump(self.sc_user_features, f_sc_train)
            pickle.dump(self.tc_user_features, f_tc_train)
            pickle.dump(self.fc_user_label, f_fc_label)
            pickle.dump(self.sc_user_label, f_sc_label)
            pickle.dump(self.tc_user_label, f_tc_label)
        pass

    def load(self, file_in, sample_rate=0):
        corpus = []
        label = []
        with open(file_in[0], 'rb') as f_train:
            user_features = pickle.load(f_train)
        with open(file_in[1], 'rb') as f_label:
            user_label = pickle.load(f_label)

        sampled_corpus = []
        sampled_label = []
        for user in user_features:
            if sample_rate != 0:
                if user_label[user] != 0:
                    if random.randint(0, 100) > sample_rate * 100:
                        sampled_corpus.append(user_features[user])
                        sampled_label.append(user_label[user])
                else:
                    sampled_corpus.append(user_features[user])
                    sampled_label.append(user_label[user])
            else:
                corpus.append(user_features[user])
                label.append(user_label[user])

        new_corpus = sampled_corpus if sample_rate != 0 else corpus
        new_label = sampled_label if sample_rate != 0 else label
        X = np.array(new_corpus)
        # X = sklearn.preprocessing.scale(np.array(new_corpus))
        Y = new_label
        return X, Y

    def check(self, file_in):
        with open(file_in[0], 'rb') as f_train:
            user_features = pickle.load(f_train)
        with open(file_in[1], 'rb') as f_label:
            user_label = pickle.load(f_label)
        trains = set(user_features.keys())
        labels = set(user_label.keys())
        print(len(trains))  # 17964
        print(len(labels))  # 17736 13624
        print(len(trains & labels))  # 17735  13624
        pass


def op_features_extract():
    '''
    剔除不常用的动作信息，将XGBoost的效果调至最优之后，提取关键动作信息
    同时这里还可以使用随机森林来共同提取，并使用关联规则进行验证
    '''
    pass

