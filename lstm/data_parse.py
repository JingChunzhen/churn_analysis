import pickle
import tensorflow as tf
import numpy as np
from gensim.models import word2vec

class Parser(object):
    '''
    store the configure infos using yaml
    '''
    def __init__(self, sql_in='./data/kbzy.db', embedding_size=50, seq_length=100):
        self.sql_in = None
        self.corpus = None
        self.labels = None
        self.action_id = None
        self.embedding_size = embedding_size
        self.seq_length = seq_length

    def sql_data_base_parse(self, day):
        '''
        Still need to rid the ops that the num lower than min_count
        Args:
            self.sql_in (string): sqlite3 data base filepath
            day (int): which day to extract
        Returns:
            corpus (list): shape = (num of users, ops length)
            labels (list): shape = (num of users, 2)
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, num_days_played, \
            FROM maidian WHERE current_day = {} ORDER BY user_id, relative_timestamp".format(day)

        self.corpuscorpus = []
        self.labels = []
        ops = []
        previous_userid = None
        previous_day = None
        self.action_id = {}
        i = 1
        for row in c.execute(query_sql):
            user_id = row[0]
            action = row[1]
            num_days_played = row[2]
            if action not in self.action_id:
                self.action_id[action] = i
                i += 1
            if previous_userid is None or user_id != previous_userid:
                label = [0, 1] if num_days_played == day else [1, 0]
                self.labels.append(label)
                self.corpus.append(ops)
                ops = []
            else:
                ops.append(self.action_id[action])
            previous_userid = user_id


    def word2vec_training(self, file_out):        
        sentences = []
        for ops in self.corpus:
            sentences.append([srt(op) for op in ops])
        self.model = word2vec.Word2Vec(sentences, self.embedding_size, min_count=1) # TODO
        self.model.save(file_out)


    def data_generator(self):
        '''
        conditions
        rid ops if the length < 16
        Args:
            corpus ():
            labels ()：        
        Returns:
        '''
        ops_length = []
        X = []
        Y = []
        padding_vector = np.random.normal(size=self.embedding_size)

        def convert_to_wv(op_id):
            return self.model.wv[str(op_id)] if op_id != 0 else padding_vector        
        
        for ops, label in zip(self.corpus, self.labels):
            ops_length.append(len(ops))        
            mask = [0] * self.seq_length # padding
            [mask[i]= ops[i] for i in range(len(ops))]        
            line = list(map(convert_to_wv, mask))
            X.append(line)
            Y.append(label)
        return np.array(X), np.array(Y), ops_length        


def batch_iter(data, labels, ops_length, batch_size, epochs, shuffle):
    '''
    Args:
        data (list)
        labels (list)
    '''
    data_size = len(data)
    data = np.array(data)
    labels = np.array(labels)
    #data_size = len(data)  # like len(list)
    num_batches_per_epoch = int(len(data) / batch_size)

    for epoch in range(epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.range(data_size))
            shuffled_data = data[shuffle_indices]
            shuffled_label = labels[shuffle_indices]
            shuffled_length = ops_length[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_label = label
            shuffled_length = ops_length

        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start: end], shuffled_label[start: end], shuffled_length[start: end]



