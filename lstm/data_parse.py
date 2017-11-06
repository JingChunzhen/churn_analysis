import pickle
import re
import sqlite3

import numpy as np
import tensorflow as tf
from gensim.models import word2vec


class Parser(object):
    '''
    store the configure infos using yaml
    '''

    def __init__(self, sql_in='../data/kbzy.db', embedding_size=50, seq_length=100):
        self.sql_in = sql_in
        self.corpus = []
        self.labels = []
        self.action_id = {}
        self.embedding_size = embedding_size
        self.seq_length = None
        self.model = None
        self.regx = re.compile(r'board_layer/board.*?')
        self.seq_length = seq_length 

    def sql_data_base_parse(self, file_out, day=1):
        '''
        replace the chat op by a substitute 
        Args:
            self.sql_in (string): sqlite3 data base filepath
            day (int): which day to extract
            file_out (list): [file path for corpus, file path for labels]
        Returns:
            corpus (list): shape = (num of users, ops length)
            labels (list): shape = (num of users, 2)
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, num_days_played \
            FROM maidian WHERE current_day = {} ORDER BY user_id, relative_timestamp".format(day)

        self.corpus = []
        self.labels = []
        ops = []
        previous_userid = None        
        self.action_id = {}
        i = 1
        for row in c.execute(query_sql):
            user_id = row[0]
            action = 'substitite_for_chat' if self.regx.match(
                row[1]) else row[1]
            num_days_played = row[2]

            if action not in self.action_id:
                self.action_id[action] = i
                i += 1
            if previous_userid is not None and user_id != previous_userid:
                label = [0, 1] if num_days_played == day else [1, 0]
                self.labels.append(label)
                self.corpus.append(ops)
                ops = []
            else:
                ops.append(self.action_id[action])
            previous_userid = user_id
        
        with open(file_out[0], 'wb') as f_ops, open(file_out[1], 'wb') as f_labels:
            pickle.dump(self.corpus, f_ops)
            pickle.dump(self.labels, f_labels)            

    def word2vec_training(self, file_out):
        sentences = []
        for ops in self.corpus:
            sentences.append([str(op) for op in ops])
        self.model = word2vec.Word2Vec(
            sentences, self.embedding_size, min_count=1)  
        self.model.save(file_out)

    def data_generator(self, *file_in):
        '''
        fix file_in -> *file_in with assert
        conditions
        rid ops if the length < 16
        Args:
            corpus ():
            labels ()：        
        Returns:
        '''       
        assert len(file_in) == 3
        ops_length = []
        X = []
        Y = []
        self.model = word2vec.Word2Vec.load(file_in[0])
        with open(file_in[1], 'rb') as f_ops, open(file_in[2], 'rb') as f_labels:
            self.corpus = pickle.load(f_ops)
            self.labels = pickle.load(f_labels)
        #self.seq_length = 100
        padding_vector = np.random.normal(size=self.embedding_size)

        def convert_to_wv(op_id):
            return self.model.wv[str(op_id)] if op_id != 0 else padding_vector
    
        for ops, label in zip(self.corpus, self.labels):
            mask = [0] * self.seq_length
            if len(ops) < self.seq_length:
                ops_length.append(len(ops))   
                for i in range(len(ops)):
                    mask[i] = ops[i]         
            else:
                ops_length.append(self.seq_length)
                for i in range(len(ops[-self.seq_length:])):
                    mask[i] = ops[i]
            
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
    # data_size = len(data)  # like len(list)
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

if __name__ == '__main__':
    parser = Parser()
    # parser.sql_data_base_parse(file_out=['./fc_ops.pkl', './fc_labels.pkl'])
    # parser.word2vec_training(file_out='wv.bin')
    x, y, _ = parser.data_generator(file_in=['wv.bin', 'fc_ops.pkl', 'fc_labels.pkl']) 
    print(np.shape(x))
    print(np.shape(y))
