import tensorflow as tf
import numpy as np
import pickle
import sklearn 
from data_parse import Parser, batch_iter

class LSTM(object):
    '''
    '''

    def __init__(self):
        self.embedding_size = None
        self.seq_max_length = None
        self.learning_rate = None
        self.l2_reg = None
        self.W = {            
            'wf': tf.Variable(tf.random_normal([self.n_hidden_lstm, 2]))
        }
        self.B = {
            'bf': tf.Variable(tf.random_normal([2]))
        }
        self.n_hidden_lstm = None
        self.X = tf.placeholder(dtype='float', shape=[None, self.seq_max_length, self.embedding_size])
        self.Y = tf.placeholder(dtype='float', shape=[None, 2])
        self.seq_lengths = tf.placeholder(dtype='int', shape=[None])        
    
    def dynamic_rnn(self, x, seqlen):
        x = tf.unstack(x, axis=1) 
        # batch_size, seq_max_length, embedding_size -> seq_max_length, batch_size, embedding_size
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_lstm)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
        
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]        
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)        
        outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden_lstm]), index)
        return tf.add(tf.matmul(outputs, self.W['wf']), self.B['bf'])

    def train(x, seqlen):
        pred = dynamic_rnn(x, seqlen)        
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=tf.cast(self.Y, tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op =  optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
        correct_predictions = tf.equal(tf.argmax(predict, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(
                _, loss, acc = [train_op, loss_op, accuracy_op],
                feed_dict={
                    self.X: x, 
                    self.Y: y,
                    self.seqlen = seqlen
                }
            )
        return loss, acc            
        

class METRIC(object):
    '''    
    TODO: model saving and loading
    TODO: early-stopping should be performed in trainning 
    TODO:
    '''

    def __init__(self, batch_size=32, epochs=1, test_size=0.2, validate_size=0.01):
        self.X_train = None
        self.X_test = None
        self.X_validate = None
        self.Y_train = None
        self.Y_test = None
        self.Y_validate = None
        self.test_size = test_size
        self.validate_size = validate_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstm = LSTM()

    def data_split(self):
        res = data_generate(file_in=[
                            '../output/act_fc_train.pkl', '../output/act_fc_label.pkl', './fc_embeddings.model'])
        X, Y = res[0], res[1]
        X_train, self.X_test, Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size)
        self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(
            X_train, Y_train, test_size=self.validate_size)
    
    def evaluate(self, x, y):
        accuracies = []
        losses = []
        for batch_data, batch_label in batch_iter(data=x, label=y, batch_size=self.batch_size,
                                                  epochs=1, shuffle=False):
            loss, accuracy = self.lstm.train(x=batch_data, y=batch_label, is_training=False)
            losses.append(loss)
            accuracies.append(accuracy)
        return np.mean(losses), np.mean(accuracies)

    def training(self):           
        i = 0 
        count = 0
        for batch_data, batch_label in batch_iter(data=self.X_train, label=self.Y_train, batch_size=self.batch_size, # error NoneType object has no len
                                                  epochs=self.epochs, shuffle=False):            
            loss_train, acc_train = self.cnn.train(x=batch_data, y=batch_label, is_training=True) 
            print('loss and acc is {}, {}'.format(loss_train, acc_train))           
            i += 1    
            if i % 10 == 0:
                loss_validate, acc_validate = self.evaluate(self.X_validate, self.Y_validate)                
                if loss_validate <= loss_train:
                    count += 1
                else:
                    count = 0
                if count >= 5:
                    break
                print('log info training loss and accuracy is {}, {}'.format(loss_train, acc_train))
                print('log info validation loss and accuracy is {}, {}'.format(loss_validate, acc_validate))
        _, acc_test = self.evaluate(self.X_test, self.Y_test)
        print('evaluation final accuracy is {}'.format(acc_test))