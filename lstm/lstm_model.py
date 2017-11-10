import tensorflow as tf
import numpy as np
import sklearn
from data_parse import Parser, batch_iter



class Dynamic_LSTM(object):
    '''
    churn predict using dynamic LSTM
    '''

    def __init__(self, embedding_size=50, seq_max_length=100, learning_rate=0.01, l2_reg=0.0, n_hidden_lstm=100):
        self.embedding_size = embedding_size
        self.seq_max_length = seq_max_length
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_hidden_lstm = n_hidden_lstm
        self.W = {
            'wf': tf.Variable(tf.random_normal([self.n_hidden_lstm, 2]))
        }
        self.B = {
            'bf': tf.Variable(tf.random_normal([2]))
        }
        self.X = tf.placeholder(dtype='float', shape=[
                                None, self.seq_max_length, self.embedding_size])
        self.Y = tf.placeholder(dtype='float', shape=[None, 2])
        self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None])

    def dynamic_rnn(self):
        x = tf.unstack(self.X, self.seq_max_length, axis=1)
        # batch_size, seq_max_length, embedding_size -> seq_max_length, batch_size, embedding_size
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_lstm)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=self.seq_lengths)

        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * \
            self.seq_max_length + (self.seq_lengths - 1)        
        outputs = tf.gather(tf.reshape(
            outputs, [-1, self.n_hidden_lstm]), index) 
        return tf.add(tf.matmul(outputs, self.W['wf']), self.B['bf'])

    def train(self, batch_size, epochs):
        pred = self.dynamic_rnn()
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=tf.cast(self.Y, tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(
            loss_op, global_step=tf.train.get_global_step())
        correct_predictions = tf.equal(
            tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(
            tf.cast(correct_predictions, "float"), name="accuracy")
        init = tf.global_variables_initializer()  

        with tf.Session() as sess:
            sess.run(init)
            
            def step(batch_x, batch_y, batch_seqlen):
                '''
                single step in training, valitdation and test 
                '''
                feed_dict = {
                    self.X: batch_x,
                    self.Y: batch_y,
                    self.seq_lengths: batch_seqlen
                }
                _, loss, acc = sess.run(
                    [train_op, loss_op, accuracy_op], feed_dict=feed_dict)
                return loss, acc

            parse = Parser()
            parse.data_split()

            def evaluate(X_eval, Y_eval, ops_length_eval):
                '''
                evaluate in test and validation 
                '''
                losses = []
                acces = []
                for x, y, seqlen in batch_iter(X_eval, Y_eval, ops_length_eval, batch_size, 1, False):
                    loss, acc = step(x, y, seqlen)
                    losses.append(loss)
                    acces.append(acc)
                return np.mean(losses), np.mean(acces)
            
            i = 0
            for x, y, seqlen in batch_iter(parse.X_train, parse.Y_train, parse.ops_length_train, batch_size, epochs, True):
                print('entering training...')
                loss, acc = step(x, y, seqlen) 
                print('training log info: loss and acc {:.4f}， {:.4f}'.format(loss, acc))
                i += 1 
                if i % 10 == 0:
                    # actually this is not true validation, it is still training 
                    loss, acc = evaluate(parse.X_validate, parse.Y_validate, parse.ops_length_validate)
                    print('validation log info: loss and acc {:.4f} {:.4f}'.format(loss, acc))

            loss, acc = evaluate(parse.X_test, parse.Y_test, parse.ops_length_test)
            print('test log info: loss and acc {:.2f} {:.2f}'.format(loss, acc))

if __name__ == '__main__':
    d_lstm = Dynamic_LSTM()
    d_lstm.train(128, 20)
    pass

