import tensorflow as tf
import numpy as np
import sklearn

import sys
sys.path.append('..')
from utils.data_parse import Parser, batch_iter

# 小波变换

class CNN(object):
    '''
    bi-gram & tri-gram feature extractor using CNN
    extract the final 50 ops, get 70% acc using xgboost
    the model should be visualizable 
    run in a bigger data set 
    TODO:Xaiver init
    TODO:Batch Norm 
    TODO:Attention
    '''

    def __init__(self, out_channels=20, dropout_rate=0.5, embedding_size=50, ops_length=100, l2_reg_loss=0.0):
        '''
        '''
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate        
        self.embedding_size = embedding_size    
        self.batch_size = None    
        self.ops_length = ops_length        
        self.l2_reg_loss = l2_reg_loss
        self.is_training = True
        # weights shape: filter_height, filter_width, channel_in, out_channels
        # try longer filter
        self.W = {
            'w_2': tf.Variable(tf.random_normal([2, self.embedding_size, 1, self.out_channels]), trainable=self.is_training),
            'w_3': tf.Variable(tf.random_normal([5, self.embedding_size, 1, self.out_channels]), trainable=self.is_training),
            'w_5': tf.Variable(tf.random_normal([10, self.embedding_size, 1, self.out_channels]), trainable=self.is_training),
            'wf_1': tf.Variable(tf.random_normal([3 * self.out_channels, 128]), trainable=self.is_training),
            'wf_2': tf.Variable(tf.random_normal([128, 2]))
        }
        self.B = {
            'b_2': tf.Variable(tf.random_normal([self.out_channels]), trainable=self.is_training),
            'b_3': tf.Variable(tf.random_normal([self.out_channels]), trainable=self.is_training),
            'b_5': tf.Variable(tf.random_normal([self.out_channels]), trainable=self.is_training),
            'bf_1': tf.Variable(tf.random_normal([128]), trainable=self.is_training),
            'bf_2': tf.Variable(tf.random_normal([2]), trainable=self.is_training)
        }
        # split the X into training, validation, test data
        # input x shape: (batch size, sequence length, embedding size)
        # reshape to batch size, x_height, x_width, in_channels (in_channels ==
        # 1)
        self.X = tf.placeholder(dtype='float', shape=[
                                None, self.ops_length, self.embedding_size])
        self.Y = tf.placeholder(dtype='float', shape=[None, 2])

    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(
            x, w, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def max_pooling(self, x, k):
        '''
        maxpool 失去了一部分信息
        '''
        return tf.nn.max_pool(value=x, ksize=[1, self.ops_length - k + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    def conv_process(self, x):
        h_2 = self.conv2d(x, self.W['w_2'], self.B['b_2'])
        h_2 = self.max_pooling(h_2, 2)
        h_3 = self.conv2d(x, self.W['w_3'], self.B['b_3'])
        h_3 = self.max_pooling(h_3, 5)
        h_5 = self.conv2d(x, self.W['w_5'], self.B['b_5'])
        h_5 = self.max_pooling(h_5, 10)
        return h_2, h_3, h_5

    def process(self, x, is_dropout):
        '''
        Args:
            x (np.array):
            y (np.array):            
            is_dropout (boolean): True for training, False for validation or test 
        '''
        inpt_x = tf.reshape(
            self.X, [self.batch_size, self.ops_length, self.embedding_size, 1])
        l2_loss = tf.constant(0.0)
        h_2, h_3, h_5 = self.conv_process(inpt_x)
        # after filtering if padding='SAME' (5, 50, 50, 20) else if padding='VALID' (5, 49, 1, 20)
        # after max pooling (5, 1, 1, 20) after reshape (5, 20)
        h_2 = tf.reshape(h_2, [-1, self.out_channels])
        h_3 = tf.reshape(h_3, [-1, self.out_channels])
        h_5 = tf.reshape(h_5, [-1, self.out_channels])
        h_4 = tf.concat([h_2, h_3, h_5], axis=1)  # shape (5, 40)
        h_6 = tf.nn.dropout(h_4, self.dropout_rate) if is_dropout else h_4

        inter_res = tf.nn.relu(
            tf.add(tf.matmul(h_6, self.W['wf_1']), self.B['bf_1']))  # shape (5, 2)
        predict = tf.nn.softmax(
            tf.add(tf.matmul(inter_res, self.W['wf_2']), self.B['bf_2']))
        return predict

    def train(self, batch_size, epochs, learning_rate):      
        self.batch_size = batch_size  
        predict = self.process(
            self.X, True) if self.is_training else self.process(x, y, False)
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(self.W['wf_1'])
        l2_loss += tf.nn.l2_loss(self.B['bf_1'])
        l2_loss += tf.nn.l2_loss(self.W['wf_2'])
        l2_loss += tf.nn.l2_loss(self.B['bf_2'])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=predict, labels=tf.cast(self.Y, tf.int32))) + self.l2_reg_loss * l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss_op, global_step=tf.train.get_global_step())
        #accuracy_op = tf.metrics.accuracy(labels=self.Y, predictions=predict)
        correct_predictions = tf.equal(
            tf.argmax(predict, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(
            tf.cast(correct_predictions, "float"), name="accuracy")
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            # do something 
            def step(batch_x, batch_y):
                '''
                single step in training, valitdation and test 
                '''                
                feed_dict = {
                    self.X: batch_x,
                    self.Y: batch_y                    
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
                self.is_training = False
                losses = []
                acces = []
                for x, y, _ in batch_iter(X_eval, Y_eval, ops_length_eval, batch_size, 1, False):
                    loss, acc = step(x, y)
                    losses.append(loss)
                    acces.append(acc)
                return np.mean(losses), np.mean(acces)

            i = 0
            for x, y, _ in batch_iter(parse.X_train, parse.Y_train, parse.ops_length_train, batch_size, epochs, False):
                print('entering training...')
                self.is_training = True
                loss, acc = step(x, y)
                print(
                    'training log info: loss and acc {:.4f}， {:.4f}'.format(loss, acc))
                i += 1
                if i % 10 == 0:                    
                    loss, acc = evaluate(
                        parse.X_validate, parse.Y_validate, parse.ops_length_validate)
                    print(
                        'validation log info: loss and acc {:.4f} {:.4f}'.format(loss, acc))

            loss, acc = evaluate(
                parse.X_test, parse.Y_test, parse.ops_length_test)
            print(
                'test log info: loss and acc {:.2f} {:.2f}'.format(loss, acc))
            _, loss, accuracy = sess.run(
                [train_op, loss_op, accuracy_op], feed_dict={self.X: x, self.Y: y})
        return loss, accuracy


if __name__ == '__main__':
    cnn = CNN()
    cnn.train(batch_size=100, epochs=3, learning_rate=0.01)
