import tensorflow as tf
import numpy as np
import sklearn
import pickle
from tensorflow.contrib import learn
from data_parser import data_generate, batch_iter


class CNN(object):
    '''
    bi-gram & tri-gram feature extractor using CNN
    仅提取最后50个动作，使用xgboost模型得到的分类准确率为70%
    TODO:Xaiver init
    TODO:Batch Norm 
    TODO:Attention
    '''

    def __init__(self, out_channels=20, dropout_rate=0.5, embedding_size=50, batch_size=5, ops_length=50):
        '''
        '''
        self.out_channels = out_channels  # 20
        self.dropout_rate = dropout_rate
        self.vocab_size = None
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.ops_length = ops_length
        # weights shape: filter_height, filter_width, channel_in, out_channels        
        self.W = {
            'w_2': tf.Variable(tf.random_normal([2, self.embedding_size, 1, self.out_channels])),
            'w_3': tf.Variable(tf.random_normal([3, self.embedding_size, 1, self.out_channels])),
            'wf': tf.Variable(tf.random_normal([2 * self.out_channels, 2]))  # TODO
        }
        self.B = {
            'b_2': tf.Variable(tf.random_normal([self.out_channels])),
            'b_3': tf.Variable(tf.random_normal([self.out_channels])),
            'bf': tf.Variable(tf.random_normal([2]))  # TODO
        }
        # X shape: batch_size, X_height, X_width, in_channels
        # split the X into training, validation, test data
        self.X = tf.placeholder(dtype='float', shape=[
                                None, self.ops_length, self.embedding_size]) # TODO
        self.Y = tf.placeholder(dtype='float', shape=[None, 2])        

    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(
            x, w, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def max_pooling(self, x, k):
        return tf.nn.max_pool(value=x, ksize=[1, self.ops_length - k + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    
    def conv_process(self, x):
        h_2 = self.conv2d(x, self.W['w_2'], self.B['b_2'])
        h_2 = self.max_pooling(h_2, 2)
        h_3 = self.conv2d(x, self.W['w_3'], self.B['b_3'])
        h_3 = self.max_pooling(h_3, 3)
        return h_2, h_3
        pass

    def process(self, x, y):
        inpt_x = tf.reshape(x, [self.batch_size, self.ops_length, self.embedding_size, 1])
        h_2, h_3 = self.conv_process(inpt_x)
        h_2 = tf.reshape(h_2, [-1, self.out_channels]) # 
        h_3 = tf.reshape(h_3, [-1, self.out_channels])
        h_4 = tf.concat([h_2, h_3], axis=1)    
        res = tf.nn.relu(tf.add(tf.matmul(x, self.W['wf']), self.B['bf']))

        # 连接提取出的bigram特征和trigram特征
        # h = tf.concat() # TODO
        # h = tf.concat(values=[], axis)
        # h = tf.add(tf.matmul(h, self.W['wf']), self.B['bf'])
        # h = tf.nn.relu(h)
        # res = tf.nn.dropout(h, self.dropout_rate)  # softmax
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            res_x, res_h_2, res_h_3, res_h_4, res_h_5 = sess.run([inpt_x, h_2, h_3, h_4, h_5], feed_dict={
                self.X: x,
                self.Y: y
            })

            '''
            ValueError: Cannot feed value of shape (5, 50, 50) for\
             Tensor 'Placeholder:0', which has shape '(5, 50, 50, 1)'
            '''
            print(type(inpt_x))
            print(np.shape(inpt_x)) 
            print(type(res_h_2))
            print(np.shape(res_h_2)) # (5, 50, 50 ,20) (5, 49 1, 20) (5, 1, 1, 20) -> (5, 20)
            print(type(res_h_3))
            print(np.shape(res_h_3)) # (5, 50, 50, 20) (5 ,48, 1, 20) (5 ,1, 1, 20) -> (5, 20)                
            print(np.shape(res_h_4)) # (5 40)            

    def test_for_variable_shape():
        pass


if __name__ == '__main__':    
    cnn = CNN()
    res = data_generate(file_in=[
        '../data/act_fc_train.pkl',
        '../data/act_fc_label.pkl',
        './fc_embeddings.model'
    ])

    data, label = res[0], res[1]
    for batch_data, batch_label in batch_iter(data, label, 5, 1, False):
        # print(batch_data.shape) 
        # print(batch_label.shape)
        cnn.process(batch_data, batch_label)
        break        
