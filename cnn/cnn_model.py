import tensorflow as tf
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from data_parser import data_generate, batch_iter


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

    def __init__(self, out_channels=20, dropout_rate=0.5, embedding_size=50, batch_size=32, ops_length=50,
                 learning_rate=1e-1, l2_reg_loss=0.0):
        '''
        '''
        self.out_channels = out_channels  
        self.dropout_rate = dropout_rate
        self.vocab_size = None
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.ops_length = ops_length
        self.learning_rate = learning_rate
        self.l2_reg_loss = l2_reg_loss
        # weights shape: filter_height, filter_width, channel_in, out_channels
        # try longer filter 
        self.W = {
            'w_2': tf.Variable(tf.random_normal([10, self.embedding_size, 1, self.out_channels])),
            'w_3': tf.Variable(tf.random_normal([20, self.embedding_size, 1, self.out_channels])),            
            'w_5': tf.Variable(tf.random_normal([30, self.embedding_size, 1, self.out_channels])),
            'wf_1': tf.Variable(tf.random_normal([3 * self.out_channels, 128])),
            'wf_2': tf.Variable(tf.random_normal([128, 2]))
        }
        self.B = {
            'b_2': tf.Variable(tf.random_normal([self.out_channels])),
            'b_3': tf.Variable(tf.random_normal([self.out_channels])),
            'b_5': tf.Variable(tf.random_normal([self.out_channels])),
            'bf_1': tf.Variable(tf.random_normal([128])),
            'bf_2': tf.Variable(tf.random_normal([2]))
        }
        # split the X into training, validation, test data
        # input x shape: (batch size, sequence length, embedding size)
        # reshape to batch size, x_height, x_width, in_channels (in_channels == 1)
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
        h_2 = self.max_pooling(h_2, 10)
        h_3 = self.conv2d(x, self.W['w_3'], self.B['b_3'])
        h_3 = self.max_pooling(h_3, 20)
        h_5 = self.conv2d(x, self.W['w_5'], self.B['b_5'])
        h_5 = self.max_pooling(h_5, 30)
        return h_2, h_3, h_5

    def process(self, x, y, is_dropout):
        '''
        Args:
            x (np.array):
            y (np.array):            
            is_dropout (boolean): True for training, False for validation or test 
        '''
        inpt_x = tf.reshape(
            x, [self.batch_size, self.ops_length, self.embedding_size, 1])
        l2_loss = tf.constant(0.0)
        h_2, h_3, h_5 = self.conv_process(inpt_x)
        # after filtering if padding='SAME' (5, 50, 50, 20) else if padding='VALID' (5, 49, 1, 20)
        # after max pooling (5, 1, 1, 20) after reshape (5, 20)
        h_2 = tf.reshape(h_2, [-1, self.out_channels])
        h_3 = tf.reshape(h_3, [-1, self.out_channels])
        h_5 = tf.reshape(h_5, [-1, self.out_channels])
        h_4 = tf.concat([h_2, h_3, h_5], axis=1)  # shape (5, 40)
        h_6 = tf.nn.dropout(h_4, self.dropout_rate) if is_dropout else h_4
        
        inter_res = tf.nn.relu(tf.add(tf.matmul(h_6, self.W['wf_1']), self.B['bf_1'])) # shape (5, 2)
        predict = tf.nn.softmax(tf.add(tf.matmul(inter_res, self.W['wf_2']), self.B['bf_2']))
        return predict

    def train(self, x, y, is_training):
        '''
        Args:            
            is_training (boolean): True for training, False for validation or test        
        '''
        predict = self.process(
            x, y, True) if is_training else self.process(x, y, False)
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(self.W['wf_1'])
        l2_loss += tf.nn.l2_loss(self.B['bf_1'])
        l2_loss += tf.nn.l2_loss(self.W['wf_2'])
        l2_loss += tf.nn.l2_loss(self.B['bf_2'])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=predict, labels=tf.cast(self.Y, tf.int32))) + self.l2_reg_loss * l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(
            loss=loss_op, global_step=tf.train.get_global_step())
        #accuracy_op = tf.metrics.accuracy(labels=self.Y, predictions=predict)
        correct_predictions = tf.equal(tf.argmax(predict, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            _, loss, accuracy = sess.run(
                [train_op, loss_op, accuracy_op], feed_dict={self.X: x, self.Y: y})
        return loss, accuracy


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
        self.cnn = CNN()

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
            loss, accuracy = self.cnn.train(x=batch_data, y=batch_label, is_training=False)
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

if __name__ == '__main__':
    metrics = METRIC()
    metrics.data_split()
    metrics.training()
