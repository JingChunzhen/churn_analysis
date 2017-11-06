import pickle
import numpy as np
from gensim.models import word2vec
'''
    split data into training data, validation data, test data
'''

def word2vec_train(file_in='../output/act_fc_train.pkl', file_out='./fc_embeddings.model', embedding_size=50):
    with open(file_in, 'rb') as f_ops:
        user_ops = pickle.load(f_ops)
        sentences = []
        for user, ops in user_ops.items():
            sent = [str(op) for op in ops]
            sentences.append(sent)
        model = word2vec.Word2Vec(sentences, embedding_size, min_count=1)
        model.save(file_out)


def data_generate(file_in):
    '''
    Args:
        file_in (list): [training data (pickle), training label (pickle), word2vec model (bin)]
    Returns:
        list: [data, label]
    '''
    model = word2vec.Word2Vec.load(file_in[2])
    with open(file_in[0], 'rb') as f_ops, open(file_in[1], 'rb') as f_label:
        user_ops = pickle.load(f_ops)
        user_label = pickle.load(f_label)
        data = []
        label = []
        for user, ops in user_ops.items():
            if len(ops) >= 50:
                # 这里除了对大于50的进行分类同样要对小于50的分
                corpus = []
                for op in ops[-50:]:
                    corpus.append(model.wv[str(op)])  # shape: (ops_length, embedding_size) == (50, 50)
                data.append(corpus)
                label.append([1, 0] if user_label[user] == 0 else [0, 1])
    return [data, label]


def batch_iter(data, label, batch_size, epochs, shuffle):
    '''
    '''
    data_size = len(data)
    data = np.array(data)
    label = np.array(label)
    #data_size = len(data)  # like len(list)
    num_batches_per_epoch = int(len(data) / batch_size)

    for epoch in range(epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.range(data_size))
            shuffled_data = data[shuffle_indices]
            shuffled_label = label[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_label = label

        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start: end], shuffled_label[start: end]


if __name__ == '__main__':
    #word2vec_train()
    res = data_generate(file_in=['../output/act_fc_train.pkl', '../output/act_fc_label.pkl', './fc_embeddings.model'])
    with open('./training_data.pkl', 'wb') as f_out:
        pickle.dump(res, f_out)    
    

