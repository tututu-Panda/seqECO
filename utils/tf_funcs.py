import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def get_weight_varible(name, shape):
    return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))


def getmask(length, max_len, out_shape):
    '''
    length shape:[batch_size]
    '''
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)


def softmax_by_length(inputs, length):
    """
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    """
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims=True) + 1e-9
    return inputs / _sum


def biLSTM(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    return tf.concat(outputs, 2)


def LSTM(inputs, sequence_length, n_hidden, scope):
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        scope=scope
    )
    return outputs


def CNNnet(inputs, sequence_length, embedding_size, num_class, filter_sizes, num_filters, keep_prob, l2_reg_lambda):
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(
                tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID', name='pool')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    with tf.name_scope('concat'):
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.name_scope('dropout'):
        h_dropout = tf.nn.dropout(h_pool_flat, keep_prob=keep_prob)

    with tf.name_scope('output'):
        W = tf.Variable(
            tf.truncated_normal(
                [num_filters_total, num_class], stddev=0.1),
            name='W')
        b = tf.Variable(tf.constant(0.1, shape=[num_class]), name='b')
        if l2_reg_lambda:
            W_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(W)
            tf.add_to_collection('losses', W_l2_loss)
        scores = tf.nn.xw_plus_b(h_dropout, W, b, name='scores')
        predictions = tf.argmax(scores, 1, name='predictions')


def att_var(inputs, length, w1, b1, w2):
    """
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    """
    max_len, n_hidden = (inputs.shape[1], inputs.shape[2])
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden])




def batch_index(length, batch_size, test=False):
    """
    用于生成minibatch训练数据
    """
    index = np.arange(length)
    if not test: np.random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if not test and len(ret) < batch_size: break
        yield ret


def acc_prf(pred_y, true_y, doc_len, average='binary'):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1
