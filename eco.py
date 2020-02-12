# encoding: utf-8
# @author: pjy
# email: 2019223049247@scu.edu.cn

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.prepare_data import *
from utils.evaluation import *
from utils.tf_funcs import *
from tensorflow.python import debug as tf_dbg

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
tf.app.flags.DEFINE_float('pos', 1.00, 'lambda2')


def bulid_model(word_embedding, pos_embedding, x, sen_len, doc_len, sentence_dis, y, keep_prob1, keep_prob2, RNN=biLSTM,
                CNN=CNNnet):
    x = tf.nn.embedding_lookup(word_embedding, x)
    sentence_dis = tf.nn.embedding_lookup(pos_embedding, sentence_dis)

    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    sh2 = 2 * FLAGS.n_hidden
    sen_len = tf.reshape(sen_len, [-1])
    doc_len = tf.reshape(doc_len, [-1])

    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    # ********* word & sentence layer **********

    with tf.name_scope('word_encode'):
        word_encode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + "word_layer")
    # word_encode = tf.reshape(word_encode, [-1, FLAGS.max_sen_len, sh2])

    with tf.name_scope('word_attention'):
        w1 = get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = get_weight_varible('word_att_b1', [sh2])
        w2 = get_weight_varible('word_att_w2', [sh2, 1])
        sentence_encode = att_var(word_encode, sen_len, w1, b1, w2)

    with tf.name_scope('sentence_layer'):
        sentence_encode = tf.reshape(sentence_encode, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        # sentence_dis = tf.reshape(sentence_dis, [-1, FLAGS.max_doc_len, FLAGS.embedding_dim_pos])
        # sentence_encode_dis = tf.concat([sentence_encode, sentence_dis], axis=2)

    with tf.name_scope("sentence_mutual"):
        # sentence_mutual_encode = RNN(sentence_encode_dis, doc_len, n_hidden=FLAGS.n_hidden,

        sentence_mutual_encode = RNN(sentence_encode, doc_len, n_hidden=FLAGS.n_hidden,
                                     scope=FLAGS.scope + 'pos_sentence_layer')
    # sentence_mutual_encode = tf.concat([sentence_mutual_encode, sentence_dis], axis=2)

    # ********* prediction label **********
    with tf.name_scope('sentence_prediction'):
        sentence_mutual_encode = tf.reshape(sentence_mutual_encode, [-1, 2 * FLAGS.n_hidden])
        s1 = tf.nn.dropout(sentence_mutual_encode, keep_prob=keep_prob2)
        w_pos = get_weight_varible('sigmoid_w_pos', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_pos = get_weight_varible('sigmoid_b_pos', [FLAGS.n_class])
        pred_temp = tf.matmul(s1, w_pos) + b_pos
        pred_pos = tf.nn.sigmoid(pred_temp)
        pred_pos = tf.reshape(pred_pos, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    reg = tf.nn.l2_loss(w_pos) + tf.nn.l2_loss(b_pos)
    return pred_pos, reg


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, doc_len, keep_prob1, keep_prob2, y_labels, batch_size, distance, test=False):
    for index in batch_index(len(y_labels), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y_labels[index], distance[index]]
        yield feed_list, len(index)


def run():
    save_dir = 'data_combine/'.format(FLAGS.scope)
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')

    print_time()
    tf.reset_default_graph()

    # ********* load word and pos embedding ********
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim,
                                                                            FLAGS.embedding_dim_pos,
                                                                            'data/clause_keywords.csv',
                                                                            FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    # ********* build model **********
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    distance = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y_labels = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y_labels, distance]
    pred_y_labels, reg = bulid_model(word_embedding, pos_embedding, x, sen_len, doc_len, distance, y_labels, keep_prob1,
                                     keep_prob2)

    # ********* model loss**********
    valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
    loss_labels = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_labels, logits=pred_y_labels)
    loss_labels = tf.reduce_sum(loss_labels) / valid_num
    # loss_labels = - tf.reduce_sum(y_labels * tf.log(pred_y_labels)) / valid_num
    # loss_labels = - tf.reduce_sum(y_labels * tf.log(pred_y_labels)) / valid_num
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_labels)

    true_y_labels_op = tf.argmax(y_labels, 2)
    pred_y_labels_op = tf.argmax(pred_y_labels, 2)
    print('build model done!\n')

    # ********* load word and pos embedding ********
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # sess = tf_dbg.LocalCLIDebugWrapperSession(sess)
        keep_rate_List, acc_subtask_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], [], []
        o_p_pair_list, o_r_pair_list, o_f1_pair_list = [], [], []

        for fold in range(1, 11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold, FLAGS)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_doc_id, tr_y_labels, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len, tr_distance = load_data(
                save_dir + train_file_name, word_id_mapping, max_sen_len=FLAGS.max_sen_len)
            te_doc_id, te_y_labels, te_y_pairs, te_x, te_sen_len, te_doc_len, te_distance = load_data(
                save_dir + test_file_name, word_id_mapping, max_sen_len=FLAGS.max_sen_len)
            max_acc_subtask, max_f1 = [-1.] * 2
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2,
                                               tr_y_labels, FLAGS.batch_size, tr_distance):
                    _, loss, pred_y, true_y, doc_len_batch = sess.run(
                        [optimizer, loss_labels, pred_y_labels_op, true_y_labels_op, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss))
                        acc, p, r, f1 = acc_prf(pred_y, true_y, doc_len_batch)
                        print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                        # acc, p, r, f1 = acc_prf(pred_y_pos, true_y_pos, doc_len_batch)
                        # print('position_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                    step = step + 1


def main(_):
    run()


if __name__ == "__main__":
    tf.app.run()
