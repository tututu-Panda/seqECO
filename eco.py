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
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
tf.app.flags.DEFINE_float('pos', 1.00, 'lambda2')


def bulid_model(word_embedding, pos_embedding,  x, sen_len, doc_len, sentence_dis, keep_prob1, keep_prob2, RNN=biLSTM, CNN=CNNnet):
    x = tf.nn.embedding_lookup(word_embedding, x)
    sentence_dis = tf.nn.embedding_lookup(pos_embedding, sentence_dis)

    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    sh2 = 2 * FLAGS.n_hidden
    sen_len = tf.reshape([sen_len, -1])

    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    # ********* word & sentence layer **********

    with tf.name_scope('word_encode'):
        word_encode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + "word_layer" )
    word_encode = tf.reshape(word_encode, [-1, FLAGS.max_sen_len, sh2])

    with tf.name_scope('word_attention'):
        w1 = get_weight_varible('word_att_w1' , [sh2, sh2])
        b1 = get_weight_varible('word_att_b1' , [sh2])
        w2 = get_weight_varible('word_att_w2' , [sh2, 1])
        sentence_encode = att_var(word_encode, sen_len, w1, b1, w2)
    sentence_encode = tf.reshape(sentence_encode, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    sentence_dis = tf.reshape(sentence_dis[:, :, 0, :], [-1, FLAGS.max_doc_len, FLAGS.embedding_dim_pos])
    sentence_encode_dis = tf.concat([sentence_encode, sentence_dis], axis=2)

    with tf.name_scope("sentence_mutual"):
        sentence_mutual_encode = RNN(sentence_encode_dis, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'pos_sentence_layer')
    sentence_mutual_encode = tf.concat([sentence_mutual_encode, sentence_dis], axis=2)


    # ********* multi label **********
    with tf.name_scope('sentence_prediction'):


def run():
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
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y_position = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause]


def main(_):
    run()


if __name__ == "__main__":
    tf.app.run()
