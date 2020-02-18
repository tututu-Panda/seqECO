# -*- coding: utf-8 -*-

# @File    : demo2.py
# @Date    : 2020-02-12
# @Author  : 3tu


import tensorflow as tf
import sys

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
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of LSTM stack')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 8, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
tf.app.flags.DEFINE_float('pos', 1.00, 'lambda2')


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, doc_len, keep_prob1, keep_prob2, y_labels, batch_size, test=False):
    for index in batch_index(len(y_labels), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y_labels[index]]
        yield feed_list, len(index)


def bulid_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, RNN = biLSTM, m_RNN = multi_biLSTM):
    """
    构造模型
    """
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)

    sen_len = tf.reshape(sen_len, [-1])
    with tf.name_scope("word_encode"):
        word_encode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + "word_layer")

    sh2 = 2 * FLAGS.n_hidden
    with tf.name_scope('word_attention'):
        w1 = get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = get_weight_varible('word_att_b1', [sh2])
        w2 = get_weight_varible('word_att_w2', [sh2, 1])
        sentence_encode = att_var(word_encode, sen_len, w1, b1, w2)

    doc_len = tf.reshape(doc_len, [-1])
    sentence_encode = tf.reshape(sentence_encode, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    with tf.name_scope("multi_RNN"):
        sentence_mutual_encode = m_RNN(FLAGS.num_layers, sentence_encode, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'pos_sentence_layer')

    with tf.name_scope('sequence_prediction'):
        s1 = tf.reshape(sentence_mutual_encode, [-1, 2 * FLAGS.n_hidden])
        s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)

        w_cause = get_weight_varible('softmax_w_cause', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_cause = get_weight_varible('softmax_b_cause', [FLAGS.n_class])
        pred_emotion = tf.nn.softmax(tf.matmul(s1, w_cause) + b_cause)
        pred_emotion = tf.reshape(pred_emotion, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)

    return pred_emotion, reg


def run():
    save_dir = 'data_combine/'.format(FLAGS.scope)
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()

    """ ********* load embedding ******** """
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim,
                                                                            FLAGS.embedding_dim_pos,
                                                                            'data/clause_keywords.csv',
                                                                            FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    """ ********* build model **********"""
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    # distance = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y_emotion]
    print('build model done!\n')

    """ ********* model loss **********"""
    pred_emotion, reg = bulid_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y_emotion)
    valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
    loss_pos = - tf.reduce_sum(y_emotion * tf.log(pred_emotion)) / valid_num
    loss_op = loss_pos * FLAGS.pos + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    true_y_pos_op = tf.argmax(y_emotion, 2)
    pred_y_pos_op = tf.argmax(pred_emotion, 2)

    """ ********* session begin **********"""
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
        acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
        p_pair_list, r_pair_list, f1_pair_list = [], [], []

        for fold in range(1, 11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold, FLAGS)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_doc_id, tr_y_position, tr_y_cause, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len = load_data_demo(
                'data_combine/' + train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            te_doc_id, te_y_position, te_y_cause, te_y_pairs, te_x, te_sen_len, te_doc_len = load_data_demo(
                'data_combine/' + test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            max_f1_cause, max_f1_pos, max_f1_avg = [-1.] * 3
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2,
                                               tr_y_position, FLAGS.batch_size):
                    _, loss, pred_y, true_y, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_pos_op, true_y_pos_op, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss))
                        acc, p, r, f1 = acc_prf(pred_y, true_y, doc_len_batch)
                        print('emotion_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                    step = step + 1
                # test
                test = [te_x, te_sen_len, te_doc_len, 1., 1., te_y_position, te_y_cause]
                loss, pred_y, true_y, doc_len_batch = sess.run(
                    [loss_op, pred_y_pos_op, true_y_pos_op, doc_len],
                    feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time() - start_time))

                acc, p, r, f1 = acc_prf(pred_y, true_y, doc_len_batch)
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('emotion_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_cause, max_p_cause,
                                                                                        max_r_cause, max_f1_cause))
                if result_avg_cause[-1] / 2. > max_f1_avg:
                    max_f1_avg = result_avg_cause[-1] / 2.
                    result_avg_cause_max = result_avg_cause

                print('Average max emotion: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                    result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2],
                    result_avg_cause_max[3]))
            print('Optimization Finished!\n')
            print('############# fold {} end ###############'.format(fold))
            # fold += 1
            acc_cause_list.append(result_avg_cause_max[0])
            p_cause_list.append(result_avg_cause_max[1])
            r_cause_list.append(result_avg_cause_max[2])
            f1_cause_list.append(result_avg_cause_max[3])

        print_training_info()
        all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, acc_pos_list, p_pos_list, r_pos_list,
                       f1_pos_list]
        acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos = map(lambda x: np.array(x).mean(),
                                                                                   all_results)
        print('\nemotion_predict: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1, 1)))
        print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
        print_time()

def main(_):
    run()


if __name__ == "__main__":
    tf.app.run()
