import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print("********** load embedding **********")

    # ********* word_embedding *********
    words = []
    inputTrainFile = open(train_file_path, 'r', encoding='utf-8')
    for line in inputTrainFile.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())

    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))

    w2v = {}
    inputFile2 = open(embedding_path, 'r', encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)

    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    # ********* pos_embedding *********
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.1, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")

    return word_idx_rev, word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len=75, max_sen_len=45):
    """
    :return:
         doc_id: 文档id
         y_labels: 当前对应的情感原因多标签
         y_pairs: 当前对应的情感原因编号
         x: 每个词对应的向量下标
         distance: 相对距离
    """
    print("load data file :{}".format(input_file))

    # ********* load data *********
    y_labels, y_pairs, x, sen_len, doc_len, distance = [], [], [], [], [], []
    doc_id = []

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)
        y_label, sen_len_tmp, x_tmp, dis_tmp = np.zeros((max_doc_len, 2)), np.zeros(max_doc_len, dtype=np.int32), np.zeros(
            (max_doc_len, max_sen_len), dtype=np.int32), np.zeros(max_doc_len)
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            if i + 1 in pos:
                y_label[i][0] = 1
            if i + 1 in cause:
                y_label[i][1] = 1
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
            dis_tmp[i] = line[0]

        y_labels.append(y_label)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        distance.append(dis_tmp)

    y_labels, x, sen_len, doc_len, distance = map(np.array, [y_labels, x, sen_len, doc_len, distance])
    for var in ['y_labels', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_labels, y_pairs, x, sen_len, doc_len, distance


# word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(200, 50, '../data/clause_keywords.csv',
#                                                                         '../data/w2v_200.txt')
# tr_doc_id, y_labels, r_y_pairs, tr_x, tr_sen_len, tr_doc_len, distance = load_data("../data_combine/train.txt", word_id_mapping)
