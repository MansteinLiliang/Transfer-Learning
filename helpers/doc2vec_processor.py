import codecs
import nltk
import re
import numpy as np
from gensim.models import Doc2Vec, doc2vec
from collections import OrderedDict
from collections import namedtuple
from random import shuffle
import cPickle
import os
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def doc2vec_build(file_path, prompt_id, sent_embedding_size, score_index=6, doc_len=100):
    """
    Return X is tuple list, each tuple means the begin and end tag of one doc
    :param file_path:
    :param prompt_id:
    :param to_lower:
    :param score_index:
    :param doc_len:
    :param sent_len:
    :return:
    """
    # PV-DBOW
    model = Doc2Vec(dm=0, size=sent_embedding_size, negative=5, hs=0, min_count=2, workers=16)
    alldocs = []
    EachDocument = namedtuple('EachDocument', 'words tags')
    with codecs.open(file_path, mode='r', encoding='utf-8', errors='ignore') as input_file:
        input_file.next()
        sent_count = 0
        essay_id_tuple = {}
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            content = content.lower()
            sents = nltk.sent_tokenize(content)
            if essay_set == prompt_id:
                doc_begin = sent_count
                for i, sent in enumerate(sents):
                    if i >= doc_len:
                        break
                    # punctuation will also be retained
                    tokens = nltk.word_tokenize(sent)
                    for index, token in enumerate(tokens):
                        if token == '@' and (index + 1) < len(tokens):
                            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
                            tokens.pop(index)
                        if is_number(token):
                            tokens[index] = '<num>'
                    alldocs.append(EachDocument(tokens, ['SEN_' + str(sent_count)]))
                    sent_count += 1
                doc_end = sent_count # alldocs[doc_begin:doc_end] is the right way
                essay_id_tuple[essay_id] = (doc_begin, doc_end)

    # get prompt query embedding
    sent = open('./data/prompt' + str(prompt_id) + '.txt', 'r')
    sent = ''.join(sent.readlines())
    tokens = nltk.word_tokenize(sent)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
        if is_number(token):
            tokens[index] = '<num>'
    alldocs.append(EachDocument(tokens, ['QUERY']))

    model.build_vocab(alldocs)
    for epoch in range(15):
        shuffle(alldocs)
        model.train(alldocs)
        model.alpha -= 0.001  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.save('./data/my_model.doc2vec')
    cPickle.dump(essay_id_tuple, open('./data/essay_id_tuple.pk', 'w'))
    return model, essay_id_tuple


def read_dataset(file_path, prompt_id, model, essay_dic, score_index=6, sent_embedding_size=100, doc_len=50):
    data_x, data_y, mask_x = [], [], []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            score = float(tokens[score_index])
            if essay_set == prompt_id:
                indices = np.zeros([doc_len, sent_embedding_size], dtype='float32')
                mask = []
                begin_id, end_id = essay_dic[essay_id]
                assert (end_id - begin_id) <= doc_len
                for i in range(end_id-begin_id):
                    mask.append(1)
                    tag = 'SEN_' + str(i+begin_id)
                    indices[i] = model.docvecs[tag]
                for i in range(end_id-begin_id, doc_len):
                    mask.append(0)
                mask_x.append(np.asarray(mask, dtype='int32').reshape((doc_len, 1)))
                data_x.append(indices.reshape((indices.shape[0], 1, indices.shape[-1])))
                data_y.append(score)
    return data_x, mask_x, data_y


def get_doc2vec_data(paths, prompt_id, doc_len, sent_embedding_size):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]
    if os.path.isfile('./data/my_model.doc2vec'):
        model = Doc2Vec.load('./data/my_model.doc2vec')
        essay_id_tuple = cPickle.load(open('./data/essay_id_tuple.pk'))
    else:
        model, essay_id_tuple = doc2vec_build(
            './data/training_set_rel3.tsv', prompt_id,
            sent_embedding_size, doc_len=doc_len
        )
    train_x, train_masks, train_y = read_dataset(train_path, prompt_id, model, essay_id_tuple, sent_embedding_size=sent_embedding_size, doc_len=doc_len)
    dev_x, dev_masks, dev_y = read_dataset(dev_path, prompt_id, model, essay_id_tuple, sent_embedding_size=sent_embedding_size, doc_len=doc_len)
    test_x, test_masks, test_y = read_dataset(test_path, prompt_id, model, essay_id_tuple, sent_embedding_size=sent_embedding_size, doc_len=doc_len)
    return (train_x, train_masks, np.array(train_y)), \
           (dev_x, dev_masks, np.array(dev_y)),\
           (test_x, test_masks, np.array(test_y)), model.docvecs['QUERY']