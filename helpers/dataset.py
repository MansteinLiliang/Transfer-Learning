# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs
import os
import itertools
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
import operator
import nltk
import logging
import re
import numpy as np
import pickle as pk
from keras.preprocessing import sequence
import pprint
pp = pprint.PrettyPrinter()
import multiprocessing
from collections import OrderedDict
from collections import namedtuple
import sys
from sklearn.utils import shuffle
from . import data_utils as D

logger = logging.getLogger(__name__)
D.set_logger(logger)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}
"""
Defines an essay set object, which encapsulates essays from training and test sets.
Performs spell and grammar checking, tokenization, and stemming.
"""
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions
if not base_path.endswith("/"):
    base_path = base_path + "/"

EOS_TOKEN = "_eos_"


def define_tensor_size(prompt_id):
    """Return sent_len, doc_len
    """
    tensor_ranges = {
        1: (60, 60),
        2: (60, 60),
        3: (60, 30),
        4: (60, 30),
        5: (60, 50),
        6: (60, 50),
        7: (60, 50),
        8: (60, 100)
    }
    return tensor_ranges[prompt_id]


def save_pkl(path, obj):
  with open(path, 'w') as f:
    pk.dump(obj, f)
    print(" [*] save %s" % path)


def load_pkl(path):
  with open(path) as f:
    obj = pk.load(f)
    print(" [*] load %s" % path)
    return obj


def save_npy(path, obj):
  np.save(path, obj)
  print(" [*] save %s" % path)


def load_npy(path):
  obj = np.load(path)
  print(" [*] load %s" % path)
  return obj


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id_array):
    '''
    :param scores_array:
    :param prompt_id_array: int or np.ndarray
    :return:
    '''
    scores_array = np.array(scores_array).astype('float32')
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = (scores_array - low) / (high - low)
    assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array
    arg_type = type(prompt_id_array)
    # assert arg_type in {int, np.ndarray}
    # if arg_type is int:
    #     low, high = asap_ranges[prompt_id_array]
    #     scores_array = scores_array * (high - low) + low
    #     # assert np.all(scores_array >= low) and np.all(scores_array <= high)
    #     scores_array = np.where(
    #         scores_array > high, high,
    #         np.where(scores_array < low, low, scores_array)
    #     )
    # else:
    #     assert scores_array.shape[0] == prompt_id_array.shape[0]
    #     dim = scores_array.shape[0]
    #     low = np.zeros(dim)
    #     high = np.zeros(dim)
    #     for ii in range(dim):
    #         low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
    #     scores_array = scores_array * (high - low) + low
    # return scores_array


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def create_vocab(file_path, prompt_id, tokenize_text, to_lower, maxlen=0, vocab_size=0):
    # assert os.path.isfile('./data/total_vocab.pk')
    # total_vocab = pkl.load(open('./data/total_vocab.pk'))

    logger.info('Creating vocabulary from: ' + file_path)
    if maxlen > 0:
        logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
    total_words, unique_words = 0, 0
    word_freqs = {}
    with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()
                if tokenize_text:
                    content = tokenize(content)
                else:
                    content = content.split()
                if maxlen > 0 and len(content) > maxlen:
                    continue
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    # ''' Vocab is too small no need
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    # '''
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len

    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1

    logger.info('  %i vocab size' % len(vocab))
    ''' Vocab is too small no need
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    '''
    return vocab


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='utf-8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids


def get_sents(string, vocab, doc_len=50, sent_len=50):
    """
    :param string:
    :return: ndarray: matrix'shape is (sent_len, doc_len), mask: shape=(doc_len, sent_len), num_hit, unk_hit, total
    """
    num_hit = 0
    unk_hit = 0
    total = 0
    sents = nltk.sent_tokenize(string)
    doc_matrix = np.zeros(shape=(doc_len, sent_len))
    mask = np.zeros(shape=(doc_len, sent_len))
    for i, sent in enumerate(sents):
        indices = []
        sent_mask = []
        if i >= doc_len:
            break
        # punctuation will also be split
        tokens = nltk.word_tokenize(sent)
        for index, token in enumerate(tokens):
            if token == '@' and (index + 1) < len(tokens):
                tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
                tokens.pop(index)

        for word in tokens:
            sent_mask.append(1)
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            # elif word in set([',','.','?','\'','"','']):
            #     continue
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1
        # If maxlen is provided, any sequence longer
        doc_matrix[i, :], mask[i, :] = sequence.pad_sequences([indices, sent_mask], sent_len, padding='post')
    # transpose the matrix set it shape=(sent_len , doc_len)
    return doc_matrix.transpose((1, 0)), mask.transpose((1, 0)), num_hit, unk_hit, total


def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=6, char_level=False, doc_len=100, sent_len=50):
    """
    Get idxs data with mask
    :returns data_x, mask_x, data_y, sent_len
    """
    logger.info('Reading dataset from: ' + file_path)
    logger.info('Removing sequences with more than ' + str(sent_len) + ' words')
    data_x, data_y, mask_x, prompt_ids = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()

                # indices = np.zeros([doc_len, sent_len])
                indices, mask, n_hit, u_hit, tt = get_sents(content, vocab, doc_len, sent_len)
                num_hit += n_hit
                unk_hit +=u_hit
                total += tt
                mask_x.append(mask)
                data_x.append(indices)
                data_y.append(score)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, mask_x, data_y, sent_len


def train_batch_generator(X, masks, y, doc_num=1, axis=1):
    """
    :param X: (max_len, batch_size)
    :param masks:
    :param y:
    :param doc_num:if it's 1, the tensor_shape=(sent_len, sent_num).Or it's shape=(sent_len, sent_num*doc_len)
    :return: X_new, masks_new, y_new
    """
    epoch = 1
    # mask_copy = [masks[0].copy() for i in range(doc_num)]
    # y_copy = [y[0].copy() for i in range(doc_num)]
    ''' Used to train just a number of data
            X_copy = [X_new[0].copy() for i in range(doc_num)]
            mask_copy = [masks_new[0].copy() for i in range(doc_num)]
            y_copy = [y_new[0].copy() for i in range(doc_num)]
            # yield epoch, np.hstack(X_copy), np.hstack(mask_copy), np.hstack(y_copy)
            '''

    while True:
        if masks is None:
            X_new, y_new = shuffle(X, y)
            print("epoch: " + str(epoch) + " begin......")
            for i in range(0, len(X_new), doc_num):
                # yield epoch, X_new[i - doc_num:i], y_new[i - doc_num:i]
                yield epoch, X_new[i:i+doc_num], y_new[i:i+doc_num]
            epoch += 1
        else:
            X_new, masks_new, y_new = shuffle(X.transpose((1,0)), masks.transpose((1, 0)), y, random_state=0)
            X_new, masks_new = X_new.transpose((1, 0)), masks.transpose((1, 0))
            print("epoch: " + str(epoch) + " begin......")
            for i in range(doc_num, X_new.shape[1], doc_num):
                yield epoch, np.concatenate(X_new[i-doc_num:i], axis=axis), np.hstack(masks_new[i-doc_num:i]), np.hstack(y_new[i-doc_num:i])
                # yield epoch, np.hstack(X_new[i-doc_num:i]), np.ones(shape=(50, 1600), dtype='int32'), np.hstack(y_new[i-doc_num:i])
            epoch += 1


def dev_test_batch_generator(X, masks, y, doc_num=1):
    """
    按照doc_num的长度来生成batch，如果最后不足batch则是动态variable的长度
    :param X:
    :param masks:
    :param y:
    :param doc_num:
    :return:
    """
    for i in range(0, len(X), doc_num):
        #   I utilize the indexing trick, out of range index will automated detected
        yield np.concatenate(X[i:i+doc_num],1), np.hstack(masks[i:i+doc_num]), np.hstack(y[i:i+doc_num])


def get_query(prompt_id, vocab, sent_len):
    return None
    sent_text = open('./data/prompt' + str(prompt_id) + '.txt', 'r').read()
    assert len(essay_text) > 5
    try:
        sent_text = sent_text.encode('ascii', 'ignore')
    except:
        sent_text.decode('utf-8', 'replace').encode('ascii', 'ignore')

    indices = []
    sent_mask = []
    # punctuation will also be split
    tokens = nltk.word_tokenize(sent_text.lower())

    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    for word in tokens:
        sent_mask.append(1)
        if is_number(word):
            indices.append(vocab['<num>'])
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])
    doc_vector, mask = sequence.pad_sequences([indices,sent_mask], sent_len, padding='post')
    return doc_vector, mask


def get_data(paths, prompt_id, vocab_size, doc_len, sent_len, tokenize_text=True, to_lower=True, sort_by_len=False,
             vocab_path=None, score_index=6):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, tokenize_text, to_lower, vocab_size=vocab_size)
        if len(vocab) < vocab_size:
            logger.warning('The vocabualry includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_masks, train_y, train_maxlen = read_dataset(train_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    dev_x, dev_masks, dev_y, dev_maxlen = read_dataset(dev_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    test_x, test_masks, test_y, test_maxlen = read_dataset(test_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)

    return ((train_x, train_masks, np.array(train_y)), (dev_x, dev_masks, np.array(dev_y)),
            (test_x, test_masks, np.array(test_y)), vocab, len(vocab))


def get_cross_data(
        path, prompt_id, vocab_size, doc_len, sent_len,
        tokenize_text=True, to_lower=True, vocab_path=None):
    if not vocab_path:
        vocab = create_vocab(path, prompt_id, tokenize_text, to_lower, vocab_size=vocab_size)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))
    train_es= read_dataset(path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    return (train_es, vocab, len(vocab))


class TextReader(object):
    """
    We need to read prompt_(x) all_data and train on train_data
    """
    def __init__(self, filepath, promp1=1, promp2=2, main_path=None):
        train_path = os.path.join(filepath, "fold_0/train.tsv")
        valid_path = os.path.join(filepath, "fold_0/dev.tsv")
        filepath = os.path.join(filepath, "training_set_rel3.tsv")
        # test_path = os.path.join(data_path, "test.txt")
        vocab_path = os.path.join(
            main_path, "tensorflow_demo/variational_text_tensorflow/vocab.pkl"
        )
        self.prompt1 = promp1
        self.prompt2 = promp2
        if os.path.exists(vocab_path):
            self._load(vocab_path)
        else:
            self._build_vocab(vocab_path, filepath, prompt_id=promp1)
        self.train_data = self._file_to_data(train_path, promp1, "train")
        self.valid_data = self._file_to_data(valid_path, promp1, "valid")
        # self.test_data = self._file_to_data(filepath,3)

        self.idx2word = {v:k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def _build_vocab(self, vocab_path, file_path, prompt_id, vocab_size=4000):
        logger.info('Creating vocabulary from: ' + file_path)
        total_words, unique_words = 0, 0
        word_freqs = {}
        with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
            input_file.next()
            for line in input_file:
                tokens = line.strip().split('\t')
                essay_id = int(tokens[0])
                essay_set = int(tokens[1])
                content = tokens[2].strip()
                score = float(tokens[6])
                if essay_set == prompt_id or prompt_id <= 0:
                    content = content.lower()
                    content = tokenize(content)
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1
                        total_words += 1
        logger.info('  %i total words, %i unique words' % (total_words, unique_words))
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
        if vocab_size <= 0:
            # Choose vocab size automatically by removing all singletons
            vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1:
                   vocab_size += 1
        vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        vcb_len = len(vocab)
        index = vcb_len
        for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
            vocab[word] = index
            index += 1

        logger.info('  %i vocab size' % len(vocab))
        ''' Vocab is too small no need
        for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
              vocab[word] = index
              index += 1
        '''
        save_pkl(vocab_path, vocab)
        self.vocab = vocab

    def get_tokens(self, sent):
        tokens = nltk.word_tokenize(sent)
        for index, token in enumerate(tokens):
            if token == '@' and (index + 1) < len(tokens):
                tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
                tokens.pop(index)
        for i,word in enumerate(tokens):
            if is_number(word):
                tokens[i] = '<num>'
            elif word not in self.vocab:
                tokens[i] = '<unk>'
        return tokens

    def _file_to_data(self, file_path, prompt_id, file_type='train'):
        data = []
        scores = []
        score_index = 6
        with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
            input_file.next()
            for line in input_file:
                tokens = line.strip().split('\t')
                essay_set = int(tokens[1])
                content = tokens[2].strip()
                score = float(tokens[score_index])
                if essay_set == prompt_id or prompt_id <= 0:
                    content = content.lower()
                    data.append(np.array(map(self.vocab.get, self.get_tokens(content))))
                    scores.append(score)
        if file_type == 'train':
            self.train_scores = scores
        else:
            self.valid_scores = scores
        # save_npy(file_path + ".npy", data)
        return data

    def _load(self, vocab_path):
        self.vocab = load_pkl(vocab_path)

    def get_data_from_type(self, data_type):
        if data_type == "train":
          raw_data = self.train_data
        elif data_type == "valid":
          raw_data = self.valid_data
        elif data_type == "test":
          raw_data = self.test_data
        else:
          raise Exception(" [!] Unkown data type %s: %s" % data_type)
        return raw_data

    def onehot(self, data, min_length=None):
        if min_length is None:
            min_length = self.vocab_size
        return np.where(np.bincount(data, minlength=min_length)>=1, 1, 0)

    def gen_nvdm_feats(self, model, data_type="train"):
        """
        Generate data features of NVDM, bag-of-words features
        """
        raw_data = self.get_data_from_type(data_type)
        return [model.get_hidden_features(self.onehot(data), data)[0]
                for data in raw_data if data != []]
        # return [self.onehot(data) for data in raw_data]

    def gen_nasm_feats(self, model, data_type="train", max_len=300):
        """
        Generate data features of NVDM, bag-of-words features
        """
        raw_data = self.get_data_from_type(data_type)
        length = self.arrays_max_len(raw_data)
        data = sequence.pad_sequences(raw_data, min(length, max_len), padding='post')
        return model.get_hidden_features(data)

    def gen_padding_mat(self, data_type="train", max_len=300):
        """
        Generate data features of NVDM, bag-of-words features
        """
        raw_data = self.get_data_from_type(data_type)
        length = self.arrays_max_len(raw_data)
        data = sequence.pad_sequences(raw_data, min(length, max_len), padding='post')
        return data

    def iterator(self, data_type="train"):
        raw_data = self.get_data_from_type(data_type)
        return itertools.cycle(([self.onehot(data), data] for data in raw_data if data != []))

    def get(self, text):
        if type(text) == str:
          text = text.lower()
          text = TreebankWordTokenizer().tokenize(text)

        try:
            data = np.array(map(self.vocab.get, text))
            return self.onehot(data), data
        except:
            unknowns = []
            for word in text:
                if self.vocab.get(word) == None:
                  unknowns.append(word)
            raise Exception(" [!] unknown words: %s" % ",".join(unknowns))

    def random(self, data_type="train"):
        raw_data = self.get_data_from_type(data_type)
        idx = np.random.randint(len(raw_data))
        data = raw_data[idx]
        return self.onehot(data), data

    def arrays_max_len(self, X):
        """
        :param X: Array of lists, [[1,2,3], [1,2]]
        :return:  max length
        """
        lengths = [len(x) for x in X]
        return max(lengths)

    def train_batch_generator(self, batch_size, supervised=True, max_len=300):
        epoch = 1
        X, y = self.gen_padding_mat("train"), \
               get_model_friendly_scores(np.array(self.train_scores).astype('float32'), self.prompt1)
        while True:
            X_new, y_new = shuffle(X, y, random_state=1)
            print("epoch: " + str(epoch) + " begin......")
            for i in range(batch_size, len(X_new), batch_size):
                batch = X_new[i - batch_size:i]
                # length = self.arrays_max_len(batch)
                # batch = sequence.pad_sequences(batch, min(length, max_len), padding='post')
                if not supervised:
                    yield epoch, batch
                else:
                    yield epoch, batch, np.hstack(y[i-batch_size:i])
            epoch += 1
