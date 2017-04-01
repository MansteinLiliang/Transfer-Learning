# -*- coding: utf-8 -*-
import codecs
import logging
import os
import pickle as pk
import re
import sys

import nltk
import numpy as np
from keras.preprocessing import sequence
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)
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
from helpers import util_functions

if not base_path.endswith("/"):
    base_path = base_path + "/"

MAXIMUM_ESSAY_LENGTH = 20000


class EssaySet(object):
    def __init__(self):
        """
        Initialize variables and check essay set type
        """
        self._score = []
        self._text = []
        self._id = []
        self._text_by_ints = []
        self._mask_by_ints = []
        self._clean_text = []
        self._tokens = []
        self._pos = []
        self._clean_stem_text = []
        self._prompt = ""
        self._spelling_errors = []
        self._markup_text = []

    def add_essay(self, essay_text, essay_score, text_by_ints, mask_by_ints):
        """
        Add new (essay_text,essay_score) pair to the essay set.
        essay_text must be a string.
        essay_score must be an int.
        essay_generated should not be changed by the user.
        Returns a confirmation that essay was added.
        """
        # Get maximum current essay id, or set to 0 if this is the first essay added
        self._text_by_ints.append(text_by_ints)
        self._mask_by_ints.append(mask_by_ints)
        if(len(self._id) > 0):
            max_id = max(self._id)
        else:
            max_id = 0
            # Verify that essay_score is an int, essay_text is a string, and essay_generated equals 0 or 1

        try:
            essay_text = essay_text.encode('ascii', 'ignore')
            if len(essay_text) < 5:
                essay_text = "Invalid essay."
        except:
            logger.exception("Could not parse essay into ascii.")

        try:
            # Try conversion of types
            essay_text = str(essay_text)
        except:
            # Nothing needed here, will return error in any case.
            logger.exception("Invalid type for essay text : {1}".format(type(essay_text)))

        if isinstance(essay_text, basestring):
            self._id.append(max_id + 1)
            self._score.append(essay_score)
            # Clean text by removing non digit/work/punctuation characters
            try:
                essay_text = str(essay_text.encode('ascii', 'ignore'))
            except:
                essay_text = (essay_text.decode('utf-8', 'replace')).encode('ascii', 'ignore')
            cleaned_essay = util_functions.sub_chars(essay_text).lower()
            if(len(cleaned_essay) > MAXIMUM_ESSAY_LENGTH):
                cleaned_essay = cleaned_essay[0:MAXIMUM_ESSAY_LENGTH]
            self._text.append(cleaned_essay)
            # Spell correct text using aspell
            cleaned_text, spell_errors, markup_text = util_functions.spell_correct(self._text[len(self._text) - 1])
            self._clean_text.append(cleaned_text)
            self._spelling_errors.append(spell_errors)
            self._markup_text.append(markup_text)
            # Tokenize text
            self._tokens.append(nltk.word_tokenize(self._clean_text[len(self._clean_text) - 1]))
            # Part of speech tag text
            try:
                self._pos.append(nltk.pos_tag(self._clean_text[len(self._clean_text) - 1].rstrip().lstrip().split(" ")))
            except:
                "can't not tag"
                exit(0)
            # Stem spell corrected text
            porter = nltk.PorterStemmer()
            por_toks = " ".join([porter.stem(w) for w in self._tokens[len(self._tokens) - 1]])
            self._clean_stem_text.append(por_toks)

            ret = "text: " + self._text[len(self._text) - 1] + " score: " + str(essay_score)
        else:
            raise util_functions.InputError(essay_text, "arguments need to be in format "
                                                        "(text,score). text needs to be string,"
                                                        " score needs to be int.")

    def update_prompt(self, prompt_text):
        """
        Update the default prompt string, which is "".
        prompt_text should be a string.
        Returns the prompt as a confirmation.
        """
        if(isinstance(prompt_text, basestring)):
            self._prompt = util_functions.sub_chars(prompt_text)
            ret = self._prompt
        else:
            raise util_functions.InputError(prompt_text, "Invalid prompt. Need to enter a string value.")
        return ret


def get_ref_dtype():
    return ref_scores_dtype


# def tokenize(string):
#     tokens = nltk.word_tokenize(string)
#     new_tokens = []
#     for index, token in enumerate(tokens):
#         if token == '@' and (index + 1) < len(tokens):
#             tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
#             # tokens.pop(index)
#         else:
#             new_tokens.append(token)
#     return new_tokens

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
        doc_matrix[i, :], mask[i, :] = sequence.pad_sequences([indices,sent_mask], sent_len, padding='post')
    # transpose the matrix set it shape=(sent_len , doc_len)
    return doc_matrix.transpose((1, 0)), mask.transpose((1, 0)), num_hit, unk_hit, total


def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=6, char_level=False, doc_len=100, sent_len=50):
    """
    :param file_path:
    :param prompt_id:
    :param vocab:
    :param to_lower:
    :param score_index:
    :param char_level:
    :return: data_x, mask_x, data_y, prompt_ids, maxlen_x
    """
    es_set = EssaySet()
    sent = open('./data/prompt' + str(prompt_id) + '.txt', 'r')

    sent = ''.join(sent.readlines())
    indices = []
    sent_mask = []
    # punctuation will also be split
    try:
        essay_text = str(sent.encode('ascii', 'ignore'))
    except:
        essay_text = (sent.decode('utf-8', 'replace')).encode('ascii', 'ignore')
    cleaned_essay = util_functions.sub_chars(essay_text).lower()

    es_set.update_prompt(cleaned_essay)

    logger.info('Reading dataset from: ' + file_path)
    logger.info('Removing sequences with more than ' + str(sent_len) + ' words')
    data_x, data_y, mask_x, prompt_ids = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    with codecs.open(file_path, mode='r', encoding='UTF8', errors='ignore') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id:
                if to_lower:
                    content = content.lower()
                # indices = np.zeros([doc_len, sent_len])
                indices, mask, n_hit, u_hit, tt = get_sents(content, vocab, doc_len, sent_len)
                num_hit += n_hit
                unk_hit +=u_hit
                total += tt
                es_set.add_essay(content, score, indices, mask)
            elif essay_set < 0:
                print essay_set
                print prompt_id
                print "No such essay_set"
                exit(0)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return es_set


def train_batch_generator(es_set, train_feats=None, doc_num=1):
    """
    :param X:
    :param train_feats: is nd_array of extracted features
    :param doc_num:if it's 1, the tensor_shape=(sent_len, sent_num).Or it's shape=(sent_len, sent_num*doc_len)

    :return: X_new, masks_new, y_new
    """
    assert hasattr(es_set, '_friendly_y')
    X, masks, y = (
        es_set._text_by_ints,
        es_set._mask_by_ints,
        es_set._friendly_y
    )
    epoch = 1
    # mask_copy = [masks[0].copy() for i in range(doc_num)]
    # y_copy = [y[0].copy() for i in range(doc_num)]
    while True:
        X_new, masks_new, y_new = shuffle(X, masks, y, random_state=0)
        ''' Used to train just a number of data
        X_copy = [X_new[0].copy() for i in range(doc_num)]
        mask_copy = [masks_new[0].copy() for i in range(doc_num)]
        y_copy = [y_new[0].copy() for i in range(doc_num)]
        # yield epoch, np.hstack(X_copy), np.hstack(mask_copy), np.hstack(y_copy)
        '''
        print "epoch: "+str(epoch)+" begin......"
        for i in xrange(0, len(X_new),doc_num):
            if train_feats!=None:
                yield epoch, np.concatenate(X_new[i:i+doc_num], axis=1), train_feats[i:i+doc_num], np.hstack(masks_new[i:i+doc_num]), np.hstack(y_new[i:i+doc_num])
            else:
                yield epoch, np.concatenate(X_new[i:i + doc_num], axis=1), np.hstack(masks_new[i:i + doc_num]), np.hstack(y_new[i:i + doc_num])
            # yield epoch, np.hstack(X_new[i-doc_num:i]), np.ones(shape=(50, 1600), dtype='int32'), np.hstack(y_new[i-doc_num:i])
        epoch += 1


def dev_test_batch_generator(X, masks, y, doc_num=1, feats=None):
    """
    按照doc_num的长度来生成batch，如果最后不足batch则是动态variable的长度
    :param X:
    :param masks:
    :param y:
    :param doc_num:
    :return:
    """
    for i in xrange(0, len(X), doc_num):
        #   I utilize the indexing trick, out of range index will automated detected
        if feats:
            yield np.concatenate(X[i:i+doc_num],1), feats[i:i+doc_num], np.hstack(masks[i:i+doc_num]), np.hstack(y[i:i+doc_num])
        else:
            yield np.concatenate(X[i:i + doc_num], 1), np.hstack(masks[i:i + doc_num]), np.hstack(y[i:i + doc_num])


def get_query(prompt_id, vocab, sent_len):
    sent = open('./data/prompt' + str(prompt_id) + '.txt', 'r')
    sent = ''.join(sent.readlines())
    indices = []
    sent_mask = []
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
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))
    train_es= read_dataset(train_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    dev_es = read_dataset(dev_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)
    test_es = read_dataset(test_path, prompt_id, vocab, to_lower, doc_len=doc_len, sent_len=sent_len)

    return (train_es, dev_es, test_es, vocab, len(vocab))

def get_cross_data(
        path, prompt_id, vocab_size, doc_len, sent_len,
        tokenize_text=True, to_lower=True,
        sort_by_len=False,vocab_path=None, score_index=6
):
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