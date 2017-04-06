#pylint: skip-file
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import time
import numpy as np
import sklearn.ensemble
import tensorflow as tf
import sys
import os
from helpers import dataset
from helpers.dataset import define_tensor_size
from tensorflow_demo.variational_text_tensorflow.utils import pp
from tensorflow_demo.variational_text_tensorflow.models import NASM, NVDM
from helpers.asap_evaluator import Evaluator
from helpers.w2vEmbReader import W2VEmbReader
from keras.preprocessing import sequence
import keras.backend as K
from helpers import asap_reader as reader


# tensorflow flags
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of rmsprop optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_iter", 5000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 50, "The dimension of latent variable [50]")
flags.DEFINE_integer("vocab_size", 4000, "The dimension of latent variable [4000]")
flags.DEFINE_integer("embed_dim", 50, "The dimension of word embeddings [500]")
flags.DEFINE_integer("max_time_steps", 500, "The max_time_steps of sentences [500]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("batch_size", 32, "batch_size of data")
flags.DEFINE_string("model", "nvdm", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("forward_only", False, "False for training, True for testing [False]")

FLAGS = flags.FLAGS


def gbdt_nvdm(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1
    reader = dataset.TextReader(prefix, main_path=main_path)
    train_y = dataset.get_model_friendly_scores(reader.train_scores, prompt_id)
    dev_y = dataset.get_model_friendly_scores(reader.valid_scores, prompt_id)
    with tf.Session() as sess:
        m = NVDM
        model = m(sess, reader, dataset=FLAGS.dataset,
                  embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
                  learning_rate=FLAGS.learning_rate, max_iter=FLAGS.max_iter,
                  checkpoint_dir=FLAGS.checkpoint_dir)
        if FLAGS.forward_only:
            model.load(FLAGS.checkpoint_dir)
        else:
            model.train(FLAGS)
        train_feats = reader.gen_nvdm_feats(model, "train")
        valid_feats = reader.gen_nvdm_feats(model, "valid")

        model = sklearn.ensemble.GradientBoostingRegressor(
            n_estimators=100, learning_rate=.05,
            max_depth=4, min_samples_leaf=3
        )
        in_start = time.time()
        model.fit(train_feats, train_y)
        test_pred = model.predict(valid_feats)
        evl = Evaluator(
            dataset, prompt_id, 'None',
            np.array(reader.valid_scores).astype('int32'),
            np.array(reader.valid_scores).astype('int32')
        )
        evl.feature_evaluate(test_pred, test_pred)
        in_time = time.time() - in_start
        print('Need Time : ', str(in_time)[:4])


def nasm_main(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1
    prefix += '/fold_0/'
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'

    (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), \
    (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = reader.get_data(
        (train_path, dev_path, test_path), prompt_id, FLAGS.vocab_size,
        0, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None
    )
    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())

    train_x = sequence.pad_sequences(train_x, FLAGS.max_time_steps, padding='post')
    dev_x = sequence.pad_sequences(dev_x, FLAGS.max_time_steps, padding='post')
    dev_y_org = dev_y.astype('int')
    test_y_org = test_y.astype('int')
    train_y = dataset.get_model_friendly_scores(train_y, prompt_id)
    dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
    train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

    # word-embedding1
    # emb_path = "./data/En_vectors.txt"
    # emb_reader = W2VEmbReader(emb_path, emb_dim=FLAGS.embed_dim)
    # U = floatX(np.random.uniform(-0.05, 0.05, size=(FLAGS.vocab_size, FLAGS.embed_dim)))
    # U[0] = np.zeros(shape=(FLAGS.embed_dim, ), dtype=theano.config.floatX)
    # U = emb_reader.get_emb_matrix_given_vocab(vocab, U)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        m = NASM
        model = m(sess, embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim, batch_size=FLAGS.batch_size,
                  learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)
        in_start = time.time()
        if FLAGS.forward_only:
            model.load(FLAGS.checkpoint_dir)
        else:
            model.train(None, train_batch)

        # train_feats = reader.gen_nasm_feats(model, "train")
        # valid_feats = reader.gen_nasm_feats(model, "valid")
        # model = sklearn.ensemble.GradientBoostingRegressor(
        #     n_estimators=100, learning_rate=.05,
        #     max_depth=4, min_samples_leaf=3
        # )
        # model.fit(train_feats, train_y)
        # pred = model.predict(valid_feats)
        # pred = model.get_pred(reader.gen_padding_mat("valid"))

        pred = model.get_pred(dev_x).flatten()
        evl = Evaluator(
            dataset, prompt_id, 'None',
            np.array(dev_y_org).astype('int32'),
            np.array(dev_y_org).astype('int32')
        )
        evl.feature_evaluate(dev_y, pred)
        in_time = time.time() - in_start
        print('Need Time : ', str(in_time)[:4])


def nasm_test(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

    prompt_id = 1
    prompt_id_1 = 2

    prefix += '/fold_0/'
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'

    (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), \
    (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = reader.get_data(
        (train_path, dev_path, test_path), prompt_id, FLAGS.vocab_size,
        0, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None
    )
    (train_x_1, train_y_1, train_pmt_1), (dev_x_1, dev_y_1, dev_pmt_1), \
    (test_x_1, test_y_1, test_pmt_1), vocab_1, vocab_size_1, overal_maxlen_1, num_outputs_1 = reader.get_data(
        (train_path, dev_path, test_path), prompt_id_1, FLAGS.vocab_size,
        0, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None
    )
    # Prompt1
    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())
    train_x = sequence.pad_sequences(train_x, FLAGS.max_time_steps, padding='post')
    dev_x = sequence.pad_sequences(dev_x, FLAGS.max_time_steps, padding='post')
    test_x = sequence.pad_sequences(test_x, FLAGS.max_time_steps, padding='post')
    dev_y_org = dev_y.astype('int')
    test_y_org = test_y.astype('int')
    train_y = dataset.get_model_friendly_scores(train_y, prompt_id)
    dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)

    # Prompt2
    train_y_1 = np.array(train_y_1, dtype=K.floatx())
    train_y_org_1 = train_y_1.astype('int')
    dev_y_1 = np.array(dev_y_1, dtype=K.floatx())
    test_y_1 = np.array(test_y_1, dtype=K.floatx())
    train_x_1 = sequence.pad_sequences(train_x_1, FLAGS.max_time_steps, padding='post')
    dev_x_1 = sequence.pad_sequences(dev_x_1, FLAGS.max_time_steps, padding='post')
    test_x_1 = sequence.pad_sequences(test_x_1, FLAGS.max_time_steps, padding='post')
    dev_y_org_1 = dev_y_1.astype('int')
    test_y_org_1 = test_y_1.astype('int')
    train_y_1 = dataset.get_model_friendly_scores(train_y_1, prompt_id_1)
    dev_y_1 = dataset.get_model_friendly_scores(dev_y_org_1, prompt_id_1)

    # Concatenate
    train_x = np.concatenate((train_x, dev_x))
    dev_x = test_x
    test_x = np.concatenate((train_x_1, dev_x_1, test_x_1))
    train_y = np.concatenate((train_y, dev_y))
    dev_y = test_y
    dev_y_org = test_y_org
    test_y = np.concatenate((train_y_1, dev_y_1, test_y_1))
    test_y_org = np.concatenate((train_y_org_1, dev_y_org_1, test_y_org_1))

    print(train_x.shape)
    print(train_y.shape)
    print(dev_x.shape)
    print(dev_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(np.array(dev_y_org).astype('int32').shape)
    train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

    FLAGS.checkpoint_dir = os.path.join(main_path, FLAGS.checkpoint_dir)
    # word-embedding
    emb_path = "./data/En_vectors.txt"
    emb_reader = W2VEmbReader(emb_path, emb_dim=FLAGS.embed_dim)
    U = np.random.uniform(-0.05, 0.05, size=(FLAGS.vocab_size, FLAGS.embed_dim)).astype(K.floatx())
    U[0] = np.zeros(shape=(FLAGS.embed_dim, ), dtype=K.floatx())
    U = emb_reader.get_emb_matrix_given_vocab(vocab, U)
    with tf.Session() as sess:
        m = NASM
        model = m(sess, embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim, batch_size=FLAGS.batch_size,
                  learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)
        evl = Evaluator(
            dataset, prompt_id, 'None',
            np.array(dev_y_org).astype('int32'),
            np.array(test_y_org).astype('int32')
        )
        if FLAGS.forward_only:  # Testing Stage
            model.load(FLAGS.checkpoint_dir)
            dev_pred = model.get_pred(dev_x).flatten()
            test_pred = model.get_pred(test_x).flatten()
            evl.feature_evaluate(dev_pred, test_pred)
        else:
            model.train(U, train_batch, evl, dev_x)


if __name__ == '__main__':
    # tf.app.run(gbdt_nvdm)
    tf.app.run(nasm_test)
