#pylint: skip-file
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import numpy as np
import sklearn.ensemble
import theano
import tensorflow as tf
import sys, os
from helpers import dataset
from helpers.dataset import define_tensor_size
from tensorflow_demo.variational_text_tensorflow.utils import pp
from tensorflow_demo.variational_text_tensorflow.models import NASM, NVDM
from helpers.asap_evaluator import Evaluator
from theano_demo.mylayers.layer_utils import floatX
from helpers.w2vEmbReader import W2VEmbReader


# tensorflow flags
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of adam optimizer [0.01]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_iter", 5000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 50, "The dimension of latent variable [50]")
flags.DEFINE_integer("vocab_size", 4000, "The dimension of latent variable [4000]")
flags.DEFINE_integer("embed_dim", 50, "The dimension of word embeddings [500]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("batch_size", 64, "batch_size of data")
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
        print 'Need Time : ', str(in_time)[:4]


def nasm_main(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    # pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1
    # reader = dataset.TextReader(prefix, main_path=main_path)
    # train_y = dataset.get_model_friendly_scores(reader.train_scores, prompt_id)
    # dev_y = dataset.get_model_friendly_scores(reader.valid_scores, prompt_id)
    # train_gen=reader.train_batch_generator(32, True)
    prefix += '/fold_0/'
    from helpers import asap_reader as reader
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
    (train_x, train_y_org, _), (dev_x, dev_y_org, _), (
        test_x, test_y_org, _), vocab, max_len = \
        reader.get_data((train_path, dev_path, test_path), prompt_id, flags.vocab_size, 500)
    from keras.preprocessing import sequence
    train_x = sequence.pad_sequences(train_x, 400, padding='post')
    dev_x = sequence.pad_sequences(dev_x, 400, padding='post')
    train_y = dataset.get_model_friendly_scores(train_y_org, prompt_id)
    dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
    train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

    # with tf.Session() as sess:
    # m = LSTM
    # word-embedding1
    emb_path = "./data/En_vectors.txt"
    emb_reader = W2VEmbReader(emb_path, emb_dim=flags.embed_dim)
    U = floatX(np.random.uniform(-0.05, 0.05, size=(flags.vocab_size, flags.embed_dim)))
    U[0] = np.zeros(shape=(flags.embed_dim, ), dtype=theano.config.floatX)
    U = emb_reader.get_emb_matrix_given_vocab(vocab, U)

    # model = m(
    #     reader, U
    # )
    # model = m(sess, reader, dataset=FLAGS.dataset,
    #           embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
    #           learning_rate=FLAGS.learning_rate, max_iter=FLAGS.max_iter,
    #           checkpoint_dir=FLAGS.checkpoint_dir)
    # if FLAGS.forward_only:
    #     model.load(FLAGS.checkpoint_dir)
    # else:
    # model.train(U, None)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras import layers

    model = Sequential()
    embedding = Embedding(flags.vocab_size, 50, input_length=400)
    embedding.W = U
    model.add(embedding)
    model.add(layers.Convolution1D(
        nb_filter=200,
        filter_length=2,
        border_mode='valid',
        activation='relu',
    ))
    model.add(layers.Dropout(0.6))
    # model.add(layers.MaxPooling1D(pool_length=5))
    model.add(layers.GRU(200))
    model.add(layers.Dropout(0.6))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(train_x, train_y, nb_epoch=40, batch_size=64)
    pred = np.reshape(model.predict(dev_x), [-1])

    # train_feats = reader.gen_nasm_feats(model, "train")
    # valid_feats = reader.gen_nasm_feats(model, "valid")

    # model = sklearn.ensemble.GradientBoostingRegressor(
    #     n_estimators=100, learning_rate=.05,
    #     max_depth=4, min_samples_leaf=3
    # )

    in_start = time.time()
    # model.fit(train_feats, train_y)
    # pred = model.predict(valid_feats)
    # pred = model.get_pred(reader.gen_padding_mat("valid"))
    evl = Evaluator(
        dataset, prompt_id, 'None',
        np.array(dev_y_org).astype('int32'),
        np.array(dev_y_org).astype('int32')
    )
    evl.feature_evaluate(pred, pred)
    in_time = time.time() - in_start
    print 'Need Time : ', str(in_time)[:4]


def nasm_test(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1

    # reader = dataset.TextReader(prefix, main_path=main_path)
    # train_y = dataset.get_model_friendly_scores(reader.train_scores, prompt_id)
    # dev_y = dataset.get_model_friendly_scores(reader.valid_scores, prompt_id)
    # train_gen=reader.train_batch_generator(32, True)

    with tf.Session() as sess:
        m = NASM
        from helpers import asap_reader as reader
        prefix += '/fold_0/'
        train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
        (train_x, train_y_org,_), (dev_x, dev_y_org,_), (
            test_x, test_y_org,_), vocab, max_len = \
            reader.get_data((train_path, dev_path, test_path), prompt_id, flags.vocab_size, 500)
        from keras.preprocessing import sequence
        train_x = sequence.pad_sequences(train_x, 500, padding='post')
        dev_x = sequence.pad_sequences(dev_x, 500, padding='post')
        train_y = dataset.get_model_friendly_scores(train_y_org, prompt_id)
        dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
        train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

        # word-embedding1
        emb_path = "./data/En_vectors.txt"
        emb_reader = W2VEmbReader(emb_path, emb_dim=flags.embed_dim)
        U = floatX(np.random.uniform(-0.05, 0.05, size=(flags.vocab_size, flags.embed_dim)))
        # U = np.zeros(shape=(vocab_size, word_embedding_size), dtype=theano.config.floatX)
        U[0] = np.zeros(shape=(flags.embed_dim, ), dtype=theano.config.floatX)
        U = emb_reader.get_emb_matrix_given_vocab(vocab, U)

        model = m(sess, dataset, dataset=FLAGS.dataset,
                  embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
                  learning_rate=FLAGS.learning_rate, max_iter=FLAGS.max_iter,
                  checkpoint_dir=FLAGS.checkpoint_dir)
        # if FLAGS.forward_only:
        #     model.load(FLAGS.checkpoint_dir)
        # else:
        model.train(U, train_batch)
        in_start = time.time()
        pred = model.get_pred(dev_x)
        evl = Evaluator(
            dataset, prompt_id, 'None',
            np.array(dev_y_org).astype('int32'),
            np.array(dev_y_org).astype('int32')
        )
        evl.feature_evaluate(pred, pred)
        in_time = time.time() - in_start
        print 'Need Time : ', str(in_time)[:4]


if __name__ == '__main__':
    # tf.app.run(gbdt_nvdm)
    tf.app.run(nasm_main)