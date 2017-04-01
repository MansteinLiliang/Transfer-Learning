import time
import numpy as np
import theano
import sys
import os
from helpers import dataset
from helpers.asap_evaluator import Evaluator
from theano_demo.mylayers.layer_utils import floatX
from helpers.w2vEmbReader import W2VEmbReader


# configuration
lr = 0.001
drop_retain_rate = 0.5

vocab_size = 4000  # 0 is define to automated infer vocab-size
batch_size = 32  # defining the doc batch_size to accelerate
hidden_size = 300
word_embedding_size = 50  # Not changed
# try: gru, lstm
cell = "lstm"


def nasm_main(args):
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    # pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1
    prefix += '/fold_0/'
    from helpers import asap_reader as reader
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
    (train_x, train_y_org, _), (dev_x, dev_y_org, _), (
        test_x, test_y_org, _), vocab, max_len = \
        reader.get_data((train_path, dev_path, test_path), prompt_id, vocab_size, 500)
    from keras.preprocessing import sequence
    train_x = sequence.pad_sequences(train_x, 400, padding='post')
    dev_x = sequence.pad_sequences(dev_x, 400, padding='post')
    train_y = dataset.get_model_friendly_scores(train_y_org, prompt_id)
    dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
    train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

    # word-embedding1
    emb_path = "./data/En_vectors.txt"
    emb_reader = W2VEmbReader(emb_path, emb_dim=word_embedding_size)
    U = floatX(np.random.uniform(-0.05, 0.05, size=(vocab_size, word_embedding_size)))
    U[0] = np.zeros(shape=(word_embedding_size, ), dtype=theano.config.floatX)
    U = emb_reader.get_emb_matrix_given_vocab(vocab, U)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.embeddings import Embedding
    from keras import layers

    model = Sequential()
    embedding = Embedding(vocab_size, 50, input_length=400)
    embedding.W = U
    model.add(embedding)
    model.add(layers.Convolution1D(
        nb_filter=200,
        filter_length=2,
        border_mode='valid',
        activation='relu',
    ))
    model.add(layers.Dropout(0.6))
    model.add(layers.GRU(200))
    model.add(layers.Dropout(0.6))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(train_x, train_y, nb_epoch=40, batch_size=64)
    pred = np.reshape(model.predict(dev_x), [-1])
    in_start = time.time()
    evl = Evaluator(
        dataset, prompt_id, 'None',
        np.array(dev_y_org).astype('int32'),
        np.array(dev_y_org).astype('int32')
    )
    evl.feature_evaluate(pred, pred)
    in_time = time.time() - in_start
    print 'Need Time : ', str(in_time)[:4]
