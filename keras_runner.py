from __future__ import print_function
import time
import numpy as np
import sys
import os
from helpers import dataset
from helpers.asap_evaluator import Evaluator
from helpers.w2vEmbReader import W2VEmbReader
import argparse
import os
import keras

# configuration
lr = 0.001
drop_retain_rate = 0.5
vocabsize = 4000  # 0 is define to automated infer vocab-size
batch_size = 32  # defining the doc batch_size to accelerate
hidden_size = 300
max_time_steps = 500
word_embedding_size = 50  # Not changed
# try: gru, lstm
cell = "lstm"


def main():
    main_path = os.path.dirname(__file__)
    sys.path.append(main_path)
    # pp.pprint(flags.FLAGS.__flags)
    prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    prompt_id = 1
    prefix += '/fold_0/'
    from helpers import asap_reader as reader
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
    # (train_x, train_y_org, _), (dev_x, dev_y_org, _), (
    #     test_x, test_y_org, _), vocab, max_len = \
    #     reader.get_data((train_path, dev_path, test_path), prompt_id, vocabsize, 500)

    (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), \
    (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = reader.get_data(
        (train_path, dev_path, test_path), prompt_id, vocabsize, 0, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None
    )
    import keras.backend as K
    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())

    from keras.preprocessing import sequence
    train_x = sequence.pad_sequences(train_x, max_time_steps, padding='post')

    dev_x = sequence.pad_sequences(dev_x, max_time_steps, padding='post')
    dev_y_org = dev_y.astype('int')
    test_y_org = test_y.astype('int')
    train_y = dataset.get_model_friendly_scores(train_y, prompt_id)
    dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)

    # train_batch = dataset.train_batch_generator(train_x, None, train_y, 32, 0)

    # word-embedding1
    emb_path = "./data/En_vectors.txt"
    emb_reader = W2VEmbReader(emb_path, emb_dim=word_embedding_size)
    U = np.random.uniform(-0.05, 0.05, size=(vocab_size, word_embedding_size)).astype(K.floatx())
    U[0] = np.zeros(shape=(word_embedding_size, ), dtype=K.floatx())
    U = emb_reader.get_emb_matrix_given_vocab(vocab, U)

    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers.embeddings import Embedding
    # from keras import layers
    # model = Sequential()
    # embedding = Embedding(vocab_size, 400)
    # # embedding.W = U
    # model.add(embedding)
    # model.add(layers.Convolution1D(
    #     nb_filter=200,
    #     filter_length=2,
    #     border_mode='valid',
    #     activation='relu',
    # ))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.GRU(200))
    # model.add(layers.Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='mse', optimizer='rmsprop')
    # model.fit(train_x, train_y, nb_epoch=3, batch_size=32)
    # pred = model.predict(dev_x).flatten()
    #
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model_type = "reg"
    args.algorithm = "rmsprop"
    args.loss = "mse"
    args.recurrent_unit = "gru"
    args.aggregation = "mot"
    args.rnn_dim = 300
    args.emb_dim = 50
    args.cnn_dim = 0
    args.dropout_prob = 0.5
    args.vocab_size = vocab_size
    args.emb_path = None

    # base_line model
    from keras_demo import models
    model = models.create_model(args, 0.6, max_time_steps, vocab)
    optim = keras.optimizers.rmsprop(lr=0.001)
    model.compile(loss="mse", optimizer=optim, metrics=["mean_squared_error"])
    model.fit(train_x, train_y, nb_epoch=10, batch_size=32)
    pred = model.predict(dev_x).flatten()
    in_start = time.time()

    evl = Evaluator(
        dataset, prompt_id, 'None',
        np.array(dev_y_org).astype('int'),
        np.array(dev_y_org).astype('int')
    )
    evl.feature_evaluate(pred, pred)
    in_time = time.time() - in_start
    print('Need Time : ', str(in_time)[:4])


if __name__ == "__main__":
    main()
