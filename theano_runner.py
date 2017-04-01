#pylint: skip-file
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import time
from helpers import data_utils as D
import numpy as np
import theano
from helpers.asap_evaluator import Evaluator
from helpers import dataset
from theano_demo.mylayers.layer_utils import floatX
from theano_demo.models import HierarchicalModel
from helpers import featurebased_dataset as dataset1
from helpers.dataset import define_tensor_size

# configuration
lr = 0.001
drop_retain_rate = 0.5
vocab_size = 4000  # 0 is define to automated infer vocab-size
doc_num = 64  # defining the doc batch_size to accelerate
hidden_size = 250
word_embedding_size = 250  # Not changed
# try: gru, lstm
cell = "lstm"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "rmsprop"


def main():
    logger = logging.getLogger(__name__)
    out_dir = './out_dir/best'
    D.set_logger(logger, out_dir=out_dir)
    for prompt_id in range(8, 9):
        sent_len, doc_len = define_tensor_size(prompt_id)  # sent_len is batch_size of tensor
        fold_best_dev = []
        fold_best_test = []
        for fold in range(0, 5):
            prefix = './data/fold_' + str(fold) + '/'
            train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
            (train_x, train_masks, train_y_org), (dev_x, dev_masks, dev_y_org), (
            test_x, test_masks, test_y_org), vocab, real_vocab_size = \
                dataset.get_data((train_path, dev_path, test_path), prompt_id, vocab_size, doc_len, sent_len)

            train_y = dataset.get_model_friendly_scores(train_y_org, prompt_id)
            dev_y = dataset.get_model_friendly_scores(dev_y_org, prompt_id)
            test_y = dataset.get_model_friendly_scores(test_y_org, prompt_id)
            query = dataset.get_query(prompt_id, vocab, sent_len)

            print 'dev_y_org as integer...'
            print "#word size = ", vocab_size
            print "#real word size = ", real_vocab_size

            # TODO We should try different word-embedding
            # word-embedding1
            # emb_path = "./data/En_vectors.txt"
            # emb_reader = W2VEmbReader(emb_path, emb_dim=word_embedding_size)
            # U = floatX(np.random.uniform(-0.05, 0.05, size=(vocab_size, word_embedding_size)))
            # # U = np.zeros(shape=(vocab_size, word_embedding_size), dtype=theano.config.floatX)
            # U[0] = np.zeros(shape=(word_embedding_size, ), dtype=theano.config.floatX)
            # U = emb_reader.get_emb_matrix_given_vocab(vocab, U)

            # word-embedding2
            U = floatX(np.random.uniform(-0.05, 0.05, size=(real_vocab_size, word_embedding_size)))
            U[0] = np.zeros((word_embedding_size,), dtype=theano.config.floatX)

            # word-embedding3: zero-embeddings
            # U = np.zeros(shape=(vocab_size, word_embedding_size), dtype=theano.config.floatX)

            '''Loading google word2vec
            # word_vectors = pkl.load(open('./data/word_vectors.pk.'))
            # for word in vocab:d
            #     U[vocab[word]] = word_vectors[word]
            '''
            model = HierarchicalModel(
                U, real_vocab_size, word_embedding_size,
                hidden_size, cell, optimizer, drop_retain_rate,
                doc_len, sent_len=sent_len
            )
            #   Original dev_y and Original test_y should be given as integer
            #   dev_y_org is like[8,10,2,10,...] not in range(0,1.0)
            #   dev_y is in range(0, 1.0)
            evl = Evaluator(dataset, prompt_id, 'None', dev_y_org.astype('int32'), test_y_org.astype('int32'))
            print "training..."
            train_batch = dataset.train_batch_generator(train_x, train_masks, train_y, doc_num)
            start = time.time()
            pre_epoch = 1
            for i in xrange(3000):
                epoch, X, mask, y = train_batch.next()
                in_start = time.time()
                true_cost, pred = model.train(X, np.asarray(mask, dtype=theano.config.floatX), lr, y, doc_num)
                if epoch > pre_epoch and epoch > 10:
                    print "Starting evaluation: " + str(epoch) + " time"
                    in_start = time.time()
                    evl.evaluate(dev_x, dev_masks, dev_y, test_x, test_masks, test_y, model, epoch-10)
                    in_time = time.time() - in_start
                    print "Evaluation: "+ str(epoch-10)+ " spent Time = " + str(in_time)[:3]
                    print "Epoch = %d, Iter = %d, Error = %s, Time = %s" % (pre_epoch, i, str(true_cost)[:6], str(in_time)[:3])
                    pre_epoch = epoch
                if epoch > 100:
                    fold_best_dev.append(evl.best_dev[0])
                    fold_best_test.append(evl.best_test[0])
                    logger.info(
                        'Prompt_%d, fold_%d, (Dev Best: {{%.3f}}) (Test Best: {{%.3f}})'
                        % (prompt_id, fold, evl.best_dev[0], evl.best_test[0])
                    )
                    break
        in_time = time.time() - in_start
        logger.info(
            'Prompt_%d, (Dev Best: {{%.3f}}) (Test Best: {{%.3f}}), Time: %s'
            % (prompt_id, np.mean(fold_best_dev), np.mean(fold_best_test), str(in_time)[:4])
        )
    print "Finished. Time = " + str(time.time() - start)[:3]
        # print "save model..."
        # save_model("./model/hed.model", model)


if __name__ == '__main__':
    main()
