import handcrafted_feature_extractor as feature_extractor
from helpers.dataset import define_tensor_size
import logging
from helpers import data_utils as D
from helpers.asap_evaluator import Evaluator
from helpers import featurebased_dataset as dataset1
from theano_demo.models import HierarchicalModel, MLP
import time
import numpy as np
import theano
from theano_demo.mylayers.layer_utils import floatX

# configuration
lr = 0.001
drop_retain_rate = 0.5

vocab_size = 4000  # 0 is define to automated infer vocab-size
doc_num = 32  # defining the doc batch_size to accelerate
hidden_size = 300
word_embedding_size = 50  # Not changed
# try: gru, lstm
cell = "lstm"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "rmsprop"
logger = logging.getLogger(__name__)
out_dir = './out_dir/best'
D.set_logger(logger, out_dir=out_dir)


def cross_deep_train():
    train_prompt, test_prompt = 1, 2
    logger = logging.getLogger(__name__)
    out_dir = './out_dir/best'
    D.set_logger(logger, out_dir=out_dir)
    prefix = './data/training_set_rel3.tsv'
    train_sent_len, train_doc_len = define_tensor_size(train_prompt)  # sent_len is batch_size of tensor
    fold_best_dev = []
    fold_best_test = []
    in_start = time.time()
    train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
    train_es, vocab, real_vocab_size = dataset1.get_cross_data(
        prefix,
        train_prompt, vocab_size, train_doc_len, train_sent_len
    )
    test_es, vocab, real_vocab_size = dataset1.get_cross_data(
        prefix,
        train_prompt, vocab_size, train_doc_len, train_sent_len
    )
    train_es._friendly_y = dataset1.get_model_friendly_scores(np.array(train_es._score), train_prompt)
    test_es._friendly_y = dataset1.get_model_friendly_scores(np.array(test_es._score), test_prompt)
    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(train_es)
    train_feats = f.gen_feats(train_es, fit=True, normalize=False)
    test_feats = f.gen_feats(test_es)

    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(train_es)
    train_feats = f.gen_feats(train_es, fit=True, normalize=False)
    test_feats = f.gen_feats(test_es)

    print 'dev_y_org as integer...'
    print "#word size = ", vocab_size
    print "#real word size = ", real_vocab_size

    # TODO We should try different word-embedding
    U = floatX(np.random.uniform(-0.05, 0.05, size=(real_vocab_size, word_embedding_size)))
    U[0] = np.zeros((word_embedding_size,), dtype=theano.config.floatX)

    model = HierarchicalModel(
        U, real_vocab_size, word_embedding_size,
        hidden_size, cell, optimizer, drop_retain_rate,
        train_doc_len, sent_len=train_sent_len
    )
    evl = Evaluator(
        dataset1, test_prompt, 'None',
        np.array(test_es._score).astype('int32'),
        np.array(test_es._score).astype('int32')
    )
    print "training..."
    train_batch = dataset1.train_batch_generator(train_es, train_feats, doc_num)
    start = time.time()
    pre_epoch = 0
    for i in xrange(3000):
        epoch, X, features, mask, y = train_batch.next()
        real_doc_num = len(y)
        true_cost, _ = model.train(
            X, np.asarray(features, dtype=theano.config.floatX),
            np.asarray(mask, dtype=theano.config.floatX),
            lr, y, real_doc_num
        )
        init_train = 0
        if epoch > 30:
            print "Starting evaluation: " + str(epoch) + " time"
            in_start = time.time()
            evl.evaluate(
                test_es._text_by_ints, test_es._mask_by_ints, test_es._friendly_y,
                test_es._text_by_ints, test_es._mask_by_ints, test_es._friendly_y, model, epoch - init_train,
                True, test_feats, test_feats
            )
            in_time = time.time() - in_start
            print "Evaluation: " + str(epoch - init_train) + " spent Time = " + str(in_time)[:3]
            print "Epoch = %d, Iter = %d, Error = %s, Time = %s" % (pre_epoch, i, str(true_cost)[:6], str(in_time)[:3])
            pre_epoch = epoch

        if epoch > 80:
            fold_best_dev.append(evl.best_dev[0])
            fold_best_test.append(evl.best_test[0])
            logger.info(
                '(Dev Best: {{%.3f}}) (Test Best: {{%.3f}})'
                % (evl.best_dev[0], evl.best_test[0])
            )

    in_time = time.time() - in_start
    logger.info(
        '(Dev Best: {{%.3f}}) (Test Best: {{%.3f}}), Time: %s'
        % (np.mean(fold_best_dev), np.mean(fold_best_test), str(in_time)[:4])
    )
    print "Finished. Time = " + str(time.time() - start)[:3]


def mlp():
    for prompt_id in range(1, 9):
        sent_len, doc_len = define_tensor_size(prompt_id)  # sent_len is batch_size of tensor
        for fold in range(0, 5):
            prefix = './data/fold_' + str(fold) + '/'
            train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
            train_es, dev_es, test_es, vocab, real_vocab_size = dataset1.get_data(
                (train_path, dev_path, test_path),
                prompt_id, vocab_size, doc_len, sent_len
            )
            train_es._friendly_y = dataset1.get_model_friendly_scores(np.array(train_es._score), prompt_id)
            dev_es._friendly_y = dataset1.get_model_friendly_scores(np.array(dev_es._score), prompt_id)
            test_es._friendly_y = dataset1.get_model_friendly_scores(np.array(test_es._score), prompt_id)
            f = feature_extractor.FeatureExtractor()
            f.initialize_dictionaries(train_es)
            train_feats = f.gen_feats(train_es, fit=True, normalize=False)
            test_feats = f.gen_feats(test_es)
            dev_feats = f.gen_feats(dev_es)
            logger.info("Features Completed!")
            model = MLP("MLP", train_feats.shape[1], 4, [400, 300, 200, 200], activation='sigmoid')
            in_start = time.time()
            for i in range(20000):
                model.train(train_feats, lr, train_es._friendly_y, len(train_feats))
            _, dev_pred = model.predict(dev_feats, dev_es._friendly_y, len(dev_feats))
            _, test_pred = model.predict(test_feats, test_es._friendly_y, len(test_feats))
            evl = Evaluator(
                dataset1, prompt_id, 'None',
                np.array(dev_es._score).astype('int32'),
                np.array(test_es._score).astype('int32')
            )
            evl.feature_evaluate(dev_pred, test_pred)
            in_time = time.time() - in_start
            print 'Need Time : ', str(in_time)[:6]


def gbdt():
    sent_len, doc_len = define_tensor_size(1)  # sent_len is batch_size of tensor
    prefix = './data/training_set_rel3.tsv'
    train_prompt_id, test_prompt_id = 1, 2
    train_es, vocab, real_vocab_size = dataset1.get_cross_data(
        prefix,
        train_prompt_id, vocab_size, doc_len, sent_len
    )
    test_es, vocab, real_vocab_size = dataset1.get_cross_data(
        prefix,
        test_prompt_id, vocab_size, doc_len, sent_len
    )
    train_es._friendly_y = dataset1.get_model_friendly_scores(np.array(train_es._score), train_prompt_id)
    test_es._friendly_y = dataset1.get_model_friendly_scores(np.array(test_es._score), test_prompt_id)
    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(train_es)
    train_feats = f.gen_feats(train_es, fit=True, normalize=False)
    test_feats = f.gen_feats(test_es)

    model = sklearn.ensemble.GradientBoostingRegressor(
        n_estimators=100, learning_rate=.05,
        max_depth=4, min_samples_leaf=3
    )
    in_start = time.time()
    model.fit(train_feats, train_es._friendly_y)
    test_pred = model.predict(test_feats)
    evl = Evaluator(
        dataset1, test_prompt_id, 'None',
        np.array(test_es._score).astype('int32'),
        np.array(test_es._score).astype('int32')
    )
    evl.feature_evaluate(test_pred, test_pred)
    in_time = time.time() - in_start
    print 'Need Time : ', str(in_time)[:4]