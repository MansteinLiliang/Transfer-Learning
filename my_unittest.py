import unittest
# from helpers import dataset
import tensorflow_demo.variational_text_tensorflow.reader
import time
import numpy as np
import sklearn.ensemble
import theano
import tensorflow as tf
import sys, os
from helpers import dataset
from helpers import asap_reader
from helpers.dataset import define_tensor_size
from tensorflow_demo.variational_text_tensorflow.utils import pp
from tensorflow_demo.variational_text_tensorflow.models import NASM, NVDM
from helpers.asap_evaluator import Evaluator
from theano_demo.mylayers.layer_utils import floatX
from helpers.w2vEmbReader import W2VEmbReader
from keras.preprocessing import sequence
import keras.backend as K
from scipy.stats import binned_statistic

# tensorflow flags
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of adam optimizer [0.01]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_iter", 5000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 50, "The dimension of latent variable [50]")
flags.DEFINE_integer("vocab_size", 4000, "The dimension of latent variable [4000]")
flags.DEFINE_integer("embed_dim", 50, "The dimension of word embeddings [500]")
flags.DEFINE_integer("batch_size", 64, "batch_size of data")
flags.DEFINE_integer("max_time_steps", 500, "max_time_steps of data")

flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("model", "nvdm", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("forward_only", False, "False for training, True for testing [False]")
FLAGS = flags.FLAGS


class Test(unittest.TestCase):
    # def test_reader_random(self):
    #     textreader = dataset.TextReader(1, 2)
    #     x, y = textreader.random()
    #     print sum(x)
    #     print y
    def test_data(self):
        main_path = os.path.dirname(__file__)
        sys.path.append(main_path)
        # pp.pprint(flags.FLAGS.__flags)
        prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/fold_0/')
        prompt_id = 1
        # reader = dataset.TextReader(prefix, main_path=main_path)
        # train_y = dataset.get_model_friendly_scores(reader.train_scores, prompt_id)
        # dev_y = dataset.get_model_friendly_scores(reader.valid_scores, prompt_id)
        # train_gen=reader.train_batch_generator(32, True)
        train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'
        (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (
            test_x, test_y, test_pmt), vocab, overal_maxlen = asap_reader.get_data(
            (train_path, dev_path, test_path), prompt_id, FLAGS.vocab_size, FLAGS.max_time_steps)
        overal_maxlen = 500
        train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
        dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
        test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)

        train_y = np.array(train_y, dtype=K.floatx())
        dev_y = np.array(dev_y, dtype=K.floatx())
        test_y = np.array(test_y, dtype=K.floatx())
        dev_y_org = dev_y.astype('int')
        test_y_org = test_y.astype('int')

        train_y = dataset.get_model_friendly_scores(train_y, prompt_id)
        dev_y = dataset.get_model_friendly_scores(dev_y, prompt_id)
        test_y = dataset.get_model_friendly_scores(test_y, prompt_id)


        # def func(x):
        #     return np.sum(x)
        #
        # results = binned_statistic(train_y, train_y, statistic=func, bins=10, range=[0, 1])
        # print results[0]
        # print results[1]


    def not_Textreader(self):
        textreader = tensorflow_demo.variational_text_tensorflow.reader.TextReader(data_path="./data/ptb")
        x, y = textreader.random()
        print(sum(x))
        print(len(x))
        print(y)

if __name__ == '__main__':
    unittest.main()
