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
flags.DEFINE_integer("max_time_steps", 300, "max_time_steps of data")

flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("model", "nvdm", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("forward_only", False, "False for training, True for testing [False]")
FLAGS = flags.FLAGS

flags = FLAGS

main_path = os.curdir
sys.path.append(main_path)
prefix = main_path+'/data/fold_0/'
prompt_id = 1


# reader = dataset.TextReader(prefix, main_path=main_path)
# train_y = dataset.get_model_friendly_scores(reader.train_scores, prompt_id)
# dev_y = dataset.get_model_friendly_scores(reader.valid_scores, prompt_id)
# train_gen=reader.train_batch_generator(32, True)


train_path, dev_path, test_path = prefix + 'train.tsv', prefix + 'dev.tsv', prefix + 'test.tsv'

# (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (
#     test_x, test_y, test_pmt), vocab, overal_maxlen = asap_reader.get_data(
#     (train_path, dev_path, test_path), prompt_id, FLAGS.vocab_size, 0)
dataset = asap_reader
(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), \
(test_x, test_y, test_pmt), \
vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
    (train_path, dev_path, test_path), prompt_id, FLAGS.vocab_size,
    0, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None
)

overal_maxlen = FLAGS.max_time_steps
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
lr = 0.001
drop_retain_rate = 0.5

vocab_size = 4000  # 0 is define to automated infer vocab-size
batch_size = 32  # defining the doc batch_size to accelerate
hidden_size = 300
max_time_steps = 300
word_embedding_size = 50  # Not changed
# try: gru, lstm
cell = "lstm"
emb_path = "./data/En_vectors.txt"
emb_reader = W2VEmbReader(emb_path, emb_dim=word_embedding_size)
U = floatX(np.random.uniform(-0.05, 0.05, size=(vocab_size, word_embedding_size)))
U[0] = np.zeros(shape=(word_embedding_size, ), dtype=theano.config.floatX)
U = emb_reader.get_emb_matrix_given_vocab(vocab, U)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras import layers



import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.model_type = "breg"
args.algorithm = "rmsprop"
args.loss = "mse"
args.recurrent_unit = "gru"
args.aggregation = "mot"
args.rnn_dim = 300
args.emb_dim = 50
args.cnn_dim = 0
args.dropout_prob = 0.5
args.vocab_size = 4000
args.skip_init_bias = True
args.emb_path=None
from keras_demo import models
train_mean = train_y.mean(axis=0)

model = models.create_model(args, train_mean, 300, vocab)
import keras
optim = keras.optimizers.rmsprop(lr=0.0005)
model.compile(loss="mse", optimizer=optim, metrics=["mean_squared_error"])
model.fit(train_x, train_y, nb_epoch=3, batch_size=32)

pred = model.predict(dev_x).flatten()
evl = Evaluator(
    dataset, prompt_id, 'None',
    np.array(dev_y_org).astype('int'),
    np.array(dev_y_org).astype('int')
)
evl.feature_evaluate(pred, pred)

pred = model.predict(test_x).flatten()
evl = Evaluator(
    dataset, prompt_id, 'None',
    np.array(test_y_org).astype('int'),
    np.array(test_y_org).astype('int')
)
evl.feature_evaluate(pred, pred)