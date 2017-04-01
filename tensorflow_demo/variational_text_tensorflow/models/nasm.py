from __future__ import absolute_import

import math
import time

import tensorflow as tf
from tensorflow.python.ops import array_ops

from helpers.theano_optimizers import *
from theano_demo.mylayers.layer_utils import init_weights, init_bias
from .base import Model


class NASM(Model):
    """Neural Answer Selection Model"""

    def __init__(self, sess, reader, dataset="ptb",
                 num_steps=3, embed_dim=50, batch_size=64, vocab_size=4000,
                 h_dim=50, learning_rate=0.01, epoch=8, rnn_dim=200,
                 checkpoint_dir="checkpoint", **kwargs):
        """Initialize Neural Varational Document Model.

        params:
          sess: TensorFlow Session object.
          reader: TextReader object for training and test.
          dataset: The name of dataset to use.
          h_dim: The dimension of document representations (h). [50, 200]
        """
        self.sess = sess
        self.reader = reader

        self.h_dim = h_dim
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir

        self.dataset = "ptb"
        self._attrs = ["batch_size", "num_steps", "embed_dim", "h_dim", "learning_rate"]
        self.build_model()

    def my_length(self, sequence):
        # used = tf.sign(tf.reduce_max(tf.abs(self.rnn_inputs), reduction_indices=2))
        # length = tf.reduce_sum(used, reduction_indices=1)
        # length = tf.cast(tf.abs(length), tf.int32)

        used = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        # length = tf.Print(length,[length],message='This is length:')
        return length

    def compute_mask(self, x):
        return tf.not_equal(x, 0)

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def build_model(self):
        # batch_size, max_len, emb_size
        self.x = tf.placeholder(tf.int32, [None, None], name='sequences')
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        self.y = tf.placeholder(tf.float32, [None], name='score')
        with tf.name_scope("embedding"):
            self._embedding_matrix = tf.get_variable(
                'embedding_matrix', [self.vocab_size, self.embed_dim],
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()
            )
        self.rnn_inputs = tf.nn.embedding_lookup(self._embedding_matrix, self.x)
        self.sequence_length = self.my_length(self.x)
        # word_vectors = tf.contrib.layers.embed_sequence(
        #     self.x, vocab_size=self.vocab_size, embed_dim=self.embed_dim, scope='words')
        # # self.embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, self.embed_dim], dtype=tf.float32)
        # # self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
        # self.rnn_inputs = word_vectors

        # self.embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, self.embed_dim], dtype=tf.float32)
        # self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
        shape = tf.shape(self.rnn_inputs)
        (self.batch_size, self.time_steps, self.indim) = shape[0], shape[1], shape[2]

        self.rnn_encoder()
        # self.attention()
        self.mse()
        self.loss = tf.reduce_mean(
            tf.square(tf.reshape(self.y, [self.batch_size]) - self.y_)
        )
        # Kullback Leibler divergence
        # self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))
        # self.e_loss = self.c_loss
        # Log likelihood
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.c_loss, tvars),
        #                                   1.0)
        # self.loss = self.c_loss
        # self.loss = tf.reduce_mean(self.lr*self.e_loss + self.c_loss)
        self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self._train_op = self.optim.apply_gradients(
        #     zip(grads, tvars),
        #     global_step=tf.contrib.framework.get_or_create_global_step())

        # _ = tf.scalar_summary("encoder loss", self.e_loss)
        # _ = tf.scalar_summary("decoder loss", self.c_loss)
        # _ = tf.scalar_summary("loss", self.loss)

    def rnn_encoder(self):
        """Inference Network. q(h|X)"""
        with tf.variable_scope("encoder"):
            q_cell = tf.contrib.rnn.GRUCell(self.rnn_dim)
            q_cell = tf.contrib.rnn.DropoutWrapper(
                q_cell, output_keep_prob=0.8)
            outputs, state = tf.nn.dynamic_rnn(
                q_cell, self.rnn_inputs, dtype=tf.float32, sequence_length=self.sequence_length
            )
            # output = tf.transpose(outputs, [1, 0, 2])
            # last = tf.gather(output, tf.shape(output)[0] - 1)
            # self.last = self.last_relevant(outputs, self.sequence_length)
            self.last = state
            # self.outputs = outputs

    def latent_variable(self):
        with tf.variable_scope("latent_variable"):
            l1 = tf.nn.relu(tf.nn.rnn_cell._linear(self.last, self.embed_dim, bias=True, scope="l1"))
            l2 = tf.nn.relu(tf.nn.rnn_cell._linear(l1, self.embed_dim, bias=True, scope="l2"))
            self.mu = tf.nn.rnn_cell._linear(l2, self.h_dim, bias=True, scope="mu")
            self.log_sigma_sq = tf.nn.rnn_cell._linear(l2, self.h_dim, bias=True, scope="log_sigma_sq")
            eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
            sigma = tf.sqrt(tf.exp(self.log_sigma_sq))
            _ = tf.histogram_summary("mu", self.mu)
            _ = tf.histogram_summary("sigma", sigma)
            self.h = self.mu + sigma * eps

    def mse(self):
        with tf.variable_scope('mse'):
            W = tf.get_variable('W', [self.rnn_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
        self.y_ = tf.nn.xw_plus_b(self.last, W, b, name='pred')

    def attention(self):
        with tf.variable_scope("attention"):
            seq_len_mask = array_ops.sequence_mask(self.sequence_length, self.time_steps, dtype=tf.float32)
            rank = self.rnn_inputs.get_shape().ndims
            rank = rank if rank is not None else array_ops.rank(self.rnn_inputs)
            extra_ones = array_ops.ones(rank - 2, dtype=tf.int32)
            seq_len_mask = array_ops.reshape(
                seq_len_mask,
                array_ops.concat(0, [array_ops.shape(seq_len_mask), extra_ones])
            )
            processed_h = array_ops.expand_dims(self.h, 1)
            v = tf.get_variable(
                "attention_v", self.embed_dim, dtype=tf.float32)
            h_tile = tf.tile(processed_h, [1, self.time_steps, 1])
            x_processed = v * tf.tanh(tf.nn.rnn_cell._linear(
                [self.rnn_inputs, h_tile], self.embed_dim, bias=True, scope="attention_fusion"))
            # x_processed = v * tf.tanh(self.rnn_inputs + processed_h)
            x_processed_mask = x_processed * seq_len_mask
            score = array_ops.expand_dims(tf.nn.softmax(tf.reduce_sum(x_processed_mask, [2])), 2)
            self.context_vector = tf.reduce_sum(score * self.rnn_inputs, [1])

    def get_hidden_features(self, x):
        return self.sess.run(
            self.last, feed_dict={self.x: x}
        )

    def get_pred(self, x):
        y_ = np.reshape(self.sess.run(
            self.y_, feed_dict={self.x: x}
        ),[-1])
        return y_

    def train(self, U, train_batch):
        sigmoid_increase = lambda x: 2.0/(1.0 + math.exp(-0.00001*x))-1
        start_time = time.time()

        # merged_sum = tf.merge_all_summaries()
        # writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

        tf.initialize_all_variables().run()
        # self.load(self.checkpoint_dir)
        # train_gen = self.reader.train_batch_generator(32, True)
        epoch = 0
        self.sess.run(self._embedding_matrix.assign(value=U))
        for idx in xrange(300000):
            # cur_epoch, x, y = train_gen.next()
            cur_epoch, x, y = train_batch.next()
            # print y[:10]
            # lr = sigmoid_increase(idx)
            lr=0
            # y_, _, loss, e_loss, g_loss, summary_str = self.sess.run(
            #     [self.y_, self.optim, self.loss, self.e_loss, self.c_loss, merged_sum],
            #     feed_dict={self.x: x, self.y: y, self.lr:lr}
            # )
            y_, y, _, loss = self.sess.run(
                [self.y_, self.y, self.optim, self.loss],
                feed_dict={self.x: x, self.y: y}
            )
            # y = dataset.convert_to_dataset_friendly_scores(np.array(y), 1)
            # print loss
            # y_ = np.reshape(y_, [-1])
            # print y_[:10]
            # print dataset.convert_to_dataset_friendly_scores(np.array(y_[:10]), 1)
            # y_ = np.rint(dataset.convert_to_dataset_friendly_scores(np.array(y_), 1))
            # print quadratic_weighted_kappa(y_, y, 2, 12)
            if cur_epoch > self.epoch:
                break
            if cur_epoch > epoch:
                epoch = cur_epoch
                print y_[:10].flatten()
                print y[:10].flatten()
                print("Epoch: [%2d] %4d time: %4.4f, loss: %.8f, e_loss: %.8f, c_loss: %.8f" \
                      % (cur_epoch, idx, time.time() - start_time, loss, loss, loss))

            # if idx % 20 == 0:
            #     writer.add_summary(summary_str, epoch)

            # if idx != 0 and idx % 1000 == 0:
            #     self.save(self.checkpoint_dir, epoch)

from theano_demo.mylayers import SentEncoderLayer
class LSTM(Model):
    """Neural Answer Selection Model"""

    def __init__(self, reader, U, dataset="ptb",
                 num_steps=3, embed_dim=50, batch_size=32, vocab_size=4000,
                 h_dim=50, learning_rate=0.01, epoch=15, rnn_dim=200,
                 checkpoint_dir="checkpoint", **kwargs):
        """Initialize Neural Varational Document Model.

        params:
          sess: TensorFlow Session object.
          reader: TextReader object for training and test.
          dataset: The name of dataset to use.
          h_dim: The dimension of document representations (h). [50, 200]
        """
        self.reader = reader
        self.h_dim = h_dim
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.params = []
        embeddings = theano.shared(value=U, name='WEmb')
        self.params.append(embeddings)
        self.idxs = T.imatrix('idxs')
        self.y = T.fvector('y')
        self.X = embeddings[self.idxs]
        self.params = []
        rng = np.random.RandomState(1234)
        self.is_train = T.iscalar('is_train')
        self.length = T.transpose(self.my_length(self.idxs))
        sent_encoder_layer = SentEncoderLayer(
            'SentEncoder', T.transpose(self.X, [1, 0, 2]), self.embed_dim, self.rnn_dim, "gru",
            1.0, self.is_train, 32, T.transpose(self.my_length(self.idxs), [1, 0]), rng
        )
        layer_input = sent_encoder_layer.activation
        sent_X = layer_input[layer_input.shape[0] - 1, :]  # sent_X shape is: (batch_size, dim)
        W_out = init_weights((self.rnn_dim, 1), 'mse_W')
        b_out = init_bias(1, 'mse_b', value=0.0)
        self.params+=[W_out, b_out]
        self.params+=sent_encoder_layer.params
        y_pred = T.dot(sent_X, W_out)+b_out
        loss = T.mean(T.square((y_pred-self.y.reshape((self.batch_size, 1)))))
        self.loss = theano.gradient.grad_clip(loss, -1.0, 1.0)
        pred = T.clip(y_pred, 0.0, 1.0)

        gparams = []
        for param in self.params:
            gparam = T.grad(self.loss, param)
            gparams.append(gparam)
        updates = eval("rmsprop")(self.params, gparams, 0.001)
        self.train_func = theano.function(
            inputs=[self.idxs, self.y],
            givens={self.is_train: np.cast['int32'](1)},
            outputs= self.loss,
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        # self.predict = theano.function(
        #     inputs=[self.idxs],
        #     givens={self.is_train: np.cast['int32'](0)},
        #     outputs=pred,
        #     on_unused_input='ignore',
        #     allow_input_downcast=True
        # )
    def my_length(self, sequence):
        used = T.sgn(self.idxs)
        length = T.cast(used, theano.config.floatX)
        return length

    # def get_pred(self, x):
    #     y_ = np.reshape(self.sess.run(
    #         self.y_, feed_dict={self.x: x}
    #     ), [-1])
    #     return y_

    def train(self, U, config):
        start_time = time.time()
        train_gen = self.reader.train_batch_generator(32, True)
        epoch = 0
        print "hhe"
        for idx in xrange(300000):
            cur_epoch, x, y = train_gen.next()
            loss = self.train_func(x, y)
            print loss
            if cur_epoch > self.epoch:
                break
            if cur_epoch > epoch:
                epoch = cur_epoch
                # print y_[:10].flatten()
                # print y[:10].flatten()
                print("Epoch: [%2d] %4d time: %4.4f, loss: %.8f, e_loss: %.8f, c_loss: %.8f" \
                      % (cur_epoch, idx, time.time() - start_time, loss, loss, loss))

            # if idx % 20 == 0:
            #     writer.add_summary(summary_str, epoch)

            # if idx != 0 and idx % 1000 == 0:
            #     self.save(self.checkpoint_dir, epoch)
