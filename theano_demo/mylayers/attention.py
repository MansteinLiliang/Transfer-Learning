#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from theano_demo.mylayers import layer_utils

init_weights = layer_utils.init_weights
init_bias = layer_utils.init_bias
ReLU = layer_utils.ReLU
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

class ConvolutionAttention(object):
    def ReLU(self, x):
        y = T.maximum(0.0, x)
        return (y)
    def __init__(self, rng, prefix, shape, sent_encs, filter_width, pre_output):
        """
        :param prefix:
        :param shape:
        :param sent_encs: shape is (batch_docs, num_sents, in_size)
        """
        prefix = prefix + "_"
        self.batch_docs, self.num_sents, self.in_size, self.out_size = shape
        self.input = sent_encs
        self.filter_shape = (self.in_size, self.in_size, 1, filter_width)
        w_bound = np.sqrt(self.in_size * 1 * filter_width)
        self.shape = (self.batch_docs, self.in_size, 1, self.num_sents)
        self.input = self.input.dimshuffle([0, 2, 1]).reshape(self.shape)

        self.non_linear = "relu"
        self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name=prefix + "W_conv")

        b_values = np.zeros((self.in_size,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name=prefix + "b_conv")
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape, border_mode='valid')
        self.W_g = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name=prefix + "W_g_conv")
        conv_gating = conv.conv2d(input=self.input, filters=self.W_g, filter_shape=self.filter_shape, border_mode='valid').reshape((self.batch_docs, self.in_size, self.num_sents-2)).dimshuffle([0, 2, 1])
        pre_output_matrix = T.tile(pre_output, (self.num_sents-2, 1, 1)).dimshuffle([1, 0, 2])
        self.W_c = init_weights((2*self.in_size, self.in_size), name=prefix + "W_c_conv")
        self.b_c = init_bias(self.in_size, name=prefix + "b_c_conv")
        gating_matrix = T.concatenate([conv_gating, pre_output_matrix], -1)
        gate = T.nnet.sigmoid(T.dot(gating_matrix, self.W_c) + self.b_c).dimshuffle([0, 2, 'x', 1])
        conv_out = gate * conv_out

        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = pool.pool_2d(input=conv_out_tanh, ds=(1, self.num_sents-2), ignore_border=True)
        elif self.non_linear == "relu":
            conv_out_relu = self.ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = pool.pool_2d(input=conv_out_relu, ds=(1, self.num_sents-2), ignore_border=True)
        else:
            pooled_out = pool.pool_2d(input=conv_out, ds=(1, self.num_sents-2), ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
        self.activation = self.output.reshape((self.batch_docs, self.in_size))


class AttentionLayer(object):
    def __init__(self, shape, sent_ens, sent_mask, auxiliary_vector=None, pre_output=None):
        """
        Base Layer of Attention
        :param prefix:
        :param shape: tuple(num_docs, num_sents, in_size, rnn_out_dim)
        :param sent_ens: shape is (num_docs, num_sents, in_size)
        :param sent_mask: sent_mask is of shape(num_docs*num_sents,), Needing reshape!
        :param auxiliary_vector: shape=(rnn_out_dim,)
        """
        self.batch_docs, self.num_sents, self.in_size, self.out_size = shape
        self.shape = shape
        self.sent_encs = sent_ens
        self.sent_mask = sent_mask
        self.pre_output = pre_output
        self.auxiliary_vector = auxiliary_vector


class DocLevelAttention(AttentionLayer):
    def __init__(self, fet_size=None, features=None, *args, **kwargs):
        """
        :param features: (num_docs, fet_size)
        """
        super(DocLevelAttention, self).__init__(*args, **kwargs)
        self.prefix = "DocLevelAttention"
        if features:
            features_mat = T.tile(features.dimshuffle([0, 'x', 1]), (1, self.num_sents, 1))
            concat = T.concatenate([features_mat, self.sent_encs], -1)
            self.W_a = init_weights((self.in_size + fet_size, self.out_size), self.prefix + "_W_a")
            self.b_a = init_bias(self.out_size, self.prefix + "_b_a")
            transform = T.tanh(T.dot(concat, self.W_a) + self.b_a)
        else:
            self.W_a = init_weights((self.in_size, self.out_size), self.prefix + "_W_a")
            self.b_a = init_bias(self.out_size, self.prefix + "_b_a")
            transform = T.tanh(T.dot(self.sent_encs, self.W_a) + self.b_a)
        self.W_u = init_weights((self.out_size, ), self.prefix + "_W_u")
        strength = T.dot(transform, self.W_u)
        sent_mask = self.sent_mask.reshape((self.batch_docs, self.num_sents))
        strength_mask = strength * sent_mask
        # a = T.nnet.softmax(strength_mask)[:, :, None]
        c = (strength_mask[:, :, None]*self.sent_encs).sum(axis=1)
        self.activation = c
        self.attention = strength_mask
        self.params = [self.W_a, self.b_a, self.W_u]


class CoherenceAttention(AttentionLayer):
    def __init__(self, begin_embedding=None, end_embedding=None, *args, **kwargs):
        super(CoherenceAttention, self).__init__(*args, **kwargs)
        self.prefix = "CoherenceAttention"
        coherence_vector = self.auxiliary_vector
        self.W_a = init_weights((self.in_size*4, self.out_size), self.prefix + "_W_a" )
        self.b_a = init_bias(self.out_size, self.prefix + "_b_a" )
        self.X = self.sent_encs
        self.sent_mask = self.sent_mask.reshape((self.batch_docs, self.num_sents))
        X_T = self.X.dimshuffle([1, 0, 2])  # shape is (sents, docs, dim)
        X_T_PRE = X_T[:self.num_sents-1, :, :]
        X_T_NEXT = X_T[1:, :, :]
        padding_vector = T.tile(init_bias(self.in_size, "sent_padding", 0.0), (1, self.batch_docs, 1))
        X_T_PRE = T.concatenate([padding_vector, X_T_PRE], 0)
        X_T_NEXT = T.concatenate([X_T_NEXT, padding_vector], 0)
        pre_output_matrix = T.tile(self.pre_output, (self.num_sents, 1, 1))
        concat = T.concatenate([X_T_PRE, X_T, X_T_NEXT, pre_output_matrix], -1)
        h_hat = T.tanh(T.dot(concat, self.W_a) + self.b_a)
        strength = T.dot(h_hat, coherence_vector)  # shape is (sents, docs, 1)

        strength_mask = strength*T.transpose(self.sent_mask)
        strength_mask_T = T.transpose(strength_mask)
        """
        note that softmax will be computed row-wised if X is matrix
        a is of shape(num_docs, num_sents)
        """
        a = T.nnet.softmax(strength_mask_T)
        c = (a[:, :, None]*self.sent_encs).sum(axis=1)
        self.attention = a
        self.activation = c
        self.params = [self.W_a, self.b_a]


class MeaningAttention(AttentionLayer):
    def __init__(self, sent_rnn_att, *args, **kwargs):
        super(MeaningAttention, self).__init__(*args, **kwargs)
        X_T = self.sent_encs.dimshuffle([1, 0, 2])  # X_T is of shape(num_sent, doc_len, hidden_embedding)
        super(MeaningAttention, self).__init__(*args, **kwargs)
        self.prefix = "Meaning_Attention_"
        concat = T.concatenate([X_T, sent_rnn_att], axis=-1)
        self.params = []
        self.W_a = init_weights([2*self.in_size, self.in_size], self.prefix+"W_a")
        query_vector = self.auxiliary_vector
        if query_vector == None:
            self.W_u = init_weights((self.in_size,), name='query_vector')
        else:
            self.W_u = query_vector
        self.b = init_bias(self.in_size, self.prefix+'b_a')
        strength = T.dot(T.tanh(T.dot(concat, self.W_a)+self.b), self.W_u)

        # strength = ReLU(strength)  # or sigmoid
        self.sent_mask = self.sent_mask.reshape((self.batch_docs, self.num_sents))
        strength_mask = strength.dimshuffle([1, 0]) * self.sent_mask
        a = T.nnet.softmax(strength_mask)[:, :, None]
        c = (a*self.sent_encs).sum(axis=1)
        self.attention = a
        self.activation = c
        self.params += [self.W_a, self.b, self.W_u]


class SyntaxAttentionLayer(AttentionLayer):
    def __init__(self, *args, **kwargs):
        super(SyntaxAttentionLayer, self).__init__(*args, **kwargs)
        syntax_vector = self.auxiliary_vector
        prefix = "Syntax_Attention_"
        self.W_a = init_weights([2*self.out_size, self.out_size], prefix+"W_a")
        self.W_u = init_weights([self.out_size, 1], prefix+"W_u")
        self.b = init_bias(self.out_size, prefix+'b_a')
        syntax_matrix = T.reshape(T.tile(syntax_vector, self.num_sents*self.batch_docs), tuple(self.shape[:-1]))
        concat = T.concatenate([self.sent_encs, syntax_matrix], axis=2)
        # TODO why or should their is a bias in the dot next?
        strength = T.dot(T.tanh(T.dot(concat, self.W_a)+self.b), self.W_u).flatten()
        sent_mask = self.sent_mask.reshape((self.batch_docs, self.num_sents))
        strength = strength.reshape((self.batch_docs, self.num_sents))
        strength_mask = strength * sent_mask
        a = T.nnet.softmax(strength_mask)[:, :, None]
        c = (a*self.sent_encs).sum(axis=1)
        self.attention = a
        self.activation = c
        self.params = [self.W_a, self.W_u, self.b]
