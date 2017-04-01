import numpy as np
import theano
import theano.tensor as T

from theano_demo.mylayers import layer_utils

init_weights = layer_utils.init_weights
init_bias = layer_utils.init_bias
ReLU = layer_utils.ReLU
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from encoding_layer import SentEncoderLayer


class DocConvolution(object):
    def __init__(self, rng, prefix, shape, X, filter_width, is_trian, p, mask):
        """
        Input sentence vectors, output the convolution result
        :param rng:
        :param prefix:
        :param shape:
        :param X: shape: (doc_num, doc_len, in_size, out_size)
        :param filter_width:
        :param is_trian:
        :param p:
        :param mask:
        """
        prefix = prefix + "_"
        self.doc_num, self.doc_len, self.in_size, self.out_size = shape
        self.hidden_size = self.out_size
        # X *= mask.reshape((self.doc_num, self.doc_len))[:, :, None]
        self.input = X
        self.filter_shape = (self.out_size, self.in_size, 1, filter_width)
        w_bound = np.sqrt(self.in_size * 1 * filter_width)
        self.shape = (self.doc_num, self.in_size, 1, self.doc_len)
        self.input = self.input.dimshuffle([0, 2, 1]).reshape(self.shape)

        self.non_linear = "relu"
        self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name=prefix + "W_conv")

        b_values = np.zeros((self.out_size,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name=prefix + "b_conv")
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape, border_mode='valid')  # shape is (doc_num, nb filters, output row=1, output col=self.sent_nums - filter_width + 1)

        self.params = [self.W, self.b]

        conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # self.output = downsample.max_pool_2d(input=conv_out_relu, ds=(1, self.doc_len - filter_width + 1),
        #                                      ignore_border=True)

        self.activation = conv_out_relu.reshape((self.doc_num, self.out_size, self.doc_len - filter_width + 1)).dimshuffle([0, 2, 1])
        self.pooling = downsample.max_pool_2d(
            input=conv_out_relu,
            ds=(1, self.doc_len - filter_width + 1),
            ignore_border=True
        ).reshape((self.doc_num, self.out_size))


class SentenceConvolution(object):
    def __init__(self, rng, prefix, shape, X, filter_width, is_train, p, mask, is_recurrent=False):
        """
        :param mask:
        :param is_train:
        :param p:
        :param is_recurrent:
        :param prefix:
        :param shape: (sent_num, sent_len, in_size, out_size)
        :param X: shape is (sent_len, batch_docs*setn_num, in_size)
        :return shape is (batch_docs*sent_nums, out_dim)
        """
        prefix = prefix + "_"
        self.sent_num, self.sent_len, self.in_size, self.out_size = shape
        self.hidden_size = self.out_size
        self.input = X.dimshuffle([1, 0, 2])
        self.filter_shape = (self.out_size, self.in_size, 1, filter_width)
        w_bound = np.sqrt(self.in_size * 1 * filter_width)
        self.shape = (self.sent_num, self.in_size, 1, self.sent_len)
        self.input = self.input.dimshuffle([0, 2, 1]).reshape(self.shape)
        self.non_linear = "relu"
        self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name=prefix + "W_conv")

        b_values = np.zeros((self.out_size,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name=prefix + "b_conv")
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape, border_mode='valid')  # shape is (sent_num, nb filters, output row=1, output col=self.sent_len - filter_width + 1)

        self.params = [self.W, self.b]
        if is_recurrent == True:
            conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            rnn_input = conv_out_relu.reshape(
                (conv_out_relu.shape[0], conv_out_relu.shape[1], conv_out_relu.shape[3])).dimshuffle([2, 0, 1])
            rnn_encoder = SentEncoderLayer(prefix+"RNN", rnn_input, self.out_size, self.out_size, 'gru', p, is_train, self.sent_num, mask[filter_width-1:], rng)
            self.activation = rnn_encoder.activation[rnn_encoder.activation.shape[0]-1:]
        else:
            if self.non_linear == "tanh":
                conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=(1, self.sent_len - filter_width + 1),
                                                     ignore_border=True)
            elif self.non_linear == "relu":
                conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                self.output = downsample.max_pool_2d(input=conv_out_relu, ds=(1, self.sent_len - filter_width + 1),
                                                     ignore_border=True)
            else:
                pooled_out = downsample.max_pool_2d(input=conv_out, ds=(1, self.sent_len - filter_width + 1),
                                                    ignore_border=True)
                self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
            self.activation = self.output.reshape((self.sent_num, self.out_size))

