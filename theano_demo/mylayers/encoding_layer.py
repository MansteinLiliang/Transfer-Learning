# pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from gru_layer import GRULayer, BdGRU
from lstm_layer import LSTMLayer, BdLSTM


class SentEncoderLayer(object):
    def __init__(self, layer_name, X, in_size, hidden_size, cell, p, is_train, total_sents, mask, rng):
        # TODO sent representation can be pooling the over whole sentence
        """
        Support for dynamic batch, which is specified by num_sens*batch_docs
        :param layer_name:
        :param rng:
        :param X:
        :param in_size:
        :param hidden_size:
        :param cell:
        :param p:
        :param is_train:
        :param batch_size:
        :param mask:
        :return Tensor: shape is (sent_len, sent_num, embedding)
        """
        self.X = X
        self.in_size = in_size  # word_embedding size
        self.hidden_size = hidden_size # sent_embedding size
        self.cell = cell
        self.drop_rate = p
        self.is_train = is_train
        self.total_sents = total_sents  # T.scalar
        self.mask = mask
        self.rng = rng
        self.layer_name = layer_name
        self.define_layers()

    def define_layers(self):
        self.params = []
        # hidden layers
        layer_input = self.X
        shape = (self.in_size, self.hidden_size)
        if self.cell == "gru":
            hidden_layer = GRULayer(
                self.rng, self.layer_name, layer_input, shape,
                self.mask, self.total_sents, self.is_train, self.drop_rate
            )
        elif self.cell == "lstm":
            hidden_layer = LSTMLayer(
                self.rng, self.layer_name, layer_input, shape,
                self.mask, self.total_sents, self.is_train, self.drop_rate
            )
        elif self.cell == "bdlstm":
            hidden_layer = BdLSTM(
                self.rng, self.layer_name+"BDLSTM_", layer_input, shape,
                self.mask, self.total_sents, self.is_train, self.drop_rate
            )
            self.hidden_size = 2*self.hidden_size
        elif self.cell == "bdgru":
            hidden_layer = BdGRU(
                self.rng, self.layer_name + "BDLSTM_", layer_input, shape,
                self.mask, self.total_sents, self.is_train, self.drop_rate
            )
            self.hidden_size = 2*self.hidden_size
        else:
            raise ValueError
        self.params += hidden_layer.params
        self.activation = hidden_layer.activation
        # hidden_size is equal to the rnn-cell state size(output a hidden state)


class DocEncoderLayer(object):
    def __init__(self, layer_name, rng, X, in_size, hidden_size, cell, p, is_train, total_docs, sent_mask):
        """
        :param rng:
        :param X: shape(sent_nums, doc_nums, in_size)
        :param in_size:
        :param hidden_size:
        :param cell:
        :param p:
        :param is_train:
        :param total_docs:
        :param sent_mask: (sent_nums, doc_nums)
        :return Tensor: shape is (sent_num, doc_num, embedding)
        """
        prefix = layer_name + "_"
        '''
        def code(j):
            i = mask[:, j].sum() - 1
            i = T.cast(i, 'int32')
            sent_x = X[i, j * self.in_size : (j + 1) * self.in_size]
            return sent_x
        sent_X, updates = theano.scan(lambda i: code(i), sequences=[T.arange(mask.shape[1])])
        '''
        self._in_size = in_size
        if cell == "gru":
            self.encoder = GRULayer(
                rng, prefix+"GRU_", X, (in_size, hidden_size),
                sent_mask, total_docs, is_train, p
            )
            self.hidden_size = hidden_size
        elif cell == "lstm":
            self.encoder = LSTMLayer(
                rng, prefix+"LSTM_", X, (in_size, hidden_size),
                sent_mask, total_docs, is_train, p
            )
            self.hidden_size = hidden_size
        elif cell == "bdlstm":
            self.encoder = BdLSTM(
                rng, prefix+"BDLSTM_", X, (in_size, hidden_size),
                sent_mask, total_docs, is_train, p
            )
            self.hidden_size = 2*hidden_size
        elif cell == "bdgru":
            self.encoder = BdGRU(
                rng, prefix+"BDGRU", X, (in_size, hidden_size),
                sent_mask, total_docs, is_train, p
            )
            self.hidden_size = hidden_size
        else:
            raise ValueError
        self.activation = self.encoder.activation

        self.params = self.encoder.params
