# pylint: skip-file
from helpers.theano_optimizers import *
from mylayers.Convolution import SentenceConvolution, DocConvolution
from mylayers.attention import SyntaxAttentionLayer, DocLevelAttention
from mylayers.encoding_layer import DocEncoderLayer
from mylayers.layer_utils import init_weights, init_bias, MultiPerceptron
import theano.tensor as T
import theano


class MLP(object):
    def __init__(self, prefix, init_dim, layer_num, hidden_size, activation='tanh', optimizer='rmsprop'):
        hidden_size.insert(0, init_dim)
        self.batch_docs = T.iscalar('batch_docs')
        self.features = T.fmatrix('features')
        self.y = T.fvector('y')
        pre_tensor = self.features
        cur_tensor = None
        self.params = []
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        for i in range(layer_num):
            prefix = prefix + '_' + str(i)
            W = init_weights((hidden_size[i], hidden_size[i+1]), prefix + '_W')
            b = init_bias(hidden_size[i+1], prefix + '_b')
            self.params += [W, b]
            if activation == 'tanh':
                cur_tensor = T.tanh(T.dot(pre_tensor, W) + b)
            else:
                cur_tensor = T.nnet.sigmoid(T.dot(pre_tensor, W) + b)
            pre_tensor = cur_tensor
        self.activation = cur_tensor
        self.define_train_test_funcs()

    def mean_squared_error(self, X, y):
        W_out = init_weights((self.hidden_size[-1], 1), 'mse_W')
        b_out = init_bias(1, 'mse_b', value=0.0)
        self.params += [W_out, b_out]
        y_pred = T.dot(X, W_out)+b_out
        print 'compile the mse'
        self.cost = T.mean(T.square((y_pred-y.reshape((self.batch_docs, 1)))))
        # self.cost = T.pow(y_pred-y.reshape((self.batch_docs, 1)), 2).mean()
        clip = theano.gradient.grad_clip(self.cost, -1.0, 1.0)
        return clip, T.clip(y_pred, 0.0, 1.0)

    def define_train_test_funcs(self):
        cost, pred = self.mean_squared_error(self.activation, self.y)
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        self.train = theano.function(
            inputs=[self.features, lr, self.y, self.batch_docs],
            outputs=[self.cost, pred],
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        self.predict = theano.function(
            inputs=[self.features, self.y, self.batch_docs],
            outputs=[self.cost, pred],
            on_unused_input='ignore',
            allow_input_downcast=True
        )


class SynModel(object):
    def __init__(self, fet_size, U, vocab_size, embedding_size, hidden_size,
                 cell="gru", optimizer="rmsprop", p=0.5, num_sents=1, sent_len=1):
        """
        :param in_size: word-embedding dimension
        :param hidden_size: lm-rnn-layer hidden_size
        :param cell:
        :param optimizer:
        :param p: probability of NOT dropping out a unit
        :param num_sents:
        """
        self.fet_size = fet_size
        self.idxs = T.imatrix('idxs')
        self.features = T.fmatrix('features')
        self.embedding_size = embedding_size
        self.sent_len = sent_len
        self.hidden_size = hidden_size
        self.cell = cell
        self.vocab_size = vocab_size
        self.drop_rate = p

        self.params = []
        self.num_sents = num_sents  # num_sents is doc_len
        self.batch_docs = T.iscalar('batch_docs')  # input_tesor.shape[1] = batch_docs*num_sents
        self.is_train = T.iscalar('is_train')  # for dropout
        self.mask = T.matrix("mask")  # dype=None means to use config.floatX
        self.sent_mask = T.switch(self.mask.sum(axis=0) > 0, 1.0, 0.0)
        self.syntax_vector = init_weights((hidden_size,), 'syntax_vector')
        self.params.append(self.syntax_vector)
        self.optimizer = optimizer
        self.y = T.fvector('y')
        self.embedding = theano.shared(value=U, name='WEmb')
        embeddings = theano.shared(value=U, name='WEmb')
        self.params.append(embeddings)
        self.X = embeddings[self.idxs]
        self.define_layers()
        self.define_train_test_funcs()

    def define_layers(self):
        rng = np.random.RandomState(1234)
        sent_encoder_layer1 = SentenceConvolution(rng, "SentEncoder1", (
            self.batch_docs * self.num_sents, self.sent_len, self.embedding_size, self.hidden_size),
            self.X, 3, self.is_train, self.drop_rate, self.sent_mask, is_recurrent=False
        )
        self.params += sent_encoder_layer1.params
        doc_sent_X = sent_encoder_layer1.activation.reshape(
            (self.batch_docs, self.num_sents, self.hidden_size)
        ) * self.sent_mask.reshape((self.batch_docs, self.num_sents))[:, :, None]

        syntax_att_layer = SyntaxAttentionLayer(
            (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
            doc_sent_X, self.sent_mask, self.syntax_vector
        )
        self.params += syntax_att_layer.params
        syntax_att_output = syntax_att_layer.activation

        # doc Convolution modelling
        # doc_encoder_layer1 = DocConvolution(
        #     rng, "DocConvolution1", (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
        #     doc_sent_X, 3, self.is_train, self.drop_rate, self.sent_mask
        # )
        # self.params += doc_encoder_layer1.params
        # doc_encoder_layer2 = DocConvolution(
        #     rng, "DocConvolution1", (self.batch_docs, self.num_sents-2, self.hidden_size, self.hidden_size),
        #     doc_encoder_layer1.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        # )
        # self.params += doc_encoder_layer2.params
        # doc_encoder_layer3 = DocConvolution(
        #     rng, "DocConvolution1", (self.batch_docs, self.num_sents-4, self.hidden_size, self.hidden_size),
        #     doc_encoder_layer2.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        # )
        # self.params += doc_encoder_layer3.params
        mlp = MultiPerceptron("hidden", self.features, self.fet_size, 2, [400, 300, 200, 200], activation='relu')
        self.params += mlp.params

        X_T = doc_sent_X.dimshuffle([1, 0, 2])
        sent_mask_T = T.transpose(self.sent_mask.reshape((self.batch_docs, -1)))
        sent_annotate_layer1 = DocEncoderLayer(
            'DocAnnotation1', rng, X_T, self.hidden_size,
            self.hidden_size, self.cell, 0, 1, self.batch_docs, sent_mask_T
        )
        meaning_att = DocLevelAttention(
            mlp.output_size, mlp.activation,
            (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
            sent_annotate_layer1.activation.dimshuffle([1, 0, 2]), self.sent_mask,
        )
        self.params += meaning_att.params
        self.params += sent_annotate_layer1.params
        # self.activation = T.concatenate([meaning_att.activation, syntax_att_output], axis=1)
        # self.activation = T.concatenate([syntax_att_output, doc_encoder_layer1.pooling, doc_encoder_layer2.pooling, doc_encoder_layer3.pooling], axis=1)

        self.activation = T.concatenate([syntax_att_output, meaning_att.activation], axis=1)

    def mean_squared_error(self, X, y):
        W_out = init_weights((2*self.hidden_size, 1), 'mse_W')
        b_out = init_bias(1, 'mse_b', value=0.0)
        self.params += [W_out, b_out]
        y_pred = T.dot(X, W_out)+b_out
        print 'compile the mse'
        self.cost = T.mean(T.square((y_pred-y.reshape((self.batch_docs, 1)))))
        # self.cost = T.pow(y_pred-y.reshape((self.batch_docs, 1)), 2).mean()
        clip = theano.gradient.grad_clip(self.cost, -1.0, 1.0)
        return clip, T.clip(y_pred, 0.0, 1.0)

    def define_train_test_funcs(self):
        cost, pred = self.mean_squared_error(self.activation, self.y)
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        updates.append(
            (self.embedding, T.set_subtensor(
                self.embedding[0],
                theano.shared(np.zeros((self.embedding_size,), dtype=theano.config.floatX))
            ))
        )
        self.train = theano.function(
            inputs=[self.idxs, self.features, self.mask, lr, self.y, self.batch_docs],
            givens={self.is_train: np.cast['int32'](1)},
            outputs=[self.cost, pred],
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        self.predict = theano.function(
            inputs=[self.idxs, self.features, self.mask, self.y, self.batch_docs],
            givens={self.is_train: np.cast['int32'](0)},
            outputs=[self.cost, pred],
            on_unused_input='ignore',
            allow_input_downcast=True
        )


class HierarchicalModel(object):
    def __init__(self, U, vocab_size, embedding_size, hidden_size,
                 cell="gru", optimizer="rmsprop", p=0.5, num_sents=1, sent_len=1):
        """
        :param in_size: word-embedding dimension
        :param hidden_size: lm-rnn-layer hidden_size
        :param cell:
        :param optimizer:
        :param p: probability of NOT dropping out a unit
        :param num_sents:
        """
        self.idxs = T.imatrix('idxs')
        self.embedding_size = embedding_size
        self.sent_len = sent_len
        self.hidden_size = hidden_size
        self.cell = cell
        self.vocab_size = vocab_size
        self.drop_rate = p
        self.params = []
        self.num_sents = num_sents  # num_sents is doc_len
        self.batch_docs = T.iscalar('batch_docs')  # input_tesor.shape[1] = batch_docs*num_sents
        self.is_train = T.iscalar('is_train')  # for dropout
        self.mask = T.matrix("mask")  # dype=None means to use config.floatX
        self.sent_mask = T.switch(self.mask.sum(axis=0) > 0, 1.0, 0.0)
        self.syntax_vector = init_weights((hidden_size,), 'syntax_vector')
        self.params.append(self.syntax_vector)
        self.optimizer = optimizer
        self.y = T.fvector('y')
        self.embedding = theano.shared(value=U, name='WEmb')
        embeddings = theano.shared(value=U, name='WEmb')
        self.params.append(embeddings)

        self.X = embeddings[self.idxs]
        self.define_layers()
        self.define_train_test_funcs()

    def define_layers(self):
        # gating_layer = GRUGatingLayer(self.hidden_size)
        # self.params += gating_layer.params
        rng = np.random.RandomState(1234)
        sent_encoder_layer1 = SentenceConvolution(rng, "SentEncoder1", (
            self.batch_docs * self.num_sents, self.sent_len, self.embedding_size, self.hidden_size),
            self.X, 3, self.is_train, self.drop_rate, self.sent_mask, is_recurrent=False
        )
        self.params += sent_encoder_layer1.params
        doc_sent_X = sent_encoder_layer1.activation.reshape(
            (self.batch_docs, self.num_sents, self.hidden_size)
        ) * self.sent_mask.reshape((self.batch_docs, self.num_sents))[:, :, None]

        # Dropout Layer
        # Doc layer
        # if self.drop_rate > 0:
        #     srng = RandomStreams(rng.randint(999999))
        #     # train_output = Dropout(srng, np.cast[theano.config.floatX](1. / p) * h, p)
        #     train_output = Dropout(srng, doc_sent_X, self.drop_rate)
        #     doc_sent_X = T.switch(T.eq(self.is_train, 1), train_output, doc_sent_X)

        # sent modelling
        syntax_att_layer = SyntaxAttentionLayer(
            (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
            doc_sent_X, self.sent_mask, self.syntax_vector
        )
        self.params += syntax_att_layer.params
        syntax_att_output = syntax_att_layer.activation

        # doc Convolution modelling
        doc_encoder_layer1 = DocConvolution(
            rng, "DocConvolution1", (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
            doc_sent_X, 3, self.is_train, self.drop_rate, self.sent_mask
        )
        self.params += doc_encoder_layer1.params
        doc_encoder_layer2 = DocConvolution(
            rng, "DocConvolution1", (self.batch_docs, self.num_sents-2, self.hidden_size, self.hidden_size),
            doc_encoder_layer1.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        )
        self.params += doc_encoder_layer2.params
        doc_encoder_layer3 = DocConvolution(
            rng, "DocConvolution1", (self.batch_docs, self.num_sents-4, self.hidden_size, self.hidden_size),
            doc_encoder_layer2.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        )
        self.params += doc_encoder_layer3.params
        doc_encoder_layer4 = DocConvolution(
            rng, "DocConvolution1", (self.batch_docs, self.num_sents-6, self.hidden_size, self.hidden_size),
            doc_encoder_layer3.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        )
        self.params += doc_encoder_layer4.params
        doc_encoder_layer5 = DocConvolution(
            rng, "DocConvolution1", (self.batch_docs, self.num_sents-8, self.hidden_size, self.hidden_size),
            doc_encoder_layer4.activation, 3, self.is_train, self.drop_rate, self.sent_mask
        )
        self.params += doc_encoder_layer5.params

        # self.cnn_features = T.max(T.concatenate([doc_encoder_layer1.pooling[None,:,:],doc_encoder_layer2.pooling[None,:,:],doc_encoder_layer3.pooling[None,:,:]],axis=0),axis=0)
        # X_T = doc_sent_X.dimshuffle([1, 0, 2])
        # sent_mask_T = T.transpose(self.sent_mask.reshape((self.batch_docs, -1)))
        # sent_annotate_layer1 = DocEncoderLayer(
        #     'DocAnnotation1', rng, X_T, self.hidden_size,
        #     self.hidden_size, self.cell, 0, 1, self.batch_docs, sent_mask_T
        # )
        # meaning_att = DocLevelAttention(
        #     (self.batch_docs, self.num_sents, self.hidden_size, self.hidden_size),
        #     sent_annotate_layer1.activation.dimshuffle([1, 0, 2]), self.sent_mask
        # )
        # self.params += meaning_att.params
        # self.params += sent_annotate_layer1.params

        # self.activation = T.concatenate([meaning_att.activation, syntax_att_output], axis=1)
        # self.activation = T.concatenate([syntax_att_output, doc_encoder_layer1.pooling, doc_encoder_layer2.pooling, doc_encoder_layer3.pooling], axis=1)
        self.activation = T.concatenate([syntax_att_output, doc_encoder_layer5.pooling], axis=1)
        # self.activation = doc_encoder_layer1.activation

    def mean_squared_error(self, X, y):
        W_out = init_weights((2*self.hidden_size, 1), 'mse_W')
        b_out = init_bias(1, 'mse_b', value=0.0)

        self.params += [W_out, b_out]

        y_pred = T.dot(X, W_out)+b_out
        print 'compile the mse'
        self.cost = T.mean(T.square((y_pred-y.reshape((self.batch_docs, 1)))))
        # self.cost = T.pow(y_pred-y.reshape((self."batch_docs, 1)), 2).mean()
        clip = theano.gradient.grad_clip(self.cost, -1.0, 1.0)
        return clip, T.clip(y_pred, 0.0, 1.0)

    def define_train_test_funcs(self):
        cost, pred = self.mean_squared_error(self.activation, self.y)

        gparams = []

        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        updates.append(
            (self.embedding, T.set_subtensor(
                self.embedding[0],
                theano.shared(np.zeros((self.embedding_size,), dtype=theano.config.floatX))
            ))
        )
        self.train = theano.function(
            inputs=[self.idxs, self.mask, lr, self.y, self.batch_docs],
            givens={self.is_train: np.cast['int32'](1)},
            outputs=[self.cost, pred],
            updates=updates,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        self.predict = theano.function(
            inputs=[self.idxs, self.mask, self.y, self.batch_docs],
            givens={self.is_train: np.cast['int32'](0)},
            outputs=[self.cost, pred],
            on_unused_input='ignore',
            allow_input_downcast=True
        )