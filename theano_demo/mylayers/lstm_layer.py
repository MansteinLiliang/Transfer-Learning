# pylint: skip-file


class LSTMLayer(object):
    def __init__(self, rng, layer_name, X, shape, mask, total_sents, is_train=1, p=0.5):
        prefix = layer_name + "_"
        self.in_size, self.out_size = shape

        self.W_xi = init_weights((self.in_size, self.out_size), prefix + "W_xi" )
        self.W_hi = init_weights((self.out_size, self.out_size), prefix + "W_hi" )
        self.W_ci = init_weights((self.out_size, self.out_size), prefix + "W_ci" )
        self.b_i = init_bias(self.out_size, prefix + "b_i" )

        self.W_xf = init_weights((self.in_size, self.out_size), prefix + "W_xf" )
        self.W_hf = init_weights((self.out_size, self.out_size), prefix + "W_hf" )
        self.W_cf = init_weights((self.out_size, self.out_size), prefix + "W_cf" )
        self.b_f = init_bias(self.out_size, prefix + "b_f" )

        self.W_xc = init_weights((self.in_size, self.out_size), prefix + "W_xc" )
        self.W_hc = init_weights((self.out_size, self.out_size), prefix + "W_hc" )
        self.b_c = init_bias(self.out_size, prefix + "b_c" )

        self.W_xo = init_weights((self.in_size, self.out_size), prefix + "W_xo")
        self.W_ho = init_weights((self.out_size, self.out_size), prefix + "W_ho" )
        self.W_co = init_weights((self.out_size, self.out_size), prefix + "W_co" )
        self.b_o = init_bias(self.out_size, prefix + "b_o" )

        self.X = X
        self.M = mask

        def _active_mask(x, m, pre_h, pre_c):
            x = T.reshape(x, (-1, self.in_size))
            pre_h = T.reshape(pre_h, (-1, self.out_size))
            pre_c = T.reshape(pre_c, (-1, self.out_size))

            i = T.nnet.sigmoid(T.dot(x, self.W_xi) + T.dot(pre_h, self.W_hi) + T.dot(pre_c, self.W_ci) + self.b_i)
            f = T.nnet.sigmoid(T.dot(x, self.W_xf) + T.dot(pre_h, self.W_hf) + T.dot(pre_c, self.W_cf) + self.b_f)
            gc = T.tanh(T.dot(x, self.W_xc) + T.dot(pre_h, self.W_hc) + self.b_c)
            c = f * pre_c + i * gc
            o = T.nnet.sigmoid(T.dot(x, self.W_xo) + T.dot(pre_h, self.W_ho) + T.dot(c, self.W_co) + self.b_o)
            h = o * T.tanh(c)

            c = c * m[:, None]
            h = h * m[:, None]
            return h, c

        [h, c], updates = theano.scan(_active_mask,
                                      sequences=[self.X, self.M],
                                      outputs_info=[T.alloc(floatX(0.), total_sents, self.out_size),
                                                    T.alloc(floatX(0.), total_sents, self.out_size)])

        h = T.reshape(h, (self.X.shape[0], total_sents, self.out_size))
        # TODO dropout
        # if p > 0:
        #     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        #     drop_mask = srng.binomial(n=1, p=1 - p, size=h.shape, dtype=theano.config.floatX)
        #     self.activation = T.switch(T.eq(is_train, 1), h * drop_mask, h * (1 - p))
        # else:
        self.activation = T.switch(T.eq(is_train, 1), h, h)

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c,
                       self.W_xo, self.W_ho, self.W_co, self.b_o]

    def _active(self, x, pre_h, pre_c):
        i = T.nnet.sigmoid(T.dot(x, self.W_xi) + T.dot(pre_h, self.W_hi) + T.dot(pre_c, self.W_ci) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_xf) + T.dot(pre_h, self.W_hf) + T.dot(pre_c, self.W_cf) + self.b_f)
        gc = T.tanh(T.dot(x, self.W_xc) + T.dot(pre_h, self.W_hc) + self.b_c)
        c = f * pre_c + i * gc
        o = T.nnet.sigmoid(T.dot(x, self.W_xo) + T.dot(pre_h, self.W_ho) + T.dot(c, self.W_co) + self.b_o)
        h = o * T.tanh(c)
        return h, c


class BdLSTM(object):
    # Bidirectional LSTM Layer.
    def __init__(self, rng, layer_prefix, X, shape, mask, total_sents, is_train=1, p=0.5):
        fwd = LSTMLayer(rng, layer_prefix + "_fwd_", X, shape, mask, total_sents, is_train=is_train, p=p)
        bwd = LSTMLayer(rng, layer_prefix + "_bwd_", X[::-1], shape, mask[::-1], total_sents, is_train=is_train, p=p)
        self.params = fwd.params + bwd.params
        # self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], axis=-1)
        self.activation = fwd.activation + bwd.activation[::-1]
        # self.activation = T.max(
        #     T.concatenate(
        #         [fwd.activation.dimshuffle(('x', 0, 1, 2)),
        #         bwd.activation[::-1].dimshuffle(('x', 0, 1, 2))],axis=0
        #     ), axis=0
        # )


