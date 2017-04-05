# pylint: skip-file
import theano.tensor as T
from .utils_pg import *

class GRULayer(object):
    def __init__(self, rng, layer_prefix, X, shape, mask, total_sents, is_train=1, p=0.5):
        prefix = "_GRU_"
        layer_name = layer_prefix + prefix
        self.in_size, self.out_size = shape

        self.W_xr = init_weights((self.in_size, self.out_size), prefix + "W_xr" + layer_name)
        self.W_hr = init_weights((self.out_size, self.out_size), prefix + "W_hr" + layer_name)
        self.b_r = init_bias(self.out_size, prefix + "b_r" + layer_name)

        self.W_xz = init_weights((self.in_size, self.out_size), prefix + "W_xz" + layer_name)
        self.W_hz = init_weights((self.out_size, self.out_size), prefix + "W_hz" + layer_name)
        self.b_z = init_bias(self.out_size, prefix + "b_z" + layer_name)

        self.W_xh = init_weights((self.in_size, self.out_size), prefix + "W_xh" + layer_name)
        self.W_hh = init_weights((self.out_size, self.out_size), prefix + "W_hh" + layer_name)
        self.b_h = init_bias(self.out_size, prefix + "b_h" + layer_name)

        self.X = X
        self.M = mask

        def _active_mask(x, m, pre_h):
            x = T.reshape(x, (-1, self.in_size))
            pre_h = T.reshape(pre_h, (-1, self.out_size))

            r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
            z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
            gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
            h = z * pre_h + (1 - z) * gh

            h = h * m[:, None] + (1 - m[:, None]) * pre_h

            # h = T.reshape(h, (batch_size * self.out_size))
            return h

        h, updates = theano.scan(_active_mask, sequences=[self.X, self.M],
                                 outputs_info=[T.alloc(floatX(0.), total_sents, self.out_size)])
        # dic to matrix
        h = T.reshape(h, (self.X.shape[0], total_sents, self.out_size))
        # dropout

        self.activation = h

        self.params = [self.W_xr, self.W_hr, self.b_r,
                       self.W_xz, self.W_hz, self.b_z,
                       self.W_xh, self.W_hh, self.b_h]

    def _active(self, x, pre_h):
        r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
        z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
        gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
        h = z * pre_h + (1 - z) * gh
        return h


class BdGRU(object):
    # Bidirectional GRU Layer.
    def __init__(self, rng, layer_prefix, X, shape, mask, total_sents, is_train=1, p=0.5):
        self.in_size, self.out_size = shape
        fwd = GRULayer(rng, layer_prefix + "_fwd_", X, shape, mask, total_sents, is_train=is_train, p=p)
        bwd = GRULayer(rng, layer_prefix + "_bwd_", X[::-1], shape, mask[::-1], total_sents, is_train=is_train, p=p)
        self.params = fwd.params + bwd.params
        # self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], -1)
        self.activation = 0.5*fwd.activation + 0.5*bwd.activation[::-1]
