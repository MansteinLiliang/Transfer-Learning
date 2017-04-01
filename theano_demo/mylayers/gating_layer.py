import theano.tensor as T

from theano_demo.mylayers import layer_utils

init_weights = layer_utils.init_weights
init_bias = layer_utils.init_bias


class HighWay(object):
    def __init__(self, vector_len):
        prefix = "HighWay_"
        prefix = "GRUGatingLayer_"
        self.W_h = init_weights([vector_len, vector_len], prefix + "W_h")
        self.b_h = init_bias(vector_len, prefix + 'b_h')
        self.params = [self.W_h, self.b_h]

    def gating_output(self, cur_output, pre_output):
        h = T.nnet.sigmoid(T.dot(pre_output, self.W_h) + self.b_h)
        output = (1 - h)*pre_output + h*cur_output
        return output


class GRUGatingLayer(object):
    def __init__(self, vector_len):
        prefix = "GRUGatingLayer_"
        self.W_hr = init_weights([vector_len, vector_len], prefix + "W_hr")
        self.W_hz = init_weights([vector_len, vector_len], prefix + "W_hz")
        self.W_xr = init_weights([vector_len, vector_len], prefix + "W_xr")
        self.W_xz = init_weights([vector_len, vector_len], prefix + "W_xz")
        self.W_hh = init_weights([vector_len, vector_len], prefix + "W_hh")
        self.W_xh = init_weights([vector_len, vector_len], prefix + "W_xh")
        self.b_h = init_bias(vector_len, prefix+'b_h')
        self.b_r = init_bias(vector_len, prefix+'b_r')
        self.b_z = init_bias(vector_len, prefix + 'b_z')
        self.params = [self.W_hr, self.W_hz, self.W_xr, self.W_xz, self.W_hh, self.W_xh, self.b_h, self.b_r, self.b_z]

    def gating_output(self, att_vector, pre_output):
        r = T.nnet.sigmoid(T.dot(att_vector, self.W_xr) + T.dot(pre_output, self.W_hr) + self.b_r)
        z = T.nnet.sigmoid(T.dot(att_vector, self.W_xz) + T.dot(pre_output, self.W_hz) + self.b_z)
        gh = T.tanh(T.dot(att_vector, self.W_xh) + T.dot(r * pre_output, self.W_hh) + self.b_h)
        cur_output = z * pre_output + (1 - z) * gh
        return cur_output


class SimpleGRUGatingLayer(object):
    def __init__(self, vector_len):
        prefix = "GRUGatingLayer_"
        self.W_hz = init_weights([vector_len, vector_len], prefix + "W_hz")
        self.W_xz = init_weights([vector_len, vector_len], prefix + "W_xz")
        self.b_z = init_bias(vector_len, prefix + 'b_z')
        self.params = [self.W_hz, self.W_xz, self.b_z]

    def gating_output(self, att_vector, pre_output):
        z = T.nnet.sigmoid(T.dot(att_vector, self.W_xz) + T.dot(pre_output, self.W_hz) + self.b_z)
        cur_output = z * pre_output + (1 - z) * att_vector
        return cur_output


class SimpleGatingLayer(object):
    def __init__(self, vector_len):

        prefix = "GRUGatingLayer_"
        self.W_h = init_weights([2*vector_len, vector_len], prefix + "W_h")
        self.b = init_bias(vector_len, prefix + 'b')
        self.params = [self.W_h, self.b]

    def gating_output(self, att_vector, pre_output):
        concat = T.concatenate([att_vector, pre_output], axis=-1)
        gating = T.nnet.sigmoid(T.dot(concat, self.W_h) + self.b)
        cur_output = gating*pre_output + (1 - gating)*att_vector
        return cur_output


class CondensedGatingLayer(object):
    def __init__(self, vector_len):
        prefix = "GRUGatingLayer_"
        self.W_h = init_weights([2 * vector_len, vector_len], prefix + "W_h")
        self.b = init_bias(vector_len, prefix + 'b')
        self.params = [self.W_h, self.b]

    def gating_output(self, att_vector, pre_output):
        concat = T.concatenate([att_vector, pre_output], axis=-1)
        cur_output = T.tanh(T.dot(concat, self.W_h) + self.b)
        return cur_output

class LinearGatingLayer(object):
    def __init__(self, vector_len):
        prefix = "LinearGatingLayer_"
        self.W_h = init_weights([vector_len, vector_len], prefix + "W_h")
        self.b = init_bias(vector_len, prefix + 'b')
        self.params = [self.W_h, self.b]

    def gating_output(self, att_vector, pre_output):
        pre_output_transform = T.dot(pre_output, self.W_h)+self.b
        cur_output = pre_output_transform + att_vector
        return cur_output