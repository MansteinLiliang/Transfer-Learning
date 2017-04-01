#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
# set use gpu programatically
import theano.sandbox.cuda


class MultiPerceptron(object):
    def __init__(self, prefix, in_tensor, init_dim, layer_num, hidden_size, activation='relu'):
        hidden_size.insert(0, init_dim)
        pre_tensor = in_tensor
        cur_tensor = None
        self.params = []
        for i in range(layer_num):
            prefix = prefix + '_' + str(i)
            W = init_weights((hidden_size[i], hidden_size[i+1]), prefix + '_W')
            b = init_bias(hidden_size[i+1], prefix + '_b')
            self.params += [W, b]
            if activation == 'tanh':
                cur_tensor = T.tanh(T.dot(pre_tensor, W) + b)
            elif activation == 'relu':
                cur_tensor = ReLU(T.dot(pre_tensor, W) + b)
            else:
                cur_tensor = T.nnet.sigmoid(T.dot(pre_tensor, W) + b)
            pre_tensor = cur_tensor
        self.activation = cur_tensor
        self.output_size = hidden_size[-1]

def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)

def Dropout(rng, activation, p=0.5):
    """
    :param rng: Is the GPU srng computed by numpy rng
    :param activation:
    :param p:
    :return:
    """
    mask = rng.binomial(n=1, p=p, size=activation.shape, dtype=theano.config.floatX)
    return activation * mask

def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name, value=0.0):
    return theano.shared(floatX(value*np.ones((size,))), name)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    return model
