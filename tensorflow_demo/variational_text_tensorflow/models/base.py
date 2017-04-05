import os
from glob import glob
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"


class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    pass

  def get_model_dir(self):
    model_dir = self.dataset
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "/%s=%s" % (attr, getattr(self, attr))
    return model_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, 
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def initialize(self, log_dir="./logs"):
    self.merged_sum = tf.merge_all_summaries()
    self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_iter = self.step.eval()

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Load SUCCESS")
      return True
    else:
      print(" [!] Load failed...")
      return False

def _linear(args, output_size, bias, bias_start=0.0, scope_name=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]
  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope, scope_name) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME+scope_name, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME+scope_name, [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return tf.nn.bias_add(res, biases)
