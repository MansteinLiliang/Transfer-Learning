from __future__ import absolute_import
import time
import numpy as np
import tensorflow as tf

from .base import Model
try:
  linear = tf.nn.rnn_cell.linear
except:
  from tensorflow.python.ops.rnn_cell import _linear as linear

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess, reader, dataset="ptb",
               decay_rate=0.96, decay_step=10000, embed_dim=500,
               h_dim=50, learning_rate=0.001, max_iter=450000,
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

    self.max_iter = max_iter
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.checkpoint_dir = checkpoint_dir
    self.step = tf.Variable(0, trainable=False)  
    self.lr = tf.train.exponential_decay(
        learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")

    _ = tf.scalar_summary("learning rate", self.lr)

    self.dataset = dataset
    self._attrs = ["h_dim", "embed_dim", "max_iter", "dataset",
                   "learning_rate", "decay_rate", "decay_step"]

    self.build_model()

  def build_model(self):
    self.x = tf.placeholder(tf.float32, [self.reader.vocab_size], name="input")
    self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")

    self.build_encoder()
    self.build_generator()

    # Kullback Leibler divergence
    self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

    # Log likelihood
    self.g_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))

    self.loss = self.e_loss + self.g_loss

    self.encoder_var_list, self.generator_var_list = [], []
    for var in tf.trainable_variables():
      if "encoder" in var.name:
        self.encoder_var_list.append(var)
      elif "generator" in var.name:
        self.generator_var_list.append(var)

    # optimizer for alternative update
    self.optim_e = tf.train.AdamOptimizer(learning_rate=self.lr) \
                         .minimize(self.e_loss, global_step=self.step, var_list=self.encoder_var_list)
    self.optim_g = tf.train.AdamOptimizer(learning_rate=self.lr) \
                         .minimize(self.g_loss, global_step=self.step, var_list=self.generator_var_list)

    # optimizer for one shot update
    self.optim = tf.train.AdamOptimizer(learning_rate=self.lr) \
                         .minimize(self.loss, global_step=self.step)

    _ = tf.scalar_summary("encoder loss", self.e_loss)
    _ = tf.scalar_summary("generator loss", self.g_loss)
    _ = tf.scalar_summary("total loss", self.loss)

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder"):
      self.l1_lin = linear(tf.expand_dims(self.x, 0), self.embed_dim, bias=True, scope="l1")
      self.l1 = tf.nn.relu(self.l1_lin)

      self.l2_lin = linear(self.l1, self.embed_dim, bias=True, scope="l2")
      self.l2 = tf.nn.relu(self.l2_lin)

      self.mu = linear(self.l2, self.h_dim, bias=True, scope="mu")
      self.log_sigma_sq = linear(self.l2, self.h_dim, bias=True, scope="log_sigma_sq")

      self.eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
      self.sigma = tf.sqrt(tf.exp(self.log_sigma_sq))

      self.h = tf.add(self.mu, tf.mul(self.sigma, self.eps))

      _ = tf.histogram_summary("mu", self.mu)
      _ = tf.histogram_summary("sigma", self.sigma)
      _ = tf.histogram_summary("h", self.h)
      _ = tf.histogram_summary("mu + sigma", self.mu + self.sigma)

  def build_generator(self):
    """Inference Network. p(X|h)"""
    with tf.variable_scope("generator"):
      self.R = tf.get_variable("R", [self.reader.vocab_size, self.h_dim])
      self.b = tf.get_variable("b", [self.reader.vocab_size])

      self.e = -tf.matmul(self.h, self.R, transpose_b=True) + self.b
      self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))

  def get_hidden_features(self, x, x_idx):
      return self.sess.run(self.mu, feed_dict={self.x: x, self.x_idx: x_idx})

  def train(self, config):
    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/%s" % self.get_model_dir(), self.sess.graph_def)

    tf.initialize_all_variables().run()
    # self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()

    iterator = self.reader.iterator()
    for step in xrange(start_iter, start_iter + self.max_iter):
      x, x_idx = iterator.next()

      """The paper update the parameters alternatively but in this repo I used oneshot update.

      _, e_loss, mu, sigma, h = self.sess.run(
          [self.optim_e, self.e_loss, self.mu, self.sigma, self.h], feed_dict={self.x: x})

      _, g_loss, summary_str = self.sess.run(
          [self.optim_g, self.g_loss, merged_sum], feed_dict={self.h: h,
                                                              self.mu: mu,
                                                              self.sigma: sigma,
                                                              self.e_loss: e_loss,
                                                              self.x_idx: x_idx})
      """

      _, loss, mu, sigma, h, summary_str = self.sess.run(
          [self.optim, self.loss, self.mu, self.sigma, self.h, merged_sum],
          feed_dict={self.x: x, self.x_idx: x_idx})

      if step % 2 == 0:
        writer.add_summary(summary_str, step)

      if step % 10 == 0:
        print("Step: [%4d/%4d] time: %4.4f, loss: %.8f" \
            % (step, self.max_iter, time.time() - start_time, loss))
        #print("Step: [%4d/%4d] time: %4.4f, loss: %.8f, e_loss: %.8f, g_loss: %.8f" \
        #    % (step, self.max_iter, time.time() - start_time, e_loss + g_loss, e_loss, g_loss))

      if step % 5000 == 0:
        self.save(self.checkpoint_dir, step)

        if self.dataset == "ptb":
          self.sample(3, "costs")
          self.sample(3, "chemical company")
          self.sample(3, "government violated")
        elif self.dataset == "toy":
          self.sample(3, "a")
          self.sample(3, "g")
          self.sample(3, "k")

  def sample(self, sample_size=20, text=None):
    """Sample the documents."""
    p = 1

    if text != None:
      try:
        x, word_idxs = self.reader.get(text)
      except Exception as e:
        print(e)
        return
    else:
      x, word_idxs = self.reader.random()
    print(" [*] Text: %s" % " ".join([self.reader.idx2word[word_idx] for word_idx in word_idxs]))

    cur_ps = self.sess.run(self.p_x_i, feed_dict={self.x: x})
    word_idxs = np.array(cur_ps).argsort()[-sample_size:][::-1]
    ps = cur_ps[word_idxs]

    for idx, (cur_p, word_idx) in enumerate(zip(ps, word_idxs)):
      print("  [%d] %-20s: %.8f" % (idx+1, self.reader.idx2word[word_idx], cur_p))
      p *= cur_p

      print(" [*] perp : %8.f" % -np.log(p))
