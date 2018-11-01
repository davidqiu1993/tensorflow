import numpy as np
import random
import tensorflow as tf

from tqdm import tqdm

import IPython


def generate_data():
  N_train = 10000
  N_valid = 2000
  N_test  = 2000

  maxlen_x = 20
  minlen_x = 2
  p_stop = 0.05
  p_one = 0.35

  N = N_train + N_valid + N_test
  X = []
  Y = []

  # generate input data
  n_samples = 0
  while n_samples < N:
    x = [[0]]
    while np.random.random() >= p_stop and len(x) < maxlen_x - 1:
      x_t = [1] if np.random.random() < p_one else [0]
      x.append(x_t)
    if len(x) >= minlen_x:
      X.append(x)
      n_samples += 1

  # generate output data
  onehot = np.eye(maxlen_x).astype(np.int)
  for i in range(len(X)):
    yn = 0
    for j in range(len(X[i])):
      yn += X[i][j][0]
    y = onehot[yn]
    Y.append(y)

  # split datasets
  X_train = X[:N_train]
  Y_train = Y[:N_train]
  X_valid = X[N_train:N_train+N_valid]
  Y_valid = Y[N_train:N_train+N_valid]
  X_test  = X[N_train+N_valid:]
  Y_test  = Y[N_train+N_valid:]

  # return
  return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


class Learner(object):
  """
  Learner for sequence with dynamic length.
  """

  def __init__(self):
    super(Learner, self).__init__()

    # configurations
    self.conf = {
      'lstm_units':      32,
      'dim_feature':     1,
      'dim_prediction':  20,
      'feature_padding': [0],
      'batch_size':      1000
    }

    # define lstm cell
    lstm_units = self.conf['lstm_units']
    self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)

    # define decoder parameters
    dim_prediction = self.conf['dim_prediction']
    self.decoder_w = tf.Variable(tf.truncated_normal([lstm_units, dim_prediction]))
    self.decoder_b = tf.Variable(tf.constant(0.1, shape=[dim_prediction]))

    # define optimizer
    self.optimizer = tf.train.AdamOptimizer()

    # build model and define operations
    X_input, Y_pred, seq_length = self._build_model()
    Y_true = tf.placeholder(tf.float32, [None, dim_prediction])
    minimize, metrics = self._define_operations(X_input, Y_pred, Y_true)
    self.seq_length = seq_length
    self.X_input = X_input
    self.Y_pred = Y_pred
    self.Y_true = Y_true
    self.minimize = minimize
    self.metrics = metrics

    # initialize session
    self.sess = tf.Session()

    # initialize all variables
    self.sess.run(tf.initialize_all_variables())


  def _pad_batch_input(self, X):
    feature_padding = self.conf['feature_padding']

    # determine the maximum length of sequence in the batch input sequences
    maxlen_x = 0
    for i in range(len(X)):
      if len(X[i]) > maxlen_x:
        maxlen_x = len(X[i])

    # construct padded input sequences with padding feature
    X_padded = []
    for i in range(len(X)):
      x = []
      for j in range(len(X[i])):
        x.append(X[i][j])
      n_paddings = maxlen_x - len(X[i])
      for j in range(n_paddings):
        x.append(feature_padding)
      X_padded.append(x)

    return X_padded


  def _build_model(self):
    dim_feature = self.conf['dim_feature']

    # define sequence length
    seq_length = tf.placeholder(tf.int32)

    # define input
    X_input = tf.placeholder(tf.float32, [None, None, dim_feature])

    # roll out lstm
    lstm_outs, lstm_state_last = tf.nn.dynamic_rnn(
      self.lstm_cell, X_input, sequence_length=seq_length, dtype=tf.float32)
    lstm_outs = tf.transpose(lstm_outs, [1, 0, 2]) # move the length dimension to the front

    # gather the lstm output at the last time step
    lstm_out_last = tf.gather(lstm_outs, seq_length - 1)

    # construct decoder
    decoder_logits = tf.matmul(lstm_out_last, self.decoder_w) + self.decoder_b
    Y_pred = tf.nn.softmax(decoder_logits)

    return X_input, Y_pred, seq_length


  def _define_loss(self, Y_pred, Y_true):
    # define cross entropy loss
    cross_entropy = -tf.reduce_sum(Y_true * tf.log(tf.clip_by_value(Y_pred, 1e-10, 1.0)))

    # l = - y * log(y_hat):
    #   y = 0, y_hat -> 0: l = 0
    #   y = 0, y_hat -> 1: l = 0
    #   y = 1, y_hat -> 0: l -> +inf
    #   y = 1, y_hat -> 1: l -> 0

    return cross_entropy


  def _define_metrics(self, Y_pred, Y_true):
    # define loss metric
    metric_loss = self._define_loss(Y_pred, Y_true)

    # define error (mean mistakes) metric
    mistakes = tf.not_equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
    metric_error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # construct metrics
    metrics = {
      'loss':  metric_loss,
      'error': metric_error
    }

    return metrics


  def _define_operations(self, X_input, Y_pred, Y_true):
    # define loss
    loss = self._define_loss(Y_pred, Y_true)

    # define loss minimization operation
    minimize = self.optimizer.minimize(loss)

    # define metrics evaluation operation
    metrics = self._define_metrics(Y_pred, Y_true)

    return minimize, metrics


  def _train_once(self, X, Y):
    # preprocess training batch
    X_train = np.array(self._pad_batch_input(X))
    Y_train = np.array(Y)
    len_x = X_train.shape[1]

    # minimize
    self.sess.run(self.minimize, {self.X_input: X_train, self.Y_true: Y_train, self.seq_length: len_x})


  def evaluate(self, X, Y):
    # preprocess samples
    X_eval = np.array(self._pad_batch_input(X))
    Y_eval = np.array(Y)
    len_x = X_eval.shape[1]

    # evaluate
    metrics_val = {}
    for o in self.metrics:
      metrics_val[o] = self.sess.run(self.metrics[o], {self.X_input: X_eval, self.Y_true: Y_eval, self.seq_length: len_x})

    return metrics_val


  def train(self, X_train, Y_train, X_valid=None, Y_valid=None):
    epochs = 50
    batch_size = self.conf['batch_size']

    N_batches = len(X_train) // batch_size
    indices_train = [ i for i in range(len(X_train)) ]

    # train for multiple epochs
    for ep in range(epochs):
      # shuffle training samples
      random.shuffle(indices_train)

      # train through all training samples
      for i_batch in tqdm(range(N_batches)):
        # determine sample indices for the training batch
        ptr_lower = i_batch * batch_size
        ptr_upper = ptr_lower + batch_size
        indices_batch = indices_train[ptr_lower:ptr_upper]

        # construct training batch
        X_batch = []
        Y_batch = []
        for i_sample in indices_batch:
          X_batch.append(X_train[i_sample])
          Y_batch.append(Y_train[i_sample])

        # train once
        self._train_once(X_batch, Y_batch)

      # evaluate
      metrics_train = self.evaluate(X_train, Y_train)
      metrics_valid = None
      if (X_valid is not None) and (Y_valid is not None):
        metrics_valid = self.evaluate(X_valid, Y_valid)

      # log
      if (X_valid is not None) and (Y_valid is not None):
        print('epoch #%04d: train_err=%.2f%%, valid_err=%.2f%%' % (
          ep+1, metrics_train['error'] * 100, metrics_valid['error'] * 100))
      else:
        print('epoch #%04d: train_err=%.2f%%' % (
          ep+1, metrics_train['error'] * 100))


  def test(self, X_test, Y_test):
    # test
    metrics_test = self.evaluate(X_test, Y_test)

    # log
    print('test: test_err=%.2f%%' % (metrics_test['error'] * 100))


def main():
  # generate datasets
  print('generating datasets..')
  X_train, Y_train, X_valid, Y_valid, X_test, Y_test = generate_data()
  print('datasets generated..')
  print('')

  # create learner
  print('creating learner..')
  learner = Learner()
  print('learner created..')
  print('')

  # train learner
  print('training learner..')
  learner.train(X_train, Y_train, X_valid, Y_valid)
  print('learner trained..')
  print('')

  # test learner
  print('testing learner..')
  learner.test(X_test, Y_test)
  print('learner tested..')
  print('')


if __name__ == '__main__':
  main()
