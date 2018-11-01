import numpy as np
import random
import tensorflow as tf

import IPython


def generate_data():
  # generate input data
  X = ['{0:016b}'.format(i) for i in range(2**16)]
  random.shuffle(X)
  X = [map(int,i) for i in X]
  ti  = []
  for i in X:
    temp_list = []
    for j in i:
      temp_list.append([j])
    ti.append(np.array(temp_list))
  X = ti

  # generate output data
  Y = []
  for i in X:
    count = 0
    for j in i:
      if j[0] == 1:
        count+=1
    temp_list = ([0]*17)
    temp_list[count]=1
    Y.append(temp_list)

  # split data
  NUM_EXAMPLES = 10000
  NUM_VALIDS = 2000
  train_X = X[:NUM_EXAMPLES]
  train_Y = Y[:NUM_EXAMPLES]
  valid_X = X[NUM_EXAMPLES:NUM_EXAMPLES+NUM_VALIDS]
  valid_Y = Y[NUM_EXAMPLES:NUM_EXAMPLES+NUM_VALIDS]
  test_X  = X[NUM_EXAMPLES+NUM_VALIDS:]
  test_Y  = Y[NUM_EXAMPLES+NUM_VALIDS:]

  # return data
  return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


def build():
  X = tf.placeholder(tf.float32, [None, 16, 1])
  Y = tf.placeholder(tf.float32, [None, 17])

  # define LSTM
  lstm_units = 24
  lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
  lstm_out, lstm_state = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

  # gather LSTM output
  lstm_out = tf.transpose(lstm_out, [1, 0, 2])
  lstm_out_last = tf.gather(lstm_out, int(lstm_out.get_shape()[0]) - 1)

  # define decoder
  decoder_w = tf.Variable(tf.truncated_normal([lstm_units, int(Y.get_shape()[1])]))
  decoder_b = tf.Variable(tf.constant(0.1, shape=[Y.get_shape()[1]]))
  decoder_out = tf.nn.softmax(tf.matmul(lstm_out_last, decoder_w) + decoder_b)

  # define loss
  cross_entropy = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(decoder_out, 1e-10, 1.0)))

  # define optimizer
  optimizer = tf.train.AdamOptimizer()
  minimize = optimizer.minimize(cross_entropy)

  # define test metrics
  mistakes = tf.not_equal(tf.argmax(Y, 1), tf.argmax(decoder_out, 1))
  error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

  return X, Y, minimize, error


def main():
  # build network
  print('building network..')
  X, Y, minimize, error = build()
  print('built network..')
  print('')

  # generate data
  print('generating data..')
  train_X, train_Y, valid_X, valid_Y, test_X, test_Y = generate_data()
  print('generated data..')
  print('')

  # initialize session
  init_op = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init_op)

  # train
  print('training..')
  batch_size = 1000
  no_of_batches = int(len(train_X) / batch_size)
  epoch = 150
  for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
      batch_X, batch_Y = train_X[ptr:ptr+batch_size], train_Y[ptr:ptr+batch_size]
      ptr += batch_size
      sess.run(minimize, {X: batch_X, Y: batch_Y})
    train_err = sess.run(error, {X: train_X, Y: train_Y})
    valid_err = sess.run(error, {X: valid_X, Y: valid_Y})
    print('epoch #%04d: train_err=%.2f%%, valid_err=%.2f%%' % (i+1, train_err * 100, valid_err * 100))
  print('trained..')
  print('')

  # test
  print('testing..')
  test_err = sess.run(error, {X: test_X, Y: test_Y})
  print('test_err: %.2f%%' % (test_err * 100))
  print('tested..')
  print()

  # close session
  sess.close()


if __name__ == '__main__':
  main()
