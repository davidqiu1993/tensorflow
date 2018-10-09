import numpy as np
import random
import tensorflow as tf

from tqdm import tqdm

import IPython


def prepare_date():
  """
  Prepare training and testing data.
  """

  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # reduce datasets
  x_train = np.array(x_train[:10240])
  y_train = np.array(y_train[:10240])
  x_test = np.array(x_test[:1024])
  y_test = np.array(y_test[:1024])

  # re-scale and add channel dimension
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = np.reshape(x_train, [ x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 ])
  x_test  = np.reshape(x_test,  [  x_test.shape[0],  x_test.shape[1],  x_test.shape[2], 1 ])

  # convert to one-hot
  y_train = np.eye(10)[y_train]
  y_test  = np.eye(10)[y_test]

  return (x_train, y_train), (x_test, y_test)


def build(im_shape, n_categories):
  """
  Build the model.
  """

  X = tf.placeholder(tf.float32, [ None, im_shape[0], im_shape[1], im_shape[2] ])

  # construct convolutional layer #1
  conv1 = tf.layers.conv2d(
    inputs      = X,
    filters     = 32,
    kernel_size = [3, 3],
    padding     = 'same',
    activation  = tf.nn.relu,
    name        = 'conv1'
  )

  # construct max pooling layer #1
  pool1 = tf.layers.max_pooling2d(
    inputs    = conv1,
    pool_size = [2, 2],
    strides   = 2,
    name      = 'pool1'
  )

  # construct convolutional layer #2
  conv2 = tf.layers.conv2d(
    inputs      = pool1,
    filters     = 64,
    kernel_size = [3, 3],
    padding     = 'same',
    activation  = tf.nn.relu,
    name        = 'conv2'
  )

  # construct max pooling layer #2
  pool2 = tf.layers.max_pooling2d(
    inputs    = conv2,
    pool_size = [2, 2],
    strides   = 2,
    name      = 'pool2'
  )

  pool2_flat = tf.reshape(pool2, [-1, pool2.get_shape()[1:].num_elements()])

  # construct fully connected layer #1
  fc1 = tf.layers.dense(
    inputs     = pool2_flat,
    units      = 512,
    activation = tf.nn.relu,
    name       = 'fc1'
  )

  # logits layer
  logits = tf.layers.dense(
    inputs = fc1, #fc1_dropout,
    units  = n_categories,
    name   = 'logits'
  )

  # softmax layer
  Y = tf.nn.softmax(logits, name='softmax')

  return X, Y


def main():
  # prepare data
  (x_train, y_train), (x_test, y_test) = prepare_date()

  # build model
  im_shape = x_train.shape[1:]
  n_categories = 10
  X, Y_hat = build(im_shape, n_categories)

  # define loss
  Y = tf.placeholder(tf.float32, [ None, n_categories ])
  cross_entropy = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(Y_hat, 1e-10, 1.0)))

  # define optimizer
  optimizer = tf.train.AdamOptimizer()
  minimize = optimizer.minimize(cross_entropy)

  # define evaluation metrics
  mistakes = tf.not_equal(tf.argmax(Y, 1), tf.argmax(Y_hat, 1))
  error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

  # initialize session
  init_op = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init_op)

  # train
  batch_size = 64
  no_of_batches = int(len(x_train) / batch_size)
  epoch = 10

  print('training...')
  for i in range(epoch):
    ptr = 0
    for j in tqdm(range(no_of_batches)):
      x_batch, y_batch = x_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
      ptr += batch_size
      sess.run(minimize, {X: x_batch, Y: y_batch})
    train_err = sess.run(error, {X: x_train, Y: y_train})
    print('epoch #%04d: train_err=%.2f%%' % (i+1, train_err * 100))

  # test
  print('testing...')
  test_err = sess.run(error, {X: x_test, Y: y_test})
  print('test: test_err:%.2f%%' % (test_err * 100))


if __name__ == '__main__':
  main()
