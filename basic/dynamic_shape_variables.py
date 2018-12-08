import tensorflow as tf
import numpy as np


def build_computation_graph():
  # define placeholders
  x = tf.placeholder(tf.float32, [None, 1])
  y_init = tf.placeholder(tf.float32, [None, 1])

  # define variables
  #y = tf.Variable(np.zeros((5,1)), dtype=tf.float32)
  #y = tf.get_variable('y', shape=[5, 1], dtype=tf.float32)
  y_initializer = tf.zeros(shape=[None, 1], dtype=tf.float32)
  y = tf.Variable(y_initializer, validate_shape=False) #TODO: not working

  # define objective
  z = tf.reduce_sum((x - y) * (x - y))

  # define initialization operation
  op_init_y = y.assign(y_init)

  # define minimization operation
  optimizer = tf.train.AdamOptimizer()
  op_min = optimizer.minimize(z)

  return x, y_init, y, z, op_init_y, op_min


def main():
  # build computation graph
  x, y_init, y, z, op_init_y, op_min = build_computation_graph()

  # initialize session
  sess = tf.Session()

  # define x value
  x_val = np.random.random((5, 1))

  # initialize global variables
  sess.run(tf.global_variables_initializer())

  # initialize y value
  sess.run(op_init_y, { y_init: np.zeros((5, 1)) })

  # minimize
  for t in range(3000):
    sess.run(op_min, { x: x_val })

  # evaluate
  z_val = sess.run(z, { x: x_val })
  y_val = sess.run(y, {})

  print('x_val = ')
  print(x_val)
  print('')

  print('y_val = ')
  print(y_val)
  print('')

  print('z_val = ')
  print(z_val)
  print('')


if __name__ == '__main__':
  main()
