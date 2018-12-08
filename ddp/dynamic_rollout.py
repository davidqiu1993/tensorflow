import numpy as np
import tensorflow as tf


def build_computation_graph(T_max=30):
  T = tf.placeholder(tf.int32, [])
  X = tf.placeholder(tf.float32, [None, 1])

  Y = tf.Variable(tf.zeros([T_max, 1], tf.float32))

  c = lambda i, T, X, Y, J: i < T
  b = lambda i, T, X, Y, J: (i + 1, T, X, Y, J + tf.reduce_sum((X[i] - Y[i]) * (X[i] - Y[i])))
  out = tf.while_loop(c, b, (tf.convert_to_tensor(0, tf.int32), T, X, Y, tf.convert_to_tensor(0.0, tf.float32)))

  J = out[4]

  optimizer = tf.train.AdamOptimizer()
  op_min = optimizer.minimize(J)

  return T, X, Y, J, op_min


def main():
  # build computation graph
  T, X, Y, J, op_min = build_computation_graph(T_max=30)

  # define value
  T_val = 5
  X_val = np.random.random((T_val, 1)) * 10

  # initialize session and global variables
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # optimize
  epochs = 10000
  for ep in range(epochs):
    if ep % 100 == 0:
      J_val = sess.run(J, { T: T_val, X: X_val })
      print('[%d/%d] J_val = %.4f' % (ep, epochs, J_val))
    sess.run(op_min, { T: T_val, X: X_val })

  # evaluate
  J_val = sess.run(J, { T: T_val, X: X_val })
  Y_val = sess.run(Y, { })

  print('X_val = ')
  print(X_val)
  print('')

  print('Y_val = ')
  print(Y_val)
  print('')

  print('J_val = ')
  print(J_val)
  print('')


if __name__ == '__main__':
  main()
