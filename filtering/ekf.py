import numpy as np
import tensorflow as tf


class ExtendedKalmanFilter(object):
  """
  Extended Kalman filter.
  """

  def __init__(self, options={}):
    """
    Initialize an extended Kalman filter.
    """

    super(ExtendedKalmanFilter, self).__init__()

    _options = {
      'x_dim': None, # state dimensions
      'u_dim': None, # control dimensions
      'z_dim': None, # measurement dimensions
      'f': None,     # process model
      'h': None      # measurement model
    }

    # load options
    for o in _options:
      if o in options:
        if _options[o] is None and o not in options:
          raise Error('Missing required option %s.' % (o))
        _options[o] = options[o]

    self.options = _options

    # define variables
    self.variables = {
      'x_hat': None,
      'Sigma_hat': None
    }

    # construct computation graph
    self.interfaces = self._construct_computation_graph()

    # initialize session
    self.sess = tf.Session()

    # initialize global variables
    self.sess.run(tf.global_variables_initializer())


  def _construct_computation_graph(self):
    """
    Construct computation graph.
    """

    x_dim = self.options['x_dim']
    u_dim = self.options['u_dim']
    z_dim = self.options['z_dim']
    f     = self.options['f']
    h     = self.options['h']

    # define placeholders
    x_prev     = tf.placeholder(tf.float32, [x_dim])        # state estimate
    Sigma_prev = tf.placeholder(tf.float32, [x_dim, x_dim]) # state uncertainty
    u_prev     = tf.placeholder(tf.float32, [u_dim])        # control
    z          = tf.placeholder(tf.float32, [z_dim])        # measurement
    Q          = tf.placeholder(tf.float32, [x_dim, x_dim]) # process noise
    R          = tf.placeholder(tf.float32, [z_dim, z_dim]) # measurement noise

    # forward the hidden state using process model
    x_pred = f(x_prev, u_prev)

    # calculate the Jacobian of the process model
    F = tf.stack([ tf.gradients(x_pred[d], x_prev)[0] for d in range(x_dim) ])

    # forward the process uncertainty with additive noise
    P = tf.matmul(F, tf.matmul(Sigma_prev, tf.transpose(F))) + Q

    # calculate the predictive measurement
    z_pred = h(x_pred)

    # calculate the measurement residual
    y = z - z_pred

    # calculate the Jacobian of the measurement model
    H = tf.stack([ tf.gradients(z_pred[d], x_pred)[0] for d in range(z_dim) ])

    # calculate the measurement uncertainty with additive noise
    S = tf.matmul(H, tf.matmul(P, tf.transpose(H))) + R

    # calculate the Kalman gain
    K = tf.matmul(P, tf.matmul(tf.transpose(H), tf.matrix_inverse(S)))

    # update the state estimate
    x = x_pred + tf.reshape(tf.matmul(K, tf.reshape(y, [z_dim, 1])), [x_dim])

    # update the state uncertainty
    Sigma = P - tf.matmul(K, tf.matmul(H, P))

    # construct interfaces
    interfaces = {
      'x_prev':     x_prev,     # previous state estimate
      'Sigma_prev': Sigma_prev, # previous state uncertainty
      'u_prev':     u_prev,     # previous control
      'z':          z,          # current measurement
      'Q':          Q,          # process noise
      'R':          R,          # measurement noise
      'F':          F,          # Jacobian of the process model
      'H':          H,          # Jacobian of the measurement model
      'x':          x,          # current state estimate
      'Sigma':      Sigma       # current state uncertainty
    }

    return interfaces


  def reset(self, x_hat, Sigma_hat):
    """
    Reset the extended Kalman filter.
    """

    self.variables['x_hat']     = x_hat
    self.variables['Sigma_hat'] = Sigma_hat


  def update(self, u_prev, z, Q, R):
    """
    Update the extended Kalman filter.
    """

    x_prev     = self.variables['x_hat']
    Sigma_prev = self.variables['Sigma_hat']

    x_hat = self.sess.run(self.interfaces['x'], {
      self.interfaces['x_prev']:     x_prev,
      self.interfaces['Sigma_prev']: Sigma_prev,
      self.interfaces['u_prev']:     u_prev,
      self.interfaces['z']:          z,
      self.interfaces['Q']:          Q,
      self.interfaces['R']:          R
    })

    Sigma_hat = self.sess.run(self.interfaces['Sigma'], {
      self.interfaces['x_prev']:     x_prev,
      self.interfaces['Sigma_prev']: Sigma_prev,
      self.interfaces['u_prev']:     u_prev,
      self.interfaces['z']:          z,
      self.interfaces['Q']:          Q,
      self.interfaces['R']:          R
    })

    self.variables['x_hat']     = x_hat
    self.variables['Sigma_hat'] = Sigma_hat


  def estimate(self):
    """
    Obtain the latest estimate.
    """

    return self.variables['x_hat'], self.variables['Sigma_hat']


def _test():
  import matplotlib.pyplot as plt

  # set up a test problem
  x_dim = 4
  u_dim = 2
  z_dim = 4

  T = 30
  dt = 0.1

  q = 0.01 ** 2
  Q = np.eye(x_dim) * q

  r = 0.1 ** 2
  R = np.eye(z_dim) * r

  def f(x, u, use_np=False):
    A = [[1, 0, dt,  0],
         [0, 1,  0, dt],
         [0, 0,  1,  0],
         [0, 0,  0,  1]]
    B = [[ 0,  0],
         [ 0,  0],
         [dt,  0],
         [ 0, dt]]

    x_next = None
    if use_np:
      x_next = np.reshape(np.matmul(A, np.reshape(x, [x_dim, 1])) + np.matmul(B, np.reshape(u, [u_dim, 1])), [x_dim])
    else:
      A = tf.constant(A, tf.float32)
      B = tf.constant(B, tf.float32)
      x_next = tf.reshape(tf.matmul(A, tf.reshape(x, [x_dim, 1])) + tf.matmul(B, tf.reshape(u, [u_dim, 1])), [x_dim])

    return x_next

  def h(x):
    return x

  def observe(x):
    z = h(x) + np.random.np.random.multivariate_normal(np.zeros(x_dim), R)
    return z

  # visualization helper
  def visualize(x, z, x_hat):
    plt.scatter([x[0]], [x[1]], c='b')
    plt.scatter([z[0]], [z[1]], c='k')
    plt.scatter([x_hat[0]], [x_hat[1]], c='r')
    plt.pause(dt)

  # generate test trajectory
  X = [ np.zeros(x_dim) ]
  U = [ np.array([1.0, 0.5]) for t in range(T) ]
  for t in range(T):
    X.append(f(X[t], U[t], use_np=True))

  x_0_hat = observe(X[0])
  Z = [ observe(X[t+1]) for t in range(T) ]

  # initialize an extended Kalman filter
  ekf = ExtendedKalmanFilter({
    'x_dim': x_dim,
    'u_dim': u_dim,
    'z_dim': z_dim,
    'f': f,
    'h': h
  })

  # reset
  ekf.reset(x_0_hat, R)

  # visualize
  plt.figure()
  plt.xlim([0.0 - 0.5, X[-1][0] + 0.5])
  plt.ylim([0.0 - 0.5, X[-1][1] + 0.5])
  visualize(X[0], x_0_hat, x_0_hat)

  # forward estimate
  for t in range(T):
    ekf.update(U[t], Z[t], Q, R)
    x_hat, Sigma_hat = ekf.estimate()

    # visualize
    visualize(X[t+1], Z[t], x_hat)

  plt.show()


if __name__ == '__main__':
  _test()
