import numpy as np
import tensorflow as tf


class TrajectoryOptimizer(object):
  """
  Trajectory optimizer.
  """

  def __init__(self, options={}):
    """
    Initialize a trajectory optimizer.
    """

    super(TrajectoryOptimizer, self).__init__()

    _options = {
      'x_dim': None, # dimensions of state
      'u_dim': None, # dimensions of control
      'dt':    0.05, # time step interval
      'f':     None, # dynamics model, x_next = f(x, u, dt)
      'c_t':   None, # running cost, c = c_t(x, u, dt)
      'c_f':   None, # final cost, c = c_f(x_f)
      'T_max': None  # maximum horizon
    }

    # load options
    for o in _options:
      if o in options:
        if _options[o] is None and o not in options:
          raise Error('Missing required option %s.' % (o))
        _options[o] = options[o]

    self.options = _options

    # initialize optimizer
    self.optimizer = tf.train.AdamOptimizer()

    # construct the computation graph
    print('constructing computation graph..')
    self.interfaces = self._construct_computation_graph()
    print('computation graph constructed..')

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
    dt    = self.options['dt']
    f     = self.options['f']
    c_t   = self.options['c_t']
    c_f   = self.options['c_f']
    T_max = self.options['T_max']

    # initialize placeholders
    T = tf.placeholder(tf.int32, [])
    x_0 = tf.placeholder(tf.float32, [x_dim])
    U_init = tf.placeholder(tf.float32, [None, u_dim])

    # initialize variables
    U = tf.Variable(tf.zeros([T_max, u_dim]), tf.float32)

    # define control trajectory initialization operation
    U_init_padded = tf.concat([ U_init, tf.zeros([T_max - T, u_dim], tf.float32) ], 0)
    op_init_U = U.assign(U_init_padded)

    # simulate forward
    X_init = tf.convert_to_tensor([x_0])
    J_init = tf.convert_to_tensor(0.0, tf.float32)

    def while_cond(t, X, J):
      return t < T

    def while_body(t, X, J):
      X = tf.concat([ X, [f(X[t], U[t], dt)] ], 0)
      J = J + c_t(X[t], U[t], dt)
      t = t + 1
      return t, X, J

    t, X, J = tf.while_loop(
      while_cond,
      while_body,
      loop_vars=[ tf.convert_to_tensor(0, tf.int32), X_init, J_init ],
      shape_invariants=[ tf.TensorShape([]), tf.TensorShape([None, x_dim]), tf.TensorShape([]) ]
    )

    J = J + c_f(X[T])

    # define cost-to-go minimization operation
    op_min_J = self.optimizer.minimize(J)

    # construct interfaces
    interfaces = {
      'T': T,
      'x_0': x_0,
      'U_init': U_init,
      'U': U,
      'X': X,
      'J': J,
      'op_init_U': op_init_U,
      'op_min_J': op_min_J
    }

    return interfaces


  def optimize(self, x_0_val, U_init_val, epsilon=1e-4, max_epochs=100, evaluate=False, verbose=False):
    """
    Optimize the action trajectory.
    """

    interfaces = self.interfaces
    sess = self.sess

    # log
    if verbose:
      print('initializing trajectory optimization..')

    # infer horizon
    T_val = len(U_init_val)

    # initialize the control trajectory
    sess.run(interfaces['op_init_U'], {
      interfaces['T']: T_val,
      interfaces['U_init']: U_init_val
    })

    # evaluate the initial trajectory
    X_init_val = None
    if evaluate:
      X_init_val = sess.run(interfaces['X'], {
        interfaces['T']: T_val,
        interfaces['x_0']: x_0_val
      })

    J_init_val = sess.run(interfaces['J'], {
      interfaces['T']: T_val,
      interfaces['x_0']: x_0_val
    })

    # log
    if verbose:
      print('x_0 = %s' % (str(x_0_val)))

      for t in range(T_val):
        print('U_init[%d] = %s' % (t, str(U_init_val[t])))

    if evaluate and verbose:
      for t in range(T_val + 1):
        print('X_init[%d] = %s' % (t, str(X_init_val[t])))

    if verbose:
      print('J_init = %.4f' % (J_init_val))

    # log
    if verbose:
      print('optimizing trajectory..')

    # run epochs
    last_J_val = None
    for ep in range(max_epochs):
      # optimize the action trajectory
      sess.run(interfaces['op_min_J'], {
        interfaces['T']: T_val,
        interfaces['x_0']: x_0_val
      })

      # evaluate
      J_val = sess.run(interfaces['J'], {
        interfaces['T']: T_val,
        interfaces['x_0']: x_0_val
      })

      # log
      if verbose:
        print('[%d/%d] J_val=%.4f' % (ep + 1, max_epochs, J_val))

      # check optimization termination conditions
      if last_J_val is not None:
        if abs(J_val - last_J_val) < epsilon:
          break

      # update last J value
      last_J_val = J_val

    # log
    if verbose:
      print('trajectory optimization finished..')

    # evaluate the optimized trajectory
    U_opt_val = sess.run(interfaces['U'], {})
    U_opt_val = U_opt_val[:T_val]

    X_opt_val = None
    if evaluate:
      X_opt_val = sess.run(interfaces['X'], {
        interfaces['T']: T_val,
        interfaces['x_0']: x_0_val
      })

    J_opt_val = sess.run(interfaces['J'], {
      interfaces['T']: T_val,
      interfaces['x_0']: x_0_val
    })

    # log
    if verbose:
      print('x_0 = %s' % (str(np.array(x_0_val).reshape(self.options['x_dim']))))

      for t in range(T_val):
        print('U_opt[%d] = %s' % (t, str(np.array(U_opt_val[t]).reshape(self.options['u_dim']))))

    if evaluate and verbose:
      for t in range(T_val + 1):
        print('X_opt[%d] = %s' % (t, str(np.array(X_opt_val[t]).reshape(self.options['x_dim']))))

    if verbose:
      print('J_opt = %.4f' % (J_opt_val))

    # construct result
    res = {
      'x_0': x_0_val,
      'U_init': U_init_val,
      'X_init': X_init_val,
      'J_init': J_init_val,
      'U_opt': U_opt_val,
      'X_opt': X_opt_val,
      'J_opt': J_opt_val
    }

    return res


def _test():
  x_dim = 4
  u_dim = 2
  dt = 1.0
  T = 10

  def f(x, u, dt):
    A = [[1, 0, dt,  0],
         [0, 1,  0, dt],
         [0, 0,  1,  0],
         [0, 0,  0,  1]]
    B = [[ 0,  0],
         [ 0,  0],
         [dt,  0],
         [ 0, dt]]

    A = tf.constant(A, tf.float32)
    B = tf.constant(B, tf.float32)

    x_next = tf.reshape(tf.matmul(A, tf.reshape(x, [x_dim, 1])) + tf.matmul(B, tf.reshape(u, [u_dim, 1])), [x_dim])

    return x_next

  def c_t(x, u, dt):
    u_lambda = 1e-6
    c = u_lambda * tf.reduce_sum(u * u) * dt
    return c

  def c_f_loc_speed(x_f):
    x_g = tf.constant(np.array([5., 5., 0., 0.]), tf.float32)
    c = tf.reduce_sum((x_g - x_f) * (x_g - x_f))
    return c

  def c_f_loc(x_f):
    loc_diff = tf.convert_to_tensor([x_f[0] - 5.0, x_f[1] - 5.0])
    c = tf.reduce_sum(loc_diff * loc_diff)
    return c

  # initialize trajectory optimizer
  options = {
    'x_dim': x_dim,
    'u_dim': u_dim,
    'dt':    dt,
    'f':     f,
    'c_t':   c_t,
    'c_f':   c_f_loc_speed,
    'T_max': T + 1
  }

  trajopt = TrajectoryOptimizer(options)

  # optimize trajectory
  x_0 = np.array([0, 0, 0, 0])
  U_init = [np.array([0, 0]) for t in range(T)]

  trajopt.optimize(x_0, U_init, epsilon=1e-4, max_epochs=1000, evaluate=True, verbose=True)


if __name__ == '__main__':
  _test()
