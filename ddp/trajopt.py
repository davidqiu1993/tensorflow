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
      'T':     None  # horizon
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
    T     = self.options['T']

    # initialize placeholders
    x_0 = tf.placeholder(tf.float32, [x_dim])
    U_init = tf.placeholder(tf.float32, [T, u_dim])
    U = tf.Variable([ [ 0.0 for i in range(u_dim) ] for j in range(T) ], tf.float32)

    # initialize control trajectory
    initialize_ctrls = U.assign(U_init)

    # simulate forward
    J = 0
    X = [x_0]
    running_costs = []
    for t in range(T):
      x_next = f(X[t], U[t], dt)
      c_next = c_t(X[t], U[t], dt, t)
      J = J + c_next
      X.append(x_next)
      running_costs.append(c_next)
    final_cost = c_f(X[-1])
    J = J + final_cost

    # define operations
    minimize = self.optimizer.minimize(J)

    # construct interfaces
    interfaces = {
      'x_0': x_0,
      'U_init': U_init,
      'U': U,
      'X': X,
      'C_t': running_costs,
      'c_f': final_cost,
      'J': J,
      'initialize': initialize_ctrls,
      'minimize': minimize
    }

    return interfaces


  def optimize(self, x_0_val, U_init_val, epsilon=1e-4, max_epochs=100, evaluate=False, verbose=False):
    """
    Optimize the action trajectory.
    """

    interfaces = self.interfaces

    # log
    if verbose:
      print('initializing trajectory optimization..')

    # initialize session
    sess = tf.Session()

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # initialize the control trajectory
    sess.run(interfaces['initialize'], {
      interfaces['U_init']: U_init_val
    })

    # evaluate the initial trajectory
    X_init_val = None
    if evaluate:
      X_init_val = []
      for t, x in enumerate(interfaces['X']):
        x_val = sess.run(x, {
          interfaces['x_0']: x_0_val
        })
        X_init_val.append(x_val)

    J_init_val = sess.run(interfaces['J'], {
      interfaces['x_0']: x_0_val
    })

    # log
    if verbose:
      print('x_0 = %s' % (str(np.array(x_0_val).reshape(self.options['x_dim']))))

      for t in range(self.options['T']):
        print('U_init[%d] = %s' % (t, str(np.array(U_init_val[t]).reshape(self.options['u_dim']))))

    if evaluate and verbose:
      for t in range(self.options['T'] + 1):
        print('X_init[%d] = %s' % (t, str(np.array(X_init_val[t]).reshape(self.options['x_dim']))))

    if verbose:
      print('J_init = %.4f' % (J_init_val))

    # log
    if verbose:
      print('optimizing trajectory..')

    # run epochs
    last_J_val = None
    for ep in range(max_epochs):
      # optimize the action trajectory
      sess.run(interfaces['minimize'], {
        interfaces['x_0']: x_0_val
      })

      # evaluate
      J_val = sess.run(interfaces['J'], {
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
    U_opt_val = sess.run(interfaces['U'], {
      interfaces['x_0']: x_0_val
    })

    X_opt_val = None
    if evaluate:
      X_opt_val = []
      for t, x in enumerate(interfaces['X']):
        x_val = sess.run(x, {
          interfaces['x_0']: x_0_val
        })
        X_opt_val.append(x_val)

    J_opt_val = sess.run(interfaces['J'], {
      interfaces['x_0']: x_0_val
    })

    # log
    if verbose:
      print('x_0 = %s' % (str(np.array(x_0_val).reshape(self.options['x_dim']))))

      for t in range(self.options['T']):
        print('U_opt[%d] = %s' % (t, str(np.array(U_opt_val[t]).reshape(self.options['u_dim']))))

    if evaluate and verbose:
      for t in range(self.options['T'] + 1):
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

  def c_t(x, u, dt, t):
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
    'T':     T
  }

  trajopt = TrajectoryOptimizer(options)

  # optimize trajectory
  x_0 = np.array([0, 0, 0, 0])
  U_init = [np.array([0, 0]) for t in range(T)]

  trajopt.optimize(x_0, U_init, epsilon=1e-4, max_epochs=1000, evaluate=True, verbose=True)


if __name__ == '__main__':
  _test()
