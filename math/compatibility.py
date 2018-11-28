import math
import numpy as np
import tensorflow as tf


def f_math(x):
    return math.sin(x)


def f_np(x):
    return np.sin(x)


def f_tf(x):
    return tf.math.sin(x)


def main():
    x = tf.Variable(1.0)

    y_math = None
    y_np   = None
    y_tf   = None

    try:
        y_math = f_math(x)
        print('math is supported.')
        print('y_math = %s' % (str(y_math)))
    except Exception as e:
        print('math is not supported.')

    try:
        y_np = f_np(x)
        print('np is supported.')
        print('y_np = %s' % (str(y_np)))
    except Exception as e:
        print('np is not supported.')

    try:
        y_tf = f_tf(x)
        print('tf is supported.')
        print('y_tf = %s' % (str(y_tf)))
    except Exception as e:
        print('tf is not supported.')


if __name__ == '__main__':
    main()
