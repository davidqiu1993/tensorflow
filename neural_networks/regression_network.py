import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import pdb, IPython

#tf.enable_eager_execution()


def generate_datasets():
    """
    Generate datasets.

    @return X_train The training dataset input data.
    @return Y_train The training dataset output data.
    @return X_valid The validation dataset input data.
    @return Y_valid The validation dataset output data.
    @return X_test The test dataset input data.
    @return Y_test The test dataset output data.
    """

    CONFIG_RANDOM_SEED = 42

    def f(X):
        return (X[:,0] + X[:,1])**2 - X[:,2] + np.random.random(len(X)) * 1.0

    # generate data
    X = np.random.random((1000, 3)) * 100
    Y = f(X)

    # split datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=CONFIG_RANDOM_SEED)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=CONFIG_RANDOM_SEED)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


class RegressionNetwork(object):
    """
    Regression network.
    """

    def __init__(self):
        """
        Initialize a regression network.
        """

        super(RegressionNetwork, self).__init__()

        # define dimensions
        self.dim = {
            'input': 3,
            'hidden_1': 32,
            'output': 1
        }

        # define input and output placeholders
        self.X = tf.placeholder('float', shape=[None, self.dim['input']])
        self.Y = tf.placeholder('float', shape=[None, self.dim['output']])

        # define weights
        self.weights = {
            'hidden_1': tf.Variable(tf.random_normal((self.dim['input'], self.dim['hidden_1']))),
            'output': tf.Variable(tf.random_normal((self.dim['hidden_1'], self.dim['output'])))
        }

        # define biases
        self.biases = {
            'hidden_1': tf.Variable(tf.random_normal((self.dim['hidden_1']))),
            'output': tf.Variable(tf.random_normal((self.dim['output'])))
        }


    def construct_network(self):
        """
        Construct the network.
        """

        self.layers = {}

        # layer: input -> hidden_1
        self.layers['hidden_1'] = tf.atanh(tf.add(tf.matmul(self.X, self.weights['hidden_1']), self.biases['hidden_1']))

        # layer: hidden_1 -> output
        self.layers['output'] = tf.add(tf.matmul(self.layers['hidden_1'], self.weights['output']), self.biases['output'])


    def train(self, X_train, Y_train, X_valid, Y_valid):
        """
        Train the network.

        @param X_train The training dataset input data.
        @param Y_train The training dataset output data.
        @param X_valid The validation dataset input data.
        @param Y_valid The validation dataset output data.
        """

        pass


    def test(self, X_test, Y_test):
        """
        Test the network.

        @param X_test The test dataset input data.
        @param Y_test The test dataset output data.
        """

        pass


def main():
    # generate datasets
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = generate_datasets()

    IPython.embed()


if __name__ == '__main__':
    main()
