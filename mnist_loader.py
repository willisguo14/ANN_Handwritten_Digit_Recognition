import pickle
import gzip

import numpy as np

def load_data():
    """Return MNIST data 

    training_data -> tuple(x, y): 
        - x: training images (28 * 28 numpy ndarray)
        - y: corresponding digit
    validation_data and test_data are similar 
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Reformat MNIST data for neural network

    training_data -> tuple(x, y):
        -x: 784-dimensional numpy.ndarray
        -y: 10-dimensional numpy.ndarray representing the corresponding digit
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Convert a digit into corresponding desired output from neural network

    Returns a 10-d unit vector with 1 in the jth position and 0 elsewhere."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e