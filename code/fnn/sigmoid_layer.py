import numpy as np
from fnn.layer import Layer

class SigmoidLayer(Layer):
    def __init__(self, do_dropout=False, dropout_rate=0.5):
        self.__output = None
        self.__do_dropout = do_dropout
        self.__dropout_rate = dropout_rate

    def get_output(self, input_matrix):
        self.__output = 1. / (1. + np.exp(-input_matrix))

        if self.__do_dropout:
            self.__output *= np.random.binomial(
                np.ones_like(input_matrix, dtype=np.int8),
                1 - self.__dropout_rate) / (1 - self.__dropout_rate)
        return np.copy(self.__output)

    def get_input_gradient(self, output_gradients):
        sigmoid_derivative = np.multiply(self.__output,
                                         (1. - self.__output))
        return np.multiply(sigmoid_derivative, output_gradients)

