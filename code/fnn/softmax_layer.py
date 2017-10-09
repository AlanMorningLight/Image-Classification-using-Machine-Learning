import numpy as np
from fnn.layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self):
        self.__output = None

    def get_output(self, input_matrix):
        input_expo = np.exp(input_matrix)
        self.__output = input_expo / np.sum(input_expo, axis=1, keepdims=True)
        return np.copy(self.__output)

    def get_input_gradient(self, targets):
        return (self.__output - targets) / targets.shape[0]

    def get_cost(self, targets):
        return -np.multiply(np.log(self.__output),
                            targets).sum() / targets.shape[0]

