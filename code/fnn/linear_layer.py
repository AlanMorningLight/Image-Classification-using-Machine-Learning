import numpy as np
from fnn.layer import Layer

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, momentum_coeff,
                 learning_rate):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(1, output_dim)
        self.momentum_coeff = momentum_coeff
        self.learning_rate = learning_rate

        self.__weights_momentum = np.zeros_like(self.weights)
        self.__bias_momentum = np.zeros_like(self.bias)

    def get_output(self, input_matrix):
        self.__input = input_matrix
        return input_matrix.dot(self.weights) + self.bias

    def get_input_gradient(self, output_gradients):
        return output_gradients.dot(self.weights.T)

    def update_parameters(self, output_gradients):
        weights_jacobian = self.__input.T.dot(output_gradients)
        bias_jacobian = np.sum(output_gradients, axis=0)

        # update with momentum
        weights_delta = (self.learning_rate * weights_jacobian +
                         self.__weights_momentum)
        self.weights -= weights_delta
        self.__weights_momentum = weights_delta * self.momentum_coeff

        bias_delta = self.learning_rate * bias_jacobian + self.__bias_momentum
        self.bias -= bias_delta
        self.__bias_momentum = bias_delta * self.momentum_coeff

    @property
    def weights(self):
        return self.__weights

    @property
    def bias(self):
        return self.__bias

    @property
    def momentum_coeff(self):
        return self.__momentum_coeff

    @property
    def learning_rate(self):
        return self.__learning_rate

    @weights.setter
    def weights(self, weights_matrix):
        self.__weights = weights_matrix

    @bias.setter
    def bias(self, a_bias):
        self.__bias = a_bias

    @momentum_coeff.setter
    def momentum_coeff(self, momentum_coeff):
        self.__momentum_coeff = momentum_coeff

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

