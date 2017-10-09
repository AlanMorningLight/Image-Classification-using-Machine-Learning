from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def get_output(self, input_matrix):
        pass

    @abstractmethod
    def get_input_gradient(self, output_gradients=None, targets=None):
        pass

