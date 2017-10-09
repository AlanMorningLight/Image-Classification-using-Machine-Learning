import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from fnn.linear_layer import LinearLayer
from fnn.sigmoid_layer import SigmoidLayer
from fnn.softmax_layer import SoftmaxLayer

class FNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_of_hidden_layers=2, hidden_layer_sizes=(50, 50),
                 batch_size=25, learning_rate=0.3, momentum_coeff=0.9,
                 do_dropout=False, dropout_rate=0.5, max_iter=40, tol=0.0001):
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum_coeff = momentum_coeff
        self.do_dropout = do_dropout
        self.dropout_rate = dropout_rate
        self.max_iter = max_iter
        self.tol = tol  # tolerance

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)

        try:
            if np.amax(self.classes_) != (self.classes_.shape[0] - 1):
                print('Input data has improper labels, reset all to 0.')
                self.y_.fill(0)
        except TypeError:
            print('Input has non-integer data, reset all targets to 0.')
            self.y_.fill(0)

        targets = np.zeros((self.y_.shape[0], self.classes_.shape[0]))
        # one-hot encoding for targets
        try:
            targets[np.arange(self.y_.shape[0]), self.y_.astype(int)] = 1
        except ValueError:
            # targets are all set to zeros in this case
            print('Invalid value type in targets. Reset all targets to 0.')
            targets[:, 0] = 1

        self.__build_neuralnet()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_, targets, test_size=0.2, random_state=42)

        train_cost = []
        val_score = []

        # start stochastic gradient descent
        for i in range(self.max_iter):
            io_batches = self.__make_batches(X_train, y_train)
            self.__sgd(io_batches)

            # store training set cost
            self.__feedforward(X_train)  # train on entire training set
            train_cost.append(self.__fnn[-1]['output'].get_cost(y_train))

            # store validation set score
            val_score.append(self.score(
                X_test, self.classes_[np.argmax(y_test, axis=1)]))

            # check whether to update learning rate
            if i > 1:
                # test for training set cost improvement
                if train_cost[i - 1] - train_cost[i] < self.tol:
                    if train_cost[i - 2] - train_cost[i - 1] < self.tol:
                        # cost over training set has failed to improve for 2
                        # consecutive epochs at this point
                        # test for improvement on validation set score
                        if val_score[i] - val_score[i - 1] < self.tol:
                            if val_score[i - 1] - val_score[i - 2] < self.tol:
                                # score on validation set has also failed to
                                # improve for 2 consecutive epochs
                                self.learning_rate /= 2

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        predictions = self.predict_proba(X)

        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X):
        return self.__feedforward(np.asarray(X))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def __build_neuralnet(self):
        # data structure for the neural network
        self.__fnn = []

        # build first hidden layer
        if self.num_of_hidden_layers == 1:
            linear = LinearLayer(self.X_.shape[1], self.classes_.shape[0],
                                 self.momentum_coeff, self.learning_rate)
        else:
            linear = LinearLayer(self.X_.shape[1], self.hidden_layer_sizes[0],
                                 self.momentum_coeff, self.learning_rate)

        activation = SigmoidLayer(do_dropout=self.do_dropout,
                                  dropout_rate=self.dropout_rate)

        self.__fnn.append({'linear': linear, 'activation': activation})

        if self.num_of_hidden_layers >= 2:
            # build more hidden layers
            for i in range(self.num_of_hidden_layers - 1):
                if i == self.num_of_hidden_layers - 2: break

                linear = LinearLayer(self.hidden_layer_sizes[i],
                                     self.hidden_layer_sizes[i + 1],
                                     self.momentum_coeff, self.learning_rate)

                activation = SigmoidLayer(do_dropout=self.do_dropout,
                                          dropout_rate=self.dropout_rate)
                self.__fnn.append({'linear': linear, 'activation': activation})

            # build the last hidden layer
            linear = LinearLayer(self.hidden_layer_sizes[-1],
                                 self.classes_.shape[0],
                                 self.momentum_coeff, self.learning_rate)

            activation = SigmoidLayer(do_dropout=self.do_dropout,
                                      dropout_rate=self.dropout_rate)
            self.__fnn.append({'linear': linear, 'activation': activation})

        # build output layer
        output = SoftmaxLayer()
        self.__fnn.append({'output': output})

    def __make_batches(self, X_train, y_train):
        num_of_batches = X_train.shape[0] // self.batch_size

        if num_of_batches <= 0:
            num_of_batches = 1

        X_rand, targets_rand = shuffle(X_train, y_train, random_state=42)

        return zip(np.array_split(X_rand, num_of_batches),
                   np.array_split(targets_rand, num_of_batches))

    def __feedforward(self, X_in):
        X_in_copy = np.copy(X_in)

        for i in range(self.num_of_hidden_layers):
            linear_out = self.__fnn[i]['linear'].get_output(X_in_copy)
            activation_out = self.__fnn[i]['activation'].get_output(linear_out)
            X_in_copy = activation_out

        return self.__fnn[-1]['output'].get_output(X_in_copy)

    def __backprop(self, targets):
        # get gradients from the output layer
        gradients = self.__fnn[-1]['output'].get_input_gradient(targets)

        # back propogate through the hidden layers
        for i in reversed(range(self.num_of_hidden_layers)):
            gradients = (
                self.__fnn[i]['activation'].get_input_gradient(gradients))
            gradients_copy = np.copy(gradients)
            gradients = self.__fnn[i]['linear'].get_input_gradient(gradients)
            self.__fnn[i]['linear'].update_parameters(gradients_copy)

    def __sgd(self, io_batches):
        for X_in, target in io_batches:
            self.__feedforward(X_in)
            # backprop and update parameters
            self.__backprop(target)

    @property
    def max_iter(self):
        return self.__max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self.__max_iter = max_iter

    @property
    def num_of_hidden_layers(self):
        return self.__num_of_hidden_layers

    @num_of_hidden_layers.setter
    def num_of_hidden_layers(self, num_of_hidden_layers):
        self.__num_of_hidden_layers = num_of_hidden_layers

    @property
    def hidden_layer_sizes(self):
        return self.__hidden_layer_sizes

    @hidden_layer_sizes.setter
    def hidden_layer_sizes(self, hidden_layer_sizes):
        if (type(hidden_layer_sizes) != tuple) or (
            len(hidden_layer_sizes) != self.num_of_hidden_layers):
            raise ValueError(
                "hidden_layer_sizes should be of type tuple, and its length "
                "must match the number of hidden layers in the neural "
                "network.")
        else:
            self.__hidden_layer_sizes = hidden_layer_sizes

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def momentum_coeff(self):
        return self.__momentum_coeff

    @momentum_coeff.setter
    def momentum_coeff(self, momentum_coeff):
        self.__momentum_coeff = momentum_coeff

    @property
    def do_dropout(self):
        return self.__do_dropout

    @do_dropout.setter
    def do_dropout(self, do_dropout):
        self.__do_dropout = do_dropout

    @property
    def dropout_rate(self):
        return self.__dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate):
        self.__dropout_rate = dropout_rate

    @property
    def tol(self):
        return self.__tol

    @tol.setter
    def tol(self, tol):
        self.__tol = tol

