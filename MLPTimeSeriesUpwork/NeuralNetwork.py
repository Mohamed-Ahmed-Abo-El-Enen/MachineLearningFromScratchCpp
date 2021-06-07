import numpy as np
import DenseLayer as Dl


class NeuralNetwork:
    def __init__(self, loss='cross-entropy', random_multiplier=0.01):
        self.layers = []
        self.random_multiplier = random_multiplier

        if loss == 'cross-entropy':
            self.loss_function = self.cross_entropy_loss
            self.loss_function_backward = self.cross_entropy_loss_grad

        elif loss == 'mean-square-error':
            self.loss_function = self.mean_square_error
            self.loss_function_backward = self.mean_square_error_grad

        else:
            raise Exception("Loss Function Not Defined")

        self.loss = loss

    def add_layer(self, input_dimension=None, units=1, activation=''):
        if input_dimension is None:
            if len(self.layers) == 0:
                raise Exception("Input Dimension Not Correct")
            input_dimension = self.layers[-1].output_dimension()
        layer = Dl.DenseLayer(input_dimension, units, activation, random_multiplier=self.random_multiplier)
        self.layers.append(layer)

    def cross_entropy_loss(self, y, A, epsilon=1e-15):
        m = y.shape[1]
        loss = -1 * (y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
        cost = 1 / m * np.sum(loss)
        return np.squeeze(cost)

    def cross_entropy_loss_grad(self, y, A):
        dA = -(np.divide(y, A) - np.divide(1 - y, 1 - A))
        return dA

    def mean_square_error(self, y, A):
        loss = np.square(y - A)
        m = y.shape[1]
        cost = 1 / m * np.sum(loss)
        return np.squeeze(cost)

    def mean_square_error_grad(self, y, A):
        dA = -2 * (y - A)
        return dA

    def cost(self, y, A):
        return self.loss_function(y, A)

    def forward(self, X):
        X = X.T
        x = np.copy(X)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, A, y):
        y = y.T
        dA = self.loss_function_backward(y, A)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate=0.01):
        for layer in self.layers:
            layer.update(learning_rate)

    def number_of_parameters(self):
        n = 0
        for layer in self.layers:
            n += np.size(layer.weights) + len(layer.bias)
        print(f'There are {n} trainable parameters in the model.')

    def __repr__(self):
        layer = ['  ' + str(ix + 1) + ' -> ' + str(x) for ix, x in enumerate(self.layers)]
        return '[\n' + '\n'.join(layer) + '\n]'


