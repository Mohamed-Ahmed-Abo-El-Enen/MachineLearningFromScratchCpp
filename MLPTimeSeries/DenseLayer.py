import numpy as np


class DenseLayer:

    def __init__(self, input_dimension, units, activation='', random_multiplier=0.01):
        self.weights, self.bias = self.initialize(input_dimension, units, random_multiplier)

        if activation == 'sigmoid':
            self.activation = activation
            self.activation_forward = self.sigmoid
            self.activation_backward = self.sigmoid_grad

        elif activation == 'relu':
            self.activation = activation
            self.activation_forward = self.relu
            self.activation_backward = self.relu_grad

        elif activation == 'tanh':
            self.activation = activation
            self.activation_forward = self.tanh
            self.activation_backward = self.tanh_grad

        elif activation == 'linear':
            self.activation = activation
            self.activation_forward = self.linear
            self.activation_backward = self.linear

        elif activation != '':
            raise Exception("Activation Does Not Exist")

        else:
            self.activation = 'none'
            self.activation_forward = self.linear
            self.activation_backward = self.linear

    def initialize(self, nx, nh, random_multiplier):
        weights = random_multiplier * np.random.randn(nh, nx)
        bias = np.zeros([nh, 1])
        return weights, bias

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def sigmoid_grad(self, dA):
        s = 1 / (1 + np.exp(-self.prevZ))
        dZ = dA * s * (1 - s)
        return dZ

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def relu_grad(self, dA):
        s = np.maximum(0, self.prevZ)
        dZ = (s > 0) * 1 * dA
        return dZ

    def tanh(self, Z):
        A = np.tanh(Z)
        return A

    def tanh_grad(self, dA):
        s = np.tanh(self.prevZ)
        dZ = (1 - s ** 2) * dA
        return dZ

    def linear(self, Z):
        return Z

    def forward(self, A):
        Z = np.dot(self.weights, A) + self.bias
        self.prevZ = Z
        self.prevA = A
        A = self.activation_forward(Z)
        return A

    def backward(self, dA):
        dZ = self.activation_backward(dA)
        m = self.prevA.shape[1]
        self.dW = 1 / m * np.dot(dZ, self.prevA.T)
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        prevdA = np.dot(self.weights.T, dZ)
        return prevdA

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dW
        self.bias = self.bias - learning_rate * self.db

    def output_dimension(self):
        return len(self.bias)

    def __repr__(self):
        act = 'none' if self.activation == '' else self.activation
        return f'Dense layer (nx={self.weights.shape[1]}, nh={self.weights.shape[0]}, activation={act})'
