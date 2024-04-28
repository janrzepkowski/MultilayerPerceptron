import numpy as np
from activation_function import sigmoid, sigmoid_derivative


class Neuron:
    def __init__(self, num_inputs, bias):
        self.inputs = None
        self.weights = np.random.uniform(-1, 1, num_inputs)
        if bias:
            self.bias = np.random.uniform(-1, 1)
        else:
            self.bias = 0
        self.output = 0
        self.weighted_sum = 0

    def forward(self, inputs):
        self.weighted_sum = np.dot(self.weights, inputs) + self.bias
        self.output = sigmoid(self.weighted_sum)
        return self.output

    def update_weights(self, weights):
        self.weights = weights

    def update_bias(self, bias):
        self.bias = bias
