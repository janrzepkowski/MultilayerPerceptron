import numpy as np
from MLP.layer import Layer
from MLP.activation_function import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, bias):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_amount = len(hidden_sizes)
        self.hidden_layers = [Layer(self.input_size if i == 0 else self.hidden_sizes[i - 1], size, bias) for i, size in enumerate(self.hidden_sizes)]
        self.output_layer = Layer(self.hidden_sizes[-1], output_size, bias)

