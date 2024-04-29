import numpy as np
from MLP.layer import Layer
from MLP.activation_function import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, bias):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_amount = len(hidden_sizes)
        # self.input_layer = Layer(input_size, hidden_sizes[0], bias)
        self.hidden_layers = [Layer(self.input_size if i == 0 else self.hidden_sizes[i - 1], size, bias) for i, size in
                              enumerate(self.hidden_sizes)]
        self.output_layer = Layer(self.hidden_sizes[-1], output_size, bias)
        self.layers = self.hidden_layers + [self.output_layer]

        activations = [np.zeros(input_size)]
        activations.extend([np.zeros(layer.num_neurons) for layer in self.layers])
        self.activations = activations

    def feed_forward(self, input_vector):
        self.activations[0] = np.array(input_vector)
        for i, layer in enumerate(self.layers):
            self.activations[i + 1] = layer.forward(self.activations[i])
        return self.activations[-1]
