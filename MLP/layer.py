import numpy as np
from MLP.neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons, bias):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.neurons = np.array([Neuron(num_inputs, bias) for _ in range(num_neurons)])
        self.outputs = None

    def forward(self, inputs):
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def weighted_sum(self):
        return np.array([neuron.weighted_sum for neuron in self.neurons])

    def update_weights(self, weights):
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(weights[i])

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def update_bias(self, bias):
        for neuron in self.neurons:
            neuron.update_bias(bias)

    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])
