import random
import numpy as np

from functions import sigmoid, sigmoid_derivative


class Network(object):
    def __init__(self, sizes, useBias):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.useBias = useBias
        if useBias:
            self.biases = [np.random.uniform(-1, 1, (y, 1)) for y in sizes[1:]]
        else:
            self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.uniform(-1, 1, (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Calculate the output of the network given the input vector `a`."""
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Train the network using stochastic gradient descent."""
        training_data = list(training_data)
        num_training_data = len(training_data)

        if test_data:
            test_data = list(test_data)
            num_test_data = len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, num_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {epoch} : {self.evaluate(test_data)} / {num_test_data}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch."""
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * gw
                        for w, gw in zip(self.weights, gradient_w)]
        if self.useBias:
            self.biases = [b - (learning_rate / len(mini_batch)) * gb
                           for b, gb in zip(self.biases, gradient_b)]

    def backpropagation(self, x, y):
        """Calculate the gradient for the cost function."""
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            gradient_b[-layer] = delta
            gradient_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (gradient_b, gradient_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        """Calculate the derivative of the cost function."""
        return output_activations - y