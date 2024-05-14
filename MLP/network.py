import random
import numpy as np
import pickle
import time

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
        self.velocity = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(self, training_data, epochs, precision, mini_batch_size, learning_rate, momentum, shuffle, error_epoch,
            test_data=None):
        start_time = time.time()
        error_log = ""
        training_data = list(training_data)
        num_training_data = len(training_data)
        prev_precision = 0
        current_precision = 0

        if test_data is not None:
            test_data = list(test_data)
            num_test_data = len(test_data)

        for epoch in range(epochs):
            if shuffle:
                random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, num_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, momentum)
            if epoch % error_epoch == 0:
                if test_data:
                    test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                                    for (x, y) in test_data]
                    num_correct = sum(int(x == y) for (x, y) in test_results)
                    current_precision = num_correct / num_test_data
                    print(f"Epoch {epoch} : {num_correct} / {num_test_data} Precision: {current_precision}")
                    if current_precision >= precision:
                        print("Desired precision reached, stopping training.")
                        with open('trainError.csv', 'w') as file:
                            file.write(error_log)
                        return
                else:
                    print(f"Epoch {epoch} complete")
                end_time = time.time()
                print(f"Time elapsed: {end_time - start_time}")
                error_log += f"{epoch}, {self.epoch_error(training_data)}\n"
        with open('trainError.csv', 'w') as file:
            file.write(error_log)

    def update_mini_batch(self, mini_batch, learning_rate, momentum):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]
        self.velocity = [momentum * v - (learning_rate / len(mini_batch)) * gw
                         for v, gw in zip(self.velocity, gradient_w)]
        self.weights = [w + v for w, v in zip(self.weights, self.velocity)]
        if self.useBias:
            self.biases = [b - (learning_rate / len(mini_batch)) * gb
                           for b, gb in zip(self.biases, gradient_b)]

    def backpropagation(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
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
        return gradient_b, gradient_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def epoch_error(self, train_data):
        error = 0
        for x, y in train_data:
            error += self.calculate_error( self.feedforward(x), y)
        return error / len(train_data)
    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def calculate_error(self, expected, output):
        return np.mean(np.power(expected - output, 2))
