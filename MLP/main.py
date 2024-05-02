import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from network import Network

# Load the iris dataset from CSV files
train_data = pd.read_csv('data_train.csv', header=None)
test_data = pd.read_csv('data_test.csv', header=None)

# Convert labels to one-hot vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(train_data.iloc[:, -1])
y_test = lb.transform(test_data.iloc[:, -1])

# Convert datasets to lists of tuples
X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values

training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_train, y_train)]
test_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_test, y_test)]

# Create a Network object with the appropriate architecture
net = Network([4, 10, 3], useBias=False)  # 4 input neurons, 10 hidden neurons, 3 output neurons

# Train the network
net.SGD(training_data, epochs=120, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Evaluate the network
correct_results = net.evaluate(test_data)
total_samples = len(test_data)
accuracy = correct_results / total_samples

print(f"Accuracy: {accuracy * 100}%")


net2 = Network([4, 10, 3], useBias=True)  # 4 input neurons, 10 hidden neurons, 3 output neurons

# Train the network
net2.SGD(training_data, epochs=120, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Evaluate the network
correct_results2 = net2.evaluate(test_data)
total_samples2 = len(test_data)
accuracy2 = correct_results2 / total_samples2

print(f"Accuracy: {accuracy2 * 100}%")