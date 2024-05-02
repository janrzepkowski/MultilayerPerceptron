from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from network import Network

# Load the iris dataset
iris = load_iris()

# Convert labels to one-hot vectors
lb = LabelBinarizer()
y = lb.fit_transform(iris.target)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.33, random_state=42)

# Convert datasets to lists of tuples
training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_train, y_train)]
test_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_test, y_test)]

# Create a Network object with the appropriate architecture
net = Network([4, 10, 3])  # 4 input neurons, 10 hidden neurons, 3 output neurons

# Train the network
net.SGD(training_data, epochs=100, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Evaluate the network
correct_results = net.evaluate(test_data)
total_samples = len(test_data)
accuracy = correct_results / total_samples

print(f"Accuracy: {accuracy * 100}%")