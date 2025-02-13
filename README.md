# Multi-Layer Perceptron (MLP)
The repository contains a Python console application that implements and evaluates a Multi-Layer Perceptron (MLP) network. The implementation is flexible and scalable, allowing users to define the architecture of the network by specifying the number of layers and neurons for each layer. The project supports both supervised learning for classification tasks and unsupervised learning tasks such as autoencoding.

## Project Description
The Multi-Layer Perceptron (MLP) is a type of artificial neural network designed to simulate the learning and decision-making processes of the human brain. It consists of multiple layers of interconnected neurons, where each neuron applies a non-linear activation function to compute its output. 

This implementation allows users to design and test neural networks of varying architectures for tasks such as classification and autoencoding. The MLP supports supervised learning where the network is trained to map input features to target outputs, as well as unsupervised tasks like reconstructing inputs.

The project uses a backpropagation learning algorithm to update weights and minimize the global error through gradient descent. With comprehensive logging and configuration options, it provides a powerful tool for machine learning experimentation and research.
## Built With
The project was built using the following tools and libraries:
- Python
- NumPy
- Matplotlib
- Pandas
- Scikit-learn 

## Getting Started
1. Clone the repository to your local machine:
```sh
git clone https://github.com/janrzepkowski/MultilayerPerceptron.git
```
2. Open IDE and select "Open an existing project." Navigate to the cloned repository and select the "MLP" directory.
3. Configure Python interpreter.
4. To install the required packages, execute the following command in the terminal:
```sh
pip install -r requirements.txt
```
5. To run application, execute the following command in the terminal:
```sh
python main.py
```

## Research Section
The application supports two main use cases for research and experimentation:
### 1. Iris Classification
- **Description**:
    - The MLP is trained to classify Iris flower species using the provided dataset.
    - The dataset is split into 70% for training and 30% for testing.

- **Performance Evaluation**: A detailed evaluation is conducted including:
    - Confusion matrix (mapping true vs predicted classifications).
    - Metrics: **Precision**, **Recall**, and **F-measure** for each class.
    - Total accuracy across all test samples is calculated.

### 2. Autoencoding Task
- **Description**:
    - A 3-layer symmetrical MLP network (4-2-4 structure) is trained to learn and reproduce 4 distinct input-output patterns.

- **Key Experiments**:
    - Evaluate the influence of using bias neurons in the learning process.
    - Train autoencoder using fixed patterns and a learning rate of 0.6 without momentum.
    - Analyze errors at the outputs and intermediate hidden layers across multiple learning configurations.

### Additional Information
Data on the various aspects considered in the comparisons are illustrated in charts and provided in the report (in Polish). For detailed experimental descriptions and results, refer to the `sise_2_sprawozdanie` document available in the repository.