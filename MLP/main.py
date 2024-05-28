import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import random

import network


def draw(precision, recall, f_measure):
    fig, ax = plt.subplots()
    names = ["precision", "recall", "f_measure"]
    counts = [precision, recall, f_measure]
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
    ax.bar(names, counts, label=bar_labels, color=bar_colors)
    ax.set_ylabel('Percentage')
    ax.set_title('Results')
    plt.show()


def prepare_data(array):
    array = np.array(array)
    target_values = []
    for genre in array:
        if genre[-1] == 0:
            target_values.append([1, 0, 0])
        elif genre[-1] == 1:
            target_values.append([0, 1, 0])
        elif genre[-1] == 2:
            target_values.append([0, 0, 1])
    x_array = array[:, :-1]
    target_values = np.array(target_values)
    combined_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_array, target_values)]
    return combined_data


def simulate(layers, train, valid, test):
    bias = False
    epochs = 1000
    error = 1.0
    sim_net = network.Network(layers, useBias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10,0.9, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, useBias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10,0.6, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, useBias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10,0.2, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, useBias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10,0.9, 0.6, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, useBias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10,0.2, 0.9, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)


def confusion(network, test):
    predicted_labels = []
    true_labels = []
    logs = "Wagi:\n" + str(network.weights)
    logs += "\n\nWejscia obciazajece:\n" + str(network.biases)
    general_error = 0.0
    for index in range(len(test)):
        test_row = test[index]
        output = network.feedforward(test_row[0])
        expected = test_row[1]
        true_labels.append(np.argmax(expected))
        predicted_labels.append(np.argmax(output))
        error = network.calculate_error(expected, output)
        general_error += error
        logs += "Wzorzec wejsciowy:\n" + str(test_row[0]) + "\n"
        logs += "Wzorzec wyjsciowy:\n" + str(expected) + "\n"
        logs += "Uzyskane wyjscia:\n" + str(output) + "\n"
        logs += "Wynik klasyfikacji: " + str(np.argmax(output) + 1) + "\n"
        logs += "Blad wyjsciowy: " + str(error) + "\n\n"

    logs += "Calkowity blad wyjsciowy: " + str(general_error) + "\n"

    with open("stats.txt", 'a') as file:
        file.write(logs)

    matrix = confusion_matrix(true_labels, predicted_labels)
    print("\nMacierz pomyłek:")
    print(matrix)
    recall = []
    i = 0
    for x in matrix:
        tmp = 0
        for a in x:
            tmp += a
        recall.append(x[i] / tmp)
        i += 1
    p = [np.array([matrix[x][y] for x in range(len(matrix))]) for y in range(len(matrix))]
    p = np.array([np.sum(x) for x in p])
    precision = []
    for x, y in zip(np.diag(matrix), p):
        if y == 0:
            precision.append(0)
        else:
            precision.append(x / y)
    f_measure = []
    for x, y in zip(precision, recall):
        if y == 0 or x == 0:
            f_measure.append(0)
        else:
            f_measure.append(2 * x * y / (x + y))

    print("\nPrecyzja (Precision):", precision)
    print("Czułość (Recall):", recall)
    print("Miara F (F-measure):", f_measure)


while True:
    open("stats.txt", 'w').close()
    print("1. Klasyfikacja irysow")
    print("2. Autoenkoder")
    print("3. Wyjscie")
    choice = int(input("Wybierz opcje: "))
    combined_train_data = np.array([])
    combined_test_data = np.array([])
    validation_data = np.array([])
    if choice == 1:
        train_data = pd.read_csv('data_train.csv', header=None)
        test_data = pd.read_csv('data_test.csv', header=None)
        combined_train_data = prepare_data(train_data)
        combined_test_data = prepare_data(test_data)
        validation_data = random.choices(combined_train_data, k=int(len(combined_train_data) / 3))
        random.shuffle(validation_data)
    elif choice == 2:
        x_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        y_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # target_values = y_array
        combined_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_array, y_array)]
        combined_test_data = combined_data
        combined_train_data = combined_data
        validation_data = combined_data
    elif choice == 3:
        exit()

    isNetworkCreated = False
    while True:
        if isNetworkCreated:
            print("1. Tryb nauki sieci")
            print("2. Tryb testowania sieci")
            print("3. Zapisz siec")
            print("4. Symulacja")
            print("5. Wyjscie")
            option = int(input("Wybierz opcje: "))
        else:
            print("Co dalej?")
            print("1. Tworzenie nowej sieci")
            print("2. Wczytanie istniejacej sieci")
            print("4. Symulacja")
            print("5. Wyjscie")
            option = int(input("Wybierz opcje: "))

        if option == 1 and not isNetworkCreated:
            print("Tworzenie nowej sieci")
            num_layers = int(input("Podaj liczbe warstw ukrytych: "))
            num_neurons = [len(combined_train_data[0][0])]
            for i in range(num_layers):
                num_neurons.append(int(input("Podaj liczbe neuronow w " + str(i + 1) + " warstwie ukrytej: ")))

            num_neurons.append(len(combined_train_data[0][1]))
            bias = int(input("Czy chcesz dodac bias?: "))

            net = network.Network(num_neurons, useBias=(False if bias == 0 else True))
            print("Siec stworzona, co dalej?")
            isNetworkCreated = True

        if option == 1 and isNetworkCreated:
            print("Podaj warunek stopu")
            print("1. Ilosc epok")
            print("2. Dokladnosc")
            epoch_number = 999
            stop_precision = 1.1
            stopCondition = int(input("Wybierz opcje: "))
            if stopCondition == 1:
                epoch_number = int(input("Podaj liczbe epok: "))
            elif stopCondition == 2:
                stop_precision = float(input("Podaj dokladnosc: "))
            learning_rate = 1
            momentum = 1
            while not (0 <= learning_rate < 1 and 0 <= momentum < 1):
                learning_rate = float(input("Podaj współczynnik nauki: "))
                momentum = float(input("Podaj współczynnik momentum: "))
            shuffle = int(input("Czy przetasowac dane? "))
            errorEpoch = int(input("Co ile epok zapisywac blad? "))
            net.train(combined_train_data, epochs=epoch_number, precision=stop_precision, mini_batch_size=10,
                      learning_rate=learning_rate, momentum=momentum, shuffle=shuffle, error_epoch=errorEpoch,
                      validation_data=validation_data, debug=True)
            print("Nauka zakonczona")

        if option == 2 and isNetworkCreated:
            net.plot_training_error()
            confusion(net, combined_test_data)
            # with open("trainStats.csv", "w") as file:
            #     pass
            # correct = [0 for _ in range(len(combined_train_data[0][1]))]
            # predicted_labels = []
            # true_labels = []
            # for index in range(len(combined_test_data)):
            #     test = combined_test_data[index]
            #     output = net.feedforward(test[0])
            #     if choice == 1:
            #         expected = test[1]
            #     else:
            #         expected = test[1]
            #     true_label = np.argmax(expected)
            #     predicted_label = np.argmax(output)
            #     true_labels.append(true_label)
            #     predicted_labels.append(predicted_label)
            #     if predicted_label == true_label:
            #         correct[true_label] += 1
            #
            #     error = net.calculate_error(expected, output)
            # #     neuronWeights = []
            # #     neuronOutputs = []
            # #
            # #     with open("trainStats.txt", "a") as file:
            # #
            # #         file.write(f"Wzorzec numer: {index}, {test[:4]}\n")
            # #         file.write(f"Popelniony blad dla wzorca: {error}\n")
            # #         file.write(f"Pozadany wzorzec odpowiedzi: {expected}\n")
            # #         for i in range(len(output)):
            # #             file.write(f"Blad popelniony na {i} wyjsciu: {output[i] - expected[i]}\n")
            # #         for i in range(len(output)):
            # #             file.write(f"Wartosc na {i} wyjsciu: {output[i]}\n")
            # #         file.write("\n\n")
            # #
            # # file.close()
            #
            # if choice == 1:
            #     print("Klasyfikacja irysow")
            #     number = len(combined_test_data) / 3
            #     accuracy = sum(correct) / number * 100 / 3
            #     print("Iris-setosa: " + str(correct[0] / number * 100) + "%")
            #     print("Iris-versicolor: " + str(correct[1] / number * 100) + "%")
            #     print("Iris-virginica: " + str(correct[2] / number * 100) + "%")
            #     print("Total: " + str(accuracy) + "%")
            # else:
            #     print("Autoenkoder")
            #     print("Popelniony blad: ", error)
            #     print("Odpowiedzi: ", output)
            #     print("Poprawne odpowiedzi: ", expected)
            # matrix = confusion_matrix(true_labels, predicted_labels)
            # print("\nMacierz pomyłek:")
            # print(matrix)
            # precision = []
            # i = 0
            # for x in matrix:
            #     tmp = 0
            #     for a in x:
            #         tmp += a
            #     precision.append(x[i]/tmp)
            #     i += 1
            # recall = [np.array([matrix[x][y] for x in range(len(matrix))]) for y in range(len(matrix))]
            # # recall = np.diag(matrix) / (len(test_data)/len(test_data[0][1]))
            # recall = np.array([np.sum(x) for x in recall])
            # recall = np.diag(matrix) / recall
            # precision = np.array(precision)
            # f_measure = 2 * (precision * recall) / (precision + recall)
            #
            # print("\nPrecyzja (Precision):", precision)
            # print("Czułość (Recall):", recall)
            # print("Miara F (F-measure):", f_measure)
            # # draw(precision, recall, f_measure)

        if option == 3 and isNetworkCreated:
            print("Zapisanie sieci")
            filename = "network.pkl"
            net.save(filename)
            print("Siec zapisana do pliku network.pkl")

        if option == 2 and not isNetworkCreated:
            print("Wczytanie sieci")
            filename = "network.pkl"
            net = network.Network.load(filename)
            print("Siec wczytana, co dalej?")
            isNetworkCreated = True
        if option == 4:
            num_layers = int(input("Podaj liczbe warstw ukrytych: "))
            num_neurons = [len(combined_train_data[0][0])]
            for i in range(num_layers):
                num_neurons.append(int(input("Podaj liczbe neuronow w " + str(i + 1) + " warstwie ukrytej: ")))

            num_neurons.append(len(combined_train_data[0][1]))
            simulate(num_neurons, combined_train_data, validation_data, combined_test_data)

        if option == 5:
            break
