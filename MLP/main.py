from ucimlrepo import fetch_ucirepo
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

import network

print("1. Klasyfikacja irysow")
print("2. Autoenkoder")
print("3. Wyjscie")
choice = int(input("Wybierz opcje: "))
training_data = np.array([])
test_data = np.array([])

if choice == 1:
    iris = fetch_ucirepo(id=53)

    X = iris.data.features
    y = iris.data.targets

    x_array = X.to_numpy()
    true_labels = y.to_numpy()
    target_values = []
    for genre in true_labels:
        if genre == "Iris-setosa":
            target_values.append([1, 0, 0])
        elif genre == "Iris-versicolor":
            target_values.append([0, 1, 0])
        elif genre == "Iris-virginica":
            target_values.append([0, 0, 1])

    target_values = np.array(target_values)

    combined_data = np.concatenate((x_array, target_values), axis=1)
    training_data = np.concatenate((combined_data[0:15], combined_data[50:65], combined_data[100:115]), axis=0)
    test_data = np.concatenate((combined_data[15:50], combined_data[65:100], combined_data[115:150]), axis=0)

elif choice == 2:
    x_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    target_values = y_array
    combined_data = np.concatenate((x_array, y_array), axis=1)
    training_data = combined_data
    test_data = combined_data
elif choice == 3:
    exit()

isNetworkCreated = False
while True:
    if isNetworkCreated:
        print("1. Tryb nauki sieci")
        print("2. Tryb testowania sieci")
        print("3. Zapisz siec")
        print("4. Wyjscie")
        option = int(input("Wybierz opcje: "))
    else:
        print("Co dalej?")
        print("1. Tworzenie nowej sieci")
        print("2. Wczytanie istniejacej sieci")
        print("4. Wyjscie")
        option = int(input("Wybierz opcje: "))

    if option == 1 and not isNetworkCreated:
        print("Tworzenie nowej sieci")
        num_layers = int(input("Podaj liczbe warstw ukrytych: "))
        num_neurons = []
        for i in range(num_layers):
            num_neurons.append(int(input("Podaj liczbe neuronow w " + str(i + 1) + " warstwie ukrytej: ")))

        num_neurons.append(choice == 1 and 3 or 4)
        isBias = int(input("Czy chcesz dodac bias?: "))
        size = [len(x_array[0]), len(num_neurons), len(target_values[0])]

        net = network.Network(size, useBias=(False if isBias == 0 else True))
        print("Siec stworzona, co dalej?")
        isNetworkCreated = True

    if option == 1 and isNetworkCreated:
        print("Podaj warunek stopu")
        print("1. Ilosc epok")
        print("2. Dokladnosc")
        epoch_number = 999
        stop_precision = 1
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
        # TODO: METODA DO NAUKI SIECI W NETWORK.PY NETWORK.SGD
        net.SGD(training_data, epochs=epoch_number, mini_batch_size=10, learning_rate=learning_rate, shuffle=shuffle,
                precision=stop_precision, momentum=momentum, test_data=test_data, error_epoch=errorEpoch)
        print("Nauka zakonczona")

    if option == 2 and isNetworkCreated:
        with open("trainStats.txt", "w") as file:
            pass
        correct = choice == 1 and [0, 0, 0] or [0, 0, 0, 0]
        predicted_labels = []
        true_labels = []
        for index in range(choice == 1 and 105 or 4):
            test = test_data[index]
            output = network.forward(test[:4])
            if choice == 1:
                expected = test[-3:]
            else:
                expected = test[-4:]
            true_label = np.argmax(expected)
            predicted_label = np.argmax(output)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            if predicted_label == true_label:
                correct[true_label] += 1
            print(expected)

            error = network.calculateError(expected, output)
            neuronWeights = []
            neuronOutputs = []
            for i in range(len(network.layers)):
                layerWeights = []
                layerOutputs = []
                for j in range(len(network.layers[i].neurons)):
                    layerWeights.append(network.layers[i].neurons[j].weights)
                    layerOutputs.append(network.layers[i].neurons[j].output)
                neuronWeights.append(layerWeights)
                neuronOutputs.append(layerOutputs)

            with open("trainStats.txt", "a") as file:

                file.write(f"Wzorzec numer: {index}, {test[:4]}\n")
                file.write(f"Popelniony blad dla wzorca: {error}\n")
                file.write(f"Pozadany wzorzec odpowiedzi: {expected}\n")
                for i in range(len(output)):
                    file.write(f"Blad popelniony na {i} wyjsciu: {output[i] - expected[i]}\n")
                for i in range(len(output)):
                    file.write(f"Wartosc na {i} wyjsciu: {output[i]}\n")
                file.write(f"Wartosci wag neuronow wyjsciowych\n {neuronWeights[-1]}\n")
                # TODO: ZAPISYWANIE WAG I WYJSCIA NEURONOW DO PLIKU
                file.write(f"Wartosci wyjsciowe neuronow ukrytych warstwy {i}: {neuronOutputs[i]}\n")
               # TODO: ZAPISYWANIE WARTOSCI WYJSCIOWYCH NEURONOW DO PLIKU
                file.write(f"Wartosci wag neuronow ukrytych warstwy {i}:\n {neuronWeights[i]}\n")
                file.write("\n\n")

        file.close()

        if choice == 1:
            print("Klasyfikacja irysow")
            accuracy = sum(correct) / (len(test_data)) * 100
            print("Iris-setosa: " + str(correct[0] / 35 * 100) + "%")
            print("Iris-versicolor: " + str(correct[1] / 35 * 100) + "%")
            print("Iris-virginica: " + str(correct[2] / 35 * 100) + "%")
            print("Total: " + str(accuracy) + "%")
        else:
            print("Autoenkoder")
            print("Popelniony blad: ", error)
            print("Odpowiedzi: ", output)
            print("Poprawne odpowiedzi: ", expected)
        matrix = confusion_matrix(true_labels, predicted_labels)
        print("\nMacierz pomyłek:")
        print(matrix)
        precision = np.diag(matrix) / np.sum(matrix, axis=0)
        recall = np.diag(matrix) / np.sum(matrix, axis=1)
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("\nPrecyzja (Precision):", precision)
        print("Czułość (Recall):", recall)
        print("Miara F (F-measure):", f_measure)

    if option == 3 and isNetworkCreated:
        print("Zapisanie sieci")
        # TODO: METODA DO ZAPISU SIECIsave(network)
        filename = "network.pkl"
        net.save(filename)
        print("Siec zapisana do pliku network.pkl")


    if option == 2 and not isNetworkCreated:
        print("Wczytanie sieci")
        #TODO: WCZYTYWANIE SIECI network = loadNetwork()
        filename = "network.pkl"
        net = net.load(filename)
        print("Siec wczytana, co dalej?")
        isNetworkCreated = True
    if option == 4:
        break

#TODO: network.errorPlot()