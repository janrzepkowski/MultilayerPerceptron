import pandas as pd
import certifi
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)

def main():
    layer_num = int(input("Podaj ilość warstw ukrytych: "))
    input_num = int(input("Podaj ilość neuronów w warstwie wejściowej: "))
    neuron_nums = []
    for i in range(layer_num):
        neuron_nums.append(int(input(f"Podaj ilość neuronów w {i + 1} warstwie ukrytej: ")))
    neuron_nums.append(int(input("Podaj ilość neuronów w warstwie wyjściowej: ")))
    bias = bool(input("Czy chcesz użyć biasu? (True/False): "))