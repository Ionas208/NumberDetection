import numpy as np
from mnist import MNIST
from random import uniform
import math

input_layer_size = 784
output_layer_size = 10

class NeuralNetwork:
    def __init__(self, hidden_layers, hidden_layer_size):
        self.activations = []
        self.hidden_layers = hidden_layers

        # Input Layer
        self.weights = []
        self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(hidden_layer_size, input_layer_size)))

        # Hidden Layers
        for i in range(hidden_layers-1):
            self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(hidden_layer_size, hidden_layer_size)))

        # Output Layer
        self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(output_layer_size, hidden_layer_size)))

    def __str__(self):
        string = ''
        output_layer = self.activations[len(self.activations)-1]
        for i in range(10):
            if output_layer[i] == np.amax(output_layer):
                string += "\033[94m"+"\033[1m" + str(i)+": "+str(round(output_layer[i], 2))+"\033[0m"+"\n"
            else:
                string += str(i)+": "+str(round(output_layer[i], 2))+"\n"
        return string

    def get_cost(self, number):
        output_layer = self.activations[len(self.activations)-1]
        cost_vector = np.zeros(10)
        cost_vector[number] = 1
        
        cost_vector = cost_vector - output_layer
        cost_vector = np.square(cost_vector)
        return np.sum(cost_vector)

    
def initialize(network, image):
    network.activations.append(sigmoid(np.array(image)))
    forward_propagate(network, 0)

def forward_propagate(network, layer):
    if layer == network.hidden_layers+1:
        return 0

    next_activations = sigmoid(network.weights[layer].dot(network.activations[layer]))
    network.activations.append(next_activations)

    forward_propagate(network, layer+1)
    
def sigmoid(x):
            return 1 / (1 + np.exp(-x))

def extract_result(result):
    return (np.where(result == np.amax(result)), np.amax(result))

mndata = MNIST('samples')
mndata.gz = True
images, labels = mndata.load_training()

nn = NeuralNetwork(5, 16)
initialize(nn, images[0])
print("Number was: "+str(labels[0]))
print("Network thinks: ")
print(nn)
print("Cost: ", end="")
print(nn.get_cost(labels[0]))