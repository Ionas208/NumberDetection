import numpy as np
from mnist import MNIST
from random import uniform
import math


class NeuralNetwork:

    def __init__(self, image):
        # Set up input layer
        self.activations = []
        self.weights = []
        self.biases = []

        activations = []
        for pixel in image:
            activations.append(self.sigmoid(pixel))
        self.activations.append(np.array(activations))

        
    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
         
    def initialize_layer(self, layer_size):
        # Randomly assign weights
        layer_weights = []
        for i in range(layer_size):
            node_weights = []
            for j in range(len(self.activations[len(self.activations)-1])):
                node_weights.append(uniform(-1,1))
            layer_weights.append(node_weights)

        layer_weights = np.matrix(layer_weights)
        self.weights.append(layer_weights)

        # Randomly assign biases
        layer_biases = []
        for i in range(layer_size):
            layer_biases.append(1)

        layer_biases = np.array(layer_biases)
        self.biases.append(layer_biases)

        #Calculate activations
        prev_activations = np.array(self.activations[len(self.activations)-1])

        # Calculate activations with follwing formula:
        # sigmoid(weights * activations + biases)
        
        print(f'{layer_weights.shape} * {layer_biases.shape}')
        layer_activations = np.array(layer_weights.dot(prev_activations))
        layer_activations = np.add(layer_activations, layer_biases)
        layer_activations = self.sigmoid(layer_activations)

        self.activations.append(layer_activations)

    

# Read Data
mndata = MNIST('samples')
mndata.gz = True
images, labels = mndata.load_training()

## Init Network
nn = NeuralNetwork(images[0])
nn.initialize_layer(16)
nn.initialize_layer(16)
nn.initialize_layer(10)