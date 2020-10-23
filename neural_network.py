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

        image = self.sigmoid(np.array(image))
        image = np.reshape(image,(784, 1))
        self.activations.append(image)

        
    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
         
    def initialize_layer(self, layer_size):
        # Randomly assign weights
        layer_weights = []
        for i in range(layer_size):
            node_weights = []
            num_prev_nodes = len(self.activations[len(self.activations)-1])
            for j in range(num_prev_nodes):
                node_weights.append(uniform(-1,1))
            layer_weights.append(node_weights)

        layer_weights = np.matrix(layer_weights)
        self.weights.append(layer_weights)

        # Randomly assign biases
        layer_biases = []
        for i in range(layer_size):
            layer_biases.append(1)

        layer_biases = np.array(layer_biases)
        layer_biases = np.reshape(layer_biases, (layer_size, 1))
        self.biases.append(layer_biases)

        prev_activations = np.array(self.activations[len(self.activations)-1])

        # Calculate activations with follwing formula:
        # sigmoid(weights * activations + biases)
        layer_activations = np.array(layer_weights.dot(prev_activations))
        layer_activations += layer_biases
        layer_activations = self.sigmoid(np.array(layer_activations))
        
        self.activations.append(layer_activations)


    def get_result(self):
        output_activations = self.activations[len(self.activations)-1]
        confidence = max(output_activations)
        for i in range(len(output_activations)):
            if output_activations[i] == confidence:
                return (i, confidence.item())
        raise Exception('No Number found')
        

    

# Read Data
mndata = MNIST('samples')
mndata.gz = True
images, labels = mndata.load_training()

## Init Network
nn = NeuralNetwork(images[0])
nn.initialize_layer(16)
nn.initialize_layer(16)
nn.initialize_layer(10)
print(nn.get_result())