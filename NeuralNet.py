# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:46:37 2021

@author: yohas
"""

import pandas as pd
import numpy as np

df = pd.read_csv('classification2.txt', header=None)
df.columns = ['column1', 'column2', 'label']
df.info()
print(f'\nLabel Value counts: \n{df.label.value_counts()}')
df.head()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


class Neuron:
    def __init__(self, input_length):
        self.bias = 1
        self.weights = self.create_random_weights(input_length)
        
    def create_random_weights(self, input_length):
        return np.random.rand(input_length)
        
    def __repr__(self):
        return f'Neuron {self.weights}'


class Layer:
    def __init__(self, input_length, n_neurons):
        self.input_length = input_length
        self.n_neurons = n_neurons
        self.create_neurons()
    
    def create_neurons(self):
        """Creates neurons of layer
        """
        self.neurons = list()
        for i in range(self.n_neurons):
            self.neurons.append(
                Neuron(self.input_length)
            )
            
    def as_weight_matrix(self):
        """Represents layer's weights as a matrix, where each column is a neuron.
        """
        return np.array([n.weights for n in self.neurons]).T
    
    def as_bias_matrix(self):
        """Represents layer's biases as a matrix, where each column is a neuron.
        """
        return [n.bias for n in self.neurons]
            
    def forward(self, input_data):
        """Calculates layer output based on input data
        """
        return np.dot(input_data, self.as_weight_matrix()) + self.as_bias_matrix()


class Net:
    def __init__(self, layer_size, hidden_layers, input_size, output_size=1):
        self.layer_size = layer_size
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.layer_list = self.create_layer_list()

        self.sig = Sigmoid()
        self.relu = ReLU()

    def create_layer_list(self):
        """Creates the list with the net layers"""
        layer_list = list()
        layer_list.append(Layer(input_length=self.input_size, n_neurons=self.layer_size))

        for v in range(self.hidden_layers - 1):
            layer_list.append(Layer(input_length=self.layer_size, n_neurons=self.layer_size))
        layer_list.append(Layer(input_length=self.layer_size, n_neurons=self.output_size))

        return layer_list

    def forward_loop(self, layer_list, input_data):
        network_size = len(layer_list)+1
        data_list = list()
        data_list.append((input_data, 0))
        for l in range(network_size-1):
            z = layer_list[l].forward(data_list[-1][0])
            a = self.sig.sigmoid(z)
            data_list.append((a, z))

        return data_list

    def back_prop(self, data_list, input_data, output_data, layer_list):
        pass


class ReLU:
    @staticmethod
    def forward(inputs):
        return np.maximum(0, inputs)
    
layer1 = Layer(input_length=X.shape[1], n_neurons=3)
layer2 = Layer(input_length=3, n_neurons=2)

print(layer1.neurons)


class Sigmoid:
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

