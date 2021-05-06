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
        
        
    def create_matrix_Delta(self):
        Delta_matrix=[]
        Delta_matrix.append(np.zeros((self.layer_size, self.input_size)))
        for l in range(self.hidden_layers-1):
            Delta_matrix.append(np.zeros((self.layer_size, self.layer_size)))
        
        Delta_matrix.append(np.zeros((self.output_size, self.layer_size)))
        
        return Delta_matrix
            
        
    
    def forward_prop(self, layer_list, input_values):
        network_size = len(layer_list)+1
        data_list = list()
        data_list.append((input_values, 0))
        for l in range(network_size-1):
            z = layer_list[l].forward(data_list[-1][0])
            a = self.sig.sigmoid(z)
            data_list.append((a, z))

        return data_list

    def get_delta(self, data_list, output_values, layer_list):
        lista_delta=[]
        
        # if data_list[-1][0].shape == (data_list[-1][0].shape[0],):
        #     lista_delta.append(np.reshape((data_list[-1][0]-output_values), (data_list[-1][0].shape[0],1)))
        # else:
        
        lista_delta.append(data_list[-1][0]-output_values)
        
        
        for l in reversed(range(len(layer_list)-1)):
            
            weight= layer_list[l+1].as_weight_matrix()
            layer_derivate=self.sig.derivative_sigmoid(data_list[l+1][1])
            
            # if layer_derivate.shape==(layer_derivate.shape[0],):
            #     layer_derivate=np.reshape(layer_derivate,(layer_derivate.shape[0],1))
            
           
            
            new_error= np.dot(weight,lista_delta[0]) * layer_derivate
            
            # print("insert"+str(l))
            lista_delta.insert(0, new_error)
        
        return lista_delta
    
    def atualiza_Delta_matrix(self, a_array, next_delta_array):
        pass
    
               # self.atualiza_Delta_matrix(forwardprop_data[l][0], delta_list[l])
    
    def back_prop(self, input_data, output_data, layer_list):
        
        Delta_matrix=self.create_matrix_Delta()
        
        if output_data.shape == (output_data.shape[0],):
            output_values=np.reshape(output_data.values, (output_data.shape[0], 1))
        else:
            output_values=output_data.values
        
        
        for k in range(input_data.shape[0]):
            forwardprop_data=self.forward_prop(self.layer_list, input_data.values[k])
            delta_list=self.get_delta(forwardprop_data, output_values[k], self.layer_list)
            
            for l in range(len(self.layer_list)):
                
                increment = np.dot(np.reshape(delta_list[l], (delta_list[l].shape[0],1)),np.reshape(forwardprop_data[l][0], (forwardprop_data[l][0].shape[0], 1)).T)
                Delta_matrix[l]=Delta_matrix[l]+increment
            
        
        print(Delta_matrix)
        
        D_matrix=self.create_matrix_Delta()
        for l in range(len(self.layer_list)):
            D_matrix[l]=1/input_data.shape[0] * Delta_matrix[l] + learning_factor * self.layer_list(l)
        
            
        

        
        
    
            
        
        
            
layer1 = Layer(input_length=X.shape[1], n_neurons=3)
layer2 = Layer(input_length=3, n_neurons=2)

print(layer1.neurons)


class Sigmoid:
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
class ReLU():
    def forward(self, inputs):
        return np.maximum(0, inputs)

NeuNet=Net(layer_size=4, hidden_layers=2, input_size=X.shape[1], output_size=1)
listaTeste=NeuNet.layer_list
NeuNet.back_prop(X, y, NeuNet.layer_list)


