import pandas as pd
import numpy as np
import sys
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

df = pd.read_csv('classification2.txt', header=None)
df.columns = ['column1', 'column2', 'label']
df.info()
print(f'\nLabel Value counts: \n{df.label.value_counts()}')
df.head()

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1]
y = y.values.reshape(y.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def print_metrics(y, preds):
    print(f'F1 : {f1_score(y, preds)}')
    print(f'Accuracy: {accuracy_score(y, preds)}')

def sigmoid(data):
    return 1 / (1 + np.exp(-data)) 

def sigmoid_derivative(data):
    return data * (1 - data)

class Neuron():
    def __init__(self, input_length):
        self.__weights = self.create_random_weights(input_length)
        
    def create_random_weights(self, input_length):
        return np.random.randn(input_length)
        
    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, new_value):
        self.__weights = new_value
    
    def __repr__(self):
        return f'Neuron {self.weights}'
    
class Layer():
    def __init__(self, input_length, n_neurons):
        self.input_length = input_length
        self.n_neurons = n_neurons
        self.create_neurons()
        self.dd = np.zeros((self.n_neurons, self.input_length))
    
    def create_neurons(self):
        """Creates neurons of layer
        """
        self.neurons = list()
        for i in range(self.n_neurons):
            self.neurons.append(
                Neuron(self.input_length)
            )
            
    def __repr__(self):
        return f'Layer \n{self.weights_matrix}'
    
    @property
    def weights_matrix(self):
        """Represents layer's weights as a matrix, where each column is a neuron.
        """
        return np.array([n.weights for n in self.neurons])
    
    @weights_matrix.setter
    def weights_matrix(self, new_weights):
        for neuron_i, neuron in enumerate(self.neurons):
            neuron.weights = new_weights[neuron_i]
    
    def transform(self, input_data, activation_function=sigmoid):
        """Calculates layer output based on input data
        """
        
        # Include bias
        #biased_input_data = np.ones((input_data.shape[0], input_data.shape[1] + 1))
        #biased_input_data[:, :-1] = input_data
        
        self.z = np.dot(input_data, self.weights_matrix.T)
        self.activation = activation_function(self.z)
        return self.activation

class NeuralNet():
    def __init__(self, *layers, learning_rate=0.1, epochs=5):
        self.layers = layers
        self.learning_rate = learning_rate
        self.costs = list()
        self.epochs = epochs
    
    def transform(self, X):
        X_transformed = X.copy()
        for layer in self.layers:
            X_transformed = layer.transform(X_transformed)
        
        return X_transformed.reshape(X.shape[0])
    
    def __repr__(self):
        return f'Neural Net \n{self.weights_matrix}'
    
    @property
    def weights_matrix(self):
        """Represents layer's weights as a matrix, where each column is a neuron.
        """
        return np.array([l.weights_matrix for l in self.layers], dtype=object)
    
    #def sample_cost(self, sample, label):
    #    log1_sample = np.log(sample)
    #    log2_sample = np.log(1 - sample)
    #    y_1 = np.multiply(log1_sample, label)
    #    y_0 = np.multiply((1 - label), log2_sample)
    #    return -(y_1 + y_0)
    
    def cost_2(self, X, y):
        hx = self.transform(X)
        return 0.5 * np.sum((np.subtract(y.reshape(len(y)), hx) ** 2)) / X.shape[0]
            
    def cost(self, X, y):
        hx = self.transform(X)
        y_reshaped = y.reshape(len(y))
        
        log1_hx = np.log(hx)
        log2_hx = np.log(1 - hx)
        
        y_1 = np.multiply(log1_hx, y_reshaped)
        y_0 = np.multiply((1 - y_reshaped), log2_hx)

        return -np.sum(
            y_1 + y_0
        ) / X.shape[0]
    
    def has_next_item(self, list_, index_):
        return (index_ + 1) == len(list_)

    def backpropagate(self, sample, sample_i, hxi, label):
        logger.debug(f'\n\n+++++++++++++++')
        logger.debug(f'Starting backpropagration\n')
        logger.debug(f'Sample {sample}')
        logger.debug(f'Hxi {hxi}')
        logger.debug(f'Label {label}')
        
        reversed_layers = self.layers[::-1]
        previous_deltas = hxi - label
        
        logger.debug(f'Initial dds: \n{previous_deltas}\n')
        
        for layer_i, layer in enumerate(reversed_layers):
            # Get activation from next layer
            if self.has_next_item(reversed_layers, layer_i):
                next_activation = sample
            else:
                next_layer = reversed_layers[layer_i + 1]
                next_activation = next_layer.activation[sample_i]
            
            # Add bias to activation
            #next_activation = np.append(next_activation, 1)
            
            # Multiply weights with deltas
            transposed_weights = layer.weights_matrix.T
            weights_dot_delta = np.dot(transposed_weights, previous_deltas)
            
            # Calculate sigmoid derivative
            sigmoid_derivate = sigmoid_derivative(next_activation)
            
            # Calculate new deltas
            new_deltas = weights_dot_delta * sigmoid_derivate
            
            # Calculate cost derivative
            dd = np.outer(previous_deltas, next_activation)
            layer.dd += dd
            previous_deltas = new_deltas#[:-1] # Exclude bias delta
            
            logger.debug(f'\n-----------------')
            logger.debug(f'Index {layer_i}, {layer}\n')
            logger.debug(f'Transposed weights: \n{transposed_weights}')
            logger.debug(f'Previous deltas: \n{previous_deltas}\n')
            logger.debug(f'Transposed Weights dd: \n{weights_dot_delta}')
            logger.debug(f'Next layer activation: \n{next_activation}')
            logger.debug(f'Sigmoid derivative: \n{sigmoid_derivate}\n')
            logger.debug(f'New deltas: \n{new_deltas}\n')
            logger.debug(f'Error derivative: \n{dd}')
            logger.debug(f'Layer error derivative: \n{layer.dd}')
            logger.debug('\n')
        
    def fit(self, X, y):
        hx = self.transform(X)
        logger.info(f'Fitting')
        cost_2 = list()
        for epoch in range(self.epochs):
            logger.info(f'Running epoch {epoch}')
            for sample_i, sample in enumerate(X):
                #logger.info(f'\n\nBackProp')
                self.backpropagate(sample, sample_i, hx[sample_i], y[sample_i])

            for layer_i, layer in enumerate(self.layers):
                #logger.info(f'Layer {layer_i} {layer}')
                #logger.info(f'Error der \n{layer.dd}')
                #logger.info('')
                layer.weights_matrix -= self.learning_rate * layer.dd / len(X)
                
            self.costs.append(self.cost(X, y))
            cost_2.append(self.cost_2(X, y))
        
        plt.plot(self.costs)
        plt.title('Cost through time')
        plt.ylabel('Cost')
        plt.show()
        
        plt.plot(cost_2)
        plt.title('Cost2')
        plt.ylabel('Cost2')
        plt.show()

#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

layers = [
    Layer(input_length=X_train.shape[1], n_neurons=2),
    Layer(input_length=2, n_neurons=1),
]

nn = NeuralNet(*layers, learning_rate=1, epochs=50)
print(nn)
print(f'Initial cost: {nn.cost(X_train, y_train)}')
nn.fit(X_train, y_train)
nn_preds = nn.transform(X_test)
nn_preds_round = nn_preds.round()