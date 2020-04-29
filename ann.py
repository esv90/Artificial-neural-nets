# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:15:38 2020

@author: a339594
"""
import numpy as np
import random

class Sequential:
    def __init__(self):
        self.layers = []
        self.has_input = False

    def add(self, layer):
        # Make the previous layer, the layer's input layer. Raise error if the model does not have an input layer
        print(type(layer))
        if self.has_input:
            try:
                layer.set_input_layer(self.layers[-1]) # Sets the input for this layer to the upstream layer in the model layer list
            except Exception as e:
                print('Layer could not be added because {}'.format(e))
        else:

            if isinstance(layer, Input):
                print(type(layer))
                self.has_input = True
            else:
                raise ValueError('The first layer of a sequential model must be an input layer')


        # Checks are good, then we can add the layer at the end of the list
        self.layers.append(layer)

    def summarize(self):
        for i, layer in enumerate(self.layers):
            print('Layer {}: {}'.format(i, str(layer)))

    def compile_model(self, optimizer, metric, loss):
        self.optimizer = optimizer
        self.metric = metric
        self.loss = loss

    def fit(self, features, labels, epochs=5, batch_size=1, validation_data=None):
        pass

    def evaluate(self, features, labels):
        pass

    def predict(self, feature, label):
        pass

# Super class
# class Layer:
#     def __init__(self):



class Input:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __str__(self):
        return "Input layer with shape = {}".format(self.input_shape)

class Dense:
    def __init__(self, neurons, stride=(1,1), activation='relu'):
        self.neurons = neurons
        self.stride = stride
        if is_activation_function(activation):
            self.activation = activation
        else:
            raise ValueError('The provided activation function is not a valid one')


        # The weights will be contained in a list of numpy lists where every numpy list contains the weights for each neuron
        self.weights = [np.array([]) for _ in range(self.neurons)] # Creates empty arrays for every neuron
        self.biases = np.zeros(shape=(neurons,)) # The bias for every neuron is a scalar

        # Initialize the neurons values
        self.values = np.random.rand(self.neurons) # Creates array of length self.neurons with random values

    def set_input_layer(self, input_layer):
        if type(input_layer) not in [Dense, Input, Flatten]:
            raise ValueError('Incompatible layer. Upstream layer must be either "Dense", "Input", or "Flatten"')

        self.input_layer = input_layer # Sets the upstream layer

    def forward_pass(self, n_weigths):
        if self.weights.size == 0:
            for neuron in range(self.neurons):
                # Initialize the weights for every neuron. The number of weights are given by the upstream layer
                self.weights[neuron] = np.random.rand(n_weights) # creates a list of size n_weights with random numbers

        for neuron in range(self.neurons):
            self.values = np.dot(self.input_layer.values, self.weights) + b



class Conv2d:
    def __init__(self, features, filter_size, stride=(1,1), activation='relu', padding='same'):
        self.features = features


class MaxPooling2D:
    pass


class Dropout:
    pass


class Flatten:
    pass


def is_activation_function(function_str):
    return function_str in ['relu', 'sigmoid', 'tanh', 'swish', 'identity', 'binary']

def relu(x):
    return x if x>0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def tanh(x):
    return np.tanh(x)

def swish(x):
    return x * sigmoid(x)






