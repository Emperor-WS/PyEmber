import numpy as np
from .module import Module
from ember.nn.activation import *


class DNN(Module):
    def __init__(self, layer_dimensions, activation_hidden=nn.ReLU()):
        """
        Initializes a Deep Neural Network (DNN).

        Parameters:
        - layer_dimensions (list): List of integers representing the dimensions of each layer.
        - activation_hidden (Activation): Activation function for hidden layers.
        """
        super().__init__()
        # Initialize layers
        self.layer_dimensions = layer_dimensions
        self.hidden_dimensions = layer_dimensions[1: -1]
        # Validate and set activation functions
        assert isinstance(activation_hidden,
                          Activation), "Unrecognized activation function type"
        self.activation_hidden = activation_hidden
        self.activation_output = Softmax(axis=1)
        # Initialize weights and biases
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes weights and biases for the neural network.

        This method initializes the weights and biases for each layer in the neural network.
        It uses a normal distribution for the weights and initializes biases to zero.

        Initialization Details:
        - Weights: Randomly sampled from a normal distribution with mean (mu) 0 and standard deviation (sigma) sqrt(2 / output_dimension).
        - Biases: Initialized to zeros.

        Parameters:
        - mu (float): Mean of the normal distribution for weight initialization.
        - variance (float): Variance of the normal distribution for weight initialization.
        - standard_deviation (float): Standard deviation of the normal distribution for weight initialization.
        - weight_shape (tuple): Shape of the weight matrix for a layer.
        - weight (numpy.ndarray): Initialized weight matrix for a layer.
        - bias (numpy.ndarray): Initialized bias vector for a layer.
        - weight_key (str): Key to store the weight matrix in the _parameters dictionary.
        - bias_key (str): Key to store the bias vector in the _parameters dictionary.
        """
        for i in range(1, len(self.layer_dimensions)):
            # Set mean, variance, and standard deviation for weight initialization
            mu = 0
            variance = 2 / self.layer_dimensions[i]
            standard_deviation = np.sqrt(variance)

            # Initialize weight matrix and bias vector
            weight_shape = (self.layer_dimensions[i - 1], self.layer_dimensions[i])
            weight = np.random.normal(
                loc=mu, scale=standard_deviation, size=weight_shape)
            bias = np.zeros((self.layer_dimensions[i], ))

            # Store the initialized weight matrix and bias vector
            weight_key = "w_" + str(i)
            self._parameters[weight_key] = weight
            bias_key = "b_" + str(i)
            self._parameters[bias_key] = bias

    def forward(self, inputs):
        """
        Performs the forward pass of the DNN.

        Parameters:
        - inputs (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output of the forward pass.
        """
        depth = len(self.layer_dimensions) - 1
        current_activation = inputs

        if self.training:
            # Cache the input activation for training purposes
            activation_key = "a_0"
            self._cache[activation_key] = current_activation
        # Iterating through the NN
        for i in range(1, depth + 1):
            weight_key = "w_" + str(i)
            bias_key = "b_" + str(i)
            current_weight = self._parameters[weight_key]
            current_bias = self._parameters[bias_key]

            # Linear transformation
            current_activation = np.dot(
                current_activation, current_weight) + current_bias

            if self.training:
                # Cache intermediate activations during training
                activation_key = "z_" + str(i)
                self._cache[activation_key] = current_activation

            if i < depth:
                # Apply activation function for hidden layers
                current_activation = self.activation_hidden(current_activation)

                if self.training:
                    # Cache intermediate activations for training
                    activation_key = "a_" + str(i)
                    self._cache[activation_key] = current_activation

        # Apply softmax activation for the output layer
        final_output = self.activation_output(current_activation)

        if self.training:
            # Cache the final output for training purposes
            activation_key = "a_" + str(depth)
            self._cache[activation_key] = final_output

        return final_output

    def backward(self, outputs, labels):
        """
        Performs the backward pass of the DNN.

        Parameters:
        - outputs (numpy.ndarray): Output of the forward pass.
        - labels (numpy.ndarray): Ground truth labels.

        Returns:
        - numpy.ndarray: Gradient with respect to the inputs.
        """
        depth = len(self.layer_dimensions) - 1

        batch_size, num_classes = outputs.shape
        coefficient = 1 / batch_size

        prev_activation_key = "a_" + str(depth - 1)
        prev_activation = self._cache[prev_activation_key]
        Jz = outputs - labels

        # Compute gradients for the output layer
        dw = coefficient * np.dot(prev_activation.T, Jz)
        db = coefficient * np.sum(Jz, axis=0)
        self._grad["dw_" + str(depth)] = dw
        self._grad["db_" + str(depth)] = db

        for i in range(depth - 1, 0, -1):
            weight_key = "w_" + str(i + 1)
            prev_activation_key = "a_" + str(i - 1)
            current_activation_key = "z_" + str(i)
            current_weight = self._parameters[weight_key]
            prev_activation = self._cache[prev_activation_key]
            current_activation = self._cache[current_activation_key]

            # Backpropagate the error to the previous layer
            Jz = self.activation_hidden.backward(
                current_activation) * np.dot(Jz, current_weight.T)
            db = coefficient * np.sum(Jz, axis=0)
            dw = coefficient * np.dot(prev_activation.T, Jz)

            # Store the gradients
            self._grad["dw_" + str(i)] = dw
            self._grad["db_" + str(i)] = db
