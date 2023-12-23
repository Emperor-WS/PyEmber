from ember.optim.optimizer import Optimizer
import ember
import numpy as np
import math
import random

class GeneticAlgorithm(Optimizer):
    """
    Genetic Algorithm optimizer updates parameters using principles of natural selection,
    crossover, and mutation.

    """

    def __init__(self, parameters, activation, error_func, x, y, num_agents=100, stop_error=0.001):
        """
        Initialize the Genetic Algorithm optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            activation (function): Activation function used in the model.
            error_func (function): Error function used in the model.
            x (numpy.array): Input data.
            y (numpy.array): Target labels.
            num_agents (int): Number of agents in each generation (default is 100).
            stop_error (float): Stop training when error falls below this threshold (default is 0.001).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.activation = activation
        self.error_func = error_func
        self.x = x
        self.y = y
        self.num_agents = num_agents
        self.stop_error = stop_error

    def generate_agents(self, n):
        """
        Generate a population of agents with random weights.

        Args:
            n (int): Number of agents in the population.

        Returns:
            numpy.array: Population of agents with random weights.

        """
        # Create random list of weights, including an extra value for the bias weight
        weights = ember.Tensor(np.random.rand(n, self.num_inputs + 1) * 2 - 1)
        return weights

    def calculate_fitness(self, current_gen):
        """
        Calculate the fitness of each agent in the current generation.

        Args:
            current_gen (numpy.array): Current generation of agents.

        Returns:
            numpy.array: Sorted weights based on fitness.
            numpy.array: Sorted errors corresponding to the weights.
            bool: True if early stopping condition is met, False otherwise.

        """
        loe = ember.Tensor([])

        for weights in current_gen:
            y_hats = self.predict(weights, self.x)
            er = self.error_func(self.y, y_hats)

            if er < self.stop_error:
                print("EARLY STOP")
                return weights, er, True

            loe = loe.append(er)

        sorted_indices = loe.argsort()
        sorted_weights = current_gen[sorted_indices]
        sorted_errors = loe[sorted_indices]

        return sorted_weights, sorted_errors, False

    def predict(self, weights, x):
        """
        Make predictions using the given weights and input data.

        Args:
            weights (numpy.array): Weights for the model.
            x (numpy.array): Input data.

        Returns:
            ember.Tensor: Predicted outputs.

        """
        neuron_vals = x.matmul(weights.transpose())
        y_hats = self.activation(neuron_vals)
        return y_hats

    def selection(self, weights, errors):
        """
        Perform selection of agents for the next generation.

        Args:
            weights (numpy.array): Weights of the current generation.
            errors (numpy.array): Errors corresponding to the weights.

        Returns:
            numpy.array: New weights for the next generation.

        """
        num_reproduced = int(math.ceil(len(weights) * 3 / 4))
        num_new = self.num_agents - num_reproduced

        # Calculate probabilities based on errors
        probabilities = 1 / errors
        probabilities /= probabilities.sum()

        # Randomly select indices based on probabilities
        index_choices = np.random.choice(len(weights), num_reproduced, p=probabilities)
        weights = weights[index_choices]

        new_weights = []

        for i in range(num_reproduced):
            parent1 = random.choice(weights)
            parent2 = random.choice(weights)
            child = parent1 if random.random() > 0.5 else parent2

            # Simulate 10% random mutations
            if random.random() < 0.1:
                child = np.random.rand(1, self.num_inputs + 1) * 2 - 1

            new_weights.append(child)

        new_weights = np.array(new_weights)
        random_weights = self.generate_agents(num_new)
        new_weights = np.concatenate((new_weights, random_weights), axis=0)

        return new_weights

    def step(self):
        """
        Perform a step of the Genetic Algorithm update rules.

        Returns:
            None

        """
        num_inputs = self.x.shape[1]
        self.num_inputs = num_inputs

        # Add a bias node to the input layer, with a value of 1
        x_with_bias = ember.cat([self.x, ember.ones((self.x.shape[0], 1))], dim=1)

        # Initial list of weights (a generation)
        low = self.generate_agents(self.num_agents)

        for generation in range(self.generations):
            sorted_weights, sorted_errors, early_quit = self.calculate_fitness(low)
            if early_quit:
                return sorted_weights, sorted_errors
            low = self.selection(sorted_weights, sorted_errors)

        best_weight, error, _ = self.calculate_fitness(low)
        return best_weight[0], error[0]
