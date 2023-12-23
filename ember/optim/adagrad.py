from ember.optim.optimizer import Optimizer
import ember


class Adagrad(Optimizer):
    """
    Adagrad optimizer. A more sophisticated version of SGD, with learning rate decay.
    """

    def __init__(self, parameters, lr=1e-2, epsilon=1e-8):
        """
        Initialize the Adagrad optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).
            epsilon (float): Small constant to prevent division by zero (default is 1e-8).
        """

        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr
        self.epsilon = epsilon

        # Initialize cache for squared gradients
        self._cache = {'decay': [ember.zeros_like(p) for p in self.parameters]}

    def step(self):
        """
        Performs Adagrad update rules.

        The Adagrad algorithm adjusts the learning rate for each parameter based on the
        historical gradient information.

        Updates are performed in-place.

        Returns:
            None
        """

        for i, parameter in enumerate(self.parameters):
            # Moving average of squared gradients
            decay = self._cache['decay'][i] ** 2

            # In-place update of the parameter using Adagrad rule
            parameter += self.lr * parameter.grad / (ember.sqrt(decay) + self.epsilon)

            # Update the cache with squared gradients
            self._cache['decay'][i] += decay
