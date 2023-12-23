from ember.optim.optimizer import Optimizer
import ember


class RMSprop(Optimizer):
    """
    RMSprop optimizer updates parameters using the Root Mean Square Propagation optimization algorithm.

    It adapts the learning rates for each parameter based on the moving average of squared gradients.

    """

    def __init__(self, parameters, lr=1e-2, beta=0.99, epsilon=1e-8):
        """
        Initialize the RMSprop optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).
            beta (float): Decay factor for the moving average of squared gradients (default is 0.99).
            epsilon (float): Small constant to prevent division by zero (default is 1e-8).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        # Initialize cache for velocity (moving average of squared gradients)
        self._cache = {'velocity': [ember.zeros_like(p) for p in self.parameters]}

    def step(self):
        """
        Performs RMSprop update rules.

        The RMSprop algorithm adapts the learning rates for each parameter based on the
        moving average of squared gradients.

        Returns:
            None
        """

        for i, parameter in enumerate(self.parameters):
            # Store a moving average of the gradients
            velocity = self._cache['velocity'][i]

            # Calculate the moving average using RMSprop rule
            velocity = self.beta * velocity + (1 - self.beta) * (parameter.grad ** 2)

            # In-place update of the parameter using RMSprop rule
            parameter -= self.lr * parameter.grad / (ember.sqrt(velocity) + self.epsilon)

            # Update the cache with the current velocity
            self._cache['velocity'][i] = velocity
