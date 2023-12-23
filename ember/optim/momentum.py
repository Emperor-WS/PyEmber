from ember.optim.optimizer import Optimizer
import ember


class Momentum(Optimizer):
    """
    Momentum optimizer enhances gradient descent by adding a fraction of the previous
    update to the current gradient. It helps accelerate convergence in the relevant direction.

    """

    def __init__(self, parameters, lr=1e-2, beta=0.9):
        """
        Initialize the Momentum optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).
            beta (float): Decay factor for the moving average of gradients (default is 0.9).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta

        # Initialize cache for momentum
        self._cache = {
            'momentum': [ember.zeros_like(p) for p in self.parameters],
        }

    def step(self):
        """
        Performs Momentum update rules.

        The Momentum optimizer enhances gradient descent by adding a fraction of the
        previous update to the current gradient.

        Returns:
            None
        """
        for i, parameter in enumerate(self.parameters):
            # Store a moving average of the gradients
            momentum = self._cache['momentum'][i]

            # Update momentum
            momentum = self.beta * momentum + (1 - self.beta) * parameter.grad

            # In-place update of the parameter using Momentum rule
            parameter -= self.lr * momentum

            # Update the cache
            self._cache['momentum'][i] = momentum
