from ember.optim.optimizer import Optimizer
import ember


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer updates parameters using the following optimization formula:
    
        updated_param_i = param_i - learning_rate * gradient_param_i
    
    This optimization process involves adjusting each parameter (param_i) by subtracting the product of the learning rate (learning_rate) and the gradient of the parameter (gradient_param_i).
    
    Momentum can be incorporated into this process for additional optimization.
    """

    def __init__(self, parameters, lr=1e-2, momentum=0):
        """
        Initialize the SGD optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).
            momentum (float): Momentum factor for SGD with momentum (default is 0).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum

        # Initialize cache for velocity (moving average of gradients)
        self._cache = {'velocity': [ember.zeros_like(p) for p in self.parameters]}

    def step(self):
        """
        Performs stochastic gradient descent with momentum.

        The SGD optimizer updates parameters by subtracting the product of the learning rate
        and the gradient of the parameters. Additionally, momentum can be added to the process.

        Returns:
            None
        """
        for i, parameter in enumerate(self.parameters):
            # Store a moving average of the gradients (velocity)
            velocity = self._cache['velocity'][i]

            # Calculate the moving average using momentum
            velocity = self.momentum * velocity - self.lr * parameter.grad

            # In-place update of the parameter using SGD rule with momentum
            parameter += velocity

            # Update the cache with the current velocity
            self._cache['velocity'][i] = velocity
