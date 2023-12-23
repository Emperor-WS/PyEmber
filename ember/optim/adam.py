from ember.optim.optimizer import Optimizer
import ember


class Adam(Optimizer):
    """
    Adam optimizer combines the concepts of momentum and RMSprop, adapting learning rates for each parameter.

    It maintains moving averages of both gradients (momentum) and squared gradients (RMSprop) to
    dynamically adjust the learning rates.

    """

    def __init__(self, parameters, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).
            beta1 (float): Decay factor for the moving average of gradients (default is 0.9).
            beta2 (float): Decay factor for the moving average of squared gradients (default is 0.999).
            epsilon (float): Small constant to prevent division by zero (default is 1e-8).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize caches for velocity, momentum, and the time step (t)
        self._cache = {
            'velocity': [ember.zeros_like(p) for p in self.parameters],
            'momentum': [ember.zeros_like(p) for p in self.parameters],
            't': 0
        }

    def step(self):
        """
        Performs Adam update rules.

        The Adam optimizer dynamically adapts learning rates based on both momentum and RMSprop.

        Returns:
            None
        """
        t = self._cache['t'] + 1
        for i, parameter in enumerate(self.parameters):
            # Store a moving average of the gradients
            velocity = self._cache['velocity'][i]
            momentum = self._cache['momentum'][i]

            # Update momentum and velocity with bias correction

            # momentum = β_1 * momentum + (1 − β_1) * ∇parameter
            momentum = self.beta1 * momentum + (1 - self.beta1) * parameter.grad
            # momentum_t = momentum / (1 - (β_1 ^ 2) )
            momentum_t = momentum / (1 - self.beta1 ** t)
            
            # velocity = β_2 * velocity+( 1 − β_2) * (∇parameter)^2
            velocity = self.beta2 * velocity + (1 - self.beta2) * (parameter.grad ** 2)
            # velocity_t = velocity / (1 - (β_2 ^ 2))
            velocity_t = velocity / (1 - self.beta2 ** t)

            # In-place update of the parameter using Adam rule
            # parameter = parameter − ( (learning_rate * momentum_t) / sqrt(velocity_t) + ϵ
            parameter -= self.lr * momentum_t / (ember.sqrt(velocity_t) + self.epsilon)

            # Update the cache
            self._cache['velocity'][i] = velocity
            self._cache['momentum'][i] = momentum

        # Update the time step
        self._cache['t'] = t
