from ember.optim.optimizer import Optimizer
import ember


class GradientDescent(Optimizer):
    """
    Vanilla Gradient Descent optimizer updates parameters by moving in the opposite direction
    of the gradient with a fixed learning rate.

    """

    def __init__(self, parameters, lr=1e-2):
        """
        Initialize the Gradient Descent optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        """
        Performs Vanilla Gradient Descent update rules.

        Updates each parameter by moving in the opposite direction of the gradient with
        a fixed learning rate.

        Returns:
            None
        """
        for i, parameter in enumerate(self.parameters):
            # In-place update of the parameter using Vanilla Gradient Descent rule
            parameter -= self.lr * parameter.grad
