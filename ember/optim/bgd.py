from ember.optim.optimizer import Optimizer
import ember


class BatchGradientDescent(Optimizer):
    """
    Batch Gradient Descent optimizer updates parameters based on the average gradient
    over the entire dataset with a fixed learning rate.

    """

    def __init__(self, parameters, lr=1e-2):
        """
        Initialize the Batch Gradient Descent optimizer.

        Args:
            parameters (iterable): Iterable containing the model's parameters.
            lr (float): Learning rate (default is 1e-2).

        """
        # Call the parent constructor to initialize parameters
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        """
        Performs Batch Gradient Descent update rules.

        Updates each parameter based on the average gradient over the entire dataset
        with a fixed learning rate.

        Returns:
            None
        """
        num_parameters = len(self.parameters)
        avg_gradients = [ember.zeros_like(p) for p in self.parameters]

        # Calculate the average gradient over the entire dataset
        for parameter in self.parameters:
            avg_gradients += parameter.grad

        avg_gradients /= num_parameters

        # Update each parameter using Batch Gradient Descent rule
        for i, parameter in enumerate(self.parameters):
            parameter -= self.lr * avg_gradients[i]
