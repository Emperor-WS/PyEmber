from abc import ABC, abstractmethod
from ember import Tensor


class Optimizer(ABC):
    """
    Optimizer. Modify a model's parameters and update its weights/biases.
    This is an abstract base class for all optimizers.
    """

    def __init__(self, parameters):
        """
        Initialize the optimizer with model parameters.

        Args:
            parameters (iterable): Iterable containing the model's parameters.

        Raises:
            TypeError: If parameters is not an iterable.
        """

        # Check if parameters is an iterable
        if isinstance(parameters, Tensor):
            raise TypeError(
                "Expected parameters to be an iterable, got {}".format(type(parameters)))
        elif isinstance(parameters, dict):
            parameters = parameters.values()

        # Convert parameters to a list
        params = list(parameters)
        self.parameters = params

    @abstractmethod
    def step(self):
        """
        Update the parameters. Should be used only with the autograd system.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement the step method.")

    def zero_grad(self):
        """
        Zero gradients for all parameters contained in the parameters attribute.

        Returns:
            None
        """

        # Loop through each parameter and zero its gradient
        for parameter in self.parameters:
            parameter.zero_grad()

    def backward(self):
        """
        Update rules without autograd.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement the backward method.")
