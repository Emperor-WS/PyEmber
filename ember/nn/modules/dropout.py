import numpy as np
from ember.module import Module
from ember.tensor import Tensor


class Dropout(Module):
    """
    Dropout layer for regularization during training.

    Dropout randomly sets a fraction of input units to zero during the forward pass,
    helping prevent overfitting.

    Attributes:
    - prob (float): Probability of dropping out a neuron, must be in the range (0, 1).

    Methods:
    - __init__(self, prob=0.5): Constructor for Dropout class.
    - forward(inputs): Performs the forward pass with dropout.
    - backward(grad): Performs the backward pass for dropout (optional).

    """

    def __init__(self, prob=0.5):
        """
        Constructor for the Dropout class.

        Args:
        - prob (float): Probability of dropping out a neuron, must be in the range (0, 1).

        """
        super().__init__()
        self.prob = prob
        self.mask = None  # Variable to store the dropout mask during forward pass

    def forward(self, inputs):
        """
        Performs the forward pass with dropout.

        The forward pass applies dropout to the input, randomly setting a fraction of
        input units to zero based on the given dropout probability.

        Args:
        - inputs: The input tensor.

        Returns:
        - Tensor: The output tensor after applying dropout.

        """
        # Scaling factor to maintain the expected value of the input
        scale = 1 / (1 - self.prob) if self.prob < 1 else 0

        # Generate random probabilities for each element in the input
        probabilities = np.random.uniform(low=0.0, high=1.0, size=inputs.shape)

        # Create a binary mask based on dropout probability
        self.mask = Tensor(np.where(probabilities > self.prob, 0, 1))

        # Apply dropout to the input
        dropout = inputs * self.mask

        return scale * dropout

    def backward(self, grad):
        """
        Performs the backward pass for dropout (optional).

        The backward pass for dropout scales the gradient by the inverse of the dropout
        probability for the retained neurons.

        Args:
        - grad: The gradient tensor.

        Returns:
        - Tensor: The scaled gradient tensor for the input.

        """
        return grad * self.mask / (1 - self.prob) if self.prob < 1 else 0
