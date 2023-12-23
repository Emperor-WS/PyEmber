from .module import Module
from ember.nn.activation import *
from ember import Parameter


class Linear(Module):
    """
    Linear layer module representing a fully connected layer in a neural network.

    Attributes:
    - input_dim: The number of input features.
    - output_dim: The number of output features.
    - bias: The bias parameter for the linear layer.
    - weight: The weight parameter for the linear layer.

    Methods:
    - __init__(self, input_dim, output_dim): Constructor for Linear class.
    - forward(self, x): Performs the forward pass for the linear layer.
    - backward(self, out): Performs the backward pass for the linear layer.
    - inner_repr(self): Provides a string representation of the inner details of the linear layer.

    """

    def __init__(self, input_dim, output_dim):
        """
        Constructor for the Linear class.

        Args:
        - input_dim: The number of input features.
        - output_dim: The number of output features.

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize bias parameter
        self.bias = Parameter.zeros(shape=(self.output_dim,))

        # Initialize weight parameter with normal distribution
        self.weight = Parameter.normal(shape=(self.input_dim, self.output_dim),
                                       mu=2 / self.input_dim,
                                       sigma=(2 / self.input_dim) ** 0.5)

    def forward(self, x):
        """
        Performs the forward pass for the linear layer.

        Args:
        - x: The input tensor.

        Returns:
        - Tensor: The output tensor after applying the linear transformation.

        Raises:
        - AssertionError: If the input dimension does not match the expected input dimension.

        """
        assert x.shape[1] == self.input_dim, 'Input dimension mismatch: ' \
                                             'Expected {}, got {}'.format(
                                                 x.shape, self._params['weight'].shape)

        output = ember.dot(x, self.weight) + self.bias

        # Cache the input for later use in the backward pass
        self._cache['x'] = x
        return output

    def backward(self, dout):
        """
        Performs the backward pass for the linear layer.

        Args:
        - out: The gradient tensor from the subsequent layer.

        Returns:
        - Tensor: The gradient tensor with respect to the input.

        """
        coef = 1 / dout.shape[0]  # Scaling factor for averaging gradients

        input_x = self._cache['x']
        weights = self.weight

        # Compute gradients for weight and bias
        weights_gradient = coef * ember.dot(input_x.T, dout)
        bias_gradient = coef * ember.sum(dout, axis=0)

        # Store gradients in the module for later use
        self._grads['b'] = bias_gradient
        self._grads['w'] = weights_gradient

        # Compute gradient with respect to the input
        dx = ember.dot(dout, weights.T)

        return dx

    def inner_repr(self):
        """
        Provides a string representation of the inner details of the linear layer.

        Returns:
        - str: String representation of the inner details.

        """

        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"bias={True if self.bias is not None else False}"
