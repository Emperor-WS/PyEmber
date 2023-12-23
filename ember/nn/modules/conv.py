import numpy as np
import ember
from ember import Parameter
from .module import Module
from ._utils import col2im, im2col


class Conv2d(Module):
    """
    2D Convolutional Layer for Convolutional Neural Networks.

    This layer applies a 2D convolution operation on the input tensor using learnable filters and biases.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (tuple of int): Size of the filters (height, width).
    - stride (int): Step size for filter movement.
    - padding (int): Zero-padding added to the input.

    Methods:
    - forward(inputs): Performs the forward pass for convolution.
    - backward(d_output): Computes the backward pass for convolution.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        """
        Constructor for Conv2d.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (tuple of int): Size of the filters (height, width).
        - stride (int): Step size for filter movement.
        - padding (int): Zero-padding added to the input.

        """
        super().__init__()
        
        # If kernel_size is an int, convert it to a tuple (square window)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # Initialize learnable parameters
        bound = 1 / (in_channels * np.product(kernel_size))
        self.weight = Parameter.uniform(
            (out_channels, in_channels, *kernel_size), -bound, bound)
        self.bias = Parameter.zeros((out_channels,))

    def forward(self, inputs):
        """
        Performs the forward pass for convolution.

        Args:
        - inputs: The input tensor.

        Returns:
        - out: The output tensor after convolution.

        """
        # Extract dimensions
        num_filters, channels, filter_height, filter_width = self.weight.shape
        num_inputs, channels, input_height, input_width = inputs.shape

        # Display a warning if the stride does not match the input image size
        if (input_height + 2 * self.pad - filter_height) % self.stride != 0 or \
           (input_width + 2 * self.pad - filter_width) % self.stride != 0:
            print("Warning: Stride may not be suitable for input image size.")

        # Calculate output dimensions
        out_height = int((input_height + 2 * self.pad -
                         filter_height) // self.stride) + 1
        out_width = int((input_width + 2 * self.pad - filter_width) // self.stride) + 1

        # Apply im2col to inputs
        input_col = im2col(inputs, filter_height, filter_width, self.stride, self.pad)
        col_weight = self.weight.reshape(num_filters, -1).T

        # Perform convolution
        out = ember.dot(input_col, col_weight) + self.bias
        out = out.reshape(num_inputs, out_height, out_width, -1).transpose(0, 3, 1, 2)

        # Cache intermediate values for backward pass
        self._cache['x'] = inputs
        self._cache['x_col'] = input_col
        self._cache['weight_col'] = col_weight

        return out

    def backward(self, d_output):
        """
        Computes the backward pass for convolution.

        Args:
        - d_output: The gradient of the output.

        Returns:
        - dx: The gradient with respect to the input.

        """
        num_filters, channels, filter_height, filter_width = self.weight.shape

        # Transpose d_output for easier manipulation
        d_output = d_output.transpose(0, 2, 3, 1).reshape(-1, num_filters)

        # Compute gradients for bias and weight
        bias_grad = ember.sum(d_output, axis=0)
        weight_col_grad = ember.dot(self._cache['x_col'].T, d_output)
        weight_grad = weight_col_grad.transpose(1, 0).reshape(
            num_filters, channels, filter_height, filter_width)

        # Compute gradient with respect to the input
        input_col_grad = ember.dot(d_output, self._cache['weight_col'].T)
        dx = col2im(input_col_grad, self._cache['x'].shape,
                    filter_height, filter_width, self.stride, self.pad)

        # Store gradients in the module for later use
        self._grads['bias'] = bias_grad
        self._grads['weight'] = weight_grad

        return dx

    def inner_repr(self):
        """
        Display the inner parameters of a CNN.
        """
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, pad={self.pad}, " \
               f"bias={True if self.bias is not None else False}"
