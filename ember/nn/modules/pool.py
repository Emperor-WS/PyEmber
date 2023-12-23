import numpy as np
import ember
from .module import Module
from ._utils import im2col, col2im


class MaxPool2d(Module):
    """
    2D Max Pooling Layer for Convolutional Neural Networks.

    This layer downsamples the spatial dimensions of the input tensor by taking the maximum value
    within a defined pool size, with optional stride and padding.

    Args:
    - kernel_size (int or tuple of int): Size of the pooling window. If int, the window is square.
    - stride (int): Step size to slide the pooling window.
    - pad (int): Zero-padding added to the input.

    Methods:
    - forward(input_data): Performs the forward pass for max pooling.
    - backward(dout): Computes the backward pass for max pooling.

    """

    def __init__(self, kernel_size, stride=1, pad=0):
        """
        Constructor for MaxPool2d.

        Args:
        - kernel_size (int or tuple of int): Size of the pooling window. If int, the window is square.
        - stride (int): Step size to slide the pooling window.
        - pad (int): Zero-padding added to the input.

        """
        super().__init__()

        # If pool_size is an int, convert it to a tuple (square window)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, input_data):
        """
        Performs the forward pass for max pooling.

        Args:
        - input_data: The input tensor.

        Returns:
        - out: The output tensor after max pooling.

        """
        # Extract dimensions
        N, C, input_height, input_width = input_data.shape

        # Calculate output dimensions
        out_h = int(1 + (input_height - self.kernel_size[0]) / self.stride)
        out_w = int(1 + (input_width - self.kernel_size[1]) / self.stride)

        # Apply im2col to input_data
        col = im2col(input_data, *self.kernel_size, self.stride, self.pad)
        col = col.reshape(-1, np.product(self.kernel_size))

        # Find the indices of the maximum values and the maximum values
        argmax = ember.argmax(col, axis=1)
        out = ember.max(col, axis=1)
        out = out.reshape(N, out_h + 2 * self.pad, out_w + 2 *
                          self.pad, C).transpose(0, 3, 1, 2)

        # Cache input_data and argmax for backward pass
        self._cache['x'] = input_data
        self._cache['argmax'] = argmax

        return out

    def backward(self, dout):
        """
        Computes the backward pass for max pooling.

        Args:
        - dout: The gradient of the output.

        Returns:
        - dx: The gradient with respect to the input.

        """
        # Transpose dout for easier manipulation
        dout = dout.transpose(0, 2, 3, 1)

        # Calculate pool size
        pool_size = np.product(self.kernel_size)

        # Create a matrix with zeros and assign gradients to max positions
        dmax = ember.zeros((dout.size, pool_size))
        x = self._cache['x']
        argmax = self._cache['argmax']
        dmax[ember.arange(argmax.size), argmax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # Reshape dmax for col2im
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # Apply col2im to get the gradient with respect to the input
        dx = col2im(dcol, x.shape, *
                    self.kernel_size, self.stride, self.pad)

        return dx

    def inner_repr(self):
        """
        Display the inner parameters of a CNN.
        """
        return f"kernel_size={self.kernel_size}, stride={self.stride}, pad={self.pad}"


class AdaptiveAvgPool2d(Module):
    """
    2D Adaptive Average Pooling Layer for Convolutional Neural Networks.

    This layer dynamically downsamples the spatial dimensions of the input tensor by taking the average value
    within a grid of a defined output size.

    Args:
    - output_size (int or tuple of int): Size of the output grid. If int, the grid is square.

    Methods:
    - forward(input_data): Performs the forward pass for adaptive average pooling.
    - backward(dout): Computes the backward pass for adaptive average pooling.

    """

    def __init__(self, output_size):
        """
        Constructor for AdaptiveAvgPool2d.

        Args:
        - output_size (int or tuple of int): Size of the output grid. If int, the grid is square.

        """
        super().__init__()

        # If output_size is an int, convert it to a tuple (square grid)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = output_size

    def forward(self, input_data):
        """
        Performs the forward pass for adaptive average pooling.

        Args:
        - input_data: The input tensor.

        Returns:
        - out: The output tensor after adaptive average pooling.

        """
        # Extract dimensions
        N, C, input_height, input_width = input_data.shape

        # Calculate output dimensions
        out_h = self.output_size[0]
        out_w = self.output_size[1]

        # Calculate step size for average pooling
        stride_h = input_height // out_h
        stride_w = input_width // out_w

        # Initialize output tensor
        out = ember.zeros((N, C, out_h, out_w),
                          requires_grad=self.requires_grad, device=self.device)

        # Perform adaptive average pooling
        for i in range(out_h):
            for j in range(out_w):
                pool_region = input_data[:, :, i *
                                         stride_h:(i + 1) * stride_h, j * stride_w:(j + 1) * stride_w]
                out[:, :, i, j] = pool_region.mean(axis=(2, 3))

        # Cache input_data and pooling indices for backward pass
        self._cache['x'] = input_data
        self._cache['pool_indices'] = (stride_h, stride_w)

        return out

    def backward(self, dout):
        """
        Computes the backward pass for adaptive average pooling.

        Args:
        - dout: The gradient of the output.

        Returns:
        - dx: The gradient with respect to the input.

        """
        # Extract cached values
        input_data = self._cache['x']
        stride_h, stride_w = self._cache['pool_indices']

        # Initialize gradient with respect to the input
        dx = ember.zeros_like(input_data)

        # Extract dimensions
        N, C, out_h, out_w = dout.shape

        # Calculate step size for average pooling
        stride_h = input_data.shape[2] // out_h
        stride_w = input_data.shape[3] // out_w

        # Iterate over the output regions to distribute gradients
        for i in range(out_h):
            for j in range(out_w):
                grad = dout[:, :, i, j][:, :, None, None]
                dx[:, :, i * stride_h:(i + 1) * stride_h, j *
                   stride_w:(j + 1) * stride_w] += grad

        return dx
