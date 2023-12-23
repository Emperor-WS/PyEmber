from abc import ABC, abstractmethod
import numpy as np
import copy
import ember
from ember.cuda import numpy_or_cupy, scalars_to_device
from .utils import inv_permutation
from .hook import Hook


class Function(ABC):
    """
    Abstract base class for defining mathematical operations as functions.

    Attributes:
    - tensors: Tensors involved in the operation.

    Methods:
    - forward(*tensors): Abstract method for the forward pass.
    - backward(grad): Abstract method for the backward pass.
    - __call__(*tensors): Invokes the function, registering hooks for gradients.
    - __repr__(): Returns a string representation of the function.
    """

    __slots__ = 'tensors'

    def __init__(self):
        super(Function, self).__init__()
        self.tensors = None

    @abstractmethod
    def forward(self, *tensors):
        """Abstract method for the forward pass."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        """Abstract method for the backward pass."""
        raise NotImplementedError

    def __call__(self, *tensors):
        """
        Invokes the function, registering hooks for gradients.

        Args:
        - *tensors: Variable number of input tensors.

        Returns:
        - Tensor: Output tensor from the forward pass.
        """
        self.tensors = (*tensors,)
        scalars_to_device(*self.tensors)

        # Perform the forward pass
        out = self.forward(*tensors)

        # Register hooks for gradients
        for tensor in self.tensors:
            if tensor.requires_grad:
                out.register_hook(Hook(tensor, self.backward))
        return out

    def __repr__(self):
        """
        Returns a string representation of the function.

        Returns:
        - str: String representation of the function.
        """
        return f'<Function: {self.__class__.__name__}>'


class Add(Function):
    """
    Addition operation.

    Methods:
    - forward(tensor1, tensor2): Performs addition.
    - single_backward(grad, tensor): Computes gradient for a single tensor.
    - backward(grad): Computes gradients for tensors involved in the backward pass.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs addition.

        Args:
        - tensor1: First input tensor.
        - tensor2: Second input tensor.

        Returns:
        - Tensor: Resultant tensor after addition.
        """
        data = tensor1.data + tensor2.data
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def single_backward(grad, tensor):
        """
        Computes gradient for a single tensor.

        Args:
        - grad: Gradient.
        - tensor: Input tensor.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        num_dims_added = grad.ndim - tensor.ndim
        for _ in range(num_dims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self, grad):
        """
        Computes gradients for tensors involved in the backward pass.

        Args:
        - grad: Gradient.

        Returns:
        - Tuple: Gradients for each input tensor.
        """
        tensor1, tensor2 = self.tensors
        return (self.single_backward(grad, tensor1),
                self.single_backward(grad, tensor2))


class Multiply(Function):
    """
    Multiplication operation.

    Methods:
    - forward(tensor1, tensor2): Performs multiplication.
    - single_backward(grad, t1, t2): Computes gradient for a single tensor.
    - backward(grad): Computes gradients for tensors involved in the backward pass.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs multiplication.

        Args:
        - tensor1: First input tensor.
        - tensor2: Second input tensor.

        Returns:
        - Tensor: Resultant tensor after multiplication.
        """
        # Determine whether to use NumPy or CuPy for element-wise multiplication
        nc = numpy_or_cupy(tensor1, tensor2)

        # Element-wise multiplication of tensor data
        data = nc.multiply(tensor1.data, tensor2.data)

        # Set attributes for the resulting tensor
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def single_backward(grad, t1, t2):
        """
        Computes gradient for a single tensor.

        Args:
        - grad: Gradient.
        - t1: First input tensor.
        - t2: Second input tensor.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        # Element-wise multiplication of gradient and second input tensor
        grad = grad * t2

        # Adjust dimensions to match the first input tensor
        num_dims_added = grad.ndim - t1.ndim
        for _ in range(num_dims_added):
            grad = grad.sum(axis=0)

        # Handle dimensions with size 1 in the first input tensor
        for i, dim in enumerate(t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def backward(self, grad):
        """
        Computes gradients for tensors involved in the backward pass.

        Args:
        - grad: Gradient.

        Returns:
        - Tuple: Gradients for each input tensor.
        """
        tensor1, tensor2 = self.tensors
        return (self.single_backward(grad, tensor1, tensor2),
                self.single_backward(grad, tensor2, tensor1))


class Dot(Function):
    """
    Dot product operation.

    Methods:
    - forward(tensor1, tensor2): Performs dot product.
    - backward(grad): Computes gradients for tensors involved in the backward pass.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs dot product.

        Args:
        - tensor1: First input tensor.
        - tensor2: Second input tensor.

        Returns:
        - Tensor: Resultant tensor after dot product.
        """
        # Matrix multiplication of input tensors
        data = tensor1.data @ tensor2.data

        # Set attributes for the resulting tensor
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    def backward(self, grad):
        """
        Computes gradients for tensors involved in the backward pass.

        Args:
        - grad: Gradient.

        Returns:
        - Tuple: Gradients for each input tensor.
        """
        tensor1, tensor2 = self.tensors
        return (grad @ tensor2.T,
                tensor1.T @ grad)


class Where(Function):
    """
    Conditional element-wise selection operation.

    Attributes:
    - cond: Boolean condition tensor.

    Methods:
    - _move_scalars_to_device(cond, tensor1, tensor2): Helper function to move scalars to the same device.
    - forward(tensor1, tensor2): Performs element-wise selection based on the condition.
    - backward(grad): Computes gradients for tensors involved in the backward pass.
    """

    def __init__(self, cond):
        """
        Initializes a Where object.

        Args:
        - cond: Boolean condition tensor.
        """
        super(Where, self).__init__()
        self.cond = cond

    @staticmethod
    def _move_scalars_to_device(cond, tensor1, tensor2):
        """
        Helper function to move scalars to the same device.

        Args:
        - cond: Boolean condition tensor.
        - tensor1: First input tensor.
        - tensor2: Second input tensor.
        """
        if tensor1.shape == ():
            if tensor2.device != 'cpu' or cond.device != 'cpu':
                tensor1.cuda()
        if tensor2.shape == ():
            if tensor1.device != 'cpu' or cond.device != 'cpu':
                tensor2.cuda()

    def forward(self, tensor1, tensor2):
        """
        Performs element-wise selection based on the condition.

        Args:
        - tensor1: First input tensor.
        - tensor2: Second input tensor.

        Returns:
        - Tensor: Resultant tensor after conditional element-wise selection.
        """
        self._move_scalars_to_device(self.cond, tensor1, tensor2)
        nc = numpy_or_cupy(tensor1, tensor2)

        # Use NumPy or CuPy to perform element-wise selection based on the condition
        data = nc.where(self.cond.data, tensor1.data, tensor2.data)

        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    def backward(self, grad):
        """
        Computes gradients for tensors involved in the backward pass.

        Args:
        - grad: Gradient.

        Returns:
        - Tuple: Gradients for each input tensor.
        """
        return (grad * ember.where(self.cond, 1, 0),
                grad * ember.where(self.cond, 0, 1))


class Sum(Function):
    """
    Summation operation along specified axis/axes.

    Attributes:
    - axis: Axis/axes along which the summation is performed.
    - keepdims: Whether to keep the dimensions of the summed axis/axes.

    Methods:
    - forward(tensor): Performs summation.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, axis=None, keepdims=False):
        """
        Initializes a Sum object.

        Args:
        - axis: Axis/axes along which the summation is performed.
        - keepdims: Whether to keep the dimensions of the summed axis/axes.
        """
        super(Sum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, tensor):
        """
        Performs summation.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after summation.
        """
        data = tensor.data.sum(axis=self.axis, keepdims=self.keepdims)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        data_keepdims = tensor.sum(axis=self.axis, keepdims=True)
        grad = grad.reshape(data_keepdims.shape) + ember.zeros_like(tensor)
        return grad


class Transpose(Function):
    """
    Transposition operation.

    Attributes:
    - indices: Tuple of indices specifying the new order of dimensions.

    Methods:
    - forward(tensor): Performs transposition.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, indices):
        """
        Initializes a Transpose object.

        Args:
        - indices: Tuple of indices specifying the new order of dimensions.
        """
        super(Transpose, self).__init__()
        self.indices = indices

    def forward(self, tensor):
        """
        Performs transposition.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after transposition.
        """
        # If indices are not provided, reverse the order of dimensions
        if self.indices is None:
            self.indices = tuple(range(tensor.ndim - 1, -1, -1))
        data = tensor.data.transpose(*self.indices)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        indices_back = tuple(inv_permutation(self.indices))
        grad = grad.transpose(*indices_back)
        return grad


class Reshape(Function):
    """
    Reshape operation.

    Attributes:
    - shape: New shape to reshape the tensor.

    Methods:
    - forward(tensor): Performs reshaping.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, shape):
        """
        Initializes a Reshape object.

        Args:
        - shape: New shape to reshape the tensor.
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        """
        Performs reshaping.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after reshaping.
        """
        data = tensor.data.reshape(*self.shape)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return grad.reshape(*tensor.shape)


class Pad(Function):
    """
    Padding operation.

    Attributes:
    - padding: Tuple specifying padding for each dimension.
    - constant_values: Value to fill padded areas with.

    Methods:
    - forward(tensor): Performs padding.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, padding, constant_values=0):
        """
        Initializes a Pad object.

        Args:
        - padding: Tuple specifying padding for each dimension.
        - constant_values: Value to fill padded areas with.
        """
        super(Pad, self).__init__()
        self.padding = padding
        self.constant_values = constant_values

    def forward(self, tensor):
        """
        Performs padding.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after padding.
        """
        nc = numpy_or_cupy(tensor)
        data = nc.pad(tensor.data, pad_width=self.padding,
                      constant_values=self.constant_values)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        return ember.unpad(grad, self.padding)


class Max(Function):
    """
    Maximum value operation.

    Attributes:
    - axis: Axis along which the maximum value is calculated.

    Methods:
    - forward(tensor): Computes the maximum value.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, axis=None):
        """
        Initializes a Max object.

        Args:
        - axis: Axis along which the maximum value is calculated.
        """
        super(Max, self).__init__()
        self.axis = axis

    def forward(self, tensor):
        """
        Computes the maximum value.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Maximum value tensor.
        """
        nc = numpy_or_cupy(tensor)
        data = nc.max(tensor.data, axis=self.axis)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        bigger_grad = ember.zeros_like(tensor)
        nc = numpy_or_cupy(grad)
        if self.axis is None:
            # If no axis is specified, find the maximum value and set the gradient there
            max_indices = ember.unravel_index(
                ember.argmax(tensor), tensor.shape)
            bigger_grad[max_indices] = grad
        else:
            # If axis is specified, find the maximum indices along the axis and set the gradient there
            max_indices = ember.argmax(tensor, axis=self.axis)
            for i, roll in enumerate(ember.rollaxis(bigger_grad, self.axis)):
                roll += (max_indices == i).astype(int) * grad

        return bigger_grad


class Neg(Function):
    """
    Negation operation.

    Methods:
    - forward(tensor): Performs negation.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self):
        """
        Initializes a Neg object.
        """
        super(Neg, self).__init__()

    def forward(self, tensor):
        """
        Performs negation.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after negation.
        """
        data = -tensor.data
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        return -grad


class Inverse(Function):
    """
    Inverse operation.

    Methods:
    - forward(tensor): Computes the element-wise inverse.
    - backward(grad): Computes gradients for the input tensor.
    """

    def forward(self, tensor):
        """
        Computes the element-wise inverse.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after element-wise inverse.
        """
        data = 1 / tensor.data
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad, *args, **kwargs):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return -1 / (tensor ** 2) * grad


class Slice(Function):
    """
    Slice operation.

    Methods:
    - __init__(indices): Initializes a Slice object.
    - forward(tensor): Slices the input tensor.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, indices):
        """
        Initializes a Slice object.

        Args:
        - indices: Indices to perform slicing.
        """
        super(Slice, self).__init__()
        self.indices = indices

    def forward(self, tensor):
        """
        Slices the input tensor.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after slicing.
        """
        data = tensor.data[self.indices]
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        bigger_grad = ember.zeros_like(tensor)
        if grad.shape != bigger_grad.shape:
            bigger_grad[self.indices] = grad
        else:
            bigger_grad = grad
        return bigger_grad


class Pow(Function):
    """
    Exponentiation operation.

    Methods:
    - __init__(power): Initializes a Pow object.
    - forward(tensor): Raises the input tensor to the given power.
    - backward(grad): Computes gradients for the input tensor.
    """

    def __init__(self, power):
        """
        Initializes a Pow object.

        Args:
        - power: Exponent power.
        """
        super(Pow, self).__init__()
        self.power = power

    def forward(self, tensor):
        """
        Raises the input tensor to the given power.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after exponentiation.
        """
        data = tensor.data ** self.power
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return grad * self.power * tensor ** (self.power - 1)


class Sqrt(Function):
    """
    Square root operation.

    Methods:
    - forward(tensor): Computes the element-wise square root.
    - backward(grad): Computes gradients for the input tensor.
    """

    def forward(self, tensor):
        """
        Computes the element-wise square root.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after square root operation.
        """
        nc = numpy_or_cupy(tensor)
        data = nc.sqrt(tensor.data)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return -1 / (2 * ember.sqrt(tensor)) * grad


class Exp(Function):
    """
    Exponential function operation.

    Methods:
    - forward(tensor): Computes the element-wise exponential function.
    - backward(grad): Computes gradients for the input tensor.
    """

    def forward(self, tensor):
        """
        Computes the element-wise exponential function.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after the exponential function.
        """
        nc = numpy_or_cupy(tensor)
        data = np.exp(tensor.data)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return grad * ember.exp(tensor)


class Log(Function):
    """
    Natural logarithm operation.

    Methods:
    - forward(tensor): Computes the element-wise natural logarithm.
    - backward(grad): Computes gradients for the input tensor.
    """

    def forward(self, tensor):
        """
        Computes the element-wise natural logarithm.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after the natural logarithm.
        """
        nc = numpy_or_cupy(tensor)
        data = nc.log(tensor.data)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return grad * ember.div(1, tensor)


class Tanh(Function):
    """
    Hyperbolic tangent operation.

    Methods:
    - forward(tensor): Computes the element-wise hyperbolic tangent.
    - backward(grad): Computes gradients for the input tensor.
    """

    def forward(self, tensor):
        """
        Computes the element-wise hyperbolic tangent.

        Args:
        - tensor: Input tensor.

        Returns:
        - Tensor: Resultant tensor after the hyperbolic tangent.
        """
        nc = numpy_or_cupy(tensor)
        data = nc.tanh(tensor.data)
        return ember.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        """
        Computes gradients for the input tensor.

        Args:
        - grad: Gradient.

        Returns:
        - Tensor: Gradient for the input tensor.
        """
        tensor, = self.tensors
        return grad * (1 - ember.tanh(tensor) ** 2)
