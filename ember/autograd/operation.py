from abc import ABC, abstractmethod
from ember.cuda import numpy_or_cupy, scalars_to_device

import ember
from .hook import Hook


class Operation(ABC):
    """
    Abstract base class for tensor operations.

    Attributes:
    - tensor1 (ember.Tensor): First input tensor.
    - tensor2 (ember.Tensor): Second input tensor.
    """

    __slots__ = 'tensor1', 'tensor2'

    def __init__(self):
        super(Operation, self).__init__()
        self.tensor1 = None
        self.tensor2 = None

    @abstractmethod
    def forward(self, tensor1, tensor2):
        """
        Forward pass of the operation.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after the forward pass.
        """
        raise NotImplementedError

    @abstractmethod
    def backward1(self, grad):
        """
        Backward pass with respect to the first input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the first input tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def backward2(self, grad):
        """
        Backward pass with respect to the second input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the second input tensor.
        """
        raise NotImplementedError

    def __call__(self, tensor1, tensor2):
        """
        Execute the operation on the given input tensors.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after applying the operation.
        """
        # Ensure scalar tensors are on the same device
        scalars_to_device(tensor1, tensor2)

        # Store the input tensors
        self.tensor1 = tensor1
        self.tensor2 = tensor2

        # Perform the forward pass of the operation
        out = self.forward(tensor1, tensor2)

        # Register hooks for backward pass if required
        if tensor1.requires_grad:
            out.register_hook(Hook(tensor1, self.backward1))
        if tensor2.requires_grad:
            out.register_hook(Hook(tensor2, self.backward2))

        return out

    def __repr__(self):
        """
        String representation of the operation.

        Returns:
        - str: String representation of the operation.
        """
        return f'<Op: {self.__class__.__name__} on {self.device.upper()}>'


class Add(Operation):
    """
    Addition operation.

    Inherits from Operation and implements the forward and backward passes for addition.

    Methods:
    - forward(tensor1, tensor2): Performs the forward pass for addition.
    - backward(grad, tensor): Performs the backward pass with respect to the given input tensor.
    - backward1(grad): Performs the backward pass with respect to the first input tensor.
    - backward2(grad): Performs the backward pass with respect to the second input tensor.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs the forward pass for addition.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after addition.
        """
        # Element-wise addition of the data in the input tensors
        data = tensor1.data + tensor2.data

        # Determine if gradients are required
        requires_grad = tensor1.requires_grad or tensor2.requires_grad

        # Determine the device of the output tensor
        device = tensor1.device

        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def backward(grad, tensor):
        """
        Performs the backward pass with respect to the given input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.
        - tensor (ember.Tensor): Input tensor.

        Returns:
        - numpy.ndarray: Gradient with respect to the input tensor.
        """
        # Determine the number of dimensions added during broadcasting
        ndims_added = grad.ndim - tensor.ndim

        # Sum along the added dimensions
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        # Sum along dimensions where the size is 1
        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def backward1(self, grad):
        """
        Performs the backward pass with respect to the first input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the first input tensor.
        """
        # Perform the backward pass with respect to the first input tensor
        grad = self.backward(grad, self.tensor1)
        return grad

    def backward2(self, grad):
        """
        Performs the backward pass with respect to the second input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the second input tensor.
        """
        # Perform the backward pass with respect to the second input tensor
        grad = self.backward(grad, self.tensor2)
        return grad


class Multiply(Operation):
    """
    Multiplication operation.

    Inherits from Operation and implements the forward and backward passes for multiplication.

    Methods:
    - forward(tensor1, tensor2): Performs the forward pass for multiplication.
    - backward(grad, t1, t2): Performs the backward pass with respect to the given input tensors.
    - backward1(grad): Performs the backward pass with respect to the first input tensor.
    - backward2(grad): Performs the backward pass with respect to the second input tensor.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs the forward pass for multiplication.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after multiplication.
        """
        # Choose the appropriate backend (NumPy or CuPy)
        nc = numpy_or_cupy(tensor1, tensor2)

        # Element-wise multiplication of the data in the input tensors
        data = nc.multiply(tensor1.data, tensor2.data)

        # Determine if gradients are required
        requires_grad = tensor1.requires_grad or tensor2.requires_grad

        # Determine the device of the output tensor
        device = tensor1.device

        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def backward(grad, t1, t2):
        """
        Performs the backward pass with respect to the given input tensors.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.
        - t1 (ember.Tensor): First input tensor.
        - t2 (ember.Tensor): Second input tensor.

        Returns:
        - numpy.ndarray: Gradient with respect to the input tensor t1.
        """
        # Element-wise multiplication of the gradient with the second input tensor
        grad = grad * t2

        # Determine the number of dimensions added during broadcasting
        ndims_added = grad.ndim - t1.ndim

        # Sum along the added dimensions
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        # Sum along dimensions where the size is 1
        for i, dim in enumerate(t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def backward1(self, grad):
        """
        Performs the backward pass with respect to the first input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the first input tensor.
        """
        # Perform the backward pass with respect to the first input tensor
        grad = self.backward(grad, self.tensor1, self.tensor2)
        return grad

    def backward2(self, grad):
        """
        Performs the backward pass with respect to the second input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the second input tensor.
        """
        # Perform the backward pass with respect to the second input tensor
        grad = self.backward(grad, self.tensor2, self.tensor1)
        return grad


class Dot(Operation):
    """
    Dot product operation.

    Inherits from Operation and implements the forward and backward passes for the dot product.

    Methods:
    - forward(tensor1, tensor2): Performs the forward pass for the dot product.
    - backward1(grad): Performs the backward pass with respect to the first input tensor.
    - backward2(grad): Performs the backward pass with respect to the second input tensor.
    """

    def forward(self, tensor1, tensor2):
        """
        Performs the forward pass for the dot product.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after the dot product.
        """
        # Compute the dot product of the data in the input tensors
        data = tensor1.data @ tensor2.data # @ is the symbol of the matrix multiplication operator in Python.

        # Determine if gradients are required
        requires_grad = tensor1.requires_grad or tensor2.requires_grad

        # Determine the device of the output tensor
        device = tensor1.device

        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    def backward1(self, grad):
        """
        Performs the backward pass with respect to the first input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the first input tensor.
        """
        # Compute the gradient with respect to the first input tensor
        return grad @ self.tensor2.T

    def backward2(self, grad):
        """
        Performs the backward pass with respect to the second input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the second input tensor.
        """
        # Compute the gradient with respect to the second input tensor
        return self.tensor1.T @ grad


class Where(Operation):
    """
    Where operation.

    Inherits from Operation and implements the forward and backward passes for the 'where' operation.

    Attributes:
    - cond (ember.Tensor): Condition tensor for the 'where' operation.

    Methods:
    - __init__(self, cond): Initializes the Where operation with the given condition tensor.
    - scalars_to_device(cond, tensor1, tensor2): Ensures that scalar tensors are on the correct device.
    - forward(tensor1, tensor2): Performs the forward pass for the 'where' operation.
    - backward1(grad): Performs the backward pass with respect to the first input tensor.
    - backward2(grad): Performs the backward pass with respect to the second input tensor.
    """

    __slots__ = 'tensor1', 'tensor2', 'cond'

    def __init__(self, cond):
        """
        Initializes the Where operation with the given condition tensor.

        Args:
        - cond (ember.Tensor): Condition tensor for the 'where' operation.
        """
        self.cond = cond

    @staticmethod
    def scalars_to_device(cond, tensor1, tensor2):
        """
        Ensures that scalar tensors are on the correct device.

        Args:
        - cond (ember.Tensor): Condition tensor for the 'where' operation.
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.
        """
        # Check if tensor1 is a scalar and move to the correct device if needed
        if tensor1.shape == ():
            if tensor2.device != 'cpu' or cond.device != 'cpu':
                tensor1.cuda()

        # Check if tensor2 is a scalar and move to the correct device if needed
        if tensor2.shape == ():
            if tensor1.device != 'cpu' or cond.device != 'cpu':
                tensor2.cuda()

    def forward(self, tensor1, tensor2):
        """
        Performs the forward pass for the 'where' operation.

        Args:
        - tensor1 (ember.Tensor): First input tensor.
        - tensor2 (ember.Tensor): Second input tensor.

        Returns:
        - ember.Tensor: Resultant tensor after the 'where' operation.
        """
        # Ensure scalars are on the correct device
        self.scalars_to_device(self.cond, tensor1, tensor2)

        # Choose the appropriate backend (NumPy or CuPy)
        nc = numpy_or_cupy(tensor1, tensor2)

        # Element-wise conditional selection of data based on the condition tensor
        data = nc.where(self.cond.data, tensor1.data, tensor2.data)

        # Determine if gradients are required
        requires_grad = tensor1.requires_grad or tensor2.requires_grad

        # Determine the device of the output tensor
        device = tensor1.device

        return ember.Tensor(data, requires_grad=requires_grad, device=device)

    def backward1(self, grad):
        """
        Performs the backward pass with respect to the first input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the first input tensor.
        """
        # Compute the gradient with respect to the first input tensor
        return grad * ember.where(self.cond, 1, 0)

    def backward2(self, grad):
        """
        Performs the backward pass with respect to the second input tensor.

        Args:
        - grad (numpy.ndarray): Gradient with respect to the output.

        Returns:
        - numpy.ndarray: Gradient with respect to the second input tensor.
        """
        # Compute the gradient with respect to the second input tensor
        return grad * ember.where(self.cond, 0, 1)
