import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

import ember
from ember.cuda import numpy_or_cupy, cuda_available


# Custom exception for backward call error
class BackwardCallError(Exception):
    pass


# Custom exception for CUDA not available error
class CUDANotAvailableError(Exception):
    pass


def tensor2string(
        tensor, prefix="",
        precision=4,
        separator=', ',
        floatmode=None,
        edgeitems=3,
        threshold=100,
        max_line_width=100,
        suppress_small=True
):
    """
    Convert a tensor to a formatted string.

    Args:
        tensor: Input tensor.
        prefix (str): Prefix for each line in the string.
        precision (int): Number of decimal places for floating-point numbers.
        separator (str): Separator between elements in the string.
        floatmode: Float formatting mode.
        edgeitems (int): Number of items at the beginning and end of each dimension to show.
        threshold (int): Total number of array elements to trigger summarization.
        max_line_width (int): Maximum width (in characters) of the string.
        suppress_small (bool): Whether to suppress small numbers.

    Returns:
        str: Formatted string representation of the tensor.
    """
    nc = numpy_or_cupy(tensor)
    array_str = nc.array_str(tensor.data,
                             precision=precision,
                             max_line_width=max_line_width,
                             suppress_small=suppress_small)

    array_str = f"\n{prefix}".join(array_str.split("\n"))
    return array_str


def to_numpy(arrayable):
    """
    Convert an array-like object to a NumPy array.

    Args:
        arrayable: Input array-like object.

    Returns:
        np.ndarray: NumPy array.
    """
    if isinstance(arrayable, Tensor):
        return np.array(arrayable.data)
    elif isinstance(arrayable, np.ndarray):
        return arrayable
    elif cuda_available() and isinstance(arrayable, cp.ndarray):
        return cp.asnumpy(arrayable)
    else:
        return np.array(arrayable)


def to_cupy(arrayable):
    """
    Convert an array-like object to a CuPy array.

    Args:
        arrayable: Input array-like object.

    Returns:
        cp.ndarray: CuPy array.

    Raises:
        CUDANotAvailableError: If CUDA is not available.
    """
    if not cuda_available():
        raise CUDANotAvailableError(
            "Could not move a tensor to GPU because CUDA was not found.")
    if isinstance(arrayable, Tensor):
        return cp.array(arrayable.data)
    elif isinstance(arrayable, np.ndarray):
        return cp.array(arrayable)
    elif isinstance(arrayable, cp.ndarray):
        return arrayable
    else:
        return cp.array(arrayable)


def to_tensor(tensorable, **kwargs):
    """
    Convert an array-like object to a Tensor.

    Args:
        tensorable: Input array-like object.
        **kwargs: Additional keyword arguments for Tensor creation.

    Returns:
        Tensor: Created Tensor object.
    """
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable, **kwargs)


class Tensor(object):
    """
    Custom Tensor class with basic operations and autograd capabilities.

    Attributes:
        _data: Underlying data array.
        requires_grad: Indicates whether gradients should be tracked for this tensor.
        _hooks: List of hooks for backpropagation.
        _grad_fn: Function to compute gradients during backpropagation.
        _grad: Computed gradients.
        _id: Unique identifier for the tensor.
        _version: Version counter for tracking changes.
        _device: Device on which the tensor is stored ('cpu' or 'cuda').

    Methods:
        to(self, device): Move the tensor to the specified device.
        cpu(self): Move the tensor to the CPU.
        cuda(self): Move the tensor to the GPU (CUDA).
        item(self): Get the Python number from a 0-dimensional tensor.
        tolist(self): Convert the tensor to a Python list.
        numpy(self): Convert the tensor to a NumPy array.
        cupy(self): Convert the tensor to a CuPy array if CUDA is available.
        sum(self, axis=None, keepdims=False): Sum the tensor elements along a specified axis.
        transpose(self, *indices): Transpose the tensor dimensions based on the provided indices.
        reshape(self, *shape): Reshape the tensor to the specified shape.
        flatten(self): Flatten the tensor to a 1D array.
        append(self, t, axis=0): Append another tensor along a specified axis.
        __repr__(self): String representation of the tensor.
        __len__(self): Return the length of the tensor.
        __gt__(self, other): Element-wise greater than comparison.
        __ge__(self, other): Element-wise greater than or equal to comparison.
        __lt__(self, other): Element-wise less than comparison.
        __le__(self, other): Element-wise less than or equal to comparison.
        __eq__(self, other): Element-wise equality comparison.
        __ne__(self, other): Element-wise inequality comparison.
        __add__(self, other): Element-wise addition.
        __radd__(self, other): Right-side element-wise addition.
        __iadd__(self, other): In-place element-wise addition.
        __neg__(self): Element-wise negation.
        __sub__(self, other): Element-wise subtraction.
        __rsub__(self, other): Right-side element-wise subtraction.
        __isub__(self, other): In-place element-wise subtraction.
        __mul__(self, other): Element-wise multiplication.
        __rmul__(self, other): Right-side element-wise multiplication.
        __imul__(self, other): In-place element-wise multiplication.
        __pow__(self, power, modulo=None): Element-wise exponentiation.
        __truediv__(self, other): Element-wise true division.
        __rtruediv__(self, other): Right-side element-wise true division.
        __itruediv__(self, other): In-place element-wise true division.
        __matmul__(self, other): Matrix multiplication.
        __getitem__(self, indices): Get a sub-tensor based on the provided indices.
        __setitem__(self, key, value): Set the value of the tensor based on the provided key.

    """

    # Setting slots free memory, and does not keep built-in functions (__builtin__ things)
    __slots__ = '_data', 'requires_grad', '_hooks', '_grad_fn', '_grad', '_id', '_version', '_device'

    _COUNTER = 0

    def __init__(self, data, requires_grad=False, device="cpu", hooks=None):
        """
        Initialize the Tensor object.

        Args:
            data: Input data for the tensor.
            requires_grad (bool): Whether gradients should be calculated for this tensor (default is False)..
            device (str): Device where the tensor is stored ("cpu" or "cuda", default is 'cpu')..
            hooks (list): List of hooks for backpropagation (default is None).

        """
        self._device = device

        # Convert data to NumPy or CuPy based on the device
        if device == 'cpu':
            data = to_numpy(data)
        else:
            data = to_cupy(data)
        self._data = data

        self.requires_grad = requires_grad
        self._hooks = hooks or []  # List to store hooks for backward pass
        self._grad_fn = None  # Gradient function for autograd
        self._grad = None  # Gradient value
        # Update the tracking
        self._version = 0  # Version for tracking changes
        self._id = Tensor._COUNTER
        Tensor._COUNTER += 1  # Increment the counter for unique IDs

    @property
    def device(self):
        return self._device.lower()

    @property
    def grad(self):
        return self._grad

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def is_leaf(self):
        if self._grad_fn is None and self._hooks == []:
            return True
        return False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

        self.detach()    

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._data = self._data.astype(new_dtype)
        self.detach()

    @property
    def version(self):
        return self._version

    @property
    def id(self):
        return self._id

    @property
    def T(self):
        return ember.transpose(self)

    def register_hook(self, hook):
        """
        Register a hook to be applied during the backward pass.

        Args:
            hook: Hook function to be registered.

        Returns:
            None
        """
        self._hooks.append(hook)

    def astype(self, new_type):
        """
        Change the data type of the tensor.

        Args:
            new_type: New data type for the tensor.

        Returns:
            Tensor: Tensor with the specified data type.
        """
        self.detach()
        return ember.astype(self, new_type)

    def detach(self):
        """
        Detach the tensor from the computation graph.

        This method clears the gradient and associated information, making the tensor a leaf node.

        Returns:
            None
        """
        self._grad = None
        self._grad_fn = None
        self._hooks = []

    def zero_grad(self):
        """
        Zero out the gradient of the tensor.

        This method sets the gradient to a tensor of zeros and detaches the tensor.

        Returns:
            None
        """
        self._grad = ember.zeros(self.shape, device=self.device, dtype='float64')
        self.detach()
        self._grad = ember.zeros(
            self.shape, device=self.device, dtype='float64')

    def backward(self, grad=None):
        """
        Perform backward pass to calculate gradients.

        Args:
            grad: Gradient value to be used in the backward pass.

        Raises:
            BackwardCallError: If the tensor does not require gradients.
            RuntimeError: If grad is not specified for a non-0-tensor.

        Returns:
            None
        """
        if not self.requires_grad:
            raise BackwardCallError(
                """Attempted backward propagation on a tensor without `requires_grad=True`.
                This may happen if the tensor was not initialized with `requires_grad=True`,
                or if gradients were set to `None` due to an inplace operation.
                Additionally, it could occur if the computational graph was split,
                and gradients are no longer linked to this branch.
                Graphs are typically split when a new tensor is created using a numeric function
                (e.g., zero, ones, eye, identity),
                and `requires_grad` was not specified."""
            )

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0, device=self.device)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        # Update the gradient
        self._grad = grad if self._grad is None else self._grad + grad

        # Back-propagation in all dependencies
        hooks = self._hooks
        if hooks is not None:
            for hook in self._hooks:
                # Compute the gradient w.r.t the operation
                backward_grad = hook.grad_fn(grad)
                # Back-propagate in the tensor used in this operation
                hook.tensor.backward(backward_grad)

    def to(self, device):
        """
        Move the tensor to the specified device.

        Args:
            device (str): The target device ('cpu' or 'cuda').

        Returns:
            Tensor: A new Tensor object with data moved to the target device.

        """
        return Tensor(self.data, requires_grad=self.requires_grad, device=device)

    def cpu(self):
        """
        Move the tensor to the CPU.

        """
        self.data = self.to('cpu').data
        self._device = 'cpu'

    def cuda(self):
        """
        Move the tensor to the GPU (CUDA).

        """
        self.data = self.to('cuda').data
        self._device = 'cuda'

    def item(self):
        """
        Get the Python number from a 0-dimensional tensor.

        Returns:
            int or float: The Python number.

        """
        self.detach()
        return self.data.item()

    def tolist(self):
        """
        Convert the tensor to a Python list.

        Returns:
            list: The Python list representation of the tensor.

        """
        self.detach()
        return self.data.tolist()

    def numpy(self):
        """
        Convert the tensor to a NumPy array.

        Returns:
            numpy.ndarray: The NumPy array representation of the tensor.

        """
        self.detach()
        return to_numpy(self.data)

    def cupy(self):
        """
        Convert the tensor to a CuPy array if CUDA is available.

        Returns:
            cupy.ndarray: The CuPy array representation of the tensor.

        Raises:
            CUDANotAvailableError: If CUDA is not available.

        """
        self.detach()
        return to_cupy(self.data)

    def sum(self, axis=None, keepdims=False):
        """
        Sum the tensor elements along a specified axis.

        Args:
            axis (int or tuple): The axis or axes along which the sum is performed.
            keepdims (bool): Whether to retain the original dimensions.

        Returns:
            Tensor: The summed tensor.

        """
        return ember.sum(self, axis=axis, keepdims=keepdims)

    def transpose(self, *indices):
        """
        Transpose the tensor dimensions based on the provided indices.

        Args:
            indices (int): The new order of dimensions.

        Returns:
            Tensor: The transposed tensor.

        """
        return ember.transpose(self, indices)

    def reshape(self, *shape):
        """
        Reshape the tensor to the specified shape.

        Args:
            shape (int): The new shape of the tensor.

        Returns:
            Tensor: The reshaped tensor.

        """
        return ember.reshape(self, shape)

    def flatten(self):
        """
        Flatten the tensor to a 1D array.

        Returns:
            Tensor: The flattened tensor.

        """
        return ember.flatten(self)

    def append(self, t, axis=0):
        """
        Append another tensor along a specified axis.

        Args:
            t (Tensor): The tensor to append.
            axis (int): The axis along which to append.

        Returns:
            Tensor: The concatenated tensor.

        """
        return ember.append(self, t, axis=axis)

    def __repr__(self):
        """
        String representation of the tensor.

        Returns:
            str: The string representation.

        """
        string_data = tensor2string(self,
                                    prefix="       ",
                                    precision=4,
                                    separator=', ',
                                    floatmode='maxprec_equal',
                                    edgeitems=3,
                                    threshold=100,
                                    max_line_width=100)
        requires_grad = "" if not self.requires_grad else f", requires_grad={self.requires_grad}"
        return f"Tensor({string_data}{requires_grad}, <{self.device.lower()}>)"

    def __len__(self):
        """
        Return the length of the tensor.

        Returns:
            int: The length of the tensor.

        """
        return len(self.data)

    def __gt__(self, other):
        """
        Element-wise greater than comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.gt(self, other)

    def __ge__(self, other):
        """
        Element-wise greater than or equal to comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.ge(self, other)

    def __lt__(self, other):
        """
        Element-wise less than comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.lt(self, other)

    def __le__(self, other):
        """
        Element-wise less than or equal to comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.le(self, other)

    def __eq__(self, other):
        """
        Element-wise equality comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.eq(self, other)

    def __ne__(self, other):
        """
        Element-wise inequality comparison.

        Args:
            other (Tensor): The tensor for comparison.

        Returns:
            Tensor: Boolean tensor indicating the element-wise comparison.

        """
        return ember.ne(self, other)

    def __add__(self, other):
        """
        Element-wise addition.

        Args:
            other (Tensor): The tensor to add.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.add(self, other)

    def __radd__(self, other):
        """
        Right-side element-wise addition.

        Args:
            other (Tensor): The tensor for addition.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.add(other, self)

    def __iadd__(self, other):
        """
        In-place element-wise addition.

        Args:
            other (Tensor): The tensor to add in-place.

        Returns:
            Tensor: The updated tensor.

        """
        self.data = self.data + ember.to_tensor(other).data
        self._version += 1
        return self

    def __neg__(self):
        """
        Element-wise negation.

        Returns:
            Tensor: The negated tensor.

        """
        return ember.neg(self)

    def __sub__(self, other):
        """
        Element-wise subtraction.

        Args:
            other (Tensor): The tensor to subtract.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.sub(self, other)

    def __rsub__(self, other):
        """
        Right-side element-wise subtraction.

        Args:
            other (Tensor): The tensor for subtraction.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.sub(other, self)

    def __isub__(self, other):
        """
        In-place element-wise subtraction.

        Args:
            other (Tensor): The tensor to subtract in-place.

        Returns:
            Tensor: The updated tensor.

        """
        self.data = self.data - ember.to_tensor(other).data
        self._version += 1
        return self

    def __mul__(self, other):
        """
        Element-wise multiplication.

        Args:
            other (Tensor): The tensor to multiply.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.multiply(self, other)

    def __rmul__(self, other):
        """
        Right-side element-wise multiplication.

        Args:
            other (Tensor): The tensor for multiplication.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.multiply(other, self)

    def __imul__(self, other):
        """
        In-place element-wise multiplication.

        Args:
            other (Tensor): The tensor to multiply in-place.

        Returns:
            Tensor: The updated tensor.

        """
        self.data = self.data * ember.to_tensor(other).data
        self._version += 1
        return self

    def __pow__(self, power, modulo=None):
        """
        Element-wise exponentiation.

        Args:
            power (float): The exponent.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.pow(self, power)

    def __truediv__(self, other):
        """
        Element-wise true division.

        Args:
            other (Tensor): The tensor for division.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.div(self, other)

    def __rtruediv__(self, other):
        """
        Right-side element-wise true division.

        Args:
            other (Tensor): The tensor for division.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.div(other, self)

    def __itruediv__(self, other):
        """
        In-place element-wise true division.

        Args:
            other (Tensor): The tensor for division.

        Returns:
            Tensor: The updated tensor.

        """
        self.data = self.data / ember.to_tensor(other).data
        self._version += 1
        return self

    def __matmul__(self, other):
        """
        Matrix multiplication.

        Args:
            other (Tensor): The tensor for multiplication.

        Returns:
            Tensor: The resulting tensor.

        """
        return ember.dot(self, other)

    def __getitem__(self, indices):
        """
        Get a sub-tensor based on the provided indices.

        Args:
            indices: The indices for slicing.

        Returns:
            Tensor: The sliced tensor.

        """
        return ember.slice(self, indices)

    def __setitem__(self, key, value):
        """
        Set the value of the tensor based on the provided key.

        Args:
            key: The key for setting the value.
            value: The value to set.

        Returns:
            Tensor: The updated tensor.

        """
        return ember.set(self, key, value)
