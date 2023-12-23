import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

import ember
from ember.cuda import numpy_or_cupy
from ember.utils import dataset

def set(t, key, value):
    """
    Set values in a tensor based on the given key.

    Parameters:
    - t (ember.Tensor): Input tensor.
    - key: The key or keys specifying the indices to set.
    - value: The value to set in the tensor.

    Returns:
    - ember.Tensor: A new tensor with updated values.

    Explanation:
    - This function allows setting values in a tensor at specific indices specified by the key.

    """
    t = ember.to_tensor(t)
    value = ember.to_tensor(value)

    # Ensure tensors are on the same device.
    if t.device == 'cpu' and value.device != 'cpu':
        t.cuda()
    elif value.device == 'cpu' and t.device != 'cpu':
        value.cuda()

    cpu = True

    # Check if any key is a tensor and not on the CPU, then move everything to GPU.
    if isinstance(key, tuple):
        for k in key:
            if isinstance(k, ember.Tensor):
                if k.device != 'cpu':
                    cpu = False

    if not cpu:
        t.cuda()
        value.cuda()
        for k in key:
            if isinstance(k, ember.Tensor):
                k.cuda()

    # Convert key to indices and update tensor data.
    if isinstance(key, ember.Tensor):
        key = key.data
    elif isinstance(key, tuple):
        keys = []
        for k in key:
            if isinstance(k, ember.Tensor):
                keys.append(k.data)
            else:
                keys.append(k)
        key = tuple(keys)

    t.data[key] = value.data

    # Detach the tensor to avoid tracking operations on it.
    t.detach()

    return t


def gt(t, other):
    """
    Element-wise greater-than comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t > other, False otherwise.

    Explanation:
    - This function performs an element-wise greater-than comparison between t and other.

    """
    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    data = t.data > other.data
    return ember.Tensor(data, device=t.device)


def ge(t, other):
    """
    Element-wise greater-than-or-equal comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t >= other, False otherwise.

    Explanation:
    - This function performs an element-wise greater-than-or-equal comparison between t and other.

    """
    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    data = t.data >= other.data
    return ember.Tensor(data, device=t.device)


def lt(t, other):
    """
    Element-wise less-than comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t < other, False otherwise.

    Explanation:
    - This function performs an element-wise less-than comparison between t and other.

    """
    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    data = t.data < other.data
    return ember.Tensor(data, device=t.device)


def le(t, other):
    """
    Element-wise less-than-or-equal comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t <= other, False otherwise.

    Explanation:
    - Performs an element-wise less-than-or-equal comparison between t and other.
    - Returns a new tensor with True where t <= other, and False otherwise.

    """

    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    data = t.data <= other.data
    return ember.Tensor(data, device=t.device)


def eq(t, other):
    """
    Element-wise equality comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t == other, False otherwise.

    Explanation:
    - Performs an element-wise equality comparison between t and other.
    - Returns a new tensor with True where t == other, and False otherwise.

    """

    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    cond = t.data == other.data
    return ember.Tensor(cond, device=t.device)


def ne(t, other):
    """
    Element-wise inequality comparison between two tensors.

    Parameters:
    - t (ember.Tensor): First input tensor.
    - other: Second input tensor or scalar.

    Returns:
    - ember.Tensor: A new tensor with True where t != other, False otherwise.

    Explanation:
    - Performs an element-wise inequality comparison between t and other.
    - Returns a new tensor with True where t != other, and False otherwise.

    """

    t = ember.to_tensor(t)
    other = ember.to_tensor(other)
    data = not t.data == other.data
    return ember.Tensor(data, device=t.device)


def unravel_index(indices, shape, order='C', requires_grad=False, device='cpu'):
    """
    Converts flat indices to multi-dimensional indices.

    Parameters:
    - indices: Flat indices.
    - shape: The shape of the multi-dimensional array.
    - order (str): Order of the indices ('C' for row-major, 'F' for column-major).
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').

    Returns:
    - ember.Tensor: A new tensor with multi-dimensional indices.

    Explanation:
    - Converts flat indices to multi-dimensional indices based on the specified shape and order.
    - The resulting tensor will have the same shape as the input shape.

    """

    if device == 'cpu':
        data = np.unravel_index(indices, shape, order=order)
    else:
        data = cp.unravel_index(indices, shape, order=order)
    return ember.Tensor(data, requires_grad=requires_grad, device=device)


def rollaxis(t, axis, start=0):
    """
    Roll the specified axis backwards.

    Parameters:
    - t (ember.Tensor): Input tensor.
    - axis (int): The axis to roll.
    - start (int): The position to start the roll.

    Returns:
    - ember.Tensor: A new tensor with the specified axis rolled.

    Explanation:
    - Rolls the specified axis of the input tensor backwards (along the specified position).
    - If start is 0, the default, it is placed as the first axis in the output tensor.
    - If start is negative, it is placed as the last axis in the output tensor.

    """

    nc = numpy_or_cupy(t)
    data = nc.rollaxis(t.data, axis, start=start)
    return ember.Tensor(data, requires_grad=t.requires_grad, device=t.device)


def zeros(shape, requires_grad=False, device='cpu', **kwargs):
    """
    Create a tensor filled with zeros.

    Parameters:
    - shape (tuple): The shape of the tensor.
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new tensor filled with zeros.

    Explanation:
    - Creates a tensor filled with zeros based on the specified shape.
    - Additional arguments are passed to the backend library for customization.

    """

    if device == 'cpu':
        data = np.zeros(shape, **kwargs)
    else:
        data = cp.zeros(shape, **kwargs)
    return ember.Tensor(data, requires_grad=requires_grad, device=device)


def zeros_like(t, **kwargs):
    """
    Create a tensor of zeros with the same shape as the input tensor.

    Parameters:
    - t (ember.Tensor): The input tensor.
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new tensor of zeros with the same shape as the input tensor.

    Explanation:
    - Creates a tensor of zeros with the same shape as the input tensor.
    - Additional arguments are passed to the backend library for customization.

    """

    return zeros(t.shape, requires_grad=t.requires_grad, device=t.device, **kwargs)


def ones(shape, requires_grad=False, device='cpu', **kwargs):
    """
    Create a tensor filled with ones.

    Parameters:
    - shape (tuple): The shape of the tensor.
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new tensor filled with ones.

    Explanation:
    - Creates a tensor filled with ones based on the specified shape.
    - Additional arguments are passed to the backend library for customization.

    """

    if device == 'cpu':
        data = np.ones(shape, **kwargs)
    else:
        data = cp.ones(shape, **kwargs)
    return ember.Tensor(data, requires_grad=requires_grad, device=device)


def ones_like(t, **kwargs):
    """
    Create a tensor of ones with the same shape as the input tensor.

    Parameters:
    - t (ember.Tensor): The input tensor.
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new tensor of ones with the same shape as the input tensor.

    Explanation:
    - Creates a tensor of ones with the same shape as the input tensor.
    - Additional arguments are passed to the backend library for customization.

    """

    return ones(t.shape, requires_grad=t.requires_grad, device=t.device, **kwargs)


def eye(size, requires_grad=False, device='cpu', **kwargs):
    """
    Create a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters:
    - size (int): The size of the square tensor.
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new 2-D tensor with ones on the diagonal.

    Explanation:
    - Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.
    - Additional arguments are passed to the backend library for customization.

    """

    if device == 'cpu':
        data = np.eye(size, **kwargs)
    else:
        data = cp.eye(size, **kwargs)
    return ember.Tensor(data, requires_grad=requires_grad, device=device)


def identity(size, requires_grad=False, device='cpu', **kwargs):
    """
    Create a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters:
    - size (int): The size of the square tensor.
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new 2-D tensor with ones on the diagonal.

    Explanation:
    - Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.
    - Additional arguments are passed to the backend library for customization.

    """

    if device == 'cpu':
        data = np.identity(size, **kwargs)
    else:
        data = cp.identity(size, **kwargs)
    return ember.Tensor(data, requires_grad=requires_grad, device=device, **kwargs)


def arange(*args, requires_grad=False, device='cpu', **kwargs):
    """
    Create a tensor with regularly spaced values.

    Parameters:
    - *args: Arguments specifying the range of values (start, stop, step, dtype).
    - requires_grad (bool): Whether to track gradients for the resulting tensor.
    - device (str): Device to store the resulting tensor ('cpu' or 'cuda').
    - **kwargs: Additional arguments to pass to the backend library (NumPy or CuPy).

    Returns:
    - ember.Tensor: A new tensor with regularly spaced values.

    Explanation:
    - Creates a tensor with regularly spaced values based on the specified arguments.
    - Additional arguments are passed to the backend library for customization.

    """

    if device == 'cpu':
        data = np.arange(*args, **kwargs)
    else:
        data = cp.arange(*args, **kwargs)

    return ember.Tensor(data, requires_grad=requires_grad, device=device)


def astype(t, new_type):
    """
    Cast the tensor to a new data type.

    Parameters:
    - t (ember.Tensor): Input tensor.
    - new_type: Desired data type for the tensor.

    Returns:
    - ember.Tensor: A new tensor with the specified data type.

    Explanation:
    - Casts the input tensor to the specified data type.
    - The resulting tensor has the same shape as the input tensor.

    """

    data = t.data.astype(new_type)
    return ember.Tensor(data, requires_grad=t.requires_grad, device=t.device)
