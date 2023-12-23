import numpy as np
import ember
from .hook import Hook
from ._utils import numpy_unpad, inv_permutation


def _T(t):
    """
    Transpose operation on the input tensor.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with the transpose operation applied.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = t.data.T  # Transpose operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad.T))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def transpose(t, indices=None):
    """
    Transpose operation on the input tensor with optional permutation of dimensions.

    Args:
    - t: Input tensor.
    - indices: Optional tuple specifying the permutation of dimensions (default is reverse order).

    Returns:
    - Tensor: Resultant tensor with the transpose operation applied.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor

    # If indices is not provided, use the reverse order of dimensions
    if indices is None:
        indices = tuple(range(t.ndim - 1, -1, -1))

    data = t.data.transpose(indices)  # Transpose operation with optional permutation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        def grad_fn(grad):
            # Inverse permutation of dimensions
            indices_back = tuple(inv_permutation(indices))
            grad = grad.transpose(indices_back)  # Transpose the gradient
            return grad

        hooks.append(Hook(t, grad_fn))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def reshape(t, shape):
    """
    Reshape operation on the input tensor.

    Args:
    - t: Input tensor.
    - shape: New shape for the tensor.

    Returns:
    - Tensor: Resultant tensor with the reshape operation applied.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = t.data.reshape(shape)  # Reshape operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad.reshape(t.shape)))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def pad(t, padding, constant_values=0):
    """
    Pad operation on the input tensor.

    Args:
    - t: Input tensor.
    - padding: Padding configuration.
    - constant_values: Constant value for padding (default is 0).

    Returns:
    - Tensor: Resultant tensor with the pad operation applied.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = np.pad(t.data, pad_width=padding,
                  constant_values=constant_values)  # Pad operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: numpy_unpad(grad, padding)))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def max(t, axis=None):
    """
    Maximum operation along the specified axis.

    Args:
    - t: Input tensor.
    - axis: Axis along which the maximum is computed (default is None, computes the global maximum).

    Returns:
    - Tensor: Resultant tensor with the maximum operation applied along the specified axis.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = np.max(t.data, axis=axis)  # Maximum operation along the specified axis
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        def grad_fn(grad):
            bigger_grad = np.zeros_like(t.data)

            # If axis is None, compute the global maximum
            if axis is None:
                max_indices = np.unravel_index(np.argmax(t.data), t.shape)
                bigger_grad[max_indices] = grad
            else:
                max_indices = np.argmax(t.data, axis=axis)

                # Roll the indices back to the original shape
                for i, roll in enumerate(np.rollaxis(bigger_grad, axis)):
                    roll += (max_indices == i).astype(int) * grad

            return bigger_grad

        hooks.append(Hook(t, grad_fn))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def argmax(t, axis=None):
    """
    Computes the index of the maximum value along the specified axis.

    Args:
    - t: Input tensor.
    - axis: Axis along which to find the maximum value (default is None, which finds the global maximum).

    Returns:
    - Tensor: Resultant tensor containing the index of the maximum value along the specified axis.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor

    if axis is None:
        # If axis is None, find the global maximum and return the unravelled index
        return ember.Tensor(np.unravel_index(np.argmax(t.data), t.shape))
    else:
        # Find the maximum value along the specified axis and return the index
        return ember.Tensor(np.argmax(t.data, axis=axis))


def flatten(t):
    """
    Flatten operation on the input tensor.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with the flatten operation applied.
    """
    return reshape(t, (t.size,))  # Reshape the input tensor to a flat shape


ITERABLE = (list, tuple)


def concatenate(iterable):
    """
    Concatenate operation on a list or tuple of tensors.

    Args:
    - iterable: Input iterable containing tensors.

    Returns:
    - Tensor: Resultant tensor with the concatenate operation applied.
    """
    assert isinstance(iterable, ITERABLE), f'iterable type {type(iterable)} unsupported for `concatenate` function.' \
                                           f'Types currently supported are list, tuple.'

    requires_grad = False  # Initialize requires_grad flag
    hooks = []  # Initialize list to store gradient computation hooks
    data = np.array([])  # Initialize data array for concatenated tensor

    # Iterate over the input iterable
    for idx, t in enumerate(iterable):
        t = ember.to_tensor(t)  # Convert each element in the iterable to a Tensor
        requires_grad = t.requires_grad or requires_grad  # Update requires_grad flag

        if data.size == 0:
            data = t.data  # If data is empty, set it to the data of the current tensor
        else:
            # Concatenate the data along the appropriate axis
            data = np.concatenate((data, t.data))

        if t.requires_grad:
            # If the current tensor requires gradient, register a gradient computation hook
            def grad_fn(grad):
                return grad[idx:idx+t.shape[0]]

            hooks.append(Hook(t, grad_fn))  # Append the hook to the list

    return ember.Tensor(data, requires_grad, hooks)  # Return the concatenated tensor


def append(t, value):
    """
    Append operation to add a tensor or value to the end of another tensor.

    Args:
    - t: Input tensor.
    - value: Tensor or value to be appended.

    Returns:
    - Tensor: Resultant tensor with the append operation applied.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    value = ember.to_tensor(value)  # Convert the value to a Tensor
    requires_grad = False  # Initialize requires_grad flag
    hooks = []  # Initialize list to store gradient computation hooks
    requires_grad = t.requires_grad or value.requires_grad  # Update requires_grad flag

    # Handle cases where either t or value has size 0
    if t.size == 0:
        data = [value.data]
    elif value.size == 0:
        data = [t.data]
    else:
        data = t.data.tolist()
        data.append(value.data)

    if t.requires_grad:
        # If the input tensor requires gradient, register a gradient computation hook
        def grad_fn(grad):
            return grad[:-1]

        hooks.append(Hook(t, grad_fn))  # Append the hook to the list

    if value.requires_grad:
        # If the value tensor requires gradient, register a gradient computation hook
        def grad_fn(grad):
            return grad[-1]

        hooks.append(Hook(value, grad_fn))  # Append the hook to the list

    # Return the tensor after append operation
    return ember.Tensor(data, requires_grad, hooks)
