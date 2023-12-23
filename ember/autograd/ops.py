import ember
import numpy as np
from .hook import Hook


def sum(t, axis=None, keepdims=False):
    """
    Calculate the sum of elements along a specified axis.

    Parameters:
    - t (ember.Tensor): Input tensor.
    - axis (int or tuple of ints, optional): Axis or axes along which a sum is performed. Default is None,
      meaning all elements will be summed.
    - keepdims (bool, optional): If True, the reduced axes are retained with a size of 1. Default is False.

    Returns:
    - ember.Tensor: A new tensor containing the sum along the specified axis or axes.

    Math:
    - Let A be the input tensor.
    - If axis is None, the output tensor B is calculated as B = sum(A).
    - If axis is specified, B[i, j, k, ...] = sum(A[:, :, :, ..., i, j, k, ...]).

    """
    data = t.data.sum(axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        def grad_fn(grad):
            # Broadcasting to match the shape of the input tensor.
            data_keepdims = t.data.sum(axis=axis, keepdims=True)
            return grad.reshape(data_keepdims.shape) + np.zeros_like(t.data)

        hooks.append(Hook(t, grad_fn))

    return ember.Tensor(data, requires_grad, hooks)


def add(t1, t2):
    """
    Element-wise addition of two tensors.

    Parameters:
    - t1 (ember.Tensor): First input tensor.
    - t2 (ember.Tensor): Second input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise sum of t1 and t2.

    Math:
    - Let A and B be the input tensors.
    - The output tensor C is calculated as C[i, j, k, ...] = A[i, j, k, ...] + B[i, j, k, ...].

    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            # Reducing dimensions added during broadcasting.
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            # Reducing dimensions added during broadcasting.
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t2, grad_fn2))

    return ember.Tensor(data, requires_grad, hooks)


def neg(t):
    """
    Element-wise negation of a tensor.

    Parameters:
    - t (ember.Tensor): Input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise negation of the input tensor.

    Math:
    - Let A be the input tensor.
    - The output tensor B is calculated as B[i, j, k, ...] = -A[i, j, k, ...].

    """
    t = ember.to_tensor(t)
    data = -t.data
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        hooks.append(Hook(t, lambda grad: -grad))

    return ember.Tensor(data, requires_grad, hooks)


def sub(t1, t2):
    """
    Element-wise subtraction of two tensors.

    Parameters:
    - t1 (ember.Tensor): First input tensor.
    - t2 (ember.Tensor): Second input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise subtraction of t2 from t1.

    Math:
    - Let A and B be the input tensors.
    - The output tensor C is calculated as C[i, j, k, ...] = A[i, j, k, ...] - B[i, j, k, ...].

    """
    return add(t1, neg(t2))


def multiply(t1, t2):
    """
    Element-wise multiplication of two tensors.

    Parameters:
    - t1 (ember.Tensor): First input tensor.
    - t2 (ember.Tensor): Second input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise product of t1 and t2.

    Math:
    - Let A and B be the input tensors.
    - The output tensor C is calculated as C[i, j, k, ...] = A[i, j, k, ...] * B[i, j, k, ...].

    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    data = np.multiply(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            # Element-wise multiplication with the data of the other tensor.
            grad = grad * t2.data

            # Reducing dimensions added during broadcasting.
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            # Element-wise multiplication with the data of the other tensor.
            grad = grad * t1.data

            # Reducing dimensions added during broadcasting.
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t2, grad_fn2))

    return ember.Tensor(data, requires_grad, hooks)


def inverse(t):
    """
    Element-wise inversion of a tensor.

    Parameters:
    - t (ember.Tensor): Input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise inverse of the input tensor.

    Math:
    - Let A be the input tensor.
    - The output tensor B is calculated as B[i, j, k, ...] = 1 / A[i, j, k, ...].

    """
    t = ember.to_tensor(t)
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        def grad_fn(grad):
            # Element-wise gradient calculation for the inverse.
            return -1 / (t.data ** 2) * grad

        hooks.append(Hook(t, grad_fn))

    return ember.Tensor(1 / t.data, requires_grad, hooks)


def div(t1, t2):
    """
    Element-wise division of two tensors.

    Parameters:
    - t1 (ember.Tensor): Dividend tensor.
    - t2 (ember.Tensor): Divisor tensor.

    Returns:
    - ember.Tensor: A new tensor containing the element-wise division of t1 by t2.

    Math:
    - Let A and B be the input tensors.
    - The output tensor C is calculated as C[i, j, k, ...] = A[i, j, k, ...] / B[i, j, k, ...].

    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)

    # Element-wise division using the multiply and inverse functions.
    return multiply(t1, inverse(t2))


def dot(t1, t2):
    """
    Dot product of two tensors.

    Parameters:
    - t1 (ember.Tensor): First input tensor.
    - t2 (ember.Tensor): Second input tensor.

    Returns:
    - ember.Tensor: A new tensor containing the dot product of t1 and t2.

    Math:
    - Let A and B be the input tensors.
    - The dot product C is calculated as C = A @ B.

    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            # Gradient calculation for the first tensor.
            return grad @ t2.data.T

        hooks.append(Hook(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            # Gradient calculation for the second tensor.
            return t1.data.T @ grad

        hooks.append(Hook(t2, grad_fn2))

    return ember.Tensor(data, requires_grad, hooks)


def slice(t, indices):
    """
    Slice a tensor based on the given indices.

    Parameters:
    - t (ember.Tensor): Input tensor.
    - indices: The indices used to slice the tensor.

    Returns:
    - ember.Tensor: A new tensor containing the sliced data.

    Math:
    - Let A be the input tensor.
    - The output tensor B is calculated as B[i] = A[indices[i]].

    """
    t = ember.to_tensor(t)

    if isinstance(indices, ember.Tensor):
        indices = indices.data

    # Slicing the tensor based on the given indices.
    data = t.data[indices]
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        def grad_fn(grad):
            # Gradient calculation for the sliced tensor.
            bigger_grad = np.zeros_like(t.data)
            if grad.shape != bigger_grad.shape:
                bigger_grad[indices] = grad
            else:
                bigger_grad = grad
            return bigger_grad

        hooks.append(Hook(t, grad_fn))

    return ember.Tensor(data, requires_grad, hooks)


def gt(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.

      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.

      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding element in `t` is greater than the corresponding element in `other`, and False otherwise.

      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 2
        >>> ember.gt(t, other)
        ember.tensor([True, True, True])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for greater than (`>`)
    cond = t > other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)


def set(t, key, value):
    """
      Sets the value of a specific element or subsequence in a tensor `t` based on a given key.

      Args:
        t: The tensor to modify.
        key: A single index, a tuple of indices, or a tensor containing indices specifying the location to set the value.
        value: The new value to set at the specified location in the tensor.

      Returns:
        The modified tensor `t` with the updated value at the specified location.

      Example:
        >>> t = ember.tensor([[1, 2, 3], [4, 5, 6]])
        >>> ember.set(t, (0, 1), 10)
        ember.tensor([[1, 10, 3], [4, 5, 6]])

      Notes:
        * If `key` is a single index, it sets the corresponding element in the tensor.
        * If `key` is a tuple of indices, it sets the corresponding subsequence at those indices.
        * If `key` is a tensor containing indices, it sets the elements at those indices based on the corresponding elements in the `value` tensor.
      """

    # Check if the key is a tensor and extract its data if necessary
    if isinstance(key, ember.Tensor):
        key = key.data

    # Handle multiple key components for subsequence modification
    elif isinstance(key, tuple):
        keys = []
        for k in key:
          if isinstance(k, ember.Tensor):
            keys.append(k.data)
          else:
            keys.append(k)
        key = tuple(keys)

    # Access and update the data within the tensor using the provided key
    t.data[key] = ember.to_tensor(value).data

    # Detach the updated tensor to prevent unwanted interactions with the original graph
    t.detach()

    # Return the modified tensor
    return t


def ge(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.
    
      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.
    
      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding element in `t` is greater than or equal to the corresponding element in `other`, and False otherwise.
    
      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 3
        >>> ember.ge(t, other)
        ember.tensor([True, True, True])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for greater than or equal to (`>=`)
    cond = t >= other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)


def lt(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.
    
      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.
    
      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding element in `t` is less than the corresponding element in `other`, and False otherwise.
    
      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 3
        >>> ember.lt(t, other)
        ember.tensor([True, False, False])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for less than (`<`)
    cond = t < other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)


def le(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.
    
      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.
    
      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding element in `t` is less than or equal to the corresponding element in `other`, and False otherwise.
    
      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 3
        >>> ember.le(t, other)
        ember.tensor([True, True, False])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for less than or equal to (`<=`)
    cond = t <= other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)


def eq(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.
    
      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.
    
      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding elements in `t` and `other` are equal, and False otherwise.
    
      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 3
        >>> ember.eq(t, other)
        ember.tensor([False, True, False])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for equality (`==`)
    cond = t == other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)


def ne(t, other):
    """
      Compares a tensor `t` with another tensor or scalar `other` element-wise, returning a new tensor of boolean values.
    
      Args:
        t: A tensor to be compared.
        other: Another tensor or scalar value to compare with `t`.
    
      Returns:
        A new tensor of the same shape as `t`, with True values at indices where the corresponding elements in `t` and `other` are not equal, and False otherwise.
    
      Example:
        >>> t = ember.tensor([1, 3, 5])
        >>> other = 3
        >>> ember.ne(t, other)
        ember.tensor([True, False, True])
      """

    # Convert both arguments to NumPy arrays for element-wise comparison
    t = ember.to_array(t)
    other = ember.to_array(other)

    # Perform element-wise comparison for not equal (`!=`)
    cond = not t == other

    # Convert the boolean comparison result back to an ember tensor
    return ember.to_tensor(cond)
