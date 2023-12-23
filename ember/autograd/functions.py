import numpy as np
import ember
from .hook import Hook


def where(cond, t1, t2):
    """
    Element-wise conditional operation.

    Args:
    - cond: Condition array.
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor based on the condition.
    """
    t1 = ember.to_tensor(t1)  # Convert the first input tensor to a Tensor
    t2 = ember.to_tensor(t2)  # Convert the second input tensor to a Tensor

    # Ensure that tensors have the same shape
    assert t1.shape == t2.shape, f"tensors should have the same shape. Got t1.shape={t1.shape}, t2.shape={t2.shape}"

    cond = ember.to_array(cond)  # Convert the condition to a NumPy array
    # Apply the conditional operation element-wise
    data = np.where(cond, t1.data, t2.data)
    # Set requires_grad based on input tensors
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []  # List to store hooks for gradient computation

    # Register hooks for gradient computation if the input tensors require it
    if t1.requires_grad:
        def grad_fn(grad):
            return grad * np.where(cond, 1, 0)  # Gradient computation for t1
        hooks.append(Hook(t1, grad_fn))

    if t2.requires_grad:
        def grad_fn(grad):
            return grad * np.where(cond, 0, 1)  # Gradient computation for t2
        hooks.append(Hook(t2, grad_fn))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def maximum(t1, t2):
    """
    Element-wise maximum operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor with maximum values element-wise.
    """
    return where(t1 > t2, t1, t2)  # Utilize where function for element-wise maximum


def minimum(t1, t2):
    """
    Element-wise minimum operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor with minimum values element-wise.
    """
    return where(t1 > t2, t2, t1)  # Utilize where function for element-wise minimum


def pow(t, power):
    """
    Element-wise power operation.

    Args:
    - t: Input tensor.
    - power: Exponent value (int).

    Returns:
    - Tensor: Resultant tensor with element-wise power operation.
    """
    assert type(power) == int, "unsupported type {} for power. Currently supported type: int".format(
        type(power))
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = t.data ** power  # Element-wise power operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []  # List to store hooks for gradient computation

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * power * t.data ** (power - 1)))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def sqrt(t):
    """
    Element-wise square root operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise square root operation.
    """
    t = ember.to_tensor(t)  # Convert the input tensor to a Tensor
    data = np.sqrt(t.data)  # Element-wise square root operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []  # List to store hooks for gradient computation

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: -1 / (2 * np.sqrt(t.data)) * grad))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def exp(t):
    """
    Element-wise exponential operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise exponential operation.
    """
    data = np.exp(t.data)  # Element-wise exponential operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []  # List to store hooks for gradient computation

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * data))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def log(t):
    """
    Element-wise natural logarithm operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise natural logarithm operation.
    """
    data = np.log(t.data)  # Element-wise natural logarithm operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []  # List to store hooks for gradient computation

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * np.divide(1, t.data)))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def softmax(x, axis=0):
    """
    Softmax operation along the specified axis.

    Args:
    - x: Input tensor.
    - axis: Axis along which the softmax is computed (default is 0).

    Returns:
    - Tensor: Resultant tensor with softmax operation applied along the specified axis.
    """
    e = ember.exp(x)  # Exponential operation element-wise
    # Sum of exponential values along the specified axis
    s = ember.sum(e, axis=axis, keepdims=True)
    t = x - ember.log(s)  # Logarithm of the softmax denominator
    soft = ember.exp(t)  # Exponential operation on the adjusted input
    return soft  # Return the softmax result


def tanh(t):
    """
    Element-wise hyperbolic tangent operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise hyperbolic tangent operation.
    """
    data = np.tanh(t.data)  # Element-wise hyperbolic tangent operation
    requires_grad = t.requires_grad  # Set requires_grad based on input tensor
    hooks = []  # List to store hooks for gradient computation

    # Register a hook for gradient computation if the input tensor requires it
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * (1 - data * data)))

    return ember.Tensor(data, requires_grad, hooks)  # Return the resultant tensor


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent function.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise derivative of tanh.
    """
    return 1 - np.power(2, np.tanh(x))  # Element-wise tanh derivative computation


def sigmoid(x):
    """
    Element-wise sigmoid operation.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise sigmoid operation.
    """
    return 1.0 / (1.0 + ember.exp(-x))  # Element-wise sigmoid operation


def sigmoid_prime(x):
    """
    Derivative of the sigmoid function.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise derivative of sigmoid.
    """
    return sigmoid(x) * (1 - sigmoid(x))  # Element-wise sigmoid derivative computation


def relu(x):
    """
    Element-wise Rectified Linear Unit (ReLU) operation.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise ReLU operation.
    """
    return maximum(ember.zeros_like(x), x)  # Element-wise ReLU operation


def relu_prime(x):
    """
    Derivative of the Rectified Linear Unit (ReLU) function.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor: Resultant tensor with element-wise derivative of ReLU.
    """
    return where(x >= 0, ember.ones_like(x), ember.zeros_like(x))  # Element-wise ReLU derivative computation


def leaky_relu(x, alpha=0.01):
    """
    Element-wise Leaky ReLU operation.

    Args:
    - x: Input tensor.
    - alpha: Slope of the negative region (default is 0.01).

    Returns:
    - Tensor: Resultant tensor with element-wise Leaky ReLU operation.
    """
    return where(x > 0, x, x * alpha)  # Element-wise Leaky ReLU operation


def leaky_relu_prime(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function.

    Args:
    - x: Input tensor.
    - alpha: Slope of the negative region (default is 0.01).

    Returns:
    - Tensor: Resultant tensor with element-wise derivative of Leaky ReLU.
    """
    return where(x > 0, ember.ones_like(x), alpha * ember.ones_like(x))  # Element-wise Leaky ReLU derivative computation
