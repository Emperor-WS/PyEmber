import numpy as np
from ember.cuda import numpy_or_cupy
import ember
import ember.autograd.function as fc
import ember.autograd.operation as op


def transpose(t, indices=None):
    """
    Transpose operation.

    Args:
    - t: Input tensor.
    - indices: Optional indices for transposition.

    Returns:
    - Tensor: Transposed tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Transpose(indices)
    return func(t)


def reshape(t, shape):
    """
    Reshape operation.

    Args:
    - t: Input tensor.
    - shape: Target shape for the tensor.

    Returns:
    - Tensor: Reshaped tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Reshape(shape)
    return func(t)


def flatten(t):
    """
    Flatten operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Flattened tensor.
    """
    return reshape(t, (t.size,))


def pad(t, padding):
    """
    Padding operation.

    Args:
    - t: Input tensor.
    - padding: Padding configuration.

    Returns:
    - Tensor: Padded tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Pad(padding)
    return func(t)


def unpad(t, padding):
    """
    Unpadding operation.

    Args:
    - t: Input tensor.
    - padding: Padding configuration.

    Returns:
    - Tensor: Unpadded tensor.
    """
    slices = []
    for pad in padding:
        e = None if pad[1] == 0 else -pad[1]
        slices.append(slice(pad[0], e))
    return t[tuple(slices)]


def max(t, axis=None):
    """
    Maximum operation.

    Args:
    - t: Input tensor.
    - axis: Axis along which the maximum is computed.

    Returns:
    - Tensor: Maximum tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Max(axis)
    return func(t)


def argmax(t, axis=None):
    """
    Argmax operation.

    Args:
    - t: Input tensor.
    - axis: Axis along which the argmax is computed.

    Returns:
    - Tensor: Tensor containing the indices of the maximum values.
    """
    t = ember.to_tensor(t)
    nc = numpy_or_cupy(t)
    if axis is None:
        data = nc.unravel_index(nc.argmax(t.data), t.shape)
    else:
        data = nc.argmax(t.data, axis=axis)
    return ember.Tensor(data, device=t.device)


def neg(t):
    """
    Negation operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Negated tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Neg()
    return func(t)


def sum(t, axis=None, keepdims=False):
    """
    Summation operation.

    Args:
    - t: Input tensor.
    - axis: Axis along which the summation is computed.
    - keepdims: Whether to keep the dimensions.

    Returns:
    - Tensor: Summation result.
    """
    t = ember.to_tensor(t)
    func = fc.Sum(axis=axis, keepdims=keepdims)
    return func(t)


def add(t1, t2):
    """
    Addition operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor after addition.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    operation = op.Add()
    return operation(t1, t2)


def sub(t1, t2):
    """
    Subtraction operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor after subtraction.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    return add(t1, neg(t2))


def multiply(t1, t2):
    """
    Multiplication operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor after multiplication.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    operation = op.Multiply()
    return operation(t1, t2)


def inverse(t):
    """
    Inverse operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after inversion.
    """
    t = ember.to_tensor(t)
    func = fc.Inverse()
    return func(t)


def div(t1, t2):
    """
    Division operation.

    Args:
    - t1: Numerator tensor.
    - t2: Denominator tensor.

    Returns:
    - Tensor: Resultant tensor after division.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    return multiply(t1, inverse(t2))


def dot(t1, t2):
    """
    Dot product operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor after dot product.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    operation = op.Dot()
    return operation(t1, t2)


def slice(t, indices):
    """
    Slice operation.

    Args:
    - t: Input tensor.
    - indices: Indices for slicing.

    Returns:
    - Tensor: Sliced tensor.
    """
    t = ember.to_tensor(t)
    func = fc.Slice(indices)
    return func(t)


def where(cond, t1, t2):
    """
    Conditional selection operation.

    Args:
    - cond: Conditional tensor.
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Resultant tensor based on the condition.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    operation = op.Where(cond)
    return operation(t1, t2)


def maximum(t1, t2):
    """
    Maximum operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Maximum tensor.
    """
    t1 = ember.to_tensor(t1)
    t2 = ember.to_tensor(t2)
    return where(t1 > t2, t1, t2)


def minimum(t1, t2):
    """
    Minimum operation.

    Args:
    - t1: First input tensor.
    - t2: Second input tensor.

    Returns:
    - Tensor: Minimum tensor.
    """
    return where(t1 > t2, t2, t1)


def pow(t, power):
    """
    Power operation.

    Args:
    - t: Input tensor.
    - power: Exponent.

    Returns:
    - Tensor: Resultant tensor after exponentiation.
    """
    t = ember.to_tensor(t)
    func = fc.Pow(power)
    return func(t)


def sqrt(t):
    """
    Square root operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after square root.
    """
    t = ember.to_tensor(t)
    func = fc.Sqrt()
    return func(t)


def exp(t):
    """
    Exponential operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after exponential.
    """
    t = ember.to_tensor(t)
    func = fc.Exp()
    return func(t)


def log(t):
    """
    Logarithm operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after logarithm.
    """
    t = ember.to_tensor(t)
    func = fc.Log()
    return func(t)


def softmax(t, axis=0):
    """
    Softmax operation.

    Args:
    - t: Input tensor.
    - axis: Axis along which softmax is computed.

    Returns:
    - Tensor: Resultant tensor after softmax.
    """
    t = ember.to_tensor(t)
    s = sum(exp(t), axis=axis, keepdims=True)
    return exp(t - log(s))


def tanh(t):
    """
    Hyperbolic tangent operation.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after hyperbolic tangent.
    """
    t = ember.to_tensor(t)
    func = fc.Tanh()
    return func(t)


def tanh_prime(t):
    """
    Derivative of the hyperbolic tangent (tanh) function.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after applying the derivative of tanh.
    """
    t = ember.to_tensor(t)
    return 1 - pow(tanh(t), 2)


def sigmoid(t):
    """
    Sigmoid activation function.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after applying the sigmoid function.
    """
    t = ember.to_tensor(t)
    return 1.0 / (1.0 + exp(-t))


def sigmoid_prime(t):
    """
    Derivative of the sigmoid activation function.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after applying the derivative of sigmoid.
    """
    t = ember.to_tensor(t)
    return sigmoid(t) * (1 - sigmoid(t))


def relu(t):
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after applying the ReLU function.
    """
    t = ember.to_tensor(t)
    return maximum(ember.zeros_like(t), t)


def relu_prime(t):
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function.

    Args:
    - t: Input tensor.

    Returns:
    - Tensor: Resultant tensor after applying the derivative of ReLU.
    """
    t = ember.to_tensor(t)
    return where(t >= 0, ember.ones_like(t), ember.zeros_like(t))


def leaky_relu(t, alpha=0.01):
    """
    Leaky ReLU activation function.

    Args:
    - t: Input tensor.
    - alpha: Slope for negative values (default is 0.01).

    Returns:
    - Tensor: Resultant tensor after applying the Leaky ReLU function.
    """
    t = ember.to_tensor(t)
    return where(t > 0, t, t * alpha)


def leaky_relu_prime(t, alpha=0.01):
    """
    Derivative of the Leaky ReLU activation function.

    Args:
    - t: Input tensor.
    - alpha: Slope for negative values (default is 0.01).

    Returns:
    - Tensor: Resultant tensor after applying the derivative of Leaky ReLU.
    """
    t = ember.to_tensor(t)
    return where(t > 0, ember.ones_like(t), alpha * ember.ones_like(t))


ITERABLE_TYPES = (list, tuple)


def concatenate(iterable):
    """
    Concatenates tensors along a new axis.

    Args:
    - iterable: An iterable containing tensors (list or tuple).

    Returns:
    - Tensor: Resultant tensor after concatenation.
    """
    assert isinstance(iterable, ITERABLE_TYPES), (
        f'iterable type {type(iterable)} unsupported for `concatenate` function.'
        f'Types currently supported are list, tuple.'
    )

    requires_grad = False  # Flag to track if any tensor in the iterable has requires_grad=True
    hooks = []  # List to store hooks for gradient computation
    # Determine whether to use NumPy or CuPy based on input tensors
    nc = numpy_or_cupy(*iterable)
    data = nc.array([])  # Initialize an empty array to store concatenated data

    # Iterate over tensors in the iterable
    for idx, t in enumerate(iterable):
        t = ember.to_tensor(t)  # Convert the element to a Tensor
        requires_grad = t.requires_grad or requires_grad  # Update requires_grad flag

        # Concatenate tensor data along the new axis
        if data.size == 0:
            data = t.data
        else:
            data = nc.concatenate((data, t.data))

        # Register a hook for gradient computation if the tensor requires it
        if t.requires_grad:
            def grad_fn(grad):
                return grad[idx:idx + t.shape[0]]

            hooks.append(ember.Hook(t, grad_fn))

    # Create a new Tensor with the concatenated data and set requires_grad based on the flag
    tensor = ember.Tensor(data, requires_grad, device=iterable[0].device)

    # Register hooks for gradient computation
    for hook in hooks:
        tensor.register_hook(hook)

    return tensor


def append(t, value):
    """
    Appends a value to a tensor along a new axis.

    Args:
    - t: Input tensor.
    - value: Value to be appended.

    Returns:
    - Tensor: Resultant tensor after appending the value.
    """
    t = ember.to_tensor(t)  # Convert input tensor to a Tensor
    value = ember.to_tensor(value)  # Convert the value to a Tensor
    requires_grad = t.requires_grad or value.requires_grad  # Update requires_grad flag

    # Initialize a list to store data for the new tensor
    if t.size == 0:
        data = [value.data]
    elif value.size == 0:
        data = [t.data]
    else:
        data = t.data.tolist()
        data.append(value.data)

    # Create a new Tensor with the appended data and set requires_grad based on the flag
    tensor = ember.Tensor(data, requires_grad, device=t.device)

    # Register hooks for gradient computation if the input tensors require it
    if t.requires_grad:
        def grad_fn(grad):
            return grad[:-1]

        tensor.register_hook(ember.Hook(t, grad_fn))

    if value.requires_grad:
        def grad_fn(grad):
            return grad[-1]

        tensor.register_hook(ember.Hook(value, grad_fn))

    return tensor
