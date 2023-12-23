import numpy
import logging

# Check if CUDA is available
CUDA_AVAILABLE = False
try:
    import cupy
    CUDA_AVAILABLE = True
except ModuleNotFoundError:
    pass

# Define the list of names to be exported when using "from cuda import *"
__all__ = [
    "cuda_available",
    "numpy_or_cupy",
    "scalars_to_device"
]


def cuda_available():
    """
    Check if CUDA is available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return CUDA_AVAILABLE


def numpy_or_cupy(*tensors):
    """
    Choose between NumPy and CuPy based on the device of input tensors.

    Args:
        *tensors: Variable number of tensors.

    Returns:
        module: NumPy or CuPy module based on the device of input tensors.

    Raises:
        RuntimeError: If tensors are on different devices.
    """
    device = numpy.mean([t.device == 'cuda' for t in tensors])
    if device == 1:
        return cupy
    elif device == 0:
        return numpy
    else:
        logging.error(f"Cannot compute from tensors on different devices. "
                      f"Got {', '.join([t.device for t in tensors])}.")


def scalars_to_device(*tensors):
    """
    Move scalar tensors to the CUDA device if available.

    Args:
        *tensors: Variable number of tensors.

    Returns:
        None
    """
    device = numpy.mean([t.device == 'cuda' for t in tensors])

    if device > 0:
        for tensor in tensors:
            if tensor.shape == ():
                tensor.cuda()
