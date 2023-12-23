import numpy as np


def _reshape_keepdims(array, axis=0):
    """
    Reshape an array while keeping dimensions.

    Args:
    - array (numpy.ndarray): Input array.
    - axis (int): Axis along which the reshaping is performed.

    Returns:
    - numpy.ndarray: Reshaped array with dimensions kept.

    """
    new_shape = [array.shape[axis]] + [1] * (array.ndim - 1)
    new_shape = tuple(new_shape)
    return np.sum(array, axis=axis).reshape(new_shape)


def _slice_keepdims(array, indices):
    """
    Slice an array while keeping dimensions.

    Args:
    - array (numpy.ndarray): Input array.
    - indices (int or slice or list): Indices for slicing.

    Returns:
    - numpy.ndarray: Sliced array with dimensions kept.

    """
    new_idx = []
    if isinstance(indices, int):
        new_idx.append(indices)
    else:
        for indice in indices:
            if isinstance(indice , int):
                new_idx.append([indice])
            else:
                new_idx.append(indice)
        new_idx = tuple(new_idx)
    return array[new_idx]


def numpy_unpad(x, pad_width):
    """
    Remove padding from an array.

    Args:
    - x (numpy.ndarray): Input array.
    - pad_width (tuple of ints): Amount of padding on each dimension.

    Returns:
    - numpy.ndarray: Unpadded array.

    """
    slices = []
    for pad in pad_width:
        end  = None if pad[1] == 0 else -pad[1]
        slices.append(slice(pad[0], end ))
    return x[tuple(slices)]


def inv_permutation(permutation):
    """
    Compute the inverse of a permutation.

    Args:
    - permutation (list): List representing a permutation.

    Returns:
    - list: Inverse permutation.

    """
    inverse = [0] * len(permutation)
    for original_idx, permuted_idx  in enumerate(permutation):
        inverse[permuted_idx] = original_idx
    return inverse
