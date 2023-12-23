import numpy as np
import ember


def im2col(input_data, filter_height, filter_width, stride=1, pad=0):
    """
    Convert image data to column data using im2col algorithm.

    Args:
    - input_data (numpy.ndarray): 4D input image data (N, C, H, W).
    - filter_height (int): Height of the filter.
    - filter_width (int): Width of the filter.
    - stride (int): Stride of the filter (default is 1).
    - padding (int): Padding size (default is 0).

    Returns:
    - numpy.ndarray: 2D column data.

    Raises:
    - AssertionError: If the parameters lead to invalid calculations.

    The im2col algorithm reshapes image data to column data for efficient matrix multiplication.
    It considers the filter size, stride, and padding to create non-overlapping blocks of the input data.

    """
    N, C, input_height, input_width = input_data.shape
    assert (input_height + 2 * pad - filter_height) % stride == 0, f'Invalid parameters for im2col: ' \
        f'(H + 2 * pad - filter_h) % stride != 0, got ' \
        f'H={input_height}, pad={pad}, filter_h={filter_height}, stride={stride}'
    assert (input_width + 2 * pad - filter_width) % stride == 0, f'Invalid parameters for im2col: ' \
                                                                     f'(W + 2 * pad - filter_w) % stride != 0, got ' \
                                                                     f'W={input_width}, pad={pad}, filter_w={filter_width}, stride={stride}'

    # Calculate the output dimensions
    output_height = (input_height + 2 * pad - filter_height) // stride + 1
    output_width = (input_width + 2 * pad - filter_width) // stride + 1

    # Apply padding to the input data
    padding_config = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    input_data_padded = ember.pad(input_data, padding_config)

    # Initialize an empty array to store the column data
    col_data = ember.zeros(
        (N, C, filter_height, filter_width, output_height, output_width))

    # Iterate over the filter and input data to fill the column array
    for row in range(filter_height):
        row_max = row + stride * output_height
        for col in range(filter_width):
            col_max = col + stride * output_width
            col_data[:, :, row, col, :, :] = input_data_padded[:,
                                                               :, row:row_max:stride, col:col_max:stride]

    # Transpose and reshape the column data to the final form
    col_data = col_data.transpose(0, 4, 5, 1, 2, 3).reshape(
        N * output_height * output_width, -1)

    return col_data


def col2im(col_data, input_shape, filter_height, filter_width, stride=1, pad=0):
    """
    Convert column data back to image data using col2im algorithm.

    Args:
    - col_data (numpy.ndarray): 2D column data.
    - input_shape (tuple): Shape of the input image (N, C, H, W).
    - filter_height (int): Height of the filter.
    - filter_width (int): Width of the filter.
    - stride (int): Stride of the filter (default is 1).
    - padding (int): Padding size (default is 0).

    Returns:
    - numpy.ndarray: 4D image data.

    Raises:
    - AssertionError: If the parameters lead to invalid calculations.

    The col2im algorithm reconstructs image data from column data. It reverses the im2col operation
    by distributing the values of the columns back to their corresponding positions in the original image.

    """
    N, C, input_height, input_width = input_shape

    assert (input_height + 2 * pad - filter_height) % stride == 0, f'Invalid parameters for col2im: ' \
        f'(H + 2 * pad - filter_h) % stride != 0, got ' \
        f'H={input_height}, pad={pad}, filter_h={filter_height}, stride={stride}'
    assert (input_width + 2 * pad - filter_width) % stride == 0, f'Invalid parameters for col2im: ' \
                                                                     f'(W + 2 * pad - filter_w) % stride != 0, got ' \
                                                                     f'W={input_width}, pad={pad}, filter_w={filter_width}, stride={stride}'

    # Calculate the output dimensions
    output_height = (input_height + 2 * pad - filter_height) // stride + 1
    output_width = (input_width + 2 * pad - filter_width) // stride + 1

    # Reshape the column data and transpose to the appropriate form
    col_data = col_data.reshape(N, output_height, output_width,
                                C, filter_height, filter_width).transpose(0, 3, 4, 5, 1, 2)

    # Initialize an empty array to store the reconstructed image data
    output_data = np.zeros((N, C, input_height + 2 * pad +
                           stride - 1, input_width + 2 * pad + stride - 1))

    # Iterate over the filter and column data to reconstruct the image
    for row in range(filter_height):
        row_max = row + stride * output_height
        for col in range(filter_width):
            col_max = col + stride * output_width
            output_data[:, :, row:row_max:stride,
                        col:col_max:stride] += col_data[:, :, row, col, :, :]

    # Return the final image data with appropriate cropping
    return output_data[:, :, pad:input_height + pad, pad:input_width + pad]
