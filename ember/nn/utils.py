import ember

def one_hot_encode_labels(labels, num_classes):
    """
    Perform one-hot encoding on a batch of categorical labels.

    Parameters:
    - labels (numpy.ndarray): Array of categorical labels to be one-hot encoded.
    - num_classes (int): The total number of classes.

    Returns:
    - numpy.ndarray: A binary matrix representing the one-hot encoded labels.
    """
    # Get the batch size from the length of the input labels.
    batch_size = len(labels)

    # Initialize a matrix to store the one-hot encoded labels, initially filled with zeros.
    one_hot_labels = ember.zeros((batch_size, num_classes))

    # Perform one-hot encoding.
    one_hot_labels[ember.arange(batch_size), labels] = 1

    # Ensure the output is of integer type before returning.
    return one_hot_labels.astype(int)
