import numpy as np
from .batch import Batch

class DataLoader:
    """
    DataLoader is a class for efficiently loading data in batches.

    Parameters:
    - dataset: The dataset to be loaded.
    - batch_size (int): The size of each batch.
    - shuffle (bool): Whether to shuffle the dataset before each epoch.
    """

    def __init__(self, dataset, batch_size=32, shuffle=False):
        """
        Initializes a DataLoader object.

        Parameters:
        - dataset: The dataset to be loaded.
        - batch_size (int): The size of each batch.
        - shuffle (bool): Whether to shuffle the dataset before each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """
        Returns an iterator over batches of the dataset.

        Returns:
        - Iterator: An iterator yielding Batch objects.
        """
        # Create an array of indices to represent the order of data points.
        starts = np.arange(0, len(self.dataset), self.batch_size)

        # Shuffle the dataset if specified.
        if self.shuffle:
            np.random.shuffle(list(self.dataset))

        # Iterate over the indices to yield batches.
        for start in starts:
            end = start + self.batch_size
            # Adjust the batch size for the last batch if it's smaller than batch_size.
            current_batch_size = min(end, len(self.dataset)) - start
            # Yield a Batch object representing the current batch.
            yield Batch(self.dataset[start:end], current_batch_size)

    def __len__(self):
        """
        Returns the number of batches in the DataLoader.

        Returns:
        - int: The number of batches.
        """
        return len(self.dataset) // self.batch_size
