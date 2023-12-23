class Batch(object):
    """
    Batch is a class representing a batch of examples.

    Attributes:
    - example: The example data in the batch.
    - batch_size (int): The size of the batch.
    """

    def __init__(self, example, batch_size):
        """
        Initializes a Batch object.

        Parameters:
        - example: The example data in the batch.
        - batch_size (int): The size of the batch.
        """
        self.example = example
        self.batch_size = batch_size

    def normalize(self):
        """
        Placeholder method for normalizing the batch.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """
        Returns an item from the example.

        Parameters:
        - item: Index or key to retrieve from the example.

        Returns:
        - Any: The item from the example.
        """
        return self.example[item]

    def __len__(self):
        """
        Returns the batch size.

        Returns:
        - int: The size of the batch.
        """
        return self.batch_size
