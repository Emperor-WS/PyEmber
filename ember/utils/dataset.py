from abc import ABC
import ember
import os
import numpy as np
from ._utils import extract_to_dir, download_from_url
from .example import Example


class Dataset(ABC):
    """
    Dataset is an abstract base class representing a collection of examples.

    Attributes:
    - examples (list): A list containing the examples in the dataset.
    - fields: The fields used in the dataset.

    Methods:
    - splits: Class method to create train, test, and validation splits of the dataset.
    - __getitem__: Method to get an item from the dataset.
    - __setitem__: Method to set an item in the dataset.
    - __len__: Method to get the length of the dataset.
    """

    urls = []
    name = ''
    dirname = ''

    def __init__(self, examples, fields):
        """
        Initializes a Dataset object.

        Parameters:
        - examples (list): A list containing the examples in the dataset.
        - fields: The fields used in the dataset.
        """
        self.examples = examples
        self.fields = fields

    @classmethod
    def splits(cls, train=None, test=None, valid=None, root='.'):
        """
        Class method to create train, test, and validation splits of the dataset.

        Parameters:
        - train: The training split of the dataset.
        - test: The testing split of the dataset.
        - valid: The validation split of the dataset.
        - root (str): The root directory for the dataset.

        Returns:
        - Dataset: The created dataset.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """
        Returns an item from the dataset.

        Parameters:
        - item: Index or key to retrieve from the dataset.

        Returns:
        - Any: The item from the dataset.
        """
        return self.examples[item]

    @classmethod
    def download(cls, root):
        """
        Downloads the dataset files from specified URLs, extracts them, and returns the path to the extracted dataset.
    
        Args:
            root (str): The base directory where the dataset should be downloaded and extracted.
    
        Returns:
            str: The path to the extracted dataset directory.
        """

        # Construct paths for dataset directory and extracted dataset
        # Combine root and dirname using path separator
        path_dirname = os.path.join(root, cls.dirname)
        # Combine path_dirname and name
        path_name = os.path.join(path_dirname, cls.name)

        # Check if the dataset directory already exists
        if not os.path.exists(path_dirname):
            # Iterate through each URL in the dataset_urls list
            for url in cls.urls:
                # Extract the filename from the URL
                filename = os.path.basename(url)
                # Construct the path for the archive file
                zpath = os.path.join(path_dirname, filename)

                # Check if the archive file already exists
                if not os.path.isfile(zpath):
                    # If the parent directory for the archive file doesn't exist, create it
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))

                    # Print a message indicating the download process
                    print(f"Downloading {filename} from {url} to {zpath}")

                    # Download the file from the URL to the specified path
                    # Assuming download_from_url is a defined function
                    download_from_url(url, zpath)

                    # Extract the contents of the archive to the extracted_dataset_path
                    # Assuming extract_to_dir is a defined function
                    extract_to_dir(zpath, path_name)

        return path_name  # Return the path to the extracted dataset

    def __repr__(self):
        """
        Returns a string representation of the dataset object, providing a clear and informative overview of its contents.
        """

        # Get the dataset class name
        name = self.__class__.__name__

        # Initialize a string to hold the representation
        string = f"Dataset {name}("

        # Set a tab character for proper indentation
        tab = "  "

        # Iterate through the object's attributes, excluding private ones
        for (key, value) in self.__dict__.items():
            # Exclude private attributes (starting with "_")
            if key[0] != "_":
                # Handle Example objects
                if isinstance(value, Example):
                    # Get the fields of the Example object
                    fields = self.fields

                    for (name, field) in fields:
                        string += f"\n{tab}({name}): {field.__class__.__name__}" \
                                  f"(transform={True if field.transform is not None else None}, dtype={field.dtype})"
                # Handle tensors (ember.Tensor) or NumPy arrays
                elif isinstance(value, ember.Tensor) or isinstance(value, np.ndarray):
                    string += f"\n{tab}({key}): {value.__class__.__name__}(shape={value.shape}, dtype={value.dtype})"
                # Handle other values
                else:
                    string += f"\n{tab}({key}): {value.__class__.__name__}"
        # Close the dataset representation and return it
        return f'{string}\n)'

    def __setitem__(self, key, value):
        """
        Sets an item in the dataset.

        Parameters:
        - key: Index or key to set in the dataset.
        - value: The value to set.

        Returns:
        - None
        """
        self.examples[key] = value

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: The length of the dataset.
        """
        return len(self.examples)
