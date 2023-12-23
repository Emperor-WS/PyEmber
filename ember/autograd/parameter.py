import numpy as np

from ember.tensor import Tensor


class Parameter(Tensor):
    """
    Subclass of Tensor representing trainable parameters.

    Attributes:
        _data: Underlying data array.
        requires_grad: Indicates whether gradients should be tracked for this parameter.

    Methods:
        __init__(self, data=None, shape=None): Constructor for Parameter class.
        scaled_weight(cls, input_dim, output_dim): Create a Parameter with scaled random weights.
        zeros(cls, shape): Create a Parameter initialized with zeros.
        uniform(cls, shape, low=-1, high=1): Create a Parameter with random values from a uniform distribution.
        normal(cls, shape, mu=0, sigma=1): Create a Parameter with random values from a normal distribution.
        orthogonal(cls, shape): Create a Parameter with orthogonal matrix initialization.
    """

    def __init__(self, data=None, shape=None):
        """
        Constructor for the Parameter class.

        Args:
            data: The underlying data for the parameter.
            shape: The shape of the parameter if data is not provided.

        Raises:
            ValueError: If neither data nor shape is specified.

        """
        # First check if there are enough information to build the Parameter
        if data is None and shape is None:
            raise ValueError('You must specify the shape or the data '
                             'to create a `Parameter`.')

        # If there is no data, generate data from a uniform distribution
        if shape is not None and data is None:
            data = np.random.randn(*shape)
        # Create the Tensor
        super().__init__(data, requires_grad=True)

    @classmethod
    def scaled_weight(cls, input_dim, output_dim):
        """
        Create a Parameter with scaled random weights.

        Args:
            input_dim (int): Number of input dimensions.
            output_dim (int): Number of output dimensions.

        Returns:
            Parameter: Parameter initialized with scaled random weights.

        """
        mu = 0
        var = 2 / input_dim
        sigma = np.sqrt(var)
        weight_shape = (input_dim, output_dim)
        
        # Generate data from a normal distribution with scaled weights
        data = np.random.normal(loc=mu, scale=sigma, size=weight_shape)
        return Parameter(data=data)

    @classmethod
    def zeros(cls, shape):
        """
        Create a Parameter initialized with zeros.

        Args:
            shape: The shape of the parameter.

        Returns:
            Parameter: Parameter initialized with zeros.

        """
        return Parameter(data=np.zeros(shape))

    @classmethod
    def uniform(cls, shape, low=-1, high=1):
        """
        Create a Parameter with random values from a uniform distribution.

        Args:
            shape: The shape of the parameter.
            low (float): The lower bound for the uniform distribution (default is -1).
            high (float): The upper bound for the uniform distribution (default is 1).

        Returns:
            Parameter: Parameter initialized with random values from a uniform distribution.

        """
        # Generate data from a uniform distribution
        data = np.random.uniform(low, high, shape)
        return Parameter(data=data)

    @classmethod
    def normal(cls, shape, mu=0, sigma=1):
        """
        Create a Parameter with random values from a normal distribution.

        Args:
            shape: The shape of the parameter.
            mu (float): The mean of the normal distribution (default is 0).
            sigma (float): The standard deviation of the normal distribution (default is 1).

        Returns:
            Parameter: Parameter initialized with random values from a normal distribution.

        """
        # Generate data from a normal distribution
        data = np.random.normal(mu, sigma, shape)
        return Parameter(data=data)

    @classmethod
    def orthogonal(cls, shape):
        """
        Create a Parameter with orthogonal matrix initialization.

        Args:
            shape: The shape of the parameter.

        Returns:
            Parameter: Parameter initialized with an orthogonal matrix.

        Raises:
            ValueError: If the shape has less than 2 dimensions.

        """
        if len(shape) < 2:
            raise ValueError(
                "only parameters with 2 or more dimensions are supported.")

        rows, cols = shape
        data = np.random.randn(rows, cols)

        if rows < cols:
            data = data.T

        # Compute QR factorization
        q, r = np.linalg.qr(data)

        # Make Q uniform
        diag = np.diag(r, 0)
        sign = np.sign(diag)
        q *= sign

        if rows < cols:
            q = q.T

        data = q
        return Parameter(data=data)
