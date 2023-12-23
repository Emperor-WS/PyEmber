

class Hook:
    """
    Hook class for attaching gradient functions to tensors.

    Hooks allow users to attach custom gradient functions to tensors for
    monitoring or modifying gradients during backpropagation.

    Attributes:
    - tensor (Tensor): The target tensor.
    - grad_fn (callable): The gradient function to be applied to the tensor.

    Methods:
    - __init__(self, tensor, grad_fn): Constructor for Hook class.
    - __repr__(self): String representation of the Hook instance.

    """

    __slots__ = 'tensor', 'grad_fn'

    def __init__(self, tensor, grad_fn):
        """
        Constructor for the Hook class.

        Args:
        - tensor (Tensor): The target tensor.
        - grad_fn (callable): The gradient function to be applied to the tensor.

        """
        self.tensor = tensor
        self.grad_fn = grad_fn

    def __repr__(self):
        """
        String representation of the Hook instance.

        Returns:
        - str: A string containing information about the tensor and its associated gradient function.

        """
        # Extract the class name from the qualified name of the gradient function
        grad_name = self.grad_fn.__qualname__.split('.')[0]
        return f"Hook(tensor_id={self.tensor.id}, grad_fn={grad_name.upper()})"
