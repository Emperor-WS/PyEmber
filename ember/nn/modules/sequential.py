from .module import Module


class Sequential(Module):
    """
    A container module that sequentially applies a list of modules.

    The Sequential module allows the user to stack multiple modules in a sequential order,
    where the output of each module serves as the input for the next one.

    Attributes:
    - modules: A dictionary to store the modules in sequential order.

    Methods:
    - __init__(self, *modules): Constructor for Sequential class, adds provided modules.
    - forward(self, inputs): Performs the forward pass by sequentially applying each module.
    - backward(self, grad): Performs the backward pass by sequentially applying each module in reverse order.

    """

    def __init__(self, *modules):
        """
        Constructor for the Sequential class.

        Args:
        - *modules: Variable number of modules to be added to the sequential container.

        """
        super(Sequential, self).__init__()
        self.add(*modules)

    def forward(self, inputs):
        """
        Performs the forward pass by sequentially applying each module.

        Args:
        - inputs: The input tensor.

        Returns:
        - Tensor: The output tensor after passing through all the modules.

        """
        for module in self.modules():
            inputs = module.forward(inputs)
        return inputs

    def backward(self, grad):
        """
        Performs the backward pass by sequentially applying each module in reverse order.

        Args:
        - grad: The gradient tensor.

        Returns:
        - Tensor: The gradient tensor after passing through all the modules in reverse order.

        """
        for module in reversed(self.modules()):
            grad = module.backward(grad)
        return grad
