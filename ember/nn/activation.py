import ember
from ember.nn.modules import Module


class Activation(Module):
    """
    Base class for activation functions.
    
    Activation functions introduce non-linearity to the neural network, allowing it to learn complex patterns.
    The forward pass applies the activation function to the input, and the backward pass computes gradients for backpropagation.


    Attributes:
    - forward_func: The forward function for the activation.
    - backward_func: The backward function for the activation.

    Methods:
    -  __init__(self, forward_func, backward_func): Constructor for Activation class.
    - forward(inputs): Performs the forward pass of the activation.
    - backward(grad): Computes the backward pass of the activation.

    """

    def __init__(self, forward_func, backward_func):
        """
        Constructor for the Activation class.

        Args:
        - forward_func: The forward function for the activation.
        - backward_func: The backward function for the activation.

        """
        super(Activation, self).__init__()
        self.forward_func = forward_func
        self.backward_func = backward_func

    def forward(self, inputs):
        """
        Performs the forward pass of the activation.
        
        The forward pass applies the activation function to the input tensor and stores the input in the cache for later use.


        Args:
        - inputs: The input tensor.

        Returns:
        - out: The output tensor after applying the activation.

        """
        self._cache['x'] = inputs
        out = self.forward_func(inputs)
        return out

    def backward(self, grad):
        """
        Computes the backward pass of the activation.
        
        The backward pass computes the gradient of the input tensor with respect to the loss, facilitating backpropagation.


        Args:
        - grad: The gradient tensor.

        Returns:
        - Tensor: Result of the backward pass.

        """
        x = self._cache['x']
        return self.backward_func(x) * grad


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU is a widely used activation that replaces negative values with zero, introducing non-linearity to the network.
    The forward pass applies ReLU to the input, and the backward pass computes gradients.


    Methods:
    - __init__(self): Constructor for ReLU class.
    - forward(inputs): Performs the forward pass of ReLU.
    - backward(grad): Computes the backward pass of ReLU.

    """

    def __init__(self):
        """
        Constructor for the ReLU class.

        """
        super().__init__(ember.relu, ember.relu_prime)


class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Tanh squashes input values between -1 and 1, aiding in modeling complex relationships.
    The forward pass applies Tanh to the input, and the backward pass computes gradients.


    Methods:
    - __init__(self): Constructor for Tanh class.
    - forward(inputs): Performs the forward pass of Tanh.
    - backward(grad): Computes the backward pass of Tanh.

    """

    def __init__(self):
        """
        Constructor for the Tanh class.

        """
        super().__init__(ember.tanh, ember.tanh_prime)


class Softmax(Activation):
    """
    Softmax activation function.
    
    Softmax is often used in the output layer for multi-class classification, converting raw scores to probabilities.
    The forward pass applies Softmax to the input, and the backward pass raises an error as the derivative is not implemented.


    Attributes:
    - axis: The axis along which the softmax is applied.

    Methods:
    -  __init__(self, axis=0): Constructor for Softmax class.
    - forward(inputs): Performs the forward pass of Softmax.
    - backward(grad): Raises NotImplementedError since softmax derivative is not implemented.


    """

    def __init__(self, axis=0):
        """
        Constructor for the Softmax class.

        Parameters:
        - axis: The axis along which the softmax is applied.

        """
        super().__init__(ember.softmax, None)
        self.axis = axis

    def forward(self, inputs):
        """
        Performs the forward pass of Softmax.
        
        Softmax converts input values into probabilities, facilitating classification.

        Args:
        - inputs: The input tensor.

        Returns:
        - out: The output tensor after applying Softmax.


        """
        return self.func(inputs, axis=self.axis)

    def backward(self, grad):
        """
        Raises NotImplementedError since softmax derivative is not implemented.

        """
        raise NotImplementedError('Softmax derivative not implemented')


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    
    Leaky ReLU allows a small, non-zero gradient for negative input values, preventing dead neurons.
    The forward pass applies Leaky ReLU to the input, and the backward pass computes gradients.

    Methods:
    - __init__(self): Constructor for LeakyReLU class.
    - forward(inputs): Performs the forward pass of Leaky ReLU.
    - backward(grad): Computes the backward pass of Leaky ReLU.

    """

    def __init__(self):
        """
        Constructor for the LeakyReLU class.

        """
        super().__init__(ember.leaky_relu, ember.leaky_relu_prime)


class Sigmoid(Activation):
    """
    Sigmoid activation function. 
    
    Sigmoid squashes input values between 0 and 1, often used in binary classification problems.
    The forward pass applies Sigmoid to the input, and the backward pass computes gradients.

    Methods:
    - __init__(self): Constructor for Sigmoid class.
    - forward(inputs): Performs the forward pass of Sigmoid.
    - backward(grad): Computes the backward pass of Sigmoid.


    """

    def __init__(self):
        """
        Constructor for the Sigmoid class.

        """
        super().__init__(ember.sigmoid, ember.sigmoid_prime)
