import ember
from ember import Parameter
from .module import Module


class BatchNorm2d(Module):
    """
    2D Batch Normalization layer.

    Parameters:
    - num_features (int): Number of input channels.
    - eps (float): Small value added to the denominator for numerical stability.
    - momentum (float): The value used for the running mean and variance computation.
    - requires_grad (bool): Whether to track gradients for the learnable parameters.
    - device (str): Device to store the parameters and perform computations ('cpu' or 'cuda').

    Methods:
    - forward(x): Forward pass through the layer.

    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, requires_grad=True, device='cpu'):
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.requires_grad = requires_grad
        self.device = device

        # Initialize learnable parameters
        self.gamma = Parameter(ember.ones((num_features,), requires_grad=requires_grad, device=device))
        self.beta = Parameter(ember.zeros((num_features,), requires_grad=requires_grad, device=device))

        # Running mean and variance
        self.running_mean = ember.zeros((num_features,), requires_grad=False, device=device)
        self.running_var = ember.ones((num_features,), requires_grad=False, device=device)

    def forward(self, x):
        """
        Forward pass through the BatchNorm2d layer.

        Parameters:
        - x (ember.Tensor): Input tensor.

        Returns:
        - ember.Tensor: Output tensor after batch normalization.

        Explanation:
        - Computes the batch normalization on the input tensor.
        - Updates the running mean and variance for inference.

        """

        # Check if training or inference
        if self.training:
            # Training mode
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            x_normalized = (x - mean) / ember.sqrt(var + self.eps)
            out = self.gamma * x_normalized + self.beta

            # Update running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # Inference mode
            x_normalized = (x - self.running_mean) / ember.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta

        return out
