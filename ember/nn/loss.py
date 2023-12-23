from abc import ABC

import ember
from ember.nn.modules import Module


class Loss(Module, ABC):
    """
    A loss function evaluates the correctness of a set of predictions regarding gold-labels.
    The predictions should be uncorrected, i.e., no transformations like Softmax should have been used before.
    The loss function will perform the transformation if necessary.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        """
        Compute the cost function.

        Args:
            predictions (numpy.array): Tensor of un-normalized floats with shape (N, c).
            labels (numpy.array): Tensor of integer values with shape (N).

        Returns:
            cost (float): The cost regarding the loss function.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def __call__(self, *inputs):
        """
        Call method for the loss function. It computes the forward pass and returns the cost.

        Args:
            *inputs: Variable number of input arguments.

        Returns:
            cost (float): The cost regarding the loss function.
        """

        cost = self.forward(*inputs)
        return cost


class MSELoss(Loss):
    """
    Mean Square Error Loss, defined as:

    MSE = (1/N) * Σ(predictions - labels)^2

    This loss measures the average squared difference between predictions and true labels.
    """

    def __init__(self):
       super().__init__()

    def forward(self, predictions, labels):
        """
        Compute the Mean Square Error (MSE) Loss.

        Args:
            predictions (numpy.array): Tensor of predicted values.
            labels (numpy.array): Tensor of true labels.

        Returns:
            mse (float): Mean Square Error loss.
        """

        assert labels.shape == predictions.shape, \
            "labels shape {} and predictions shape {} should match".format(
                labels.shape, predictions.shape)

        # Compute the MSE loss: (1/N) * Σ(predictions - labels)^2
        return ember.sum((predictions - labels) ** 2) / len(predictions)

    def backward(self, predictions, labels):
        """
       Compute the gradient of the Mean Square Error (MSE) Loss with respect to predictions.

       Args:
           predictions (numpy.array): Tensor of predicted values.
           labels (numpy.array): Tensor of true labels.

       Returns:
           gradient (numpy.array): Gradient of the MSE loss with respect to predictions.
       """

        assert labels.shape == predictions.shape, \
            "labels shape {} and predictions shape {} should match".format(
                labels.shape, predictions.shape)

        # Compute the gradient of the MSE loss: 2 * (predictions - labels)
        return 2 * (predictions - labels)


class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss. This class represents the cross-entropy loss used in classification tasks.
    It includes both forward and backward computations.
    """

    def __init__(self, epsilon=1e-10):
        super().__init__()
        # epsilon is used  to prevent log0
        self.epsilon = epsilon

    def forward(self, predictions, labels):
        """
        Forward pass for the Cross Entropy Loss.

        Args:
        - predictions (ember.Tensor): Predicted probabilities for each class.
        - labels (ember.Tensor): True labels for each example.

        Returns:
        - cost (ember.Tensor): Computed cross-entropy loss.
        """

        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(
            predictions.dtype)

        batch_size, _ = predictions.shape

        # Ensure predictions are mapped to [0, 1] using softmax transformation
        predictions = ember.softmax(predictions, axis=1)

        # Compute cross-entropy loss using the formula: -1/N * Σ(labels_i * log(pred_i))
        cost = ember.Tensor(-1 / batch_size, device=predictions.device) * \
            ember.sum(ember.log(predictions + self.epsilon) * labels)

        return cost

    def backward(self, predictions, labels):
        """
        Backward pass for the Cross Entropy Loss.

        Args:
        - predictions (ember.Tensor): Predicted probabilities for each class.
        - labels (ember.Tensor): True labels for each example.

        Returns:
        - gradient (ember.Tensor): Gradient of the loss with respect to predictions.
        """

        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(
            predictions.dtype)

        # Ensure predictions are mapped to [0, 1] using softmax transformation
        predictions = ember.softmax(predictions)

        # Compute the gradient of the loss with respect to predictions
        return (predictions + self.epsilon) - labels


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross Entropy Loss. This class represents the binary cross-entropy loss used in binary classification tasks.
    It includes both forward and backward computations.
    """

    def __init__(self, epsilon=1e-10):
        super().__init__()
        # epsilon is used to prevent log(0)
        self.epsilon = epsilon

    def forward(self, predictions, labels):
        """
        Forward pass for the Binary Cross Entropy Loss.

        Args:
        - predictions (ember.Tensor): Predicted probabilities for the positive class.
        - labels (ember.Tensor): True labels for each example (binary, 0 or 1).

        Returns:
        - cost (ember.Tensor): Computed binary cross-entropy loss.
        """

        assert labels.dtype == int, "unsupported labels type {} for binary cross entropy loss".format(
            predictions.dtype)

        batch_size, _ = predictions.shape

        # Ensure predictions are mapped to [0, 1] using sigmoid transformation
        predictions = ember.sigmoid(predictions)

        # Compute binary cross-entropy loss using the formula: -1/N * Σ(labels_i * log(pred_i) + (1 - labels_i) * log(1 - pred_i))
        cost = ember.Tensor(-1 / batch_size, device=predictions.device) * \
            (ember.sum(labels * ember.log(predictions + self.epsilon) + (1 - labels) * ember.log(1 - predictions + self.epsilon)))

        return cost

    def backward(self, predictions, labels):
        """
        Backward pass for the Binary Cross Entropy Loss.

        Args:
        - predictions (ember.Tensor): Predicted probabilities for the positive class.
        - labels (ember.Tensor): True labels for each example (binary, 0 or 1).

        Returns:
        - gradient (ember.Tensor): Gradient of the loss with respect to predictions.
        """

        assert labels.dtype == int, "unsupported labels type {} for binary cross entropy loss".format(
            predictions.dtype)

        # Ensure predictions are mapped to [0, 1] using sigmoid transformation
        predictions = ember.sigmoid(predictions)

        # Compute the gradient of the loss with respect to predictions
        return -labels / (predictions + self.epsilon) + (1 - labels) / ((1 - predictions) + self.epsilon)
