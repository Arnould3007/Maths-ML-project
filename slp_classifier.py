"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import logistic, softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = 32,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
    ) -> None:
        """
        Initialize the SLP classifier.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        batch_size : int or None
            Number of samples per gradient step (mini-batch); None = full batch
        early_stopping : bool
            If True, stop when the validation loss stops improving
        validation_fraction : float
            Share of the training data held out for validation (early stopping only)
        n_iter_no_change : int
            Number of epochs without improvement before stopping
        tol : float
            Minimum decrease of the validation loss that counts as an improvement
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, random_state,
            batch_size, early_stopping, validation_fraction, n_iter_no_change, tol,
        )

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        activation, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_  # (n, d) @ (d, h) + (h,) -> (n, h)
        a1 = activation(z1)  # (n, h)
        z2 = a1 @ self.W2_ + self.b2_  # (n, h) @ (h, k) + (k,) -> (n, k)
        # Binary: 1 output neuron + sigmoid = P(y = class 1). Multi-class: k neurons + softmax.
        y_pred = logistic(z2) if self.n_outputs_ == 1 else softmax(z2)  # (n, k)
        return z1, a1, z2, y_pred

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        _, activation_derivative = self._get_activation_function()

        # Output layer error: for sigmoid + binary CE and for softmax + CE,
        # dL/dz2 simplifies to (y_pred - y) (course, session 3, slide 29).
        # The loss is a sum over samples (not a mean), so there is no division by n.
        dz2 = y_pred - y  # (n, k)
        dW2 = a1.T @ dz2  # (h, n) @ (n, k) -> (h, k), same shape as W2
        db2 = dz2.sum(axis=0)  # (k,), sum over the samples

        # Hidden layer error: send dz2 back through W2, then through the activation
        da1 = dz2 @ self.W2_.T  # (n, k) @ (k, h) -> (n, h)
        dz1 = da1 * activation_derivative(z1)  # (n, h), element-wise (chain rule)
        dW1 = X.T @ dz1  # (d, n) @ (n, h) -> (d, h), same shape as W1
        db1 = dz1.sum(axis=0)  # (h,)

        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Clip predictions to avoid log(0) = -inf
        eps = 1e-15
        p = np.clip(y_pred, eps, 1 - eps)
        # Sum over samples, exactly as E(w) on slide 29 (its gradient is (y_pred - y))
        if self.n_outputs_ == 1:
            # Binary CE: -sum[y log p + (1 - y) log(1 - p)]
            return float(-np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        # Multi-class CE: -sum_i sum_k y_ik log p_ik (only the true class counts)
        return float(-np.sum(y_true * np.log(p)))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self._validate_params()
        X, y = self._validate_input(X, y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,), got shape {y.shape}")

        self.classes_ = np.unique(y)  # sorted labels, e.g. [0, 1, 2]
        n_classes = len(self.classes_)
        if n_classes <= 2:
            # Binary (or a single class): 1 output neuron, target = 1 for the last class
            Y = (y == self.classes_[-1]).astype(float).reshape(-1, 1)  # (n, 1)
        else:
            # Multi-class: one-hot encoding, Y[i, k] = 1 if sample i is of class k
            Y = (y.reshape(-1, 1) == self.classes_).astype(float)  # (n, k)

        self.n_features_in_ = X.shape[1]  # d
        self.n_outputs_ = Y.shape[1]  # k
        self._train(X, Y)  # mini-batch gradient descent (+ early stopping), see slp_base.py
        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        if self.W1_ is None:
            raise RuntimeError("This SimpleSLPClassifier is not fitted yet. Call fit() first.")
        X, _ = self._validate_input(X)
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 1))  # only one possible class
        _, _, _, y_pred = self._forward_propagation(X)  # (n, k)
        if self.n_outputs_ == 1:
            # Sigmoid gives P(class 1) only: add P(class 0) = 1 - p, like sklearn
            return np.hstack([1 - y_pred, y_pred])  # (n, 2)
        return y_pred  # softmax output, rows already sum to 1

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)  # (n, n_classes)
        # Column index of the highest probability -> original label
        return self.classes_[np.argmax(proba, axis=1)]  # (n,)

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        # Share of correct predictions
        return float(np.mean(self.predict(X) == np.asarray(y)))
