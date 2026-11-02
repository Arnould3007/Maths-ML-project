"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.
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
        Initialize the SLP regressor.

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
        y_pred = z2  # identity output: a regression target is unbounded
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
            Target values
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        _, activation_derivative = self._get_activation_function()
        n_samples = X.shape[0]

        # Output layer error: dL/dz2 for L = 1/(2n) * sum((y_pred - y)^2), identity output
        dz2 = (y_pred - y) / n_samples  # (n, k)
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
        Compute mean squared error loss.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        # 1/(2n) * sum of squared errors: the 1/2 cancels the 2 of the derivative,
        # so the matching output gradient is exactly (y_pred - y) / n
        n_samples = y_true.shape[0]
        return float(np.sum((y_pred - y_true) ** 2) / (2 * n_samples))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self._validate_params()
        X, y = self._validate_input(X, y)
        y = y.astype(float)
        self._y_is_1d = y.ndim == 1
        Y = y.reshape(-1, 1) if self._y_is_1d else y  # always (n, k) inside the network

        self.n_features_in_ = X.shape[1]  # d
        self.n_outputs_ = Y.shape[1]  # k
        self._train(X, Y)  # mini-batch gradient descent (+ early stopping), see slp_base.py
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        if self.W1_ is None:
            raise RuntimeError("This SimpleSLPRegressor is not fitted yet. Call fit() first.")
        X, _ = self._validate_input(X)
        _, _, _, y_pred = self._forward_propagation(X)  # (n, k)
        return y_pred.ravel() if self._y_is_1d else y_pred  # back to the shape of y

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X).reshape(y.shape)
        ss_res = np.sum((y - y_pred) ** 2, axis=0)  # error left by the model (per output)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)  # error of always predicting the mean
        # Constant target (ss_tot = 0): R² is undefined. Like sklearn.metrics.r2_score,
        # return 1 for a perfect prediction and 0 otherwise instead of dividing by zero.
        constant = ss_tot == 0
        r2 = 1.0 - ss_res / np.where(constant, 1.0, ss_tot)
        r2 = np.where(constant, np.where(ss_res == 0, 1.0, 0.0), r2)
        return float(np.mean(r2))  # average over outputs, like sklearn
