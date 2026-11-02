"""
Base class for SLP Classifier and Regressor
"""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    identity,
    identity_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
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
        Initialize the SLP estimator.

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
            If True, hold out validation_fraction of the training data and stop
            when the validation loss stops improving
        validation_fraction : float
            Share of the training data used for validation (early stopping only)
        n_iter_no_change : int
            Number of epochs without improvement before stopping
        tol : float
            Minimum decrease of the validation loss that counts as an improvement
        """
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.batch_size: Optional[int] = batch_size
        self.early_stopping: bool = early_stopping
        self.validation_fraction: float = validation_fraction
        self.n_iter_no_change: int = n_iter_no_change
        self.tol: float = tol

        # To be initialized in fit()
        self.n_features_in_: Optional[int] = None  # d: number of input features
        self.n_outputs_: Optional[int] = None  # k: number of output neurons
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training
        self.validation_loss_curve_: list[float] = []  # Only filled with early_stopping
        self.n_iter_: int = 0  # Number of epochs actually run

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "identity":
            return identity, identity_derivative
        elif self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _validate_params(self) -> None:
        """Check the hyperparameters given to __init__ (called at the start of fit)."""
        if not isinstance(self.hidden_layer_size, (int, np.integer)) or self.hidden_layer_size < 1:
            raise ValueError(f"hidden_layer_size must be a positive integer, got {self.hidden_layer_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {self.max_iter}")
        self._get_activation_function()  # raises ValueError for an unknown activation
        if self.batch_size is not None and (
            not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1
        ):
            raise ValueError(f"batch_size must be a positive integer or None, got {self.batch_size}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction}")
        if not isinstance(self.n_iter_no_change, (int, np.integer)) or self.n_iter_no_change < 1:
            raise ValueError(f"n_iter_no_change must be a positive integer, got {self.n_iter_no_change}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")

    def _validate_input(
        self, X: NDArray, y: Optional[NDArray] = None
    ) -> Tuple[NDArray[np.floating], Optional[NDArray]]:
        """
        Convert X to a 2D float array (and y to an array) and check their shapes.

        Without y (predict time), also checks that X has as many features as in fit().
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        if y is None:
            if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}"
                )
            return X, None
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y have different numbers of samples: {X.shape[0]} vs {y.shape[0]}")
        return X, y

    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases.

        Requires self.n_features_in_ (d) and self.n_outputs_ (k), set in fit().
        Weights ~ N(0, 2 / (fan_in + fan_out)) (Glorot/Xavier, as in scikit-learn):
        random to break the symmetry between hidden neurons, and scaled so that
        the signal keeps the same variance from one layer to the next.
        A fixed std of 0.01 (the scaffold's example) made W2 so small that the
        hidden layer received almost no gradient and the classifier did not learn.
        Biases start at 0 (the random weights already break the symmetry).
        """
        # Local generator: same seed -> same weights, without touching NumPy's global state.
        # Kept in self._rng so fit() can reuse it (e.g. to shuffle mini-batches).
        self._rng = np.random.default_rng(self.random_state)

        d, h, k = self.n_features_in_, self.hidden_layer_size, self.n_outputs_
        self.W1_ = self._rng.normal(loc=0.0, scale=np.sqrt(2.0 / (d + h)), size=(d, h))  # (d, h)
        self.b1_ = np.zeros(h)  # (h,)
        self.W2_ = self._rng.normal(loc=0.0, scale=np.sqrt(2.0 / (h + k)), size=(h, k))  # (h, k)
        self.b2_ = np.zeros(k)  # (k,)

    def _train(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> None:
        """
        Mini-batch gradient descent loop, shared by the classifier and the regressor.

        X : (n, d) inputs. Y : (n, k) targets, already in the network's output format
        (one-hot for the classifier). Requires n_features_in_ and n_outputs_ to be set.
        One iteration (epoch) = one pass over all training samples, in shuffled mini-batches.
        """
        self._initialize_weights()  # also creates self._rng
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        self.n_iter_ = 0

        if self.early_stopping:
            # Hold out a random validation set that is never used for gradient steps
            n_val = max(1, int(round(self.validation_fraction * X.shape[0])))
            if X.shape[0] - n_val < 1:
                raise ValueError("Not enough samples to hold out a validation set for early stopping")
            perm = self._rng.permutation(X.shape[0])
            X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
            X, Y = X[perm[n_val:]], Y[perm[n_val:]]
            best_val_loss = np.inf
            best_params = None
            epochs_no_improve = 0

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)

        for epoch in range(self.max_iter):
            # Weights at the end of the previous epoch (finite, or we would have stopped):
            # restored if this epoch diverges, so the model is never left with NaN weights
            last_finite_params = (self.W1_.copy(), self.b1_.copy(), self.W2_.copy(), self.b2_.copy())

            # New random order every epoch, so the mini-batches differ between epochs
            order = self._rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):  # loop over batches, not samples
                idx = order[start : start + batch_size]
                z1, a1, z2, y_pred = self._forward_propagation(X[idx])
                dW1, db1, dW2, db2 = self._backward_propagation(X[idx], Y[idx], z1, a1, z2, y_pred)

                # Gradient descent: move each parameter against its gradient
                self.W1_ -= self.learning_rate * dW1
                self.b1_ -= self.learning_rate * db1
                self.W2_ -= self.learning_rate * dW2
                self.b2_ -= self.learning_rate * db2

            # Loss on the whole training set at the end of the epoch
            loss = self._compute_loss(Y, self._forward_propagation(X)[3])
            if not np.isfinite(loss):
                # Divergence: go back to the last finite weights, keep only the finite losses
                self.W1_, self.b1_, self.W2_, self.b2_ = last_finite_params
                warnings.warn(
                    f"Training diverged at epoch {epoch + 1} (loss is inf/NaN): weights restored "
                    "from the last finite epoch. Lower learning_rate or standardize the target.",
                    RuntimeWarning,
                )
                break
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

            if self.early_stopping:
                val_loss = self._compute_loss(Y_val, self._forward_propagation(X_val)[3])
                self.validation_loss_curve_.append(val_loss)
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    best_params = (self.W1_.copy(), self.b1_.copy(), self.W2_.copy(), self.b2_.copy())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        break

        if self.early_stopping and best_params is not None:
            # Keep the weights of the best validation epoch, not the last one
            self.W1_, self.b1_, self.W2_, self.b2_ = best_params

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
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

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
