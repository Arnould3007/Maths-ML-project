"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # ReLU(z) = max(0, z): negative values become 0, positive values are kept.
    # np.maximum compares element by element (np.max would return a single number).
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # ReLU'(z) = 1 if z > 0, else 0 (by convention 0 at z = 0).
    # (z > 0) gives a boolean array, converted to 1.0 / 0.0.
    return (z > 0).astype(float)


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # tanh(z) squashes values into (-1, 1), centered on 0.
    return np.tanh(z)


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # tanh'(z) = 1 - tanh(z)^2
    return 1 - np.tanh(z) ** 2


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # sigmoid(z) = 1 / (1 + e^(-z)), squashes values into (0, 1).
    # z is clipped to [-500, 500] so that np.exp(-z) cannot overflow.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)), maximum 0.25 at z = 0.
    s = logistic(z)
    return s * (1 - s)


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # softmax(z)_i = e^(z_i) / sum_j e^(z_j), computed row by row (one row = one sample).
    # Subtracting the row maximum avoids overflow and does not change the result.
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
