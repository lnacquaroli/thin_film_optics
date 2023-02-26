"""Collection of loss functions for the fitting procedure.
"""

from typing import Any
import numpy as np


def mae_loss_function(array_1: Any, array_2: Any) -> Any:
    """Calculates the loss function between two ndarrays, using the mean-abs error criteria.

    Args:
        array_1 (ndarray): first array
        array_2 (ndarray): second array

    Returns:
        (float) : mean-abs error loss value.
    """
    return np.mean(np.abs(array_1 - array_2))


def mse_loss_function(array_1: Any, array_2: Any) -> Any:
    """Calculates the mean square error function between two ndarrays.

    Args:
        array_1 (ndarray): first array
        array_2 (ndarray): second array

    Returns:
        (float) : mse loss value.
    """
    return np.mean((array_1 - array_2) ** 2)
