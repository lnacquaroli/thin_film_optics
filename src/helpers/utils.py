"""Module with all the utils used in the main package.
"""

from typing import Any

import numpy as np


def find_closest(x_scalar: int, x_array: Any) -> Any:
    """Returns the index of the value in the 1d-array x closest to the scalar value a.

    Args:
        x_scalar (int): value to search inside the array.
        x_array (nd.array): array to search the closest value.

    Raises:
        TypeError: input x_array must be a numpy ndarray

    Returns:
        (int): Index with the closest value inside the array.

    Example:
    >>> x_scalar = 2
    >>> x_array = np.array([1.0, 2.5, 5.0, 10.9])
    >>> find_closest(x_scalar, x_array)
    1
    """
    if isinstance(x_array, np.ndarray):
        diffabs = np.abs(x_scalar - x_array)
        return np.where(np.min(diffabs) == diffabs)[0][0]
    else:
        raise TypeError("input x_array must be a numpy ndarray")


def _neg_eps_finfo():
    """Returns the negative machine eps.

    Returns:
        (ndarray)
    """
    return -np.finfo(float).eps