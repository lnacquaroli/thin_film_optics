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

    raise TypeError("input x_array must be a numpy ndarray")


def _neg_eps_finfo():
    """Returns the negative machine eps.

    Returns:
        (ndarray)
    """
    return -np.finfo(float).eps


def phase_shift(
    constant: float,
    thickness: Any,
    index_refraction: complex,
    cos_angle_inc: complex,
    wavelength: float,
) -> complex:
    """Calculates the phase shift.

    Args:
        constant (float): 2*pi or 4*pi depending on the method.
        thickness (float): thickness of the layer.
        index_refraction (ndarray): index of refraction of the layer.
        cos_angle_inc (float): cosine of the angle of incidence to the layer.
        wavelength (ndarray): wavelength array.

    Returns:
        (ndarray): phase shift.
    """
    return constant * thickness * index_refraction * cos_angle_inc / wavelength


def admittance_p(index_refraction: Any, cosangle: Any) -> Any:
    """Admittance of p-wave.

    Args:
        index_refraction (ndarray, complex): index of refraction
        cosangle (ndarray, complex): cosine of angle of incidence

    Returns:
        adm_p (ndarray, complex): admittance of p-wave
    """
    _check_admittance_angle_index_array(index_refraction, cosangle)
    return index_refraction / cosangle


def admittance_s(index_refraction: Any, cosangle: Any) -> Any:
    """Admittance of s-wave.

    Args:
        index_refraction (ndarray, complex): index of refraction
        cosangle (ndarray, complex): cosine of angle of incidence

    Returns:
        adm_s (ndarray, complex): admittance of s-wave
    """
    _check_admittance_angle_index_array(index_refraction, cosangle)
    return index_refraction * cosangle


def _check_admittance_angle_index_array(n: Any, a: Any) -> Any:
    assert len(n) == len(a), "index of refraction and angle lengths must match"


def snell_cosine_law(
    index_refraction_1: Any,
    index_refraction_2: Any,
    cos_angle: Any,
) -> Any:
    """Snell's law in cosine form. Returns the cosine already.

    Args:
        index_refraction_1 (ndarray): index of refraction of medium 1
        index_refraction_2 (ndarray): index of refraction of medium 2
        cos_angle (float): cosine of angle of incidence of medium 1

    Returns:
        (float): cosine of angle of refraction in medium 2

    Examples:
    >>> n_1, n_2 = 1.0, 1.5
    >>> theta = np.deg2rad(15)
    >>> cos_theta = np.cos(theta)
    >>> cos_theta_2 = snell_cosine_law(n_1, n_2, cos_theta)
    >>> cos_theta_2
    0.9850014555865657
    >>> theta_2 = np.arccos(cos_theta_2)
    >>> theta_2
    0.17341388536678448
    >>> np.rad2deg(theta_2)
    9.935883740482216
    """
    return np.sqrt(
        1.0 - (index_refraction_1 / index_refraction_2) ** 2 * (1.0 - cos_angle**2)
    )
