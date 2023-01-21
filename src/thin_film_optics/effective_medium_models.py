"""Module containing effective refractive index models.

- Need to add references.
"""

from typing import Any

import numpy as np


def _check_index_input(refractive_index: Any) -> Any:
    """Check input for the refractive index.

    Args:
        refractive_index (_type_): _description_

    Returns:
        (ndarray): refractive_index
    """
    if isinstance(refractive_index, list):
        return np.array(refractive_index, dtype = complex)

    elif isinstance(refractive_index, np.ndarray):
        if not isinstance(refractive_index.dtype, complex):
            return np.array(refractive_index, dtype = complex)

    return refractive_index

def _check_fraction_input(fraction):
    """Check input for the fraction.

    Args:
        fraction (float): _description_

    Returns:
        (float): fraction
    """
    if isinstance(fraction, (float, int)):
        if not (0 <= fraction <= 1):
            raise ValueError("The input fraction value is out of bounds.")
    if isinstance(fraction, (list, np.ndarray)):
        if not np.all(0 <= fraction <= 1):
            raise ValueError("The input fraction values are out of bounds.")

def lorentz_lorenz(
    refractive_index_1: Any,
    refractive_index_2: Any,
    fraction_1: Any,
) -> Any:
    """Effective index of refraction of a binary mix using the Lorentz-Lorenz model.

    Args:
        refractive_index_1 (ndarray[complex]) : Index of refraction of the component 1
        refractive_index_2 (ndarray[complex]) : Index of refraction of the component 2
        fraction_1 (Any): Fraction of component 1 in the mixture.

    Returns:
        (ndarray[complex]): Effective refractive index.
    """
    n1 = _check_index_input(refractive_index_1)
    n2 = _check_index_input(refractive_index_2)
    _check_fraction_input(fraction_1)

    n1sq, n2sq = n1**2, n2**2
    aux = fraction_1*(n1sq - 1.0)/(n1sq  + 2.0) + (1.0 - fraction_1)*(n2sq - 1.0)/(n2sq + 2.0)

    effective_index = np.sqrt(-1.0 - 2.0*aux)/np.sqrt(aux - 1.0)

    return effective_index

def maxwell_garnett(
    refractive_index_1: Any,
    refractive_index_2: Any,
    fraction_1: Any,
) -> Any:
    """Effective index of refraction of a binary mix using the Maxwell-Garnett model.

    Args:
        refractive_index_1 (ndarray[complex]) : Index of refraction of the component 1
        refractive_index_2 (ndarray[complex]) : Index of refraction of the component 2
        fraction_1 (Any): Fraction of component 1 in the mixture.

    Returns:
        (ndarray[complex]): Effective refractive index.
    """
    n1 = _check_index_input(refractive_index_1)
    n2 = _check_index_input(refractive_index_2)
    _check_fraction_input(fraction_1)

    e1, e2 = n2**2, n1**2 # flipped so f belongs to n_1
    e2_times_e1 = e2*e1

    denom = (-3.0*e2 + fraction_1*(e2 - e1))

    effective_index = np.sqrt(
        (-3.0*e2_times_e1 + 2.0*fraction_1*(e2_times_e1 - e2**2)) / denom
    )

    return effective_index


def bruggeman(
    refractive_index_1: Any,
    refractive_index_2: Any,
    fraction_1: Any,
) -> Any:
    """Effective index of refraction of a binary mix using the Bruggeman model.

    Args:
        refractive_index_1 (ndarray[complex]) : Index of refraction of the component 1
        refractive_index_2 (ndarray[complex]) : Index of refraction of the component 2
        fraction_1 (Any): Fraction of component 1 in the mixture.

    Returns:
        (ndarray[complex]): Effective refractive index.
    """
    n1 = _check_index_input(refractive_index_1)
    n2 = _check_index_input(refractive_index_2)
    _check_fraction_input(fraction_1)

    e1, e2 = n2**2, n1**2 # flipped so f belongs to n_1
    e1_times_2 = 2.0*e1
    A = 3.0*fraction_1*(e2 - e1)

    effective_index = np.sqrt((e1_times_2 - e2 + A + np.sqrt(8.0*e1*e2 + (e1_times_2 - e2 + A)**2))/4.0)

    return effective_index


def looyenga(
    refractive_index_1: Any,
    refractive_index_2: Any,
    fraction_1: Any,
) -> Any:
    """Effective index of refraction of a binary mix using the Looyenga-Landau-Lifshitz model.

    Args:
        refractive_index_1 (ndarray[complex]) : Index of refraction of the component 1
        refractive_index_2 (ndarray[complex]) : Index of refraction of the component 2
        fraction_1 (Any): Fraction of component 1 in the mixture.

    Returns:
        (ndarray[complex]): Effective refractive index.
    """
    n1 = _check_index_input(refractive_index_1)
    n2 = _check_index_input(refractive_index_2)
    _check_fraction_input(fraction_1)

    effective_index = (((1.0 - fraction_1)*(n2**(2/3))) + ((n1**(2/3))*fraction_1))**(3/2)

    return effective_index


def inverse_looyenga(fraction: Any, effective_optical_thickness: Any) -> Any:
    """Returns the physical thickness for a given fraction and optical thickness using the Looyenga-Landau-Lifshitz model with two components.

    Args:
        fraction (Any): fraction of the mixture.
        effective_optical_thickness (ndarray, complex, float): effective optical thickness.

    Returns:
        float: physical thickness
    """
    _check_fraction_input(fraction)

    physical_thickness = effective_optical_thickness/(1.6*(1.0 - fraction) + 1.0)**(1.5)

    return physical_thickness
