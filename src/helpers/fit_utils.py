""" Utility functions for the fitting procedures.
It needs the effective_index_models.py and refractive_index_db.py files in the path.
"""

from typing import Any, Callable

# import numpy as np

from src.thin_film_optics.effective_medium_models import looyenga, inverse_looyenga
from src.thin_film_optics.reflectance import reflectance_fresnel_binary_ema
from .loss_functions_utils import mae_loss_function  # , mse_loss_function

# from .utils import find_closest


def loss_fresnel_ema_binary(
    *,
    params: Any,
    beam: Any,
    ref_spectrum: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    ema_binary_func: Callable = looyenga,
    inverse_ema_func: Callable = inverse_looyenga,
    loss_function: Callable = mae_loss_function,
) -> Any:
    """Calculates the loss function between the experimental reflection spectrum and a
    calculated one using the Fresnel coefficients.
    It works for a system of 3 media: incident,thin-film, substrate.

    Args:
        params (ndarray): initial guesses of the paramters to fit.
            params[0] -> thickness in nanometers
            params[1] -> porosity in range (0, 1)
        beam (namedtuple): beam parameters
        n_incident (ndarray): index of refraction of the incident medium as a function of
        wavelengths
        n_substrate (ndarray): index of refraction of the substrate medium as a function of
        wavelengths
        n_void (ndarray): index of refraction of the void medium as a function of
        wavelengths inside the effective medium
        n_matrix (ndarray): index of refraction of the matrix medium as a function of
        wavelengths inside the effective medium
        ema_binary_func (Callable): effective medium approximation. Defaults to looyenga.
        inverse_ema_func (Callable): inverse of ema_binary_func to retrieve the physical
        thickness. Defaults to inverse_looyenga.
        loss_function (Callable): function to calculate the loss between two arrays.
        Defaults to mae_loss_function.

    Returns:
        (float): loss value between the experimental and calculated reflectances spectra.
    """
    calculated_reflectance = reflectance_fresnel_binary_ema(
        params=params,
        beam=beam,
        n_incident=n_incident,
        n_substrate=n_substrate,
        n_void=n_void,
        n_matrix=n_matrix,
        ema_binary_func=ema_binary_func,
        inverse_ema_func=inverse_ema_func,
    )

    loss = loss_function(calculated_reflectance, ref_spectrum)

    return loss
