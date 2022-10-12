"""Main module for the optimisation of single layers.

- Three media system: incident, thin-film, substrate.
"""

from typing import Any, NamedTuple, Tuple
from collections import namedtuple

import numpy as np
import scipy.interpolate

import effective_medium_models as ema
import refractive_index_database as ridb
from ..helpers.reflectance_utils import reflectance_fresnel_binary_ema


def loss_mean_abs(x, y):
    """Mean abs loss function.

    Args:
        x (ndarray): Array 1
        y (ndarray): Array 2

    Returns:
        (float): Mean absolute value.
    """
    return np.mean(np.abs(x - y))

def cost_function_binary_ema(
    *,
    params: Any,
    beam: NamedTuple,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    ref_experimental: Any,
    ema_binary_func: function = ema.looyenga,
    inverse_ema_func: function = ema.inverse_looyenga,
):
    """Returns the mean-abs cost value between the experimental and calculated reflectance spectra.

    - Uses a binary mixing rule.

    Args:
        params (Tuple, List): Thickness and fraction of component n_void.
        beam (NamedTuple): Beam parameter structure.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        ema_binary_func (function, optional): Mixing rule to use. Defaults to ema.looyenga.
        inverse_ema_func (function, optional): Inverse of the mixing rule selected. Defaults to ema.inverse_looyenga. Needs to return the physical thickness of the layer.

    Returns:
        (float): mean-abs difference between calculated and reflectance spectra.
    """

    reflectance = reflectance_fresnel_binary_ema(
        params = params,
        beam = beam,
        n_incident = n_incident,
        n_substrate = n_substrate,
        n_void = n_void,
        n_matrix = n_matrix,
        ema_binary_func = ema_binary_func,
        inverse_ema_func = inverse_ema_func,
    )

    cost = loss_mean_abs(reflectance, ref_experimental)

    return cost

def linear_search_binary_ema(
    LB: Tuple,
    UB: Tuple,
    beam: Any,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    NUM_GRID: int = 40,
    cost_func: function =  cost_function_binary_ema,
    ema_binary_func: function = ema.looyenga,
    inverse_ema_func: function = ema.inverse_looyenga, # EMA to recover the physical thickness
) -> Any:
    """Returns a linear search for the input lower and upper bounds, calculating the loss between the experimental and theoretical reflectance, using a binary mixing rule.

    Args:
        LB (Tuple): Lower bounds of the thickness and fraction.
        UB (Tuple): Upper bounds of the thickness and fraction.
        beam (NamedTuple): Beam parameter structure.
        ref_experimental (ndarray): Experimental reflectance for a range of wavelengths.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        NUM_GRID (int, optional): Number of point to generate the grids. Defaults to 40.
        cost_func (function, optional): Cost function to estimate the loss. Defaults to cost_function_binary_ema.
        ema_binary_func (function, optional): Mixing rule to use.. Defaults to ema.looyenga.
        inverse_ema_func (function, optional): Inverse of the mixing rule selected. Defaults to ema.inverse_looyenga.

    Returns:
        (NamedTuple): LinearSearchSolSpace:
            error_surface
            min_error_surface
            optimal_params
            grid_params_0
            grid_params_1
    """

    aux1 = np.linspace(LB[0], UB[0], NUM_GRID)
    aux2 = np.linspace(LB[1], UB[1], NUM_GRID)
    error_surface = np.zeros((len(aux1), len(aux2)))

    for j in range(error_surface.shape[0]):
        for k in range(error_surface.shape[1]):
            error_surface[j, k] = cost_function_binary_ema(
                params = [aux1[j], aux2[k]],
                ref_experimental = ref_experimental,
                n_incident = n_incident,
                n_substrate = n_substrate,
                beam = beam,
                n_void = n_void,
                n_matrix = n_matrix,
                ema_binary_func = ema_binary_func,
                inverse_ema_func = inverse_ema_func,
            )

    # Find the minimum of the solution space
    i1, i2 = np.where(error_surface == np.min(error_surface))
    s = [aux1[i1][0], aux2[i2][0]]

    solution = linear_search_solution_space(
        error_surface = error_surface,
        min_error_surface = np.min(error_surface),
        optimal_params = s,
        grid_params_0 = aux1,
        grid_params_1 = aux2,
    )

    return solution

def linear_search_solution_space(
    *,
    error_surface,
    min_error_surface,
    optimal_params,
    grid_params_0,
    grid_params_1,
):

    LinearSearchSolSpace = namedtuple(
        "LinearSearchSolSpace", [
            "error_surface",
            "min_error_surface",
            "optimal_params",
            "grid_params_0",
            "grid_params_1",
        ]
    )

    return LinearSearchSolSpace(
        error_surface,
        min_error_surface,
        optimal_params,
        grid_params_0,
        grid_params_1,
    )