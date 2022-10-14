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
from ..helpers.reflectance_utils import reflectance_layered
from .layer_information import tmmo_layer
from .beam_parameters import beam_parameters


def loss_mean_abs(x, y):
    """Mean abs loss function.

    Args:
        x (ndarray): Array 1
        y (ndarray): Array 2

    Returns:
        (float): Mean absolute value.
    """
    return np.mean(np.abs(x - y))

def objfunc_binary_ema(
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
) -> float:
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

def _naive_search(
    *,
    aux1,
    aux2,
    objective_func,
    ref_experimental,
    n_incident,
    n_substrate,
    n_void,
    n_matrix,
    beam,
    ema_binary_func,
    inverse_ema_func,
):
    """Generates the surface solution given the grids input.

    - Computes the objective_func cost comparing the calculated and experimental reflectance spectra.

    Args:
        aux1 (ndarray): Grid 1.
        aux2 (ndarray): Grid 2.
        objective_func (function): Objective function to use.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        beam (NamedTuple): Beam parameter structure.
        ema_binary_func (function, optional): Mixing rule to use. Defaults to ema.looyenga.
        inverse_ema_func (function, optional): Inverse of the mixing rule selected. Defaults to ema.inverse_looyenga. Needs to return the physical thickness of the layer.

    Returns:
        _type_: _description_
    """
    error_surface = np.zeros((len(aux1), len(aux2)))

    for j in range(error_surface.shape[0]):
        for k in range(error_surface.shape[1]):
            error_surface[j, k] = objective_func(
                params = [aux1[j], aux2[k]],
                ref_experimental = ref_experimental,
                n_incident = n_incident,
                n_substrate = n_substrate,
                beam = beam,
                n_void = n_void,
                n_matrix = n_matrix,
                ema_binary_func = ema_binary_func,
                inverse_ema_func = inverse_ema_func,
            )  # type: ignore

    # Find the minimum of the solution space
    i1, i2 = np.where(error_surface == np.min(error_surface))
    s = [aux1[i1][0], aux2[i2][0]]

    return error_surface, np.min(error_surface), s

def linear_search_binary_ema(
    LB: Tuple,
    UB: Tuple,
    beam: Any,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_grid: int = 40,
    objective_func: function = objective_function_binary_ema,
    ema_binary_func: function = ema.looyenga,
    inverse_ema_func: function = ema.inverse_looyenga, # EMA to recover the physical thickness
) -> NamedTuple:
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
        num_grid (int, optional): Number of point to generate the grids. Defaults to 40.
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

    def _linear_search_solution_space(
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

    aux1 = np.linspace(LB[0], UB[0], num_grid)
    aux2 = np.linspace(LB[1], UB[1], num_grid)

    error_surface, min_error_surface, s = _naive_search(
        aux1 = aux1,
        aux2 = aux2,
        objective_func = objective_func,
        ref_experimental = ref_experimental,
        n_incident = n_incident,
        n_substrate = n_substrate,
        n_void = n_void,
        n_matrix = n_matrix,
        beam = beam,
        ema_binary_func = ema_binary_func,
        inverse_ema_func = inverse_ema_func,
    )

    solution = _linear_search_solution_space(
        error_surface = error_surface,
        min_error_surface = min_error_surface,
        optimal_params = s,
        grid_params_0 = aux1,
        grid_params_1 = aux2,
    )

    return solution

def random_search_binary_ema(
    LB: Tuple,
    UB: Tuple,
    beam: Any,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_grid: int = 40,
    objective_func: function = objective_function_binary_ema,
    ema_binary_func: function = ema.looyenga,
    inverse_ema_func: function = ema.inverse_looyenga, # EMA to recover the physical thickness
    random_seed: int = 1234,
) -> NamedTuple:
    """Returns a random search for the input lower and upper bounds, calculating the loss between the experimental and theoretical reflectance, using a binary mixing rule.

    Args:
        LB (Tuple): Lower bounds of the thickness and fraction.
        UB (Tuple): Upper bounds of the thickness and fraction.
        beam (NamedTuple): Beam parameter structure.
        ref_experimental (ndarray): Experimental reflectance for a range of wavelengths.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        num_grid (int, optional): Number of point to generate the grids. Defaults to 40.
        cost_func (function, optional): Cost function to estimate the loss. Defaults to cost_function_binary_ema.
        ema_binary_func (function, optional): Mixing rule to use.. Defaults to ema.looyenga.
        inverse_ema_func (function, optional): Inverse of the mixing rule selected. Defaults to ema.inverse_looyenga.
        random_seed (int, optional): Seed for the generation of random search. Defaults to 1234.

    Returns:
        (NamedTuple): RandomSearchSolSpace:
            error_surface
            min_error_surface
            optimal_params
            grid_params_0
            grid_params_1
    """

    def _random_search_solution_space(
        *,
        error_surface,
        min_error_surface,
        optimal_params,
        grid_params_0,
        grid_params_1,
    ):

        RandomSearchSolSpace = namedtuple(
            "RandomSearchSolSpace", [
                "error_surface",
                "min_error_surface",
                "optimal_params",
                "grid_params_0",
                "grid_params_1",
            ]
        )

        return RandomSearchSolSpace(
            error_surface,
            min_error_surface,
            optimal_params,
            grid_params_0,
            grid_params_1,
        )

    np.random.seed(random_seed)
    rand_nums = np.random.rand(num_grid)

    aux1 = LB[0] + rand_nums * (UB[0] - LB[0])
    aux2 = LB[1] + rand_nums * (UB[1] - LB[1])

    error_surface, min_error_surface, s = _naive_search(
        aux1 = aux1,
        aux2 = aux2,
        objective_func = objective_func,
        ref_experimental = ref_experimental,
        n_incident = n_incident,
        n_substrate = n_substrate,
        n_void = n_void,
        n_matrix = n_matrix,
        beam = beam,
        ema_binary_func = ema_binary_func,
        inverse_ema_func = inverse_ema_func,
    )

    solution = _random_search_solution_space(
        error_surface = error_surface,
        min_error_surface = min_error_surface,
        optimal_params = s,
        grid_params_0 = aux1,
        grid_params_1 = aux2,
    )

    return solution

def objfunc_binary_ema_fraction_gradient(
    *,
    params: Any,
    beam: NamedTuple,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_layers: int = 100,
    ema_binary_func: function = ema.looyenga,
    cost_func: function = loss_mean_abs,
    gradient_function: function = linear_porosity,
):
    """Returns the loss function between the ref_experimental and the calculated reflectance spectra.

    - It uses a three layers system in which the single layer between the two outter media is represented by a stack of layers with a fraction that changes in position. This simulates the inhomegeneity of the dissolution process in the anodization of the material. (porosity gradient in depth)

    Args:
        params (Tuple, List): Thickness, fraction and alpha. The alpha represents the weight of the linear variation.
        beam (NamedTuple): Beam parameter structure.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        ema_binary_func (function, optional): Mixing rule to use. Defaults to ema.looyenga.
        cost_func (function): Loss function to calculate the cost. Defaults to loss_mean_abs.
        gradient_function (function): Type of inhomogenity to simulate. Defaults to linear_porosity, that builds a linear variation of the fraction in terms of the thickness.

    Returns:
        (ndarray): Reflectance of the gradient fraction layer.
    """

    pvec, dvec = gradient_function(params = params, num_layers = num_layers)

    reflectance = np.zeros(len(beam.wavelength), dtype = np.float)
    for w in range(len(beam.wavelength)):

        # Build the layer system
        layers = list()

        # Effective layers
        gradient_ema_layers = ema_binary_func(n_void[w], n_matrix[w], pvec)

        # Build the index of refraction for all the layers
        for d, grad_layer in zip(dvec, gradient_ema_layers):
            N = np.concatenate([
                n_incident[w],
                grad_layer,
                n_substrate[w],
            ])

            layers.append(
                tmmo_layer(
                    index_refraction = N,
                    thickness = d,
                )
            )

        reflectance[w] = reflectance_layered(
            layers = layers,
            beam = beam_parameters(
                wavelength = beam.wavelength[w],
                angle_inc_degree = beam.angle_inc_degrees,
                polarisation = beam.polarisation,
                wavelength_0 = beam.wavelength_0,
            ),
        )

    cost = cost_func(reflectance, ref_experimental)

    return cost

def linear_porosity(*, params, num_layers):
    """Build the linear porosity array variation in terms of the thickness.

    porosity[i] = params[1] + params[2]*i/num_layers
    thickness[i] = params[0]*i/num_layers

    Args:
        params (Tuple, List): (thickness, fraction, alpha)
        num_layers (int): Number of layers to build up.

    Raises:
        ValueError: If the number of parameters is not 3.

    Returns:
        (Tuple): (Thickness array, Fraction array)
    """

    if len(params) == 3:

        pvec = params[1] \
            + params[2]*(1.0 - np.linspace(1, num_layers, num = num_layers)/num_layers)

        dvec = np.ones(num_layers)*params[0]/num_layers

        return pvec, dvec

    raise ValueError(
        "the input params should have three values: thickness, fraction and alpha parameters."
    )

