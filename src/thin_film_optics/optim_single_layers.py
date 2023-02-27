"""Main module for the optimisation of single layers.

- Three media system: incident, thin-film, substrate.
"""

from typing import Any, NamedTuple, Tuple, List, Callable
from collections import namedtuple

import numpy as np

# import scipy.interpolate

from src.helpers.loss_functions_utils import mae_loss_function  # , mse_loss_function
from .reflectance import reflectance_fresnel_binary_ema
from .reflectance import reflectance_layered
from .layer_information import tmmo_layer
from .beam_parameters import beam_parameters
from . import effective_medium_models as ema

# from . import refractive_index_database as ridb


def objective_func_binary_ema(
    *,
    params: Any,
    beam: NamedTuple,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    ref_experimental: Any,
    ema_binary_func: Callable = ema.looyenga,
    inverse_ema_func: Callable = ema.inverse_looyenga,
    loss_func: Callable = mae_loss_function,
) -> Any:
    """Returns the mean-abs cost value between the experimental and calculated reflectance
    spectra.

    - Uses a binary mixing rule.

    Args:
        params (Tuple, List or ndarray): Thickness and fraction of component n_void.
        beam (NamedTuple): Beam parameter structure.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        ema_binary_func (Callable, optional): Mixing rule to use. Defaults to ema.looyenga.
        inverse_ema_func (Callable, optional): Inverse of the mixing rule selected.
        Defaults to ema.inverse_looyenga. Needs to return the physical thickness of the
        layer.
        loss_func (Callable): Objective function to calculate the cost. Defaults to
        mae_loss_function.

    Returns:
        (float): mean-abs difference between calculated and reflectance spectra.
    """

    reflectance = reflectance_fresnel_binary_ema(
        params=params,
        beam=beam,
        n_incident=n_incident,
        n_substrate=n_substrate,
        n_void=n_void,
        n_matrix=n_matrix,
        ema_binary_func=ema_binary_func,
        inverse_ema_func=inverse_ema_func,
    )

    cost = loss_func(reflectance, ref_experimental)

    return cost


def _naive_search(
    *,
    aux1: Any,
    aux2: Any,
    objective_func: Callable,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    beam: NamedTuple,
    ema_binary_func: Callable,
    inverse_ema_func: Callable,
    loss_func: Callable,
) -> Any:
    """Generates the surface solution given the grids input.

    - Computes the objective_func cost comparing the calculated and experimental
    reflectance spectra.

    Args:
        aux1 (ndarray): Grid 1.
        aux2 (ndarray): Grid 2.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        beam (NamedTuple): Beam parameter structure.
        ema_binary_func (Callable, optional): Mixing rule to use.
        inverse_ema_func (Callable, optional): Inverse of the mixing rule selected. Needs
        to return the physical thickness of the layer.
        loss_func (Callable): Estimation of the cost.

    Returns:
        _type_: _description_
    """
    error_surface = np.zeros((len(aux1), len(aux2)))

    for j in range(error_surface.shape[0]):
        for k in range(error_surface.shape[1]):

            error_surface[j, k] = objective_func(
                params=[aux1[j], aux2[k]],
                ref_experimental=ref_experimental,
                n_incident=n_incident,
                n_substrate=n_substrate,
                beam=beam,
                n_void=n_void,
                n_matrix=n_matrix,
                ema_binary_func=ema_binary_func,
                inverse_ema_func=inverse_ema_func,
                loss_function=loss_func,
            )  # type: ignore

    # Find the minimum of the solution space
    i1, i2 = np.where(error_surface == np.min(error_surface))
    s = [aux1[i1][0], aux2[i2][0]]

    return error_surface, np.min(error_surface), s


def linear_search_binary_ema(
    LB: List[Any],
    UB: List[Any],
    beam: Any,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_grid: int = 40,
    objective_func: Callable = objective_func_binary_ema,
    loss_func: Callable = mae_loss_function,
    ema_binary_func: Callable = ema.looyenga,
    inverse_ema_func: Callable = ema.inverse_looyenga,  # EMA to recover the thickness
) -> NamedTuple:
    """Returns a linear search for the input lower and upper bounds, calculating the loss
    between the experimental and theoretical reflectance, using a binary mixing rule.

    Args:
        LB (Tuple or List): Lower bounds of the thickness and fraction.
        UB (Tuple or List): Upper bounds of the thickness and fraction.
        beam (NamedTuple): Beam parameter structure.
        ref_experimental (ndarray): Experimental reflectance for a range of wavelengths.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        num_grid (int, optional): Number of point to generate the grids. Defaults to 40.
        objective_funct (Callable, optional): Objective function to optimize the loss.
        Defaults to objective_func_binary_ema.
        loss_func (Callable, optional): Cost function to estimate the loss. Defaults to
        mae_loss_function.
        ema_binary_func (Callable, optional): Mixing rule to use.. Defaults to ema.looyenga.
        inverse_ema_func (Callable, optional): Inverse of the mixing rule selected.
        Defaults to ema.inverse_looyenga.

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
        error_surface: Any,
        min_error_surface: float,
        optimal_params: List[Any],
        grid_params_0: Any,
        grid_params_1: Any,
    ) -> NamedTuple:

        LinearSearchSolSpace = namedtuple(
            "LinearSearchSolSpace",
            [
                "error_surface",
                "min_error_surface",
                "optimal_params",
                "grid_params_0",
                "grid_params_1",
            ],
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
        aux1=aux1,
        aux2=aux2,
        objective_func=objective_func,
        ref_experimental=ref_experimental,
        n_incident=n_incident,
        n_substrate=n_substrate,
        n_void=n_void,
        n_matrix=n_matrix,
        beam=beam,
        ema_binary_func=ema_binary_func,
        inverse_ema_func=inverse_ema_func,
        loss_func=loss_func,
    )

    solution = _linear_search_solution_space(
        error_surface=error_surface,
        min_error_surface=min_error_surface,
        optimal_params=s,
        grid_params_0=aux1,
        grid_params_1=aux2,
    )

    return solution


def random_search_binary_ema(
    LB: List[Any],
    UB: List[Any],
    beam: Any,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_grid: int = 40,
    objective_func: Callable = objective_func_binary_ema,
    loss_func: Callable = mae_loss_function,
    ema_binary_func: Callable = ema.looyenga,
    inverse_ema_func: Callable = ema.inverse_looyenga,  # EMA to recover the thickness
    random_seed: int = 1234,
) -> NamedTuple:
    """Returns a random search for the input lower and upper bounds, calculating the loss
    between the experimental and theoretical reflectance, using a binary mixing rule.

    Args:
        LB (Tuple or List): Lower bounds of the thickness and fraction.
        UB (Tuple or List): Upper bounds of the thickness and fraction.
        beam (NamedTuple): Beam parameter structure.
        ref_experimental (ndarray): Experimental reflectance for a range of wavelengths.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        num_grid (int, optional): Number of point to generate the grids. Defaults to 40.
        objective_funct (Callable, optional): Objective function to optimize the loss.
        Defaults to objective_func_binary_ema.
        loss_func (Callable, optional): Cost function to estimate the loss. Defaults to
        mae_loss_function.
        ema_binary_func (Callable, optional): Mixing rule to use.. Defaults to ema.looyenga.
        inverse_ema_func (Callable, optional): Inverse of the mixing rule selected.
        Defaults to ema.inverse_looyenga.
        random_seed (int, optional): Seed for the generation of random search. Defaults to
        1234.

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
        error_surface: Any,
        min_error_surface: float,
        optimal_params: List[Any],
        grid_params_0: Any,
        grid_params_1: Any,
    ) -> NamedTuple:

        RandomSearchSolSpace = namedtuple(
            "RandomSearchSolSpace",
            [
                "error_surface",
                "min_error_surface",
                "optimal_params",
                "grid_params_0",
                "grid_params_1",
            ],
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
        aux1=aux1,
        aux2=aux2,
        objective_func=objective_func,
        ref_experimental=ref_experimental,
        n_incident=n_incident,
        n_substrate=n_substrate,
        n_void=n_void,
        n_matrix=n_matrix,
        beam=beam,
        ema_binary_func=ema_binary_func,
        inverse_ema_func=inverse_ema_func,
        loss_func=loss_func,
    )

    solution = _random_search_solution_space(
        error_surface=error_surface,
        min_error_surface=min_error_surface,
        optimal_params=s,
        grid_params_0=aux1,
        grid_params_1=aux2,
    )

    return solution


def linear_porosity(*, params: Any, num_layers: int) -> Tuple[Any, Any]:
    """Build the linear porosity array variation in terms of the thickness.

    porosity[i] = params[1] + params[2]*i/num_layers
    thickness[i] = params[0]*i/num_layers

    Args:
        params (Tuple, List, or ndarray): (thickness, fraction, alpha)
        num_layers (int): Number of layers to build up.

    Raises:
        ValueError: If the number of parameters is not 3.

    Returns:
        (Tuple): (Thickness array, Fraction array)
    """

    if len(params) == 3:

        pvec = params[1] + params[2] * (
            1.0 - np.linspace(1, num_layers, num=num_layers) / num_layers
        )

        dvec = np.ones(num_layers) * params[0] / num_layers

        return pvec, dvec

    raise ValueError(
        "the input params should have three values: \n"
        + "thickness, fraction and alpha parameters."
    )


def objective_func_binary_ema_fraction_gradient(
    *,
    params: Any,
    beam: NamedTuple,
    ref_experimental: Any,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    num_layers: int = 100,
    ema_binary_func: Callable = ema.looyenga,
    loss_func: Callable = mae_loss_function,
    gradient_function: Callable = linear_porosity,
) -> Any:
    """Returns the loss function between the ref_experimental and the calculated
    reflectance spectra.

    - It uses a three layers system in which the single layer between the two outter media
    is represented by a stack of layers with a fraction that changes in position. This
    simulates the inhomegeneity of the dissolution process in the anodization of the
    material. (porosity gradient in depth)

    Args:
        params (Tuple, List): Thickness, fraction and alpha. The alpha represents the
        weight of the linear variation.
        beam (NamedTuple): Beam parameter structure.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        ema_binary_func (Callable, optional): Mixing rule to use. Defaults to ema.looyenga.
        loss_func (Callable): Loss function to calculate the cost. Defaults to
        mae_loss_function.
        gradient_function (Callable): Type of inhomogenity to simulate. Defaults to
        linear_porosity, that builds a linear variation of the fraction in terms of the
        thickness.

    Returns:
        float: Cost of the objective function.
    """

    pvec, dvec = gradient_function(params=params, num_layers=num_layers)

    reflectance = np.zeros(len(beam.wavelength), dtype=np.float)
    for index_w, value_w in enumerate(beam.wavelength):

        # Build the layer system
        layers = []

        # Effective layers
        ema_layers = ema_binary_func(n_void[index_w], n_matrix[index_w], pvec)

        for thickness, grad_layer in zip(dvec, ema_layers):

            # Build refractive index of layers per wavelength
            n_layers = np.concatenate(
                [n_incident[index_w], grad_layer, n_substrate[index_w]]
            )

            # Build layer system
            layers.append(
                tmmo_layer(
                    index_refraction=n_layers,
                    thickness=thickness,
                )
            )

        reflectance[index_w] = reflectance_layered(
            layers=layers,
            beam=beam_parameters(
                wavelength=value_w,
                angle_inc_degree=beam.angle_inc_degrees,
                polarisation=beam.polarisation,
                wavelength_0=beam.wavelength_0,
            ),
        )

    cost = loss_func(reflectance, ref_experimental)

    return cost
