"""Main module for the optimisation of multilayers.

- Most of the functions are customized to binary 1d photonic crystals with two layers.
- Some functionality for Fabry-Perot type offered also.
"""

from typing import Any, NamedTuple

import numpy as np

from ..helpers.loss_functions_utils import mae_loss_function, mse_loss_function
from .layer_information import tmmo_layer
from .beam_parameters import beam_parameters
from ..helpers.reflectance_utils import reflectance_layered

import effective_medium_models as ema


def objective_func_binary_ema_alpha_depth(
    *,
    params: Any,
    beam: NamedTuple,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    ref_experimental: Any,
    ema_binary_func: function = ema.looyenga,
    loss_func: function = mae_loss_function,
    num_layers_bragg: int = 4,
    num_layers_defect: int = 2,
) -> float:
    """Returns the optimization cost fitting the calculated and experimental reflectance spectra of a multilayer with a binary EMA.

    - It alternates two different layers.
    - It also fits an alpha parameter to simulate a variation of the thicknesses of each layer in depth (from surface to bottom) of the multilayer.

    Args:
        params (Tuple, List or ndarray): Thicknesses, fractions and alpha.
        beam (NamedTuple): Beam parameter structure.
        n_incident (ndarray): Incident index of refraction.
        n_substrate (ndarray): Substrate index of refraction.
        n_void (ndarray): Component 1 of the mixture.
        n_matrix (ndarray): Component 2 of the mixture.
        ref_experimental (ndarray): Experimental reflectance spectrum.
        ema_binary_func (function, optional): Mixing rule to use. Defaults to ema.looyenga.
        loss_func (function): Loss function to calculate the cost. Defaults to mae_loss_function.
        num_layers_bragg (int, optional): Number of layers in the Bragg mirrors at the side of the defect. Defaults to 4.
        num_layers_defect (int, optional): Number of layers in the central defect of the multilayer. Defaults to 2.

    Returns:
        float: Cost of the objective function.
    """

    def _tile_vectors(param, num_layers_bragg, num_layers_defect):

        _aux = np.tile(param, num_layers_bragg)

        arr = np.concatenate(
            (_aux, np.tile(param[0], num_layers_defect), np.flip(_aux))
        )

        return arr

    fractions, thicknesses, alpha = params[0:2], params[2:5], params[5]

    fractions_vec = _tile_vectors(fractions, num_layers_bragg, num_layers_defect) #pvec

    vec_1 = np.arange(2*num_layers_bragg + num_layers_defect)
    thicknesses_vec = _tile_vectors(
        thicknesses, num_layers_bragg, num_layers_defect,
    )*np.concatenate(([1], alpha**vec_1))  #dvec

    reflectance = np.zeros(len(beam.wavelength), dtype = np.float)
    for w in range(len(beam.wavelength)):

        # Build the layer system
        layers = list()

        # Effective layers
        ema_layers = ema_binary_func(n_void[w], n_matrix[w], fractions_vec)

        for d, grad_layer in zip(thicknesses_vec, ema_layers):

            # Build refractive index of layers per wavelength
            N = np.concatenate([n_incident[w], grad_layer, n_substrate[w]])

            # Build layer system
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

    cost = loss_func(reflectance, ref_experimental)

    return cost
