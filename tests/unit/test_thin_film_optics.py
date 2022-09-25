"""Unit tests for the main module for the transfer matrix method.
"""

import numpy as np

from thin_film_optics.transfer_matrix_method import TMMOptics
from thin_film_optics.beam_parameters import beam_parameters
from thin_film_optics.layer_information import tmmo_layer
from thin_film_optics import __version__


def test_tmm_optics_version():
    assert __version__ == "0.1.0"

def test_tmm_optics_constructor():
    index_refraction = np.array([3.4, 3.4, 3.4, 3.4, 3.4], dtype = complex)
    layers = [
        tmmo_layer(index_refraction = index_refraction),
        tmmo_layer(index_refraction = index_refraction),
        tmmo_layer(index_refraction = index_refraction),
    ]
    beam = beam_parameters(
        wavelength = range(400, 901),
        angle_inc_degree = 15.0,
        polarisation = 0.5,
    )
    system = TMMOptics(beam = beam, layers = layers)
    assert isinstance(system, TMMOptics)
