"""Main module for the beam parameters.

- The beam structures is based on named tuples to keep readability of the code.
"""


from typing import Any, NamedTuple
from collections import namedtuple

import numpy as np

from helpers.utils import find_closest, _neg_eps_finfo


NEG_EPS = _neg_eps_finfo()


def beam_parameters(
    *,
    wavelength : Any,
    angle_inc_degree : Any = 0.0,
    polarisation : Any = 0.5,
    wavelength_0 : Any = NEG_EPS,
) -> NamedTuple:
    """Builds a structure with the beam parameters.

    Args:
        wavelength (ndarrray, range, int, float): wavelentgh range.
        angle_inc_degree (ndarrray, range, int, float, optional): Angle of incidence in degrees. Defaults to 0.0.
        polarisation (float, optional): polarisation of the wave. Defaults to 0.5.
        wavelength_0 (float, optional): reference wavelength. Defaults to -np.finfo(float).eps.

    Returns:
        (namedtuple):
            wavelength: range of wavelength
            angle_inc_degrees: range of angles of incidence in degree
            angle_inc_radians: range of angles of incidence in radians
            polarisation: polarisation of the wave
            wavelength_0: reference wavelength
            wavelength_0_index: index of the reference wavelength
    """
    wavelength_, angle_, polarisation_ = _check_beam_input(
        wavelength, angle_inc_degree, polarisation
    )
    wavelength_0_, wavelength_0_index_ = _check_wavelength_0(wavelength_, wavelength_0)

    BeamParameters = namedtuple(
        "BeamParameters",
        [
            "wavelength",
            "angle_inc_degrees",
            "angle_inc_radians",
            "polarisation",
            "wavelength_0",
            "wavelength_0_index",
        ]
    )

    return BeamParameters(
        wavelength_,
        angle_,
        np.deg2rad(angle_),
        polarisation_,
        wavelength_0_,
        wavelength_0_index_,
    )


def _check_beam_input(wavelength, angle_inc_degree, polarisation):
        """Checks and validates input of the beam parameters.
        """
        assert(0.0 <= polarisation <= 1.0), "the polarisation should be between 0 and 1"

        if isinstance(wavelength, (int, float, range, list)):
            wavelength = np.array([wavelength]).reshape(-1)
        assert(wavelength.all() > 0.0), "the wavelength should be > 0"

        if isinstance(angle_inc_degree, (int, float, range, list)):
            angle_inc_degree = np.array([angle_inc_degree]).reshape(-1)

        assert(0.0 <= angle_inc_degree.all() <= 90.0), "the angle of incidence should be between 0 and 90 degrees"

        return wavelength, angle_inc_degree, polarisation


def _check_wavelength_0(wavelength: Any, wavelength_0: Any) -> Any:
        """Checks if the input reference wavelength is other than default.

        Args:
            wavelength_0 (ndarray): reference wavelength.

        Returns:
            (ndarray): reference wavelength.
            (float): index of the reference wavelength.
        """
        if np.allclose(wavelength_0, NEG_EPS):
            wln = np.mean(wavelength)
        else:
            wln = wavelength_0

        return wln, find_closest(wavelength_0, wavelength)