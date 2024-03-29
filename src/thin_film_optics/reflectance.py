"""Collection of functions to deal with the reflectance.
"""

from typing import Any, NamedTuple, Tuple, List, Callable
import numpy as np

from src.helpers.utils import snell_cosine_law, phase_shift
from . import effective_medium_models as ema
from .transfer_matrix_method import TMMOptics


FOURPI = 4.0 * np.pi


def reflectance_fresnel_binary_ema(
    *,
    params: Any,
    beam: NamedTuple,
    n_incident: Any,
    n_substrate: Any,
    n_void: Any,
    n_matrix: Any,
    ema_binary_func: Callable = ema.looyenga,
    inverse_ema_func: Callable = ema.inverse_looyenga,
) -> Any:
    """Calculates the reflection spectrum of a thin film single layer deposited on a
    substrate.
    It uses the binary Looyenga model for the calculation of the index of refraction as
    default.
    It computes the polarisation averaged reflectance for a given angle of incidence other
    than zero.

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
        ema_binary_func (Callable): effective medium approximation (default: looyenga)
        inverse_ema_func (Callable): inverse of ema_binary_func to retrieve the physical
        thickness (default: inverse_looyenga)

    Returns:
        (ndarray): Polarisation averaged reflectance spectrum.
    """
    # Effective index of refraction
    n_effective = ema_binary_func(n_void, n_matrix, params[0])

    # Calculation of the angle of incidence inside each media with Snell law of refraction
    cos_angle_inc_0 = np.cos(beam.angle_inc_radians)
    cos_angle_inc_1 = snell_cosine_law(n_incident, n_effective, cos_angle_inc_0)  # bulk
    cos_angle_inc_2 = snell_cosine_law(
        n_effective, n_substrate, cos_angle_inc_1
    )  # substrate

    # Fresnel coefficients of reflection for both polarizations
    rp01 = fresnel_coefficient(
        n_incident,
        cos_angle_inc_1,
        n_effective,
        cos_angle_inc_0,
    )
    rp12 = fresnel_coefficient(
        n_effective,
        cos_angle_inc_2,
        n_substrate,
        cos_angle_inc_1,
    )
    rs01 = fresnel_coefficient(
        n_incident,
        cos_angle_inc_0,
        n_effective,
        cos_angle_inc_1,
    )
    rs12 = fresnel_coefficient(
        n_effective,
        cos_angle_inc_1,
        n_substrate,
        cos_angle_inc_2,
    )

    phase_shift_ = phase_shift(
        FOURPI,
        inverse_ema_func(params),
        n_effective,
        cos_angle_inc_1,
        beam.wavelength,
    )

    # Complex reflectance
    exp_phase_shift = np.exp(1j * phase_shift_)
    Ap, As = rp12 * exp_phase_shift, rs12 * exp_phase_shift
    rp012 = (rp01 + Ap) / (1.0 + rp01 * Ap)
    rs012 = (rs01 + As) / (1.0 + rs01 * As)

    # Reflectance
    refp, refs = (rp012 * np.conj(rp012)).real, (rs012 * np.conj(rs012)).real

    return ((1.0 - beam.polarisation) * refs) + (beam.polarisation * refp)


def fresnel_coefficient(
    index_refraction_1: Any,
    angle_inc_1: Any,
    index_refraction_2: Any,
    angle_inc_2: Any,
) -> Any:
    """Calculates the Fresnel coefficient of reflection.

    Args:
        index_refraction_1 (npdarray): index of refraction of medium 1
        angle_inc_1 (np.float): angle of incidence of medium 1 (radians)
        index_refraction_2 (npdarray): index of refraction of medium 2
        angle_inc_2 (np.float): angle of incidence of medium 2 (radians)

    Returns:
        (ndarray): Fresnel coefficient of reflection
    """
    prod_1 = index_refraction_1 * angle_inc_1
    prod_2 = index_refraction_2 * angle_inc_2
    return (prod_1 - prod_2) / (prod_1 + prod_2)


def normalize_experimental_reflectance(
    *,
    ref_spectrum: Any,
    ref_reference: Any,
    ref_theory: Any,
    beam: Any,
) -> Any:
    """Normalize the reflectance spectrum dividing by the reference of the material and
    multiplying by the theoretical spectrum of the material.

    Args:
        ref_spectrum (ndarray, 2): experimental reflectance. First column wavelength,
        second column the reflectance spectrum.
        ref_reference (ndarray, 2): reference reflectance. First column wavelength, second
        column the reflectance spectrum.
        ref_theory (ndarray, 2): theoretical reflectance. First column wavelength, second
        column the reflectance spectrum.
        beam (namedtuple): beam parameters.

    Returns:
        ndarray: normalized experimental reflectance spectum interpolated for each beam.
        wavelength value.
    """
    if len(ref_theory) != len(beam.wavelength):
        raise IndexError(
            "length of reference spectrum does not match wavelength array."
        )

    itp_spectrum = np.interp(beam.waelength, ref_spectrum[:, 0], ref_spectrum[:, 1])
    itp_reference = np.interp(beam.waelength, ref_reference[:, 0], ref_reference[:, 1])

    return itp_spectrum / itp_reference * ref_theory


def reflectance_layered(
    *,
    layers: List[NamedTuple],
    beam: NamedTuple,
) -> Tuple[Any, Any, Any]:
    """Returns the reflectance of a n_layers structure for a given beam structure.

    Args:
        beam (NamedTuple): beam parameters build with beam_parameters function.
        layers (List[NamedTuple]): list of tmmo_layer for each layer in the system.

    Returns:
        (ndarray): reflectance of the input structure.
    """
    tmm_optics = TMMOptics(beam=beam, layers=layers)
    tmm_optics.tmm_spectra()
    spectra = tmm_optics.spectra
    return (
        spectra.reflectance_p,
        spectra.reflectance_s,
        spectra.reflectance,
    )
