"""Main module for the transfer matrix method.

- The beam parameter and the layer structures are based on dictionaries to keep mutability and readability of the code. This is useful for simulations that requires changes in the parameters inside these structures. Otherwise, named tuples would have been preferred.
"""

from typing import Any, Dict, List
from functools import reduce
#from collections import namedtuple
import logging

import numpy as np

#from ..helpers.reflectance_utils import snell_cosine_law, phase_shift
from ..helpers.utils import find_closest


logger = logging.getLogger(__name__)


TWOPI = 2.0*np.pi
NEG_EPS = -np.finfo(float).eps


# def tmm_layer_information(
#     *,
#     index_refraction : Any,
#     thickness: Any = np.nan,
#     layer_type: str = "GT",
#     n_wavelength: Any = NEG_EPS,
# ) -> Dict[str, Any]:
#     """Builds a layer structure to be simulated.

#     Args:
#         index_refraction (ndarray, complex) : index of refraction of the layer.
#         thickness (float, optional): thickness of the layer. Defaults to np.nan.
#         layer_type (str, optional): "GT" (geometrical thickness), or "OT" (optical thickness). Defaults to "GT".
#         n_wavelength (float, optional): Central or reference lambda (mostly for multilayer structures). Defaults to -np.finfo(float).eps.

#     Returns:
#         (dict): layer information.
#     """
#     if isinstance(index_refraction, (complex, float, int)):
#         index_refraction = np.array(index_refraction)

#     return {
#         "index_refraction" : index_refraction,
#         "thickness" : thickness,
#         "layer_type" : layer_type,
#         "n_wavelength" : n_wavelength,
#     }


# def beam_parameters(
#     *,
#     wavelength: Any,
#     angle_inc_degree: Any = 0.0,
#     polarisation: Any = 0.5,
#     wavelength_0: Any = NEG_EPS,
# ) -> Dict[str, Any]:
#     """Builds a structure with the beam parameters.

#     Args:
#         wavelength (ndarrray, range, int, float): wavelentgh range.
#         angle_inc_degree (ndarrray, range, int, float, optional): Angle of incidence in degrees. Defaults to 0.0.
#         polarisation (float, optional): polarisation of the wave. Defaults to 0.5.
#         wavelength_0 (float, optional): reference wavelength. Defaults to -np.finfo(float).eps.

#     Returns:
#         (dict):
#             wavelength: range of wavelength
#             angle_inc_degrees: range of angles of incidence in degree
#             angle_inc_radians: range of angles of incidence in radians
#             polarisation: polarisation of the wave
#             wavelength_0: reference wavelength
#             wavelength_0_index: index of the reference wavelength
#     """
#     wavelength_, angle_, polarisation_ = _check_beam_input(
#         wavelength, angle_inc_degree, polarisation
#     )
#     wavelength_0_, wavelength_0_index_ = _check_wavelength_0(wavelength_, wavelength_0)

#     return {
#         "wavelength" : wavelength_,
#         "angle_inc_degrees" : angle_,
#         "angle_inc_radians" : np.deg2rad(angle_),
#         "polarisation" : polarisation_,
#         "wavelength_0" : wavelength_0_,
#         "wavelength_0_index" : wavelength_0_index_,
#     }


# def _check_beam_input(wavelength, angle_inc_degree, polarisation):
#         """Checks and validates input of the beam parameters.
#         """
#         assert(0.0 <= polarisation <= 1.0), "the polarisation should be between 0 and 1"

#         if isinstance(wavelength, (int, float, range, list)):
#             wavelength = np.array([wavelength]).reshape(-1)
#         assert(wavelength.all() > 0.0), "the wavelength should be > 0"

#         if isinstance(angle_inc_degree, (int, float, range, list)):
#             angle_inc_degree = np.array([angle_inc_degree]).reshape(-1)

#         assert(0.0 <= angle_inc_degree.all() <= 90.0), "the angle of incidence should be between 0 and 90 degrees"

#         return wavelength, angle_inc_degree, polarisation


# def _check_wavelength_0(wavelength: Any, wavelength_0: Any) -> Any:
#         """Checks if the input reference wavelength is other than default.

#         Args:
#             wavelength_0 (ndarray): reference wavelength.

#         Returns:
#             (ndarray): reference wavelength.
#             (float): index of the reference wavelength.
#         """
#         if np.allclose(wavelength_0, NEG_EPS):
#             wln = np.mean(wavelength)
#         else:
#             wln = wavelength_0

#         return wln, find_closest(wavelength_0, wavelength)


class TMMOptics():
    """Constructs the structure of the simulation and provides methods for it.
    """

    def __init__(self, beam: Dict[str, Any], layers: List[Dict[str, Any]]) -> None:
        """The initial constructor needs the beam and layers information.

        Args:
            beam (Dict[str, Any]): beam parameters build with beam_parameters function.
            layers (List[Dict[str, Any]]): list of tmm_layer_information for each layer in the system.
        """
        self._beam = beam
        self._layers = layers

    def tmm_spectra(self) -> Any:
        """Calculates the reflection and transmission spectra.

        Args:
            self

        Returns:
            (self): Adds
                reflectance
                reflectance_coeff
                transmittance
                transmittance_coeff
        """
        if not self.__spectra_calculated_yet():
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()
            self = self._transfer_matrix_spectra()
            self.__spectra_calculated = True

        return self

    def __spectra_calculated_yet(self) -> bool:
        """Check if the spectra were calculated already.

        Returns:
            (boolean)
        """
        return hasattr(self, "__spectra_calculated")

    def _add_gt_thickness(self) -> Any:
        """Build physical thicknesses and n_wavelength_0 depending on the input.
        It follows some logic depending on whether n_wavelength_0 was input.

        Args:
            self

        Returns:
            self
                n_wavelength_0
                physical_thickness
        """
        num_layers = len(self._layers)
        n_wavelength_0 = np.zeros(num_layers)
        physical_thickness = np.zeros(num_layers)

        for i, x in enumerate(self._layers):
            # n_wavelength_0 depending on the input
            if x["n_wavelength_0"] == NEG_EPS: # not specified
                n_wavelength_0[i] = np.real(
                    x["index_refraction"][self._beam["wavelength_0_index"]],
                )
            else:
                n_wavelength_0[i] = np.real(x["n_wavelength_0"])
            # Build thickness depending on the input
            physical_thickness[i] = x["thickness"]
            if x["layer_type"] == "OT":
                physical_thickness[i] *= self._beam["wavelength_0"]/n_wavelength_0[i]
        self._n_wavelength_0 = n_wavelength_0
        self._physical_thickness = physical_thickness

        return self

    def _transfer_matrix_spectra(self):
        # Needs to return a dict spectra with the parameters
        # self._spectra["reflectance"] = ...
        # Lo mismo para los otros
        pass

    def tmm_emf(self, layers_split: int = 10) -> Any:
        """Calculates the electromagnetic field distribution.

        This method needs tmm_spectra to be executed before. If you want to calculate everything, use tmm_spectra_emf directly.

        Args:
            self
            layers_split (int, optional): Number of sub-layers to use for the calculation. Defaults to 10.

        Returns:
            self: Adds:
                emfp
                emfs
                depth_sublayers
        """
        if self.__spectra_calculated_yet() and not self.__emf_calculated_yet():
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()

            self = self._transfer_matrix_emf(layers_split)
            self._depth_sublayers = self._layers_depth(layers_split)
            self.__emf_calculated = True
            return self

        return logger.warning("you need to execute tmm_spectra before")

    def __emf_calculated_yet(self) -> bool:
        """Check if the EMF was calculated already.

        Returns:
            (boolean)
        """
        return hasattr(self, "__emf_calculated")

    def _transfer_matrix_emf(self, h):
        pass

    def _layers_depth(self, h: int) -> Any:
        """Provides the multilayer depth considering the h division

        Args:
            self
            h (int, float): number of sub-layers.

        Returns:
            depth_sublayers
        """
        d = self._physical_thickness[1:-1]
        d = d.reshape(len(d), 1)
        l = (d/h)*np.ones((1, h)) # outer product
        l = np.insert(l, 0, 0)
        l = np.cumsum(l)[:-1] # remove last from cumsum

        return l

    def tmm_spectra_emf(self, layers_split: int = 10) -> Any:
        """Calculates the reflectance and transmittance spectra, and also the electromagnetic field distribution.

        Args:
            self
            layers_split (int, optional): Number of sub-layers to use for the calculation. Defaults to 10.

        Returns:
            self: Adds:
                reflectance
                reflectance_coeff
                transmittance
                transmittance_coeff
                emfp
                emfs
                depth_sublayers
        """
        if not self.__emf_calculated_yet() and not self.__spectra_calculated_yet():
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()
            self = self._transfer_matrix_spectra_emf(layers_split)
            self._depth_sublayers = self._layers_depth(layers_split)

            return self

        return logger.info("you already calculated the specta or the EMF.")

    def _transfer_matrix_spectra_emf(self, h: int):
        pass

    def photonic_dispersion(self) -> Any:
        """Calculates photonic dispersion for a perfect infinite crystal.

        Args:
            self

        Returns:
            self: Adds:
                crystal_period
                wavevector_qz
                omega_h
                omega_l
        """

        assert(len(self._layers) > 3), "the number of layers must be greater than 3."

        if not self.__pbg_calculated_yet():
            d = self._physical_thickness[1:3]

            n = [
                self._layers[1]["index_refraction"][self._beam["wavelength_0_index"]],
                self._layers[2]["index_refraction"][self._beam["wavelength_0_index"]],
            ]

            self._crystal_period = np.sum(d)

            # parallel wavevector qz
            self._wavevector_qz = np.sin(self._beam["angle_inc_radians"])*np.pi/2.0

            self = self._photonic_dispersion(d, n)
            self = self._omega_h_l(d)
            self.__pbg_calculated = True

            return self

        return logger.info("you already calculated the photonic disperion")

    def __pbg_calculated_yet(self) -> bool:
        """Check if the photonic dispersion was calculated already.

        Returns:
            (boolean)
        """
        return hasattr(self, "__pbg_calculated")

    def _photonic_dispersion(self, d, n):
        pass

    def _omega_h_l(self, d: Any) -> Any:
        """Calculates the frequencies h and l.

        Args:
            d (ndarray): indexes of the two different layers of the crystal.

        Returns:
            self: Adds:
                omega_h
                omega_l
        """
        n0, n1, n2 = self._n_wavelength_0[0:3]

        self._omega_h = self._crystal_period/np.pi/(d[0]*n1 + d[1]*n2)*np.arccos(-np.abs(n1 - n2)/(n1 + n2))

        self._omega_l = self._crystal_period/np.pi/(d[1]*np.sqrt(n2**2 - n0**2) + d[0]*np.sqrt(n1**2 - n0**2))*np.arccos(np.abs((n1**2*np.sqrt(n2**2 - n0**2) - n2**2*np.sqrt(n1**2 - n0**2))/(n1**2*np.sqrt(n2**2 - n0**2) + n2**2*np.sqrt(n1**2 - n0**2))))

        return self

    @property
    def beam(self):
        """The beam parameters."""
        return self._beam

    @property
    def layers(self):
        """The layers structure."""
        return self._layers

    @property
    def spectra(self):
        """The spectra parameters."""
        return self._spectra

    @property
    def pbg_disperion(self):
        """The photonic dispersion parameters."""
        return self._pbg_dispersion

    @property
    def emf(self):
        """The EMF structure."""
        return self._emf

    @property
    def misc(self):
        """The phase shift, delta, etc parameters."""
        return self._misc




