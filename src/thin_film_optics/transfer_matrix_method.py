"""Main module for the transfer matrix method.

"""

from collections import namedtuple
from typing import Any, NamedTuple, List
from functools import reduce
import logging

import numpy as np

#from helpers.reflectance_utils import snell_cosine_law, phase_shift
from helpers.utils import find_closest, _neg_eps_finfo
from .beam_parameters import beam_parameters
from .layer_information import tmmo_layer


logger = logging.getLogger(__name__)

TWOPI = 2.0*np.pi
NEG_EPS = _neg_eps_finfo()


class TMMOptics():
    """Constructs the structure of the simulation and provides methods for it.

    After building your beam parameters and the created a list of layers, you can pass them to this class constructor.
    """

    def __init__(self, beam: NamedTuple, layers: List[NamedTuple]) -> None:
        """The initial constructor needs the beam and layers information.

        Args:
            beam (NamedTuple): beam parameters build with beam_parameters function.
            layers (List[NamedTuple]): list of tmmo_layer for each layer in the system.
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
            if x.n_wavelength_0 == NEG_EPS: # not specified
                n_wavelength_0[i] = np.real(
                    x.index_refraction[self._beam.wavelength_0_index],
                )
            else:
                n_wavelength_0[i] = np.real(x.n_wavelength_0)
            # Build thickness depending on the input
            physical_thickness[i] = x.thickness
            if x.layer_type == "OT":
                physical_thickness[i] *= self._beam.wavelength_0/n_wavelength_0[i]
        self._n_wavelength_0 = n_wavelength_0
        self._physical_thickness = physical_thickness

        return self

    def _transfer_matrix_spectra(self):
        """Computes the reflection and transmission coefficients and spectra with the transfer matrix method.

        Returns:
            self: _description_
        """
        # Initialize variables
        len_ang_inc = len(self._beam.angle_inc_radians)
        len_wavelength = len(self._beam.wavelength)
        num_layers = len(self._layers) - 1 # number of layers in the structure
        ts = np.zeros((len_wavelength, len_ang_inc), dtype=complex)
        tp, rs, rp = ts.copy(), ts.copy(), ts.copy()
        delta = np.zeros((len_wavelength, len_ang_inc, num_layers + 1), dtype=complex)
        adm_p, adm_s = delta.copy(), delta.copy()
        Ms, Mp = np.eye(2, dtype=complex), np.eye(2, dtype=complex)
        I = np.eye(2, dtype=complex)[:, :, np.newaxis]
        cosphi = np.zeros(num_layers + 1, dtype=complex)
        n = cosphi.copy()
        for l, wavelen in enumerate(self._beam.wavelength):
            for a in range(len_ang_inc):
                cosphi[0] = np.cos(self._beam.angle_inc_radians[a])
                adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = _complete_transfer_matrix(self._physical_thickness, cosphi, self._layers, l, wavelen, I, n)
                # calculation of the spectra
                rs[l, a], rp[l, a], ts[l, a], tp[l, a] = _r_t_coefficients(
                    adm_s[l, a, 0], adm_s[l, a, num_layers], Ms,
                    adm_p[l, a, 0], adm_p[l, a, num_layers], Mp,
                )

        self = self._save_spectra_data(adm_p, adm_s, tp, ts, rp, rs)
        self = self._save_phase_admittance(adm_p, adm_s, delta)

        return self

    def _save_spectra_data(self, adm_p, adm_s, tp, ts, rp, rs):
        Spectra = namedtuple(
            "Spectra",
            [
                "coeff_reflection_p",
                "coeff_reflection_s",
                "coeff_transmission_p",
                "coeff_transmission_s",
                "reflectance_p",
                "reflectance_s",
                "reflectance",
                "transmittance_p",
                "transmittance_s",
                "transmittance",
            ]
        )
        Rp, Rs = np.abs(rp)**2, np.abs(rs)**2
        R = (1.0 - self._beam.polarisation)*Rs + self._beam.polarisation*Rp
        Tp = np.real(adm_p[:, :, 0]*adm_p[:, :, -1])*np.abs(tp)**2
        Ts = np.real(adm_s[:, :, 0]*adm_s[:, :, -1])*np.abs(ts)**2
        T = (1.0 - self._beam.polarisation)*Ts + self._beam.polarisation*Tp
        self._spectra = Spectra(
            rp, rs, tp, ts,
            Rp, Rs, R,
            Tp, Ts, T,
        )

        return self

    def _save_phase_admittance(self, admp, adms, delta):
        PhaseAdmittance = namedtuple(
            "PhaseAdmittance",
            [
                "admittance_p",
                "admittance_s",
                "phase",
            ],
        )
        self._phase_admittance = PhaseAdmittance(
            admp, adms, delta,
        )

        return self

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

        return logger.warning("you need to execute tmm_spectra() before")

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
                self._layers[1].index_refraction[self._beam.wavelength_0_index],
                self._layers[2].index_refraction[self._beam.wavelength_0_index],
            ]

            self._crystal_period = np.sum(d)

            # parallel wavevector qz
            self._wavevector_qz = np.sin(self._beam.angle_inc_radians)*np.pi/2.0

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




