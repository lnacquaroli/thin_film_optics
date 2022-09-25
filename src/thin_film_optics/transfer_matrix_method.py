"""Main module for the transfer matrix method.

"""

from collections import namedtuple
from typing import Any, NamedTuple, List, Tuple
import functools
import logging

import numpy as np

from helpers.reflectance_utils import snell_cosine_law
from helpers.reflectance_utils import phase_shift
from helpers.reflectance_utils import admittance_p, admittance_s

from helpers.utils import find_closest, _neg_eps_finfo
#from .beam_parameters import beam_parameters
#from .layer_information import tmmo_layer


logger = logging.getLogger(__name__)


# Constants
TWOPI = 2.0*np.pi
NEG_EPS = _neg_eps_finfo()


class TMMOptics():
    """Constructs the structure of the simulation and provides methods for it.

    After building your beam parameters and the created a list of layers, you can pass them to this class constructor.
    """

    def __init__(self, *, beam: NamedTuple, layers: List[NamedTuple]) -> None:
        """The initial constructor needs the beam and layers information.

        Args:
            beam (NamedTuple): beam parameters build with beam_parameters function.
            layers (List[NamedTuple]): list of tmmo_layer for each layer in the system.
        """
        self._beam = beam
        self._layers = layers
        self.__pbg_calculated = False
        self.__emf_calculated = False
        self.__spectra_calculated = False
        self.__adm_phase_calculated = False

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
        if not self.__spectra_calculated:
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()
            self = self._transfer_matrix_spectra()
            self.__spectra_calculated = True
            self.__adm_phase_calculated = True

            return self

        return logger.info("you already calculated the spectra.")

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

    def _transfer_matrix_spectra(self) -> Any:
        """Computes the reflection and transmission coefficients and spectra with the transfer matrix method.

        Returns:
            self: _description_
        """
        # Warm-up
        len_ang_inc, len_wavelength, num_layers = self._initialize_lengths()
        ts, tp, rs, rp = self._initialize_coefficients(len_wavelength, len_ang_inc)
        delta, adm_p, adm_s = self._initialize_admittance_phase(
            len_wavelength, len_ang_inc, num_layers,
        )
        Ms, Mp, I = self._initialize_matrices()
        cosphi, n = self._initialize_cosphi_index(num_layers)

        for l, wavelen in enumerate(self._beam.wavelength):
            for a in range(len_ang_inc):
                cosphi[0] = np.cos(self._beam.angle_inc_radians[a])
                adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = self._complete_transfer_matrix(cosphi, l, wavelen, I, n)
                # calculation of the spectra
                rs[l, a], rp[l, a], ts[l, a], tp[l, a] = self._r_t_coefficients(
                    adm_s[l, a, 0], adm_s[l, a, num_layers], Ms,
                    adm_p[l, a, 0], adm_p[l, a, num_layers], Mp,
                )

        self = self._save_spectra_data(adm_p, adm_s, tp, ts, rp, rs)
        self = self._save_phase_admittance(adm_p, adm_s, delta)

        return self

    def _initialize_lengths(self) -> Tuple[int, int, int]:
        """Initialize length of angle of incidence, wavelength and number of layers.

        Args:
            self

        Returns:
            length of angle of incidence array (int)
            length of wavelength array (int)
            number of layers (int)
        """
        return (
            len(self._beam.angle_inc_radians),
            len(self._beam.wavelength),
            len(self._layers) - 1, # number of layers in the structure
        )

    def _initialize_coefficients(
        self,
        len_wavelength: int,
        len_ang_inc: int,
    ) -> Tuple[Any, Any, Any, Any]:
        """Initialize the arrays for the complex coefficients.

        Args:
            len_wavelength (int): number of wavelengths
            len_ang_inc (int): number of angles of incidence

        Returns:
            ts, tp, rs, rp: complex placeholders for the coefficients
        """
        ts = np.zeros((len_wavelength, len_ang_inc), dtype=complex)
        tp, rs, rp = ts.copy(), ts.copy(), ts.copy()

        return ts, tp, rs, rp

    def _initialize_admittance_phase(
        self,
        len_wavelength: int,
        len_ang_inc: int,
        num_layers: int,
    ) -> Tuple[Any, Any, Any]:
        """Initialize the arrays for the complex admittances and phase shift.

        Args:
            len_wavelength (int): number of wavelengths
            len_ang_inc (int): number of angles of incidence

        Returns:
            delta, adm_p, adm_s: placeholders for the phase shift and admittances.
        """
        delta = np.zeros((len_wavelength, len_ang_inc, num_layers + 1), dtype=complex)
        adm_p, adm_s = delta.copy(), delta.copy()

        return delta, adm_p, adm_s

    def _initialize_matrices(self) -> Tuple[Any, Any, Any]:
        """Initialize complex 2x2 matrices.

        Returns:
            Ms, Mp, I: placeholders for total transfer and identity matrices.

        """
        Ms, Mp = np.eye(2, dtype=complex), np.eye(2, dtype=complex)
        I = np.eye(2, dtype=complex)[:, :, np.newaxis]

        return Ms, Mp, I

    def _complete_transfer_matrix(
        self,
        cosphi: complex,
        l: int,
        wavelen: float,
        I: Any,
        n: Any,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        numlay = len(self._layers)
        n[0] = self._layers[0].index_refraction[l]
        for c in range(1, numlay):
            n[c] = self._layers[c].index_refraction[l]
            # compute angles inside each layer according to the Snell law
            cosphi[c] = snell_cosine_law(n[c - 1], n[c], cosphi[c - 1])
        # phase shifts for each layer: 2*pi = 6.283185307179586
        delta = phase_shift(
            TWOPI, n, self._physical_thickness, cosphi, wavelen,
        ).reshape(-1)
        adm_s, adm_p = admittance_s(n, cosphi), admittance_p(n, cosphi)

        Ms = self._total_transfer_matrix(I, delta, adm_s)
        Mp = self._total_transfer_matrix(I, delta, adm_p)

        return adm_s, adm_p, delta, Ms, Mp

    def _initialize_cosphi_index(self, num_layers: int) -> Tuple[Any, Any]:
        """Initialize the index of refraction and angle of incidence arrays.

        Args:
            num_layers (int): number of layers

        Returns:
            cosphi, n: complex placeholders for the index of refraction and angle of incidence.
        """
        cosphi = np.zeros(num_layers + 1, dtype=complex)
        n = cosphi.copy()

        return cosphi, n

    def _total_transfer_matrix(self, I: Any, delta: Any, adm: Any) -> Any:
        """Total 2x2 transfer matrix.

        Args:
            I (ndarray): identity matrix
            delta (ndarray): complex phase shift
            adm (ndarray): complex admittance

        Returns:
            M: total 2x2 tranfer matrix
        """
        M = functools.reduce(
            np.matmul,
            np.append(I, self._tmatrix(delta[1:-1], adm[1:-1]), 2).T,
        ).T
        M[0, 0], M[1, 1] = M[1, 1], M[0, 0]

        return M

    def _tmatrix(self, beta: complex, adm: complex) -> Any:
        """Optical 2x2 transfer matrix.

        Args:
            beta (complex): phase shift
            adm (complex): admittance

        Returns:
            Q: 2x2 transfer matrix
        """
        cos_beta, sin_beta = np.cos(beta), -1j*np.sin(beta)
        Q = np.array(
            [
                [cos_beta, sin_beta/adm],
                [adm*sin_beta, cos_beta],
            ],
        )

        return Q

    def _r_t_coefficients(
        self,
        adm_s_0: Any,
        adm_s_m: Any,
        Ms: Any,
        adm_p_0: Any,
        adm_p_m: Any,
        Mp: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        """Computes the reflection and transmission coefficients given the admittance and transfer matrix of the whole structure per wavelenth and angle of incidence.

        Args:
            adm_s_0 (complex): admittance of first layer s-wave
            adm_s_m (complex): admittance of last layer s-wave
            Ms (ndarray): total transfer matrix s-wave
            adm_p_0 (complex): admittance of first layer p-wave
            adm_p_m (complex): admittance of last layer p-wave
            Mp (ndarray): total transfer matrix p-wave

        Returns:
            rs: complex reflection coefficient s-wave
            rp: complex reflection coefficient p-wave
            ts: complex transmission coefficient s-wave
            tp: complex transmission coefficient p-wave
        """
        b, c, d = Ms[1,0]/adm_s_0, Ms[0,1]*adm_s_m, Ms[1,1]*adm_s_m/adm_s_0
        rs = (Ms[0,0] - b + c - d)/(Ms[0,0] + b + c + d)
        b, c, d = Mp[1,0]/adm_p_0, Mp[0,1]*adm_p_m, Mp[1,1]*adm_p_m/adm_p_0
        rp = (Mp[0,0] - b + c - d)/(Mp[0,0] + b + c + d)
        ts = 2.0/(adm_s_0*Ms[0,0] + Ms[1, 0] + adm_s_0*adm_s_m*Ms[0, 1] + adm_s_m*Ms[1, 1])
        tp = 2.0/(adm_p_0*Mp[0,0] + Mp[1, 0] + adm_p_0*adm_p_m*Mp[0, 1] + adm_p_m*Mp[1, 1])
        return rs, rp, ts, tp

    def _save_spectra_data(
        self,
        adm_p: Any,
        adm_s: Any,
        tp: Any,
        ts: Any,
        rp: Any,
        rs: Any,
    ) -> Any:
        """Small utility to save the spectra data.

        Args:
            adm_p (ndarray): admittance with p-wave
            adm_s (ndarray): admittance with s-wave
            tp (ndarray): transmission coefficient with p-wave
            ts (ndarray): transmission coefficient with s-wave
            rp (ndarray): reflection coefficient with p-wave
            rs (ndarray): reflection coefficient with p-wave

        Returns:
            self: Adds
                _spectra (namedtuple): results of the spectral simulations
        """
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

    def _save_phase_admittance(
        self,
        admp: Any,
        adms: Any,
        delta: Any,
    ) -> Any:
        """Small utility to save the phase and admittance data.

        Args:
            adm_p (ndarray): admittance with p-wave
            adm_s (ndarray): admittance with s-wave
            delta (ndarray): phase shift

        Returns:
            self: Adds
                _phase_admittance (namedtuple)
        """
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

        Args:
            self
            layers_split (int, optional): Number of sub-layers to use for the calculation. Defaults to 10.

        Returns:
            self: Adds:
                emfp
                emfs
                depth_sublayers
        """
        if not self.__emf_calculated:
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()

            self._num_layers_split = int(layers_split)
            self = self._transfer_matrix_emf()
            self._depth_sublayers = self._layers_depth()
            self.__emf_calculated = True
            self.__adm_phase_calculated = True

            return self

        return logger.info("you already calculated the EMF.")

    def _transfer_matrix_emf(self) -> Any:
        """Calculates the electromagnetic field distribution.

        Returns:
            self
        """
        # Warm-up
        len_ang_inc, len_wavelength, num_layers = self._initialize_lengths()
        # ts, tp, rs, rp = self._initialize_coefficients(len_wavelength, len_ang_inc)
        delta, adm_p, adm_s = self._initialize_admittance_phase(
            len_wavelength, len_ang_inc, num_layers,
        )
        Ms, Mp, I = self._initialize_matrices()
        cosphi, n = self._initialize_cosphi_index(num_layers)
        emfs, emfp = self._initialize_emf(len_wavelength, len_ang_inc)

        for l, wavelen in enumerate(self._beam.wavelength):
            for a in range(len_ang_inc):
                cosphi[0] = np.cos(self._beam.angle_inc_rad[a])
                adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = self._complete_transfer_matrix(cosphi, l, wavelen, I, n)
                emfs[l, a, :] = self._emfield(delta[l, a, :], adm_s[l, a, :], Ms, num_layers + 1)
                emfp[l, a, :] = self._emfield(delta[l, a, :], adm_p[l, a, :], Mp, num_layers + 1)

        self = self._save_emf_data(emfs, emfp)
        self = self._save_phase_admittance(adm_p, adm_s, delta)

        return self

    def _emfield(
        self,
        delta: Any,
        adm: Any,
        M: Any,
        numlay: int,
    ) -> Any:
        """Computes the EMF.

        Args:
            delta (ndarray[complex]): phase shift
            adm (ndarray[complex]): admittance
            M (ndarray[complex]): transfer matrix
            numlay (int): number of layers

        Returns:
            field_intensity (ndarray): amplitude of the EMF
        """
        m0 = np.zeros((2, 2), dtype=complex)
        m1 = np.eye(2, 2, dtype=complex) # Identity 2x2 matrix
        g11 = np.zeros((numlay - 2)*self._num_layers_split, dtype=complex)
        g12 = g11.copy()

        # Divide the phase shift by num_layers_split but keep adm as is for each layer
        m_delta = delta/self._num_layers_split
        for c in range(1, numlay - 1):
            _m1 = self._inverse_tmatrix(m_delta[c], adm[c])
            for j in range(self._num_layers_split):
                k = self._num_layers_split*(c - 1) + j
                m1 = np.matmul(_m1, m1)
                m0 = np.matmul(m1, M)
                g11[k] = m0[0, 0]
                g12[k] = m0[0, 1]

        field_intensity = self._field_intensity(g11, g12, adm[0], adm[-1], M)

        return field_intensity

    def _inverse_tmatrix(self, beta: complex, adm: complex) -> Any:
        """Inverse of the 2x2 optic transfer matrix.

        Args:
            beta (complex): phase shift
            adm (complex): admittance

        Returns:
            Xi: inverse matrix
        """
        cos_beta, sin_beta = np.cos(beta), 1j*np.sin(beta)
        Xi = np.array([
            [cos_beta, sin_beta/adm],
            [adm*sin_beta, cos_beta],
        ])

        return Xi

    def _field_intensity(self, g11, g12, adm_0, adm_m, M):
        """Compute the field intensity.

        Args:
            g11 (ndarray[complex]): matrix
            g12 (ndarray[complex]): matrix
            adm_0 (complex): admittance of the incident medium
            adm_m (complex): admittance of the substrate
            M (ndarray[complex]): transfer matrix

        Returns:
            field_intensity: Field intensity
        """
        fi = np.abs(
            (g11 + adm_m*g12) \
            /(0.25*(adm_0*M[0, 0] + M[1, 0] + adm_0*adm_m*M[0,1] + adm_m*M[1,1]))
        )**2

        return fi

    def _layers_depth(self) -> Any:
        """Provides the multilayer depth considering the h division

        Args:
            self

        Returns:
            depth_sublayers
        """
        thicknesses = self._physical_thickness[1:-1]
        thicknesses = thicknesses.reshape(len(thicknesses), 1)
        # outer product
        l = (thicknesses/self._num_layers_split)*np.ones((1, self._num_layers_split))
        l = np.insert(l, 0, 0)
        l = np.cumsum(l)[:-1] # remove last from cumsum

        return l

    def _initialize_emf(self, len_wavelength: int, len_ang_inc: int) -> Tuple[Any, Any]:
        """Initialize the EMF arrays.

        Args:
            len_wavelength (int): number of wavelengths
            len_ang_inc (int): number of angles of incidence

        Returns:
            emfs, emfp: placeholders for the EMF arrays.
        """
        emfs = np.zeros(
            (len_wavelength, len_ang_inc, (len(self._layers) - 2)*self._num_layers_split),
        )
        emfp = emfs.copy()

        return emfs, emfp

    def _save_emf_data(self, emfs: Any, emfp: Any) -> Any:
        """Small utility to save the EMF data.

        Args:
            emfs (ndarray): EMF s-wave
            emfp (ndarray): EMF p-wave

        Returns:
            self: Adds
                _emf (namedtuple)
        """
        EMF = namedtuple(
            "EMF",
            [
                "emf_p",
                "emf_s",
            ],
        )
        self._emf = EMF(emfp, emfs)

        return self

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
        if not self.__emf_calculated and not self.__spectra_calculated:
            # Build the sequence of n and d depending on the input
            self = self._add_gt_thickness()

            self._num_layers_split = int(layers_split)

            self = self._transfer_matrix_spectra_emf()
            self._depth_sublayers = self._layers_depth()

            self.__emf_calculated = True
            self.__spectra_calculated = True
            self.__adm_phase_calculated = True

            return self

        return logger.info("you already calculated the specta or the EMF.")

    def _transfer_matrix_spectra_emf(self) -> Any:
        """Computes the reflection and transmission coefficients and spectra, and the EMF with the transfer matrix method.

        Args:
            self

        Returns:
            self
        """
        # Warm-up
        len_ang_inc, len_wavelength, num_layers = self._initialize_lengths()
        ts, tp, rs, rp = self._initialize_coefficients(len_wavelength, len_ang_inc)
        delta, adm_p, adm_s = self._initialize_admittance_phase(
            len_wavelength, len_ang_inc, num_layers,
        )
        Ms, Mp, I = self._initialize_matrices()
        cosphi, n = self._initialize_cosphi_index(num_layers)
        emfs, emfp = self._initialize_emf(len_wavelength, len_ang_inc)

        for l, wavelen in enumerate(self._beam.wavelength):
            for a in range(len_ang_inc):
                cosphi[0] = np.cos(self._beam.angle_inc_radians[a])
                adm_s[l, a, :], adm_p[l, a, :], delta[l, a, :], Ms, Mp = self._complete_transfer_matrix(cosphi, l, wavelen, I, n)
                # calculation of the spectra
                rs[l, a], rp[l, a], ts[l, a], tp[l, a] = self._r_t_coefficients(
                    adm_s[l, a, 0], adm_s[l, a, num_layers], Ms,
                    adm_p[l, a, 0], adm_p[l, a, num_layers], Mp,
                )
                emfs[l, a, :] = self._emfield(delta[l, a, :], adm_s[l, a, :], Ms, num_layers + 1)
                emfp[l, a, :] = self._emfield(delta[l, a, :], adm_p[l, a, :], Mp, num_layers + 1)

        self = self._save_spectra_data(adm_p, adm_s, tp, ts, rp, rs)
        self = self._save_phase_admittance(adm_p, adm_s, delta)
        self = self._save_emf_data(emfs, emfp)

        return self

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

        if not self.__pbg_calculated:
            d = self._physical_thickness[1:3]
            n = [
                self._layers[1].index_refraction[self._beam.wavelength_0_index],
                self._layers[2].index_refraction[self._beam.wavelength_0_index],
            ]

            self._crystal_period = np.sum(d)
            self._wavevector_qz = np.sin(self._beam.angle_inc_radians)*np.pi/2.0

            self = self._photonic_dispersion(d, n)
            self = self._omega_h_l(d)
            self.__pbg_calculated = True

            return self

        return logger.info("you already calculated the photonic dispersion.")

    def _photonic_dispersion(self, d: Any, n: Any) -> Any:
        """Calculates the PBG for a binary perfect crystals.

        Args:
            d (Any): thicknesses of the two layers
            n (Any): index of refraction of the two layers

        Returns:
            self: Adds
                _bloch
        """

        def _adm_factor(adm_1, adm_2):
            x = 0.5*(adm_1**2 + adm_2**2)/adm_1/adm_2
            return x

        def _bloch_wavevector(a1, a2, f):
            x = np.cos(a1)*np.cos(a2) - f*np.sin(a1)*np.sin(a2)
            return x

        def _remove_nans(kappa):
            knan = np.isnan(kappa)
            kappa[knan] = kappa[np.logical_not(knan)].max()
            return kappa

        def _initialize_wavevectors(self):
            kpr = np.ones((len(self._beam.wavelength), len(self._beam.angle_inc_rad)))
            return kpr, kpr.copy(), kpr.copy(), kpr.copy()

        kpr, kpi, ksr, ksi = _initialize_wavevectors(self)

        self._omega = 2.0*np.pi/self._beam.wavelength # Angular frequency

        # Angle of incidence of the second layer with Snell's law of cosine
        cosphi_1 = np.cos(self._beam.angle_inc_rad)
        cosphi_2 = np.array([snell_cosine_law(n[0], n[1], a) for a in cosphi_1])

        # Prefactor for Bloch wavevector
        factor_s = _adm_factor(n[0]*cosphi_1, n[1]*cosphi_2)
        factor_p = _adm_factor(n[0]/cosphi_1, n[1]/cosphi_2)
        fsr, fsi = np.real(factor_s), np.imag(factor_s)
        fpr, fpi = np.real(factor_p), np.imag(factor_p)

        # Bloch wavevectors: I split into real and imag because the arccos seems to have a problem with complexes. It is better but not solved this way.
        for a in range(len(cosphi_1)):
            for b in range(len(self._omega)):
                a1 = d[0]*self._omega[b]*n[0]*cosphi_1[a]
                a2 = d[1]*self._omega[b]*n[1]*cosphi_2[a]
                kpr[b, a] = np.arccos(_bloch_wavevector(a1, a2, fpr[a]))
                ksr[b, a] = np.arccos(_bloch_wavevector(a1, a2, fsr[a]))
                kpi[b, a] = np.arccos(_bloch_wavevector(a1, a2, fpi[a]))
                ksi[b, a] = np.arccos(_bloch_wavevector(a1, a2, fsi[a]))

        kp = kpr + kpi*1j
        ks = ksr + ksi*1j
        kp, ks = _remove_nans(kp), _remove_nans(ks)

        self = self._save_bloch_data(kp, ks)

        return self

    def _save_bloch_data(self, kp: Any, ks: Any) -> Any:
        """Utility to save the data of the PBG.

        Args:
            kp (ndarray): Bloch wavevector p-wave
            ks (ndarray): Bloch wavevector s-wave

        Returns:
            self: Adds
                _bloch (namedtuple)
        """
        Bloch = namedtuple(
            "Bloch",
            [
                "bloch_vector_p",
                "bloch_vector_p",
            ],
        )
        self._bloch = Bloch(
            kp / self._crystal_period,
            ks / self._crystal_period,
        )

        return self

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
        """The spectra results."""
        return self._spectra

    @property
    def pbg_dispersion(self):
        """The photonic dispersion results."""
        return self._pbg_dispersion

    @property
    def emf(self):
        """The EMF structure."""
        return self._emf

    @property
    def misc(self):
        """The phase shift, delta, and others results."""
        return self._misc




