"""Module that tests the beam parameters.
"""

import numpy as np

from thin_film_optics.beam_parameters import beam_parameters


def test_beam_parameters_scalars():
    """Tests the creation of the namedtuple with scalars in wavelength and angle of incidence.
    """
    wavelength = 400
    angle = 15
    polarisation = 1.0
    beam = beam_parameters(
        wavelength = wavelength,
        angle_inc_degree = angle,
        polarisation = polarisation,
    )

    assert beam.wavelength == wavelength
    assert beam.angle_inc_degrees == angle
    assert beam.angle_inc_radians == np.deg2rad(angle)
    assert beam.polarisation == polarisation
    assert beam.wavelength_0 == wavelength
    assert beam.wavelength_0_index == 0


def test_beam_parameters_arrays():
    """Tests the creation of the namedtuple with arrays in wavelength and angle of incidence.
    """
    wavelength = np.array([400, 500, 600, 700, 800, 900])
    angle = np.array([0, 5, 10, 15, 20, 25, 30])
    polarisation = 0.5
    beam = beam_parameters(
        wavelength = wavelength,
        angle_inc_degree = angle,
        polarisation = polarisation,
    )

    assert np.allclose(beam.wavelength, wavelength)
    assert np.allclose(beam.angle_inc_degrees, angle)
    assert np.allclose(beam.angle_inc_radians, np.deg2rad(angle))
    assert beam.polarisation == polarisation
    assert beam.wavelength_0 == np.mean(wavelength)
    assert beam.wavelength_0_index == 0


def test_beam_parameters_arrays_reference_wavelength():
    """Tests the creation of the namedtuple with arrays in wavelength and angle of incidence, with a reference wavelength.
    """
    wavelength = np.array([400, 500, 600, 700, 800, 900])
    angle = np.array([0, 5, 10, 15, 20, 25, 30])
    polarisation = 0.5
    wavelength_0 = 800
    beam = beam_parameters(
        wavelength = wavelength,
        angle_inc_degree = angle,
        polarisation = polarisation,
        wavelength_0 = wavelength_0,
    )

    assert np.allclose(beam.wavelength, wavelength)
    assert np.allclose(beam.angle_inc_degrees, angle)
    assert np.allclose(beam.angle_inc_radians, np.deg2rad(angle))
    assert beam.polarisation == polarisation
    assert beam.wavelength_0 == wavelength_0
    assert beam.wavelength_0_index == 4


def test_beam_parameters_range():
    """Tests the creation of the namedtuple with ranges in wavelength and angle of incidence.
    """
    wavelength = range(400, 901)
    angle = range(0, 21)
    polarisation = 0.5
    wavelength_0 = 400
    beam = beam_parameters(
        wavelength = wavelength,
        angle_inc_degree = angle,
        polarisation = polarisation,
        wavelength_0 = wavelength_0,
    )

    assert np.allclose(beam.wavelength, wavelength)
    assert np.allclose(beam.angle_inc_degrees, angle)
    assert np.allclose(beam.angle_inc_radians, np.deg2rad(angle))
    assert beam.polarisation == polarisation
    assert beam.wavelength_0 == wavelength_0
    assert beam.wavelength_0_index == 0