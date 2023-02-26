"""Module that tests the tmmo_layer namedtuple.
"""

import numpy as np

from src.thin_film_optics.layer_information import tmmo_layer


def test_tmmo_layer_scalar_index_refraction():
    """Tests with index of refraction scalar."""
    index_refraction = 3.4
    layer = tmmo_layer(index_refraction=index_refraction)

    assert layer.index_refraction == index_refraction
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_array_index_refraction():
    """Tests with index of refraction array."""
    index_refraction = np.array([3.4, 3.4, 3.4, 3.4, 3.4])
    layer = tmmo_layer(index_refraction=index_refraction)

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_list_index_refraction():
    """Tests with index of refraction list."""
    index_refraction = [3.4, 3.4, 3.4, 3.4, 3.4]
    layer = tmmo_layer(index_refraction=index_refraction)

    assert layer.index_refraction == index_refraction
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_array_complex_index_refraction():
    """Tests with index of refraction array complex."""
    index_refraction = np.array([3.4, 3.4, 3.4, 3.4, 3.4], dtype=complex)
    layer = tmmo_layer(index_refraction=index_refraction)

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_thickness():
    """Tests with thickness."""
    index_refraction = 3.4
    thickness = 100
    layer = tmmo_layer(
        index_refraction=index_refraction,
        thickness=thickness,
    )

    assert np.allclose(layer.index_refraction, index_refraction)
    assert layer.thickness == thickness
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_layer_type():
    """Tests with layer_type."""
    index_refraction = 3.4
    layer = tmmo_layer(
        index_refraction=index_refraction,
        layer_type="OT",
    )

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "OT"
    assert np.allclose(layer.n_wavelength_0, -np.finfo(float).eps)


def test_tmmo_layer_n_wavelength_0_scalar():
    """Tests with n_wavelength_0 scalar."""
    index_refraction = 3.4
    n_wavelength_0 = 2.5
    layer = tmmo_layer(
        index_refraction=index_refraction,
        n_wavelength_0=n_wavelength_0,
    )

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, n_wavelength_0)


def test_tmmo_layer_n_wavelength_0_array():
    """Tests with n_wavelength_0 array."""
    index_refraction = 3.4
    n_wavelength_0 = np.array([2.5])
    layer = tmmo_layer(
        index_refraction=index_refraction,
        n_wavelength_0=n_wavelength_0,
    )

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, n_wavelength_0)


def test_tmmo_layer_n_wavelength_0_array_complex():
    """Tests with n_wavelength_0 array complex."""
    index_refraction = 3.4
    n_wavelength_0 = np.array([2.5], dtype=complex)
    layer = tmmo_layer(
        index_refraction=index_refraction,
        n_wavelength_0=n_wavelength_0,
    )

    assert np.allclose(layer.index_refraction, index_refraction)
    assert np.isnan(layer.thickness)
    assert layer.layer_type == "GT"
    assert np.allclose(layer.n_wavelength_0, n_wavelength_0)


def test_tmm_optics_layers_list_length():
    silicon = np.array([3.4, 3.4, 3.4, 3.4, 3.4], dtype=complex)
    glass = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=complex)
    air = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=complex)
    layers = [
        tmmo_layer(index_refraction=air),
        tmmo_layer(index_refraction=glass),
        tmmo_layer(index_refraction=silicon),
    ]
    assert len(layers) == 3
