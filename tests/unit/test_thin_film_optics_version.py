"""File to test the version of the package.
"""

from thin_film_optics import __version__


def test_version():
    assert __version__ == "0.1.0"