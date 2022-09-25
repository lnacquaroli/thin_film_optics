"""Main module for the layers structure.

- The layer structures is based on named tuples to keep readability of the code.
"""


from typing import Any, NamedTuple
from collections import namedtuple

import numpy as np

from helpers.utils import _neg_eps_finfo


# Constants
NEG_EPS = _neg_eps_finfo()


def tmmo_layer(
    *,
    index_refraction : Any,
    thickness : Any = np.nan,
    layer_type : str = "GT",
    n_wavelength_0 : Any = NEG_EPS,
) -> NamedTuple:
    """Builds a layer structure to be simulated.

    Args:
        index_refraction (ndarray, complex) : index of refraction of the layer.
        thickness (float, optional): thickness of the layer. Defaults to np.nan.
        layer_type (str, optional): "GT" (geometrical thickness), or "OT" (optical thickness). Defaults to "GT".
        n_wavelength (float, optional): Central or reference lambda (mostly for multilayer structures). Defaults to -np.finfo(float).eps.

    Returns:
        (namedtuple): layer information.
            index_refraction
            thickness
            layer_type
            n_wavelength_0
    """
    if isinstance(index_refraction, (complex, float, int)):
        index_refraction = np.array(index_refraction)

    TmmoLayer = namedtuple(
        "TmmoLayer", [
            "index_refraction",
            "thickness",
            "layer_type",
            "n_wavelength_0",
        ]
    )

    return TmmoLayer(
        index_refraction,
        thickness,
        layer_type,
        n_wavelength_0,
    )

# class TmmoLayer(NamedTuple):
#     index_refraction : Any
#     thickness: Any = np.nan
#     layer_type: str = "GT"
#     n_wavelength: Any = NEG_EPS

