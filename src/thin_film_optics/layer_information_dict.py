"""Main module for the layers structure.

- The layer structures is based on dictionaries to keep mutability and readability of the code. This is useful for simulations that requires changes in the parameters inside these structures. Otherwise, named tuples would have been preferred.
"""


from typing import Any, Dict
from collections import namedtuple

import numpy as np


NEG_EPS = -np.finfo(float).eps


def tmm_layer_information(
    *,
    index_refraction : Any,
    thickness: Any = np.nan,
    layer_type: str = "GT",
    n_wavelength: Any = NEG_EPS,
) -> Dict[str, Any]:
    """Builds a layer structure to be simulated.

    Args:
        index_refraction (ndarray, complex) : index of refraction of the layer.
        thickness (float, optional): thickness of the layer. Defaults to np.nan.
        layer_type (str, optional): "GT" (geometrical thickness), or "OT" (optical thickness). Defaults to "GT".
        n_wavelength (float, optional): Central or reference lambda (mostly for multilayer structures). Defaults to -np.finfo(float).eps.

    Returns:
        (dict): layer information.
    """
    if isinstance(index_refraction, (complex, float, int)):
        index_refraction = np.array(index_refraction)

    return {
        "index_refraction" : index_refraction,
        "thickness" : thickness,
        "layer_type" : layer_type,
        "n_wavelength" : n_wavelength,
    }