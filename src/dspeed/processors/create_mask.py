from __future__ import annotations

import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:])",
        "void(float64[:], float64[:])",
    ],
    "(n),(n)",
)
def create_mask(pb_img_polar: np.ndarray, mask: np.ndarray):
    """
    Parameters
    ----------
    pb_img_polar
        pb_img_polar
    mask
        output mask
    """

    mask[:] = np.nan

    for i in range(len(pb_img_polar)):
        if pb_img_polar[i] == -1:
            mask[i] = False
        else:
            mask[i] = True
