from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:])",
        "void(float64[:], float64[:])",
    ],
    "(n),(n)"
)

def create_mask(pB_img_polar: np.ndarray, mask: np.ndarray):
    """
    Parameters
    ----------
    pB_img_polar
        pB_img_polar
    mask
        output mask
    """

    mask[:] = np.nan

    for i in range(len(pB_img_polar)):
        if pB_img_polar[i] == -1:
            mask[i] = False
        else:
            mask[i] = True

