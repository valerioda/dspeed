from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(p),(m)",
    **nb_kwargs,
)
def inl_correction(
    w_in: np.ndarray, inl: np.ndarray, w_out: np.ndarray
) -> None:
    """INL correction.

    Note
    ----
    This processor correct the input waveform by applying the INL.

    Parameters
    ----------
    w_in
        the input waveform.
    inl
        inl correction array.
    w_out
        corrected waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

      "wf_corr": {
          "function": "inl_correction",
          "module": "dspeed.processors",
          "args": ["w_in", "inl", "w_out"],
          "unit": "ADC"
       }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.isnan(inl).any():
        raise DSPFatal("INL has nan")

    for i in range(len(w_in)):
        w_out[i] = w_in[i] - inl[int(w_in[i])]
