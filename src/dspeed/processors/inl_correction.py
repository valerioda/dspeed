from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32[:])",
        "void(float64[:], float64[:], float64, float64[:])",
    ],
    "(n),(p),(),(n)",
    **nb_kwargs,
)
def inl_correction(
    w_in: np.ndarray, inl: np.ndarray, factor: int, w_out: np.ndarray
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
    factor
        sign to apply inl correction.

        * ``+1``: positive
        * ``-1``: negative

    w_out
        corrected waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

      "wf_corr": {
          "function": "inl_correction",
          "module": "dspeed.processors",
          "args": ["w_in", "inl", "pos", "w_out"],
          "unit": "ADC"
       }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(inl).any():
        return

    if (factor != 1) & (factor != -1):
        raise DSPFatal("factor type not found.")

    for i in range(len(w_in)):
        adc_code = int(w_in[i])
        w_out[i] = w_in[i] + factor * inl[adc_code]
