from __future__ import annotations

import numba as nb
import numpy as np
from numba import guvectorize
from pygama.utils import numba_math_defaults as nb_defaults


@nb.njit(**nb_defaults(parallel=False))
def custom_polynomial(x: np.ndarray, coeffs_exps: np.ndarray) -> np.ndarray:
    midpoint = len(coeffs_exps) // 2
    coeffs, exps = np.split(coeffs_exps, [midpoint])

    result = np.zeros_like(x)
    for i in range(midpoint):
        result += coeffs[i] * x ** (-exps[i])
    return result


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32, float32, float32, float32, float32, float32[:])",
        "void(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64[:])",
    ],
    "(n),(n),(),(),(),(),(),(),(m)",
)
def minuit_polynomial_fit(
    pb_img_polar: np.ndarray,
    mask: np.ndarray,
    deg: int,
    r_sun_pixel: float,
    dphi: float,
    dr: float,
    in_fov: float,
    out_fov: float,
    pb_coeffs_exps: np.ndarray,
):
    """
    Parameters
    ----------
    pb_img_polar
        always prepend the 'r' for raw string (e.g.: r"C:path\to\file.h5")
    deg
        number of monomials in the polynomial regression
    """

    pb_coeffs_exps[:] = np.nan

    params = np.ones(int(deg * 2))  # da rimettere come argomento

    num = np.around((out_fov - in_fov) / (dr * r_sun_pixel), decimals=0)
    r_arr = np.linspace(in_fov, out_fov, int(num) + 1) / r_sun_pixel

    x_data = r_arr[mask == 1]
    y_data = pb_img_polar[mask == 1] * 10**8

    learning_rate = 0.01
    epochs = 1000

    for _ in range(epochs):
        grad = np.zeros_like(params)
        epsilon = 1e-2
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            res_plus = np.sum((custom_polynomial(x_data, params_plus) - y_data) ** 2)
            res_minus = np.sum((custom_polynomial(x_data, params_minus) - y_data) ** 2)
            grad[i] = (res_plus - res_minus) / (2 * epsilon)
        params -= learning_rate * grad

    pb_coeffs_exps[:] = params
