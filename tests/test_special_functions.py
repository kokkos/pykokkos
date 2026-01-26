"""
Test the Kokkos Special Functions defined in
Kokkos_MathematicalSpecialFunctions.hpp

    - expint1
    - erf
    - erfcx
    - cyl_bessel_j0
    - cyl_bessel_y0
    - cyl_bessel_j1
    - cyl_bessel_y1
    - cyl_bessel_i0
    - cyl_bessel_k0
    - cyl_bessel_i1
    - cyl_bessel_k1
    - cyl_bessel_h10
    - cyl_bessel_h11
    - cyl_bessel_h20
    - cyl_bessel_h21

For np.float32, np.float64, np.complex64 and np.complex128 dtypes
"""

import pytest
import numpy as np
import pykokkos as pk
import scipy.special as spsp


@pytest.mark.parametrize(
    "pk_func, sp_func",
    [
        (pk.expint1, spsp.exp1),
        (pk.erf, spsp.erf),
        (pk.erfcx, spsp.erfcx),
        (pk.cyl_bessel_j0, spsp.cyl_bessel_j0),
        (pk.cyl_bessel_y0, spsp.cyl_bessel_y0),
        (pk.cyl_bessel_j1, spsp.cyl_bessel_j1),
        (pk.cyl_bessel_y1, spsp.cyl_bessel_y1),
        (pk.cyl_bessel_i0, spsp.cyl_bessel_i0),
        (pk.cyl_bessel_k0, spsp.cyl_bessel_k0),
        (pk.cyl_bessel_i1, spsp.cyl_bessel_i1),
        (pk.cyl_bessel_k1, spsp.cyl_bessel_k1),
        (pk.cyl_bessel_h10, spsp.cyl_bessel_h10),
        (pk.cyl_bessel_h11, spsp.cyl_bessel_h11),
        (pk.cyl_bessel_h20, spsp.cyl_bessel_h20),
        (pk.cyl_bessel_h21, spsp.cyl_bessel_h21),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_kokkos_special_functions(pk_func, sp_func, dtype):
    # generate random numpy data
    rng = np.random.default_rng()
    arr = rng.uniform(low=-100, high=100, size=400, dtype=dtype)

    expected = sp_function(arr)
    actual = pk_function(arr)

    np.testing.assert_allequal(actual, expected)
