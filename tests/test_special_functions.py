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
from functools import partial


@pk.workunit
def special_function_workunit(tid, out, arr, flag):
    if flag == 0:
        out[tid] = expint1(arr[tid])
    elif flag == 1:
        out[tid] = erf(arr[tid])
    elif flag == 2:
        out[tid] = erfcx(arr[tid])
    elif flag == 3:
        out[tid] = cyl_bessel_j0(arr[tid])
    elif flag == 4:
        out[tid] = cyl_bessel_y0(arr[tid])
    elif flag == 5:
        out[tid] = cyl_bessel_j1(arr[tid])
    elif flag == 6:
        out[tid] = cyl_bessel_y1(arr[tid])
    elif flag == 7:
        out[tid] = cyl_bessel_i0(arr[tid])
    elif flag == 8:
        out[tid] = cyl_bessel_k0(arr[tid])
    elif flag == 9:
        out[tid] = cyl_bessel_i1(arr[tid])
    elif flag == 10:
        out[tid] = cyl_bessel_k1(arr[tid])
    elif flag == 11:
        out[tid] = cyl_bessel_h10(arr[tid])
    elif flag == 12:
        out[tid] = cyl_bessel_h11(arr[tid])
    elif flag == 13:
        out[tid] = cyl_bessel_h20(arr[tid])
    elif flag == 14:
        out[tid] = cyl_bessel_h21(arr[tid])


@pytest.mark.parametrize(
    "flag, sp_func",
    [
        (0, spsp.exp1),
        (1, spsp.erf),
        (2, spsp.erfcx),
        (3, partial(spsp.jv, v=0)),
        (4, partial(spsp.yv, v=0)),
        (5, partial(spsp.jv, v=1)),
        (6, partial(spsp.yv, v=1)),
        (7, partial(spsp.iv, v=0)),
        (8, partial(spsp.kv, v=0)),
        (9, partial(spsp.iv, v=1)),
        (10, partial(spsp.kv, v=1)),
        (11, partial(spsp.hankel1, v=0)),
        (12, partial(spsp.hankel1, v=1)),
        (13, partial(spsp.hankel2, v=0)),
        (14, partial(spsp.hankel2, v=1)),
    ],
)
@pytest.mark.parametrize("dtype", [np.complex128])  # TODO: add more types
def test_kokkos_special_functions(flag, sp_func, dtype):
    # generate random numpy data
    N = 400
    rng = np.random.default_rng()
    if dtype == np.complex64:
        real = rng.random(size=N, dtype=np.float32) * N - (N // 2)
        imag = rng.random(size=N, dtype=np.float32) * N - (N // 2)
        arr = real + 1j * imag
    elif dtype == np.complex128:
        real = rng.random(size=N, dtype=np.float32) * N - (N // 2)
        imag = rng.random(size=N, dtype=np.float32) * N - (N // 2)
        arr = real + 1j * imag
    else:
        arr = rng.random(size=N, dtype=dtype) * N - (N // 2)

    expected = np.empty_like(arr)
    pk.parallel_for(N, special_function_workunit, out=expected, arr=arr, flag=flag)
    actual = pk_function(arr)

    np.testing.assert_allequal(actual, expected)
