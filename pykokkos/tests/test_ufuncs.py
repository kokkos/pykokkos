import pykokkos as pk

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest


@pk.workunit
def sqrt_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = sqrt(view[tid])


@pk.workunit
def exp_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = exp(view[tid])


@pk.workunit
def exp2_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = exp2(view[tid])


@pk.workunit
def positive_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = positive(view[tid])


@pk.workunit
def negative_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = negative(view[tid])


@pk.workunit
def absolute_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = abs(view[tid])


@pk.workunit
def fabsolute_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = fabs(view[tid])


@pk.workunit
def rint_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = rint(view[tid])


@pk.workunit
def conjugate_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = conj(view[tid])


@pk.workunit
def sign_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = sign(view[tid])


@pk.workunit
def log_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = log(view[tid])


@pk.workunit
def log2_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = log2(view[tid])


@pk.workunit
def log10_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = log10(view[tid])


@pk.workunit
def expm1_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = expm1(view[tid])


@pk.workunit
def log1p_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = log1p(view[tid])


@pk.workunit
def square_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = square(view[tid])


@pk.workunit
def cbrt_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = cbrt(view[tid])


@pk.workunit
def reciprocal_workunit(tid: int, view: pk.View1D[pk.double]) -> None:
    view[tid] = reciprocal(view[tid])


@pytest.mark.parametrize(
    "kokkos_workunit, numpy_ufunc",
    [
        (sqrt_workunit, np.sqrt),
        (exp_workunit, np.exp),
        pytest.param(
            exp2_workunit, np.exp2, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        pytest.param(
            positive_workunit, np.positive, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        pytest.param(
            negative_workunit, np.negative, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        (absolute_workunit, np.absolute),
        (fabsolute_workunit, np.fabs),
        pytest.param(
            rint_workunit, np.rint, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        pytest.param(
            conjugate_workunit,
            np.conjugate,
            marks=pytest.mark.xfail(reason="see gh-27"),
        ),
        pytest.param(
            sign_workunit, np.sign, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        (log_workunit, np.log),
        (log2_workunit, np.log2),
        (log10_workunit, np.log10),
        (expm1_workunit, np.expm1),
        (log1p_workunit, np.log1p),
        pytest.param(
            square_workunit, np.square, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        pytest.param(
            cbrt_workunit, np.cbrt, marks=pytest.mark.xfail(reason="see gh-27")
        ),
        pytest.param(
            reciprocal_workunit,
            np.reciprocal,
            marks=pytest.mark.xfail(reason="see gh-27"),
        ),
    ],
)
def test_1d_unary_ufunc_vs_numpy(kokkos_workunit, numpy_ufunc):
    # verify that we can easily recreate the functionality
    # of most NumPy "unary" ufuncs on 1D views/arrays without much
    # custom code
    # NOTE: maybe we directly provide i.e., pk.sqrt(view)
    # "pykokkos ufuncs" some day?
    view: pk.View1d = pk.View([10], pk.double)
    view[:] = np.arange(10, dtype=np.float64)
    pk.parallel_for(10, kokkos_workunit, view=view)
    actual = view
    expected = numpy_ufunc(range(10))
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.reciprocal, np.reciprocal),
        (pk.log, np.log),
        (pk.log2, np.log2),
        (pk.log10, np.log10),
        (pk.log1p, np.log1p),
        (pk.sqrt, np.sqrt),
        (pk.sign, np.sign),
        (pk.negative, np.negative),
        (pk.positive, np.positive),
        (pk.square, np.square),
        (pk.sin, np.sin),
        (pk.cos, np.cos),
        (pk.tan, np.tan),
        (pk.logical_not, np.logical_not),
        (pk.exp, np.exp),
        (pk.exp2, np.exp2),
        (pk.mean, np.mean),
        (pk.var, np.var),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
def test_1d_exposed_ufuncs_vs_numpy(pk_ufunc, numpy_ufunc, pk_dtype, numpy_dtype):
    # test the ufuncs we have exposed in the pk namespace
    # vs. their NumPy equivalents
    expected = numpy_ufunc(np.arange(10, dtype=numpy_dtype))

    view: pk.View1d = pk.View([10], pk_dtype)
    view[:] = np.arange(10, dtype=numpy_dtype)
    actual = pk_ufunc(view=view)
    # log10 single-precision needs relaxed tol
    # for now
    if numpy_ufunc in {np.log10, np.cos, np.tan} and numpy_dtype == np.float32:
        assert_allclose(actual, expected, rtol=1.5e-7)
    else:
        assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.add, np.add),
        (pk.subtract, np.subtract),
        (pk.multiply, np.multiply),
        (pk.divide, np.divide),
        (pk.np_matmul, np.matmul),
        (pk.power, np.power),
        (pk.fmod, np.fmod),
        (pk.greater, np.greater),
        (pk.logaddexp, np.logaddexp),
        (pk.floor_divide, np.floor_divide),
        (pk.true_divide, np.true_divide),
        (pk.logaddexp2, np.logaddexp2),
        (pk.logical_and, np.logical_and),
        (pk.logical_or, np.logical_or),
        (pk.logical_xor, np.logical_xor),
        (pk.fmax, np.fmax),
        (pk.fmin, np.fmin),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
def test_multi_array_1d_exposed_ufuncs_vs_numpy(
    pk_ufunc, numpy_ufunc, pk_dtype, numpy_dtype
):

    # test the multi array ufuncs we have exposed
    # in the pk namespace vs. their NumPy equivalents
    expected = numpy_ufunc(
        np.arange(10, dtype=numpy_dtype), np.full(10, 5, dtype=numpy_dtype)
    )

    viewA: pk.View1d = pk.View([10], pk_dtype)
    viewA[:] = np.arange(10, dtype=numpy_dtype)
    viewB: pk.View1d = pk.View([10], pk_dtype)
    viewB[:] = np.full(10, 5, dtype=numpy_dtype)

    actual = pk_ufunc(viewA, viewB)

    assert_allclose(actual, expected)


# TODO: There may be more funcs that support scalars
@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [(pk.add, np.add), (pk.multiply, np.multiply), (pk.divide, np.divide)],
)
@pytest.mark.parametrize("numpy_dtype", [np.float64, np.float32])
def test_scalar_operations_vs_numpy(pk_ufunc, numpy_ufunc, numpy_dtype):
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected = numpy_ufunc(np.array(data, dtype=numpy_dtype), 1)
    actual = pk_ufunc(pk.array(np.array(data, dtype=numpy_dtype)), 1)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.matmul, np.matmul),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
def test_matmul_1d_exposed_ufuncs_vs_numpy(
    pk_ufunc, numpy_ufunc, pk_dtype, numpy_dtype
):
    expected = numpy_ufunc(
        np.arange(10, dtype=numpy_dtype), np.full((10, 1), 2, dtype=numpy_dtype)
    )

    viewA = pk.View([10], pk_dtype)
    viewB = pk.View([10, 1], pk_dtype)
    viewA[:] = np.arange(10, dtype=numpy_dtype)
    viewB[:] = np.full((10, 1), 2, dtype=numpy_dtype)

    with pytest.raises(RuntimeError) as e_info:
        viewC = pk.View([11], pk_dtype)
        viewC[:] = np.arange(11, dtype=numpy_dtype)
        pk_ufunc(viewC, viewB)

    assert (
        e_info.value.args[0]
        == "Input operand 1 has a mismatch in its core dimension (Size 11 is different from 10)"
    )

    actual = pk_ufunc(viewA, viewB)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([4, -1, np.inf]),
        np.array([-np.inf, np.nan, np.inf]),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
def test_1d_sqrt_negative_values(arr, pk_dtype, numpy_dtype):
    # verify sqrt behavior for negative reals,
    # NaN and infinite values
    expected = np.sqrt(arr, dtype=numpy_dtype)
    view: pk.View1d = pk.View([arr.size], pk_dtype)
    view[:] = arr
    actual = pk.sqrt(view=view)
    assert_allclose(actual, expected)


def test_caching():
    # regression test for gh-34
    expected = np.reciprocal(np.arange(10, dtype=np.float32))
    for i in range(300):
        view: pk.View1d = pk.View([10], pk.float)
        view[:] = np.arange(10, dtype=np.float32)
        actual = pk.reciprocal(view=view)
        assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.reciprocal, np.reciprocal),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
def test_2d_exposed_ufuncs_vs_numpy(pk_ufunc, numpy_ufunc, pk_dtype, numpy_dtype):
    rng = default_rng(123)
    in_arr = rng.random((5, 5)).astype(numpy_dtype)
    expected = numpy_ufunc(in_arr)

    view: pk.View2d = pk.View([5, 5], pk_dtype)
    view[:] = in_arr
    actual = pk_ufunc(view=view)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.np_matmul, np.matmul),
    ],
)
@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
@pytest.mark.parametrize(
    "test_dim", [[4, 4, 4, 4], [4, 3, 3, 4], [1, 1, 1, 1], [2, 5, 5, 1]]
)
def test_np_matmul_2d_2d_vs_numpy(
    pk_ufunc, numpy_ufunc, pk_dtype, numpy_dtype, test_dim
):

    N1 = test_dim[0]
    M1 = test_dim[1]
    N2 = test_dim[2]
    M2 = test_dim[3]
    rng = default_rng(123)
    np1 = rng.random((N1, M1)).astype(numpy_dtype)
    np2 = rng.random((N2, M2)).astype(numpy_dtype)
    expected = numpy_ufunc(np1, np2)

    view1: pk.View2d = pk.View([N1, M1], pk_dtype)
    view1[:] = np1
    view2: pk.View2d = pk.View([N2, M2], pk_dtype)
    view2[:] = np2
    actual = pk_ufunc(view1, view2)

    assert_allclose(actual, expected, rtol=1.5e-7)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.np_matmul, np.matmul),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
@pytest.mark.parametrize("test_dim", [[4, 4, 4], [4, 3, 3], [1, 1, 1], [2, 5, 5]])
def test_np_matmul_2d_1d_vs_numpy(pk_ufunc, numpy_ufunc, numpy_dtype, test_dim):

    N1 = test_dim[0]
    M1 = test_dim[1]
    N2 = test_dim[2]
    rng = default_rng(123)
    np1 = rng.random((N1, M1)).astype(numpy_dtype)
    np2 = rng.random(N2).astype(numpy_dtype)
    expected = numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2)
    actual = pk_ufunc(view1, view2)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.np_matmul, np.matmul),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
@pytest.mark.parametrize("test_dim", [[4, 4, 4], [3, 3, 6], [1, 1, 1], [5, 5, 1]])
def test_np_matmul_1d_2d_vs_numpy(pk_ufunc, numpy_ufunc, numpy_dtype, test_dim):

    N1 = test_dim[0]
    N2 = test_dim[1]
    M2 = test_dim[2]
    rng = default_rng(123)
    np1 = rng.random(N1).astype(numpy_dtype)
    np2 = rng.random((N2, M2)).astype(numpy_dtype)
    expected = numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2)
    actual = pk_ufunc(view1, view2)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
@pytest.mark.parametrize(
    "test_dim", [[4, 3, 3], [3, 1, 6], [1, 4, 2], [5, 6, 1], [4, 3, 2, 1], [2, 3, 2, 4]]
)
def test_np_matmul_fails(numpy_dtype, test_dim):
    N1 = None
    N2 = None
    M1 = None
    M2 = None
    np1 = None
    rng = default_rng(123)

    if len(test_dim) == 3:
        N1 = test_dim[0]
        N2 = test_dim[1]
        M2 = test_dim[2]
        np1 = rng.random(N1).astype(numpy_dtype)

    if len(test_dim) == 4:
        N1 = test_dim[0]
        M1 = test_dim[1]
        N2 = test_dim[2]
        M2 = test_dim[3]
        np1 = rng.random((N1, M1)).astype(numpy_dtype)

    np2 = rng.random((N2, M2)).astype(numpy_dtype)

    with pytest.raises(RuntimeError) as e_info:
        view1 = pk.array(np1)
        view2 = pk.array(np2)
        pk.np_matmul(view1, view2)  # Should fail with 1d x 2d

    err_np_matmul = (
        "Matrix dimensions are not compatible for multiplication: {} and {}".format(
            view1.shape, view2.shape
        )
    )
    assert e_info.value.args[0] == err_np_matmul

    with pytest.raises(RuntimeError) as e_info:
        pk.np_matmul(view2, view1)  # should fail with 2d x 1 as well

    err_np_matmul = (
        "Matrix dimensions are not compatible for multiplication: {} and {}".format(
            view2.shape, view1.shape
        )
    )
    assert e_info.value.args[0] == err_np_matmul


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [(pk.subtract, np.subtract), (pk.add, np.add), (pk.multiply, np.multiply)],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
def test_multi_array_2d_exposed_ufuncs_vs_numpy(pk_ufunc, numpy_ufunc, numpy_dtype):
    N = 4
    M = 7
    rng = default_rng(123)
    np1 = rng.random((N, M)).astype(numpy_dtype)
    np2 = rng.random((N, M)).astype(numpy_dtype)
    expected = numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2)
    actual = pk_ufunc(view1, view2)

    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.subtract, np.subtract),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
@pytest.mark.parametrize(
    "test_dim",
    [[4, 3, 1, 1], [4, 3, 1, 3], [4, 3, 4, 1], [4, 3, 1], [4, 3, 3], [4, 3], [4]],
)
def test_broadcast_array_exposed_ufuncs_vs_numpy(
    pk_ufunc, numpy_ufunc, numpy_dtype, test_dim
):

    np1 = None
    np2 = None
    rng = default_rng(123)
    scalar = 3.0

    if len(test_dim) == 4:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = rng.random((test_dim[2], test_dim[3])).astype(numpy_dtype)
    elif len(test_dim) == 3:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = rng.random((test_dim[2])).astype(numpy_dtype)
    elif len(test_dim) == 2:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = scalar  # 2d with scalar
    elif len(test_dim) == 1:
        np1 = rng.random((test_dim[0])).astype(numpy_dtype)
        np2 = scalar  # 1d with scalar
    else:
        raise NotImplementedError(
            "Invalid test conditions: Broadcasting operations are only supported uptil 2D"
        )

    assert (
        np1 is not None and np2 is not None
    ), "Invalid test conditions: Are parameters uptil 2D?"

    expected = numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2) if isinstance(np2, np.ndarray) else np2
    actual = pk_ufunc(view1, view2)

    assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "pk_dtype, numpy_dtype",
    [
        (pk.double, np.float64),
        (pk.float, np.float32),
    ],
)
@pytest.mark.parametrize(
    "in_arr",
    [
        np.array([-5, 4.5, np.nan]),
        np.array([np.nan, np.nan, np.nan]),
    ],
)
def test_sign_1d_special_cases(in_arr, pk_dtype, numpy_dtype):
    in_arr = in_arr.astype(numpy_dtype)
    view: pk.View1D = pk.View([in_arr.size], pk_dtype)
    view[:] = in_arr
    expected = np.sign(in_arr)
    actual = pk.sign(view=view)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.copyto, np.copyto),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
def test_copyto_1d(pk_ufunc, numpy_ufunc, numpy_dtype):
    N = 4
    M = 7
    rng = default_rng(123)
    np1 = rng.random((N, M)).astype(numpy_dtype)
    np2 = rng.random((N, M)).astype(numpy_dtype)
    numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2)
    pk_ufunc(view1, view2)

    assert_allclose(np1, view1)


@pytest.mark.parametrize(
    "pk_ufunc, numpy_ufunc",
    [
        (pk.subtract, np.subtract),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        (np.float64),
        (np.float32),
    ],
)
@pytest.mark.parametrize(
    "test_dim",
    [
        [4, 3, 4, 3],
        [4, 3, 1, 1],
        [4, 3, 1, 3],
        [4, 3, 4, 1],
        [4, 3, 1],
        [4, 3, 3],
        [4, 3],
        [4],
    ],
)
def test_copyto_broadcast_2d(pk_ufunc, numpy_ufunc, numpy_dtype, test_dim):
    np1 = None
    np2 = None
    rng = default_rng(123)
    scalar = 3.0

    if len(test_dim) == 4:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = rng.random((test_dim[2], test_dim[3])).astype(numpy_dtype)
    elif len(test_dim) == 3:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = rng.random((test_dim[2])).astype(numpy_dtype)
    elif len(test_dim) == 2:
        np1 = rng.random((test_dim[0], test_dim[1])).astype(numpy_dtype)
        np2 = scalar  # 2d with scalar
    elif len(test_dim) == 1:
        np1 = rng.random((test_dim[0])).astype(numpy_dtype)
        np2 = scalar  # 1d with scalar
    else:
        raise NotImplementedError(
            "Invalid test conditions: Broadcasting operations are only supported uptil 2D"
        )

    assert (
        np1 is not None and np2 is not None
    ), "Invalid test conditions: Are parameters uptil 2D?"

    numpy_ufunc(np1, np2)

    view1 = pk.array(np1)
    view2 = pk.array(np2) if isinstance(np2, np.ndarray) else np2
    pk_ufunc(view1, view2)

    assert_allclose(np1, view1)


@pytest.mark.parametrize(
    "input_dtype",
    [
        pk.double,
        pk.float,
    ],
)
@pytest.mark.parametrize(
    "pk_ufunc",
    [
        pk.floor,
        pk.round,
        pk.ceil,
        pk.trunc,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1],
        [1, 1],
        [1, 1, 1],
    ],
)
def test_rounding_dtype_preservation(input_dtype, pk_ufunc, shape):
    # at the time of writing the array API standard
    # conformance test suite doesn't appear to probe
    # floating point data types for many of the rounding
    # functions

    # for now, we simply test data type preservation
    # of output vs. input so that we flush these codepaths
    # a bit
    view = pk.View(shape, input_dtype)
    actual_dtype = pk_ufunc(view).dtype
    assert actual_dtype.value == input_dtype.value
