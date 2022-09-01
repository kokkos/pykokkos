import pykokkos as pk

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

@pk.workload
class SqrtView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = sqrt(self.view[tid])


@pk.workload
class ExpView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = exp(self.view[tid])

@pk.workload
class Exp2View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = exp2(self.view[tid])

@pk.workload
class PositiveView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = positive(self.view[tid])

@pk.workload
class NegativeView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = negative(self.view[tid])

@pk.workload
class AbsoluteView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = abs(self.view[tid])

@pk.workload
class FabsoluteView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = fabs(self.view[tid])

@pk.workload
class RintView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = rint(self.view[tid])

@pk.workload
class ConjugateView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = conj(self.view[tid])

@pk.workload
class SignView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = sign(self.view[tid])

@pk.workload
class LogView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log(self.view[tid])

@pk.workload
class Log2View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log2(self.view[tid])

@pk.workload
class Log10View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log10(self.view[tid])

@pk.workload
class Expm1View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = expm1(self.view[tid])

@pk.workload
class Log1pView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log1p(self.view[tid])

@pk.workload
class SquareView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = square(self.view[tid])

@pk.workload
class CbrtView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = cbrt(self.view[tid])

@pk.workload
class ReciprocalView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = reciprocal(self.view[tid])

@pytest.mark.parametrize("kokkos_test_class, numpy_ufunc", [
    (SqrtView1D, np.sqrt),
    (ExpView1D, np.exp),
    pytest.param(Exp2View1D, np.exp2, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(PositiveView1D, np.positive, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(NegativeView1D, np.negative, marks=pytest.mark.xfail(reason="see gh-27")),
    (AbsoluteView1D, np.absolute),
    (FabsoluteView1D, np.fabs),
    pytest.param(RintView1D, np.rint, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(ConjugateView1D, np.conjugate, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(SignView1D, np.sign, marks=pytest.mark.xfail(reason="see gh-27")),
    (LogView1D, np.log),
    (Log2View1D, np.log2),
    (Log10View1D, np.log10),
    (Expm1View1D, np.expm1),
    (Log1pView1D, np.log1p),
    pytest.param(SquareView1D, np.square, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(CbrtView1D, np.cbrt, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(ReciprocalView1D, np.reciprocal, marks=pytest.mark.xfail(reason="see gh-27")),
])
def test_1d_unary_ufunc_vs_numpy(kokkos_test_class, numpy_ufunc):
    # verify that we can easily recreate the functionality
    # of most NumPy "unary" ufuncs on 1D views/arrays without much
    # custom code
    # NOTE: maybe we directly provide i.e., pk.sqrt(view)
    # "pykokkos ufuncs" some day?
    view: pk.View1d = pk.View([10], pk.double)
    view[:] = np.arange(10, dtype=np.float64)
    kokkos_instance = kokkos_test_class(threads=10, view=view)
    pk.execute(pk.ExecutionSpace.Default, kokkos_instance)
    actual = kokkos_instance.view
    expected = numpy_ufunc(range(10))
    assert_allclose(actual, expected)


@pytest.mark.parametrize("pk_ufunc, numpy_ufunc", [
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
])
@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
def test_1d_exposed_ufuncs_vs_numpy(pk_ufunc,
                                    numpy_ufunc,
                                    pk_dtype,
                                    numpy_dtype):
    # test the ufuncs we have exposed in the pk namespace
    # vs. their NumPy equivalents
    expected = numpy_ufunc(np.arange(10, dtype=numpy_dtype))

    view: pk.View1d = pk.View([10], pk_dtype)
    view[:] = np.arange(10, dtype=numpy_dtype)
    actual = pk_ufunc(view=view)
    # log10 single-precision needs relaxed tol
    # for now
    if numpy_ufunc == np.log10 and numpy_dtype == np.float32:
        assert_allclose(actual, expected, rtol=1.5e-7)
    else:
        assert_allclose(actual, expected)


@pytest.mark.parametrize("pk_ufunc, numpy_ufunc", [
        (pk.add, np.add),
        (pk.subtract, np.subtract),
        (pk.multiply, np.multiply),
        (pk.divide, np.divide),
        (pk.power, np.power),
        (pk.fmod, np.fmod),
        (pk.greater, np.greater),
        (pk.logaddexp, np.logaddexp),
])
@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
def test_multi_array_1d_exposed_ufuncs_vs_numpy(pk_ufunc,
                                    numpy_ufunc,
                                    pk_dtype,
                                    numpy_dtype):

    # test the multi array ufuncs we have exposed 
    # in the pk namespace vs. their NumPy equivalents
    expected = numpy_ufunc(
        np.arange(10, dtype=numpy_dtype),
        np.full(10, 5, dtype=np.float32))

    viewA: pk.View1d = pk.View([10], pk_dtype)
    viewA[:] = np.arange(10, dtype=numpy_dtype)
    viewB: pk.View1d = pk.View([10], pk_dtype)
    viewB[:] = np.full(10, 5, dtype=numpy_dtype)

    actual = pk_ufunc(viewA=viewA, viewB=viewB)

    assert_allclose(actual, expected)


@pytest.mark.parametrize("pk_ufunc, numpy_ufunc", [
        (pk.matmul, np.matmul),
])
@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
def test_matmul_1d_exposed_ufuncs_vs_numpy(pk_ufunc,
                                    numpy_ufunc,
                                    pk_dtype,
                                    numpy_dtype):
    expected = numpy_ufunc(
        np.arange(10, dtype=numpy_dtype),
        np.full((10, 1), 2, dtype=numpy_dtype))


    viewA = pk.View([10], pk_dtype)
    viewB = pk.View([10, 1], pk_dtype)
    viewA[:] = np.arange(10, dtype=numpy_dtype)
    viewB[:] = np.full((10, 1), 2, dtype=numpy_dtype)

    with pytest.raises(RuntimeError) as e_info:
        viewC = pk.View([11], pk_dtype)
        viewC[:] = np.arange(11, dtype=numpy_dtype)
        pk_ufunc(viewC, viewB)

    assert e_info.value.args[0] == "Input operand 1 has a mismatch in its core dimension (Size 11 is different from 10)"

    actual = pk_ufunc(viewA, viewB)

    assert_allclose(actual, expected)

@pytest.mark.parametrize("arr", [
    np.array([4, -1, np.inf]),
    np.array([-np.inf, np.nan, np.inf]),
])
@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
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


@pytest.mark.parametrize("pk_ufunc, numpy_ufunc", [
        (pk.reciprocal, np.reciprocal),
])
@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
def test_2d_exposed_ufuncs_vs_numpy(pk_ufunc,
                                    numpy_ufunc,
                                    pk_dtype,
                                    numpy_dtype):
    rng = default_rng(123)
    in_arr = rng.random((5, 5)).astype(numpy_dtype)
    expected = numpy_ufunc(in_arr)

    view: pk.View2d = pk.View([5, 5], pk_dtype)
    view[:] = in_arr
    actual = pk_ufunc(view=view)
    assert_allclose(actual, expected)


@pytest.mark.parametrize("pk_dtype, numpy_dtype", [
        (pk.double, np.float64),
        (pk.float, np.float32),
])
@pytest.mark.parametrize("in_arr", [
    np.array([-5, 4.5, np.nan]),
    np.array([np.nan, np.nan, np.nan]),
])
def test_sign_1d_special_cases(in_arr, pk_dtype, numpy_dtype):
    in_arr = in_arr.astype(numpy_dtype)
    view: pk.View1D = pk.View([in_arr.size], pk_dtype)
    view[:] = in_arr
    expected = np.sign(in_arr)
    actual = pk.sign(view=view)
    assert_allclose(actual, expected)
