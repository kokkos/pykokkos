import pykokkos as pk

import numpy as np
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
    sqrt_kokkos_instance = kokkos_test_class(threads=10, view=view)
    pk.execute(pk.ExecutionSpace.Default, sqrt_kokkos_instance)
    actual = sqrt_kokkos_instance.view
    expected = numpy_ufunc(range(10))
    assert_allclose(actual, expected)

