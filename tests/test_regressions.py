import pykokkos as pk


@pk.functor
class Workload_gh_173:
    def __init__(self, s: int):
        self.size: pk.int = s
        self.result: pk.View1D[pk.int64] = pk.View([1], dtype= pk.int64)


    @pk.workunit
    def store_result(self, i: int):
        self.result[i] = self.size


def test_gh_173():
    # this is a more tractable test case for gh-173
    # since it provides a retrieved value rather than
    # a printed value
    w = Workload_gh_173(900)
    pk.parallel_for(1, w.store_result)
    assert w.result[0] == 900
    w.size = 10
    pk.parallel_for(1, w.store_result)
    assert w.result[0] == 10
