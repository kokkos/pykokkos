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



@pk.workunit
def sin(tid: int, view: pk.View1D[pk.double]):
    view[tid] = sin(view[tid])


def test_gh_192_1():
    #Regression Test to check if workunits can have the same name as functions called within
    #this previously errored out in the compilation stage as we used tags with the name of the workunit.
    #The error originated from the compiler using the tag's ctor call instead of the function that we wanted to use in the workunit 
    v = pk.View([10], dtype=pk.float64)
    pk.parallel_for(1, sin, view=v)

def test_gh_192_2():
    #the same has to hold if we manually specify a policy
    v = pk.View([10], dtype=pk.float64)
    policy = pk.RangePolicy(pk.ExecutionSpace.Default,0,1)
    pk.parallel_for(policy,sin,view=v)
