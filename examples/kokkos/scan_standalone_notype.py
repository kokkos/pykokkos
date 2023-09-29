import pykokkos as pk

@pk.workunit
def init(i: int, view: pk.View1D[pk.int32]):
    view[i] = i

@pk.workunit
def scan(i: int, acc: pk.Acc[pk.double], last_pass: bool, view: pk.View1D[pk.int32]):
    acc += view[i]
    if last_pass:
        view[i] = acc

if __name__ == "__main__":
    N = 10

    A: pk.View1D[pk.int32] = pk.View([N], pk.int32)
    p = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)
    pk.parallel_for(p, init, view=A)

    timer = pk.Timer()
    result = pk.parallel_scan(p, scan, view=A)
    timer_result = timer.seconds()

    print(f"{A} total={result} time({timer_result})")
