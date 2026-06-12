import pykokkos as pk


@pk.workunit
def init(i: int, A: pk.View1D[pk.int32]):
    A[i] = i


@pk.workunit
def scan(i: int, acc: pk.Acc[pk.double], last_pass: bool, A: pk.View1D[pk.int32]):
    acc += A[i]
    if last_pass:
        A[i] = acc


def run() -> None:
    N: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    A: pk.View1D[pk.int32] = pk.View([N], pk.int32)

    p = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)

    pk.parallel_for(p, init, A=A)

    timer = pk.Timer()
    result = pk.parallel_scan(p, scan, A=A)
    timer_result = timer.seconds()

    print(f"{A} total={result} time({timer_result})")


if __name__ == "__main__":
    run()
