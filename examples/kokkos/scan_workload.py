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

    pk.parallel_for(N, init, A=A)

    timer = pk.Timer()
    result: int = pk.parallel_scan(N, scan, A=A)
    timer_result: float = timer.seconds()

    print(f"{A} total={result} time({timer_result})")


if __name__ == "__main__":
    run()
