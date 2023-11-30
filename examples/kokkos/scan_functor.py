import pykokkos as pk

@pk.functor
class Workload:
    def __init__(self, N: int):
        self.A: pk.View1D[pk.int32] = pk.View([N], pk.int32)

    @pk.workunit
    def init(self, i: int):
        self.A[i] = i

    @pk.workunit
    def scan(self, i: int, acc: pk.Acc[pk.double], last_pass: bool):
        acc += self.A[i]
        if last_pass:
            self.A[i] = acc

def run() -> None:
    N = 10
    w = Workload(N)
    p = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)
    pk.parallel_for(p, w.init)

    timer = pk.Timer()
    result = pk.parallel_scan(p, w.scan)
    timer_result = timer.seconds()

    print(f"{w.A} total={result} time({timer_result})")

if __name__ == "__main__":
    run()
