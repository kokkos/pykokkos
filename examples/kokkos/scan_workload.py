import pykokkos as pk

@pk.workload
class Workload:
    def __init__(self, N: int):
        self.N: int = N
        self.A: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        self.result: int = 0
        self.timer_result: float = 0

    @pk.main
    def run(self):
        pk.parallel_for(self.N, lambda i: i, self.A)

        timer = pk.Timer()

        self.result = pk.parallel_scan(self.N, self.scan)

        self.timer_result = timer.seconds()

    @pk.callback
    def results(self):
        print(f"{self.A} total={self.result} time({self.timer_result})")

    @pk.workunit
    def scan(self, i: int, acc: pk.Acc[pk.double], last_pass: bool):
        acc += self.A[i]
        if last_pass:
            self.A[i] = acc

def run() -> None:
    pk.execute(pk.ExecutionSpace.OpenMP, Workload(10))

if __name__ == "__main__":
    run()
