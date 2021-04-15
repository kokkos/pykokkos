import pykokkos as pk


@pk.classtype
class TestClass:
    def __init__(self, x: float):
        self.x: float = x

    def test(self) -> float:
        return self.x * 2


@pk.workload
class Workload:
    def __init__(self, total_threads: int):
        self.total_threads: int = total_threads

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.total_threads, self.work)

    @pk.workunit
    def work(self, tid: int) -> None:
        pk.printf("%d\n", tid)

    @pk.function
    def fun(self, f: TestClass) -> None:
        f.x = 3
        x: float = f.x + 5

    @pk.function
    def test(self) -> TestClass:
        return TestClass(3.5)


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.Default, Workload(10))
