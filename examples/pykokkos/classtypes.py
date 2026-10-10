import pykokkos as pk


@pk.classtype
class TestClass:
    def __init__(self, x: float):
        self.x: float = x

    def test(self) -> float:
        return self.x * 2


@pk.workunit
def work(tid: int, acc: pk.Acc[pk.double]) -> None:
    tc: TestClass = TestClass(float(tid))
    acc += tc.test()


def main():
    total_threads: int = 10
    result: float = pk.parallel_reduce(total_threads, work)
    expected: float = sum(2.0 * float(i) for i in range(total_threads))
    print(f"parallel_reduce: {result} (expected {expected})")


if __name__ == "__main__":
    main()
