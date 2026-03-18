import pykokkos as pk


@pk.classtype
class TestClass:
    def __init__(self, x: float):
        self.x: float = x

    def test(self) -> float:
        return self.x * 2


@pk.workunit
def work(tid: int):
    pk.printf("%d\n", tid)


def main():
    total_threads: int = 10
    pk.parallel_for(total_threads, work)


if __name__ == "__main__":
    main()
