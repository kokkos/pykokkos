import pykokkos as pk


@pk.workload
class HelloWorld:
    def __init__(self, n):
        self.N: int = n

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.hello)

    @pk.workunit
    def hello(self, i: int):
        pk.printf("Hello from i = %d\n", i)


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, HelloWorld(10))
