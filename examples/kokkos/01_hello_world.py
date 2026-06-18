import pykokkos as pk


@pk.workunit
def hello(i: int):
    pk.printf("Hello from i = %d\n", i)


def main():
    N: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    pk.parallel_for(N, hello)


if __name__ == "__main__":
    main()
