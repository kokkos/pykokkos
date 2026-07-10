import pykokkos as pk


def main():
    N: int = 10
    pk.parallel_for(N, lambda i: pk.printf("Hello from i = %d\n", i))


if __name__ == "__main__":
    main()
