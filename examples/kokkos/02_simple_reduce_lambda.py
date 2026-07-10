import pykokkos as pk


def main():
    N: int = 10

    total = pk.parallel_reduce(N, lambda i, acc: acc + i * i)

    print("Sum:", total)


if __name__ == "__main__":
    main()
