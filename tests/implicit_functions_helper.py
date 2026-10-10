import pykokkos as pk


@pk.function
def external_helper(i: int) -> int:
    return i + 3
