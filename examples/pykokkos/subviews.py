import pykokkos as pk

if pk.get_default_space() in pk.DeviceExecutionSpace:
    import cupy as np
else:
    import numpy as np


def main():
    n: int = 10
    # 2D view; each thread takes a row subview and a column-range subview inside the workunit.
    view = np.zeros((n, n), dtype=np.int32)
    pk.parallel_for(n, work, view=view)
    print(
        "PyKokkos subviews: each iteration builds Kokkos::subview row = view[i, :] and "
        "band = view[i, 2:5], then writes through those 1D subviews (not view[i][j] alone).\n"
    )
    print(view)


@pk.workunit
def work(i: int, view: pk.View2D[pk.int32]):
    # Plain `name = view[...]` becomes Kokkos::subview (annotated assign does not).
    row = view[i, :]
    band = view[i, 2:5]
    row[i] = 1
    band[i % 3] = i + 1


if __name__ == "__main__":
    main()
