import numpy as np
import pykokkos as pk


def main():
    view = np.zeros((10, 10), dtype=np.int32)
    subview = view[3, 2:5]
    pk.parallel_for(10, work, view=view)
    print(view)


@pk.workunit
def work(i: int, view: pk.View2D[pk.int32]):
    view[i][i] = 1


if __name__ == "__main__":
    main()
