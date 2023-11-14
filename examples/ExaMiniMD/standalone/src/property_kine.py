import pykokkos as pk

from comm import Comm
from system import System


@pk.workunit
def work(i: int, KE: pk.Acc[float], v: pk.View2D[float], mass: pk.View1D[float], type: pk.View1D[int]) -> None:
    index: int = type[i]
    KE += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * mass[index]


class KinE:
    def __init__(self, comm: Comm):
        self.comm = comm

    def compute(self, system: System) -> float:
        KE = pk.parallel_reduce("KinE", system.N_local, work, v=system.v, mass=system.mass, type=system.type)

        factor: float = 0.5 * system.mvv2e
        self.comm.reduce_float(KE, 1)

        return KE * factor
