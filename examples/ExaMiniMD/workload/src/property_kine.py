import pykokkos as pk

from types_h import t_v, t_mass, t_type
from comm import Comm
from system import System


@pk.workload
class KinE:
    def __init__(self, comm: Comm):
        self.comm = comm

        self.v: pk.View2D[pk.double] = t_v(0, 3)
        self.mass: pk.View1D[pk.double] = t_mass(0)
        self.type: pk.View1D[pk.int32] = t_type(0)

        self.N_local: int = 0

        # Reduction result
        self.KE: float = 0

    def compute(self, system: System) -> float:
        self.v = system.v
        self.mass = system.mass
        self.type = system.type

        self.N_local = system.N_local
        pk.execute(pk.ExecutionSpace.Default, self)

        self.v = t_v(0, 3)
        self.mass = t_mass(0)
        self.type = t_type(0)

        factor: float = 0.5 * system.mvv2e

        self.comm.reduce_float(self.KE, 1)
        return self.KE * factor

    @pk.main
    def run(self) -> None:
        self.KE = pk.parallel_reduce("KinE", self.N_local, self.work)

    @pk.workunit
    def work(self, i: int, acc: pk.Acc[pk.double]) -> None:
        index: int = self.type[i]
        acc += (self.v[i][0] * self.v[i][0] + self.v[i][1] * self.v[i][1]
               + self.v[i][2] * self.v[i][2]) * self.mass[index]
