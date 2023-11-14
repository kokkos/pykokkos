import pykokkos as pk

from comm import Comm
from system import System
from types_h import t_v, t_mass, t_type


@pk.workload
class Temperature:
    def __init__(self, comm: Comm):
        self.comm = comm
        self.v: pk.View2D[pk.double] = t_v(0, 3)
        self.mass: pk.View1D[pk.double] = t_mass(0)
        self.type: pk.View1D[pk.int32] = t_type(0)
        self.N_local: int = 0

        # Reduction result
        self.T: float = 0.0

    def compute(self, system: System) -> float:
        self.v = system.v
        self.mass = system.mass
        self.type = system.type

        self.N_local = system.N_local
        pk.execute(pk.ExecutionSpace.Default, self)

        dof: int = 3 * system.N - 3
        factor: float = system.mvv2e / (1.0 * dof * system.boltz)

        self.comm.reduce_float(self.T, 1)

        return self.T * factor

    @pk.main
    def run(self) -> None:
        self.T = pk.parallel_reduce("Temperature", self.N_local, self.compute_workunit)

    @pk.workunit
    def compute_workunit(self, i: int, acc: pk.Acc[pk.double]) -> None:
        mass_index: int = self.type[i]
        acc += (self.v[i][0] * self.v[i][0] + self.v[i][1] * self.v[i]
                [1] + self.v[i][2] * self.v[i][2]) * self.mass[mass_index]
