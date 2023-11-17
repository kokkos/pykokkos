import pykokkos as pk

from integrator import Integrator
from system import System


@pk.workunit
def initial_integrate(
    i: int, x: pk.View2D[float], v: pk.View2D[float],
    f: pk.View2D[float], type: pk.View1D[int], mass: pk.View1D[float],
    dtf: float, dtv: float
) -> None:
    index: int = type[i]
    dtfm: float = dtf / mass[index]
    v[i][0] += dtfm * f[i][0]
    v[i][1] += dtfm * f[i][1]
    v[i][2] += dtfm * f[i][2]
    x[i][0] += dtv * v[i][0]
    x[i][1] += dtv * v[i][1]
    x[i][2] += dtv * v[i][2]


@pk.workunit
def final_integrate(
    i: int, v: pk.View2D[float], f: pk.View2D[float],
    type: pk.View1D[int], mass: pk.View1D[float],
    dtf: float
) -> None:
    index: int = type[i]
    dtfm: float = dtf / mass[index]
    v[i][0] += dtfm * f[i][0]
    v[i][1] += dtfm * f[i][1]
    v[i][2] += dtfm * f[i][2]


class IntegratorNVE(Integrator):
    def __init__(self, s: System):
        super().__init__(s)
        self.dtv: float = self.system.dt
        self.dtf: float = 0.5 * self.system.dt / self.system.mvv2e

        self.step: int = 1

    def initial_integrate(self) -> None:
        pk.parallel_for("IntegratorNVE::initial_integrate", self.system.N_local, initial_integrate, 
            x=self.system.x, v=self.system.v, f=self.system.f, type=self.system.type, mass=self.system.mass, dtf=self.dtf, dtv=self.dtv)

        self.step += 1

    def final_integrate(self) -> None:
        pk.parallel_for("IntegratorNVE::final_integrate", self.system.N_local, final_integrate, 
            v=self.system.v, f=self.system.f, type=self.system.type, mass=self.system.mass, dtf=self.dtf)

        self.step += 1
