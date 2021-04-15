import pykokkos as pk

from integrator import Integrator
from system import System
from types_h import t_x, t_v, t_f, t_type, t_mass, t_id


@pk.workload(x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class InitialIntegrateFunctor:
    def __init__(
            self, x: t_x, v: t_v, f: t_f, t: t_type,
            mass: t_mass, i: t_id,
            dtf: float, dtv: float, step: int,
            N_local: int
    ):
        self.x: pk.View2D[pk.double] = x
        self.v: pk.View2D[pk.double] = v
        self.f: pk.View2D[pk.double] = f
        self.type: pk.View1D[pk.int32] = t
        self.mass: pk.View1D[pk.double] = mass
        self.id: pk.View1D[pk.int32] = i

        self.dtf: float = dtf
        self.dtv: float = dtv
        self.step: int = step

        self.N_local: int = N_local

    @pk.main
    def run(self) -> None:
        pk.parallel_for("IntegratorNVE::initial_integrate", self.N_local, self.integrate)

    @pk.workunit
    def integrate(self, i: int) -> None:
        index: int = self.type[i]
        dtfm: float = self.dtf / self.mass[index]
        self.v[i][0] += dtfm * self.f[i][0]
        self.v[i][1] += dtfm * self.f[i][1]
        self.v[i][2] += dtfm * self.f[i][2]
        self.x[i][0] += self.dtv * self.v[i][0]
        self.x[i][1] += self.dtv * self.v[i][1]
        self.x[i][2] += self.dtv * self.v[i][2]

@pk.workload(x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class FinalIntegrateFunctor:
    def __init__(
            self, v: t_v, f: t_f, t: t_type,
            mass: t_mass, dtf: float, dtv: float,
            i: t_id, step: int, x: t_x,
            N_local: int
    ):
        self.x: pk.View2D[pk.double] = x
        self.v: pk.View2D[pk.double] = v
        self.f: pk.View2D[pk.double] = f
        self.type: pk.View1D[pk.int32] = t
        self.mass: pk.View1D[pk.double] = mass
        self.id: pk.View1D[pk.int32] = i

        self.dtf: float = dtf
        self.dtv: float = dtv
        self.step: int = step

        self.N_local: int = N_local

    @pk.main
    def run(self) -> None:
        pk.parallel_for("IntegratorNVE::final_integrate", self.N_local, self.integrate)

    @pk.workunit
    def integrate(self, i: int) -> None:
        index: int = self.type[i]
        dtfm: float = self.dtf / self.mass[index]
        self.v[i][0] += dtfm * self.f[i][0]
        self.v[i][1] += dtfm * self.f[i][1]
        self.v[i][2] += dtfm * self.f[i][2]


class IntegratorNVE(Integrator):
    def __init__(self, s: System):
        super().__init__(s)
        self.dtv: float = self.system.dt
        self.dtf: float = 0.5 * self.system.dt / self.system.mvv2e

        self.step: int = 1

    def initial_integrate(self) -> None:
        workload = InitialIntegrateFunctor(
            self.system.x, self.system.v, self.system.f,
            self.system.type, self.system.mass, self.system.id,
            self.dtf, self.dtv, self.step, self.system.N_local)

        pk.execute(pk.ExecutionSpace.Default, workload)
        self.step += 1

    def final_integrate(self) -> None:
        workload = FinalIntegrateFunctor(
            self.system.v, self.system.f, self.system.type,
            self.system.mass, self.dtf, self.dtv, self.system.id,
            self.step, self.system.x, self.system.N_local)

        pk.execute(pk.ExecutionSpace.Default, workload)
        self.step += 1
