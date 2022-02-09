from system import System
from typing import List

import pykokkos as pk

from binning import Binning
from force import Force
from neighbor import Neighbor
from neighbor_types.neighbor_2d import NeighList2D
from types_h import MAX_TYPES_STACKPARAMS, t_f


@pk.workload(x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class ForceLJNeigh(Force):
    class t_fparams(pk.View):
        def __init__(self, x: int = 0, y: int = 0, data_type: pk.DataTypeClass = pk.double):
            super().__init__([x, y], data_type)

    class t_fparams_rnd(pk.View):
        def __init__(self, x: int = 0, y: int = 0, data_type: pk.DataTypeClass = pk.double):
            super().__init__([x, y], data_type)

    def __init__(self, args: List[str], system: System, half_neigh: bool):
        super().__init__(args, system, half_neigh)

        self.ntypes: int = system.ntypes

        self.half_neigh: bool = False
        self.use_stackparams: bool = False
        # self.use_stackparams: bool = self.ntypes <= MAX_TYPES_STACKPARAMS

        self.lj1: pk.View2D[pk.double] = self.t_fparams()
        self.lj2: pk.View2D[pk.double] = self.t_fparams()
        self.cutsq: pk.View2D[pk.double] = self.t_fparams()

        if not self.use_stackparams:
            self.lj1 = self.t_fparams(self.ntypes, self.ntypes)
            self.lj2 = self.t_fparams(self.ntypes, self.ntypes)
            self.cutsq = self.t_fparams(self.ntypes, self.ntypes)

        self.rnd_lj1: pk.View2D[pk.double] = self.t_fparams()
        self.rnd_lj2: pk.View2D[pk.double] = self.t_fparams()
        self.rnd_cutsq: pk.View2D[pk.double] = self.t_fparams()

        self.nbinx: int = 0
        self.nbiny: int = 0
        self.nbinz: int = 0

        self.N_local: int = 0
        self.nhalo: int = 0
        self.step: int = 0

        self.stack_lj1: List[List[float]] = [[0 for i in range(
            MAX_TYPES_STACKPARAMS + 1)] for j in range(MAX_TYPES_STACKPARAMS + 1)]
        self.stack_lj2: List[List[float]] = [[0 for i in range(
            MAX_TYPES_STACKPARAMS + 1)] for j in range(MAX_TYPES_STACKPARAMS + 1)]
        self.stack_cutsq: List[List[float]] = [[0 for i in range(
            MAX_TYPES_STACKPARAMS + 1)] for j in range(MAX_TYPES_STACKPARAMS + 1)]

        self.energy: float = 0.0

        # parallel_for and parallel_reduce are called separately
        # so this boolean is used in run to decide which one
        # to call
        self.parallel_for: bool = True

        # copied from NeighList2D
        self.num_neighs_view: pk.View1D[pk.int32] = pk.View([0], pk.int32)
        self.neighs_view: pk.View2D[pk.int32] = pk.View([0, 0], pk.int32)

        # copied from system
        self.x: pk.View2D[pk.double] = system.x
        self.f: pk.View2D[pk.double] = system.f
        # TODO: this should be atomic. Disabled since it is
        # overwriting f
        #self.f_a: pk.View2D[pk.double] = system.f
        self.id: pk.View1D[pk.int32] = system.id
        self.type: pk.View1D[pk.int32] = system.type

    def init_coeff(self, nargs: int, args: List[str]) -> None:
        self.step = 0

        one_based_type: int = 1
        t1: int = int(args[1]) - one_based_type
        t2: int = int(args[2]) - one_based_type
        eps: float = float(args[3])
        sigma: float = float(args[4])
        cut: float = float(args[5])

        if self.use_stackparams:
            for i in range(self.ntypes):
                for j in range(self.ntypes):
                    self.stack_lj1[i][j] = 48.0 * eps * (sigma ** 12.0)
                    self.stack_lj2[i][j] = 24.0 * eps * (sigma ** 6.0)
                    self.stack_cutsq[i][j] = cut*cut

        else:
            self.lj1[t1][t2] = 48.0 * eps * (sigma ** 12.0)
            self.lj2[t1][t2] = 24.0 * eps * (sigma ** 6.0)
            self.lj1[t2][t1] = self.lj1[t1][t2]
            self.lj2[t2][t1] = self.lj2[t1][t2]
            self.cutsq[t1][t2] = cut * cut
            self.cutsq[t2][t1] = cut * cut

            self.rnd_lj1 = self.lj1
            self.rnd_lj2 = self.lj2
            self.rnd_cutsq = self.cutsq

    def compute(self, system: System, binning: Binning, neighbor: Neighbor) -> None:
        neigh_list: NeighList2D = neighbor.get_neigh_list()
        self.num_neighs_view: pk.View1D = neigh_list.num_neighs
        self.neighs_view: pk.View2D = neigh_list.neighs

        self.N_local = system.N_local
        self.x = system.x
        self.f = system.f
        # self.f_a = system.f
        self.type = system.type
        self.id = system.id

        self.parallel_for = True
        pk.execute(pk.ExecutionSpace.Default, self)

        self.step += 1

    def compute_energy(self, system: System, binning: Binning, neighbor: Neighbor) -> float:
        neigh_list: NeighList2D = neighbor.get_neigh_list()
        self.num_neighs_view: pk.View1D = neigh_list.num_neighs
        self.neighs_view: pk.View2D = neigh_list.neighs

        self.N_local = system.N_local
        self.x = system.x
        self.f = system.f
        # self.f_a = system.f
        self.type = system.type
        self.id = system.id

        self.parallel_for = False
        pk.execute(pk.ExecutionSpace.Default, self)

        self.step += 1
        return self.energy

    def name(self) -> str:
        if self.half_neigh:
            return "ForceLJNeighHalf"
        else:
            return "ForceLJNeighFull"

    @pk.main
    def run(self) -> None:
        if self.parallel_for:
            if self.half_neigh:
                pk.parallel_for("ForceLJNeigh::compute", self.N_local, self.halfneigh_for)
            else:
                pk.parallel_for("ForceLJNeigh::compute", self.N_local, self.fullneigh_for)
        else:
            if self.half_neigh:
                self.energy = pk.parallel_reduce("ForceLJNeigh::compute_energy", self.N_local, self.halfneigh_reduce)
            else:
                self.energy = pk.parallel_reduce("ForceLJNeigh::compute_energy", self.N_local, self.fullneigh_reduce)

    @pk.workunit
    def fullneigh_for(self, i: int) -> None:
        x_i: float = self.x[i][0]
        y_i: float = self.x[i][1]
        z_i: float = self.x[i][2]
        type_i: int = self.type[i]

        fxi: float = 0.0
        fyi: float = 0.0
        fzi: float = 0.0

        num_neighs: int = self.num_neighs_view[i]
        for jj in range(num_neighs):
            j: int = self.neighs_view[i][jj]
            dx: float = x_i - self.x[j][0]
            dy: float = y_i - self.x[j][1]
            dz: float = z_i - self.x[j][2]

            type_j: int = self.type[j]
            rsq: float = dx * dx + dy * dy + dz * dz

            cutsq_ij: float = self.rnd_cutsq[type_i][type_j]

            if rsq < cutsq_ij:
                lj1_ij: float = self.rnd_lj1[type_i][type_j]
                lj2_ij: float = self.rnd_lj2[type_i][type_j]

                r2inv: float = 1.0 / rsq
                r6inv: float = r2inv * r2inv * r2inv
                fpair: float = (r6inv * (lj1_ij * r6inv - lj2_ij)) * r2inv
                fxi += dx * fpair
                fyi += dy * fpair
                fzi += dz * fpair

        self.f[i][0] += fxi
        self.f[i][1] += fyi
        self.f[i][2] += fzi

    @pk.workunit
    def halfneigh_for(self, i: int) -> None:
        x_i: float = self.x[i][0]
        y_i: float = self.x[i][1]
        z_i: float = self.x[i][2]
        type_i: int = self.type[i]

        fxi: float = 0.0
        fyi: float = 0.0
        fzi: float = 0.0

        num_neighs: int = self.num_neighs_view[i]
        for jj in range(num_neighs):
            j: int = self.neighs_view[i][jj]
            dx: float = x_i - self.x[j][0]
            dy: float = y_i - self.x[j][1]
            dz: float = z_i - self.x[j][2]

            type_j: int = self.type[j]
            rsq: float = dx * dx + dy * dy + dz * dz

            cutsq_ij: float = 0.0
            if self.use_stackparams:
                pass
            else:
                cutsq_ij = self.rnd_cutsq[type_i][type_j]

            if rsq < cutsq_ij:
                lj1_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj1_ij = self.rnd_lj1[type_i][type_j]

                lj2_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj2_ij = self.rnd_lj2[type_i][type_j]

                r2inv: float = 1.0 / rsq
                r6inv: float = r2inv * r2inv * r2inv
                fpair: float = (r6inv * (lj1_ij * r6inv - lj2_ij)) * r2inv
                fxi += dx * fpair
                fyi += dy * fpair
                fzi += dz * fpair
                self.f[j][0] -= dx * fpair
                self.f[j][1] -= dy * fpair
                self.f[j][2] -= dz * fpair
                # self.f_a[j][0] -= dx * fpair
                # self.f_a[j][1] -= dy * fpair
                # self.f_a[j][2] -= dz * fpair

        self.f[i][0] += fxi
        self.f[i][1] += fyi
        self.f[i][2] += fzi
        # self.f_a[i][0] += fxi
        # self.f_a[i][1] += fyi
        # self.f_a[i][2] += fzi

    @pk.workunit
    def fullneigh_reduce(self, i: int, PE: pk.Acc[pk.double]) -> None:
        x_i: float = self.x[i][0]
        y_i: float = self.x[i][1]
        z_i: float = self.x[i][2]
        type_i: int = self.type[i]
        shift_flag: bool = True

        num_neighs: int = self.num_neighs_view[i]
        for jj in range(num_neighs):
            j: int = self.neighs_view[i][jj]
            dx: float = x_i - self.x[j][0]
            dy: float = y_i - self.x[j][1]
            dz: float = z_i - self.x[j][2]

            type_j: int = self.type[j]
            rsq: float = dx * dx + dy * dy + dz * dz

            cutsq_ij: float = 0.0
            if self.use_stackparams:
                pass
            else:
                cutsq_ij = self.rnd_cutsq[type_i][type_j]

            if rsq < cutsq_ij:
                lj1_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj1_ij = self.rnd_lj1[type_i][type_j]

                lj2_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj2_ij = self.rnd_lj2[type_i][type_j]

                r2inv: float = 1.0 / rsq
                r6inv: float = r2inv * r2inv * r2inv
                PE += 0.5 * r6inv * (0.5 * lj1_ij * r6inv - lj2_ij) / 6.0

                if shift_flag:
                    r2invc: float = 1.0 / cutsq_ij
                    r6invc: float = r2invc * r2invc * r2invc

                    PE -= 0.5 * r6invc * (0.5 * lj1_ij * r6invc - lj2_ij) / 6.0

    @pk.workunit
    def halfneigh_reduce(self, i: int, PE: pk.Acc[pk.double]) -> None:
        x_i: float = self.x[i][0]
        y_i: float = self.x[i][1]
        z_i: float = self.x[i][2]
        type_i: int = self.type[i]
        shift_flag: bool = True

        num_neighs: int = self.num_neighs_view[i]
        for jj in range(num_neighs):
            j: int = self.neighs_view[i][jj]
            dx: float = x_i - self.x[j][0]
            dy: float = y_i - self.x[j][1]
            dz: float = z_i - self.x[j][2]

            type_j: int = self.type[j]
            rsq: float = dx * dx + dy * dy + dz * dz

            cutsq_ij: float = 0.0
            if self.use_stackparams:
                pass
            else:
                cutsq_ij = self.rnd_cutsq[type_i][type_j]

            if rsq < cutsq_ij:
                lj1_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj1_ij = self.rnd_lj1[type_i][type_j]

                lj2_ij: float = 0.0
                if self.use_stackparams:
                    pass
                else:
                    lj2_ij = self.rnd_lj2[type_i][type_j]

                r2inv: float = 1.0 / rsq
                r6inv: float = r2inv * r2inv * r2inv
                fac: float = 0.0
                if j < self.N_local:
                    fac = 1.0
                else:
                    fac = 0.5

                PE += fac * r6inv * (0.5 * lj1_ij * r6inv - lj2_ij) / 6.0

                if shift_flag:
                    r2invc: float = 1.0 / cutsq_ij
                    r6invc: float = r2invc * r2invc * r2invc

                    PE -= fac * r6invc * (0.5 * lj1_ij * r6invc - lj2_ij) / 6.0
