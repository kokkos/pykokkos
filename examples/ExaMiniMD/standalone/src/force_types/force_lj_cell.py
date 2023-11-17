from typing import List

import pykokkos as pk

from binning import Binning
from neighbor import Neighbor
from force import Force
from system import System
from types_h import t_x, t_type, t_f


@pk.classtype
class t_scalar3:
    def __init__(self):
        self.x: int = 0
        self.y: int = 0
        self.z: int = 0

    def init_copy(self, rhs: t_scalar3) -> t_scalar3:
        init: t_scalar3 = t_scalar3()

        init.x = rhs.x
        init.y = rhs.y
        init.z = rhs.z

        return init

    def init_scalar(self, x_: int, y_: int, z_: int) -> t_scalar3:
        init: t_scalar3 = t_scalar3()

        init.x = x_
        init.y = y_
        init.z = z_

        return init


@pk.workload
class ForceLJCell(Force):
    class t_fparams(pk.View2D):
        def __init__(self, x: int = 0, y: int = 0, data_type: pk.DataType = pk.double):
            super().__init__(x, y, data_type)

    def __init__(self, args: List[str], system: System, half_neigh: bool):
        super().__init__(args, system, half_neigh)

        self.lj1: pk.View2D[pk.double] = self.t_fparams(system.ntypes, system.ntypes)
        self.lj2: pk.View2D[pk.double] = self.t_fparams(system.ntypes, system.ntypes)
        self.cutsq: pk.View2D[pk.double] = self.t_fparams(system.ntypes, system.ntypes)

        self.bin_offsets: pk.View3D[pk.int32] = pk.View([0, 0, 0], pk.int32)
        self.bin_count: pk.View3D[pk.int32] = pk.View([0, 0, 0], pk.int32)
        self.permute_vector: pk.View1D[pk.int32] = pk.View([0], pk.int32)

        self.x: pk.View2D[pk.double] = pk.View([0, 0], pk.double)
        self.type: pk.View1D[pk.int32] = pk.View([0], pk.int32)
        self.f: pk.View2D[pk.double] = pk.View([0, 0], pk.double)

        self.N_local: int = 0
        self.nbinx: int = 0
        self.nbiny: int = 0
        self.nbinz: int = 0
        self.nbins: int = 0

        # Defined as a static variable in ForceLJCell::compute
        self.step_i: int = 0

        # parallel_for and parallel_reduce are called separately
        # so this boolean is used in run to decide which one
        # to call
        self.parallel_for: bool = True

        self.PE: float = 0

    def init_coeff(self, nargs: int, args: List[str]) -> None:
        one_based_type: int = 1
        t1: int = int(args[1]) - one_based_type
        t2: int = int(args[2]) - one_based_type
        eps: float = float(args[3])
        sigma: float = float(args[4])
        cut: float = float(args[5])

        self.lj1[t1][t2] = 48.0 * eps * (sigma ** 12.0)
        self.lj2[t1][t2] = 24.0 * eps * (sigma ** 6.0)
        self.lj1[t2][t1] = self.lj1[t1][t2]
        self.lj2[t2][t1] = self.lj2[t1][t2]
        self.cutsq[t1][t2] = cut * cut
        self.cutsq[t2][t1] = cut * cut

    def compute(self, system: System, binning: Binning, neighbor: Neighbor, fill: bool) -> None:
        self.x = system.x
        self.f = system.f
        self.id = system.id
        self.type = system.type
        self.N_local = system.N_local

        self.step = self.step_i
        self.bin_count = binning.bincount
        self.bin_offsets = binning.binoffsets
        self.permute_vector = binning.permute_vector

        self.nhalo = binning.nhalo
        self.nbinx = binning.nbinx
        self.nbiny = binning.nbiny
        self.nbinz = binning.nbinz

        if fill:
            self.f.fill(0)
        else:
            for i in range(self.f.x):
                for j in range(self.f.y):
                    self.f[i][j] = 0.0

        self.nbins: int = self.nbinx * self.nbiny * self.nbinz

        self.parallel_for = True
        pk.execute(self)

        self.step_i += 1
        self.x = t_x()
        self.type = t_type()
        self.f = t_f()

    def compute_energy(self, system: System, binning: Binning, neighbor: Neighbor) -> float:
        self.x = system.x
        self.id = system.id
        self.type = system.type
        self.N_local = system.N_local

        self.bin_count = binning.bincount
        self.bin_offsets = binning.binoffsets
        self.permute_vector = binning.permute_vector

        self.nhalo = binning.nhalo
        self.nbinx = binning.nbinx
        self.nbiny = binning.nbiny
        self.nbinz = binning.nbinz

        self.nbins: int = self.nbinx * self.nbiny * self.nbinz

        self.parallel_for = False
        pk.execute(self, dependencies=[t_scalar3])

        self.x = t_x()
        self.type = t_type()
        self.f = t_f()

        return self.PE

    @pk.main
    def run(self) -> None:
        if self.parallel_for:
            pk.parallel_for(pk.TeamPolicy(self.nbins, 1, 8), self.pfor)
        else:
            self.PE = pk.parallel_reduce(
                pk.TeamPolicy(self.nbins, 1, 8), self.preduce)

    @pk.workunit
    def pfor(self, team: pk.TeamMember) -> None:
        bx: int = team.league_rank() // (self.nbiny * self.nbinz)
        by: int = (team.league_rank() // self.nbinz) % self.nbiny
        bz: int = team.league_rank() % self.nbinz

        i_offset: int = self.bin_offsets[bx][by][bz]

        def team_thread_for(bi: int):
            i: int = self.permute_vector[i_offset + bi]
            if i >= self.N_local:
                return

            x_i: float = self.x[i][0]
            y_i: float = self.x[i][1]
            z_i: float = self.x[i][2]
            type_i: int = self.type[i]

            f_i: t_scalar3 = t_scalar3()

            bx_j_start: int = bx
            if bx > 0:
                bx_j_start = bx - 1

            bx_j_stop: int = bx + 1
            if bx + 1 < self.nbinx:
                bx_j_stop = bx + 2

            by_j_start: int = by
            if by > 0:
                by_j_start = by - 1

            by_j_stop: int = by + 1
            if by + 1 < self.nbiny:
                by_j_stop = by + 2

            bz_j_start: int = bz
            if bz > 0:
                bz_j_start = bz - 1

            bz_j_stop: int = bz + 1
            if bz + 1 < self.nbinx:
                bz_j_stop = bz + 2

            for bx_j in range(bx_j_start, bx_j_stop):
                for by_j in range(by_j_start, by_j_stop):
                    for bz_j in range(bz_j_start, bz_j_stop):
                        j_offset: int = self.bin_offsets[bx_j][by_j][bz_j]

                        f_i_tmp: t_scalar3 = t_scalar3()

                        def thread_vector_reduce_x(bj: int, lf_i: pk.Acc[pk.double]):
                            j: int = self.permute_vector[j_offset + bj]

                            dx: float = x_i - self.x[j][0]
                            dy: float = y_i - self.x[j][1]
                            dz: float = z_i - self.x[j][2]

                            type_j: int = self.type[j]
                            rsq: float = (dx * dx) + (dy * dy) + (dz * dz)

                            if rsq < self.cutsq[type_i][type_j] and i != j:
                                r2inv: float = 1.0 / rsq
                                r6inv: float = r2inv * r2inv * r2inv
                                fpair: float = (
                                    r6inv * (self.lj1[type_i][type_j] * r6inv - self.lj2[type_i][type_j])) * r2inv

                                lf_i += dx * fpair

                        def thread_vector_reduce_y(bj: int, lf_i: pk.Acc[pk.double]):
                            j: int = self.permute_vector[j_offset + bj]

                            dx: float = x_i - self.x[j][0]
                            dy: float = y_i - self.x[j][1]
                            dz: float = z_i - self.x[j][2]

                            type_j: int = self.type[j]
                            rsq: float = (dx * dx) + (dy * dy) + (dz * dz)

                            if rsq < self.cutsq[type_i][type_j] and i != j:
                                r2inv: float = 1.0 / rsq
                                r6inv: float = r2inv * r2inv * r2inv
                                fpair: float = (
                                    r6inv * (self.lj1[type_i][type_j] * r6inv - self.lj2[type_i][type_j])) * r2inv

                                lf_i += dy * fpair

                        def thread_vector_reduce_z(bj: int, lf_i: pk.Acc[pk.double]):
                            j: int = self.permute_vector[j_offset + bj]

                            dx: float = x_i - self.x[j][0]
                            dy: float = y_i - self.x[j][1]
                            dz: float = z_i - self.x[j][2]

                            type_j: int = self.type[j]
                            rsq: float = (dx * dx) + (dy * dy) + (dz * dz)

                            if rsq < self.cutsq[type_i][type_j] and i != j:
                                r2inv: float = 1.0 / rsq
                                r6inv: float = r2inv * r2inv * r2inv
                                fpair: float = (
                                    r6inv * (self.lj1[type_i][type_j] * r6inv - self.lj2[type_i][type_j])) * r2inv

                                lf_i += dz * fpair

                        thread_vector_count: int = self.bin_count[bx_j][by_j][bz_j]
                        f_i_tmp_x: float = pk.parallel_reduce(
                            pk.ThreadVectorRange(team, thread_vector_count), thread_vector_reduce_x)
                        f_i_tmp_y: float = pk.parallel_reduce(
                            pk.ThreadVectorRange(team, thread_vector_count), thread_vector_reduce_y)
                        f_i_tmp_z: float = pk.parallel_reduce(
                            pk.ThreadVectorRange(team, thread_vector_count), thread_vector_reduce_z)

                        f_i.x += f_i_tmp_x
                        f_i.y += f_i_tmp_y
                        f_i.z += f_i_tmp_z

            self.f[i][0] = f_i.x
            self.f[i][1] = f_i.y
            self.f[i][2] = f_i.z

        team_thread_count: int = self.bin_count[bx][by][bz]
        pk.parallel_for(pk.TeamThreadRange(
            team, team_thread_count), team_thread_for)

    @pk.workunit
    def preduce(self, team: pk.TeamMember, PE_bi: pk.Acc[pk.double]) -> None:
        bx: int = team.league_rank() // (self.nbiny * self.nbinz)
        by: int = (team.league_rank() // self.nbinz) % self.nbiny
        bz: int = team.league_rank() % self.nbinz

        shift_flag: bool = True
        i_offset: int = self.bin_offsets[bx][by][bz]

        def team_thread_reduce(bi: int, PE_i: pk.Acc[pk.double]):
            i: int = self.permute_vector[i_offset + bi]
            if i >= self.N_local:
                return

            x_i: float = self.x[i][0]
            y_i: float = self.x[i][1]
            z_i: float = self.x[i][2]
            type_i: int = self.type[i]

            bx_j_start: int = bx
            if bx > 0:
                bx_j_start = bx - 1

            bx_j_stop: int = bx + 1
            if bx + 1 < self.nbinx:
                bx_j_stop = bx + 2

            by_j_start: int = by
            if by > 0:
                by_j_start = by - 1

            by_j_stop: int = by + 1
            if by + 1 < self.nbiny:
                by_j_stop = by + 2

            bz_j_start: int = bz
            if bz > 0:
                bz_j_start = bz - 1

            bz_j_stop: int = bz + 1
            if bz + 1 < self.nbinx:
                bz_j_stop = bz + 2

            for bx_j in range(bx_j_start, bx_j_stop):
                for by_j in range(by_j_start, by_j_stop):
                    for bz_j in range(bz_j_start, bz_j_stop):
                        j_offset: int = self.bin_offsets[bx_j][by_j][bz_j]

                        def thread_vector_reduce(bj: int, PE_ibj: pk.Acc[pk.double]):
                            j: int = self.permute_vector[j_offset + bj]

                            dx: float = x_i - self.x[j][0]
                            dy: float = y_i - self.x[j][1]
                            dz: float = z_i - self.x[j][2]

                            type_j: int = self.type[j]
                            rsq: float = (dx * dx) + (dy * dy) + (dz * dz)

                            if rsq < self.cutsq[type_i][type_j] and i != j:
                                r2inv: float = 1.0 / rsq
                                r6inv: float = r2inv * r2inv * r2inv

                                PE_ibj += 0.5 * r6inv * \
                                    (0.5 * self.lj1[type_i][type_j] *
                                     r6inv - self.lj2[type_i][type_j]) / 6.0

                                if shift_flag:
                                    r2invc: float = 1.0 / \
                                        self.cutsq[type_i][type_j]
                                    r6invc: float = r2inv * r2inv * r2inv

                                    PE_ibj -= 0.5 * r6invc * \
                                        (0.5 * self.lj1[type_i][type_j] *
                                         r6invc - self.lj2[type_i][type_j]) / 6.0

                        thread_vector_count: int = self.bin_count[bx_j][by_j][bz_j]
                        PE_ibj: float = pk.parallel_reduce(pk.ThreadVectorRange(
                            team, thread_vector_count), thread_vector_reduce)
                        PE_i += PE_ibj

        team_thread_count: int = self.bin_count[bx][by][bz]
        PE_i: float = pk.parallel_reduce(pk.TeamThreadRange(
            team, team_thread_count), team_thread_reduce)
