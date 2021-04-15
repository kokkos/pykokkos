import pykokkos as pk

from binning import Binning
from neighbor import Neighbor
from system import System


class NeighList2D:
    def __init__(self):
        self.maxneighs: int = 16

        self.num_neighs: pk.View1D = pk.View([0], pk.int32)
        self.neighs: pk.View2D = pk.View([0, 0], pk.int32)

    def get_neighs(self, i: int) -> pk.View:
        return self.neighs[i, :]

    def get_num_neighs(self, i: int) -> int:
        return self.num_neighs[i]


@pk.workload(x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class Neighbor2D(Neighbor):
    def __init__(self):
        self.neigh_cut: float = 0.0

        self.resize: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.new_maxneighs: pk.View1D[pk.int32] = pk.View([1], pk.int32)

        self.comm_newton: bool = False

        self.N_local: int = 0
        self.nhalo: int = 0
        self.nbinx: int = 0
        self.nbiny: int = 0
        self.nbinz: int = 0
        self.nbins: int = 0

        # copied from NeighList2D
        self.maxneighs: int = 0
        self.num_neighs: pk.View1D[pk.int32] = pk.View([0], pk.int32)
        self.neighs: pk.View2D[pk.int32] = pk.View([0, 0], pk.int32)

        # copied from self.create_neigh_list()
        self.half_neigh: bool = True
        self.bin_offsets: pk.View3D[pk.int32] = pk.View([0, 0, 0], pk.int32)
        self.bin_count: pk.View3D[pk.int32] = pk.View([0, 0, 0], pk.int32)
        self.permute_vector: pk.View1D[pk.int32] = pk.View([0], pk.int32)

        # copied from System
        self.x: pk.View2D[pk.double] = pk.View([0, 0], pk.double)
        self.type: pk.View1D[pk.int32] = pk.View([0], pk.int32)
        self.id: pk.View1D[pk.int32] = pk.View([0], pk.int32)

    def init(self, neigh_cut: float) -> None:
        self.neigh_cut = neigh_cut

        self.neigh_list = NeighList2D()

    def create_neigh_list(self, system: System, binning: Binning, half_neigh: bool, b: bool, fill: bool) -> None:
        self.N_local = system.N_local
        self.x = system.x
        self.type = system.type
        self.id = system.id
        self.half_neigh = half_neigh

        if self.neigh_list.num_neighs.extent(0) < self.N_local + 1:
            self.neigh_list.num_neighs = pk.View([self.N_local + 1], pk.int32)

        self.nhalo = binning.nhalo
        self.nbinx = binning.nbinx - 2 * self.nhalo
        self.nbiny = binning.nbiny - 2 * self.nhalo
        self.nbinz = binning.nbinz - 2 * self.nhalo

        self.nbins = self.nbinx * self.nbiny * self.nbinz

        self.bin_offsets = binning.binoffsets
        self.bin_count = binning.bincount
        self.permute_vector = binning.permute_vector

        self.bind_views()
        condition: bool = True

        while condition:
            if self.neigh_list.neighs.extent(0) < self.N_local + 1 or self.neigh_list.neighs.extent(1) < self.neigh_list.maxneighs:
                self.neigh_list.neighs = pk.View([self.N_local + 1, self.neigh_list.maxneighs], pk.int32)

            if fill:
                self.neigh_list.num_neighs.fill(0)
            else:
                for i in range(len(self.neigh_list.num_neighs)):
                    self.neigh_list.num_neighs[i] = 0

            self.resize[0] = 0

            self.bind_views()
            pk.execute(pk.ExecutionSpace.Default, self)

            if self.resize[0] != 0:
                self.neigh_list.maxneighs = int(self.new_maxneighs[0] * 1.2)

            condition = self.resize[0] != 0

    def bind_views(self) -> None:
        self.neighs = self.neigh_list.neighs
        self.num_neighs = self.neigh_list.num_neighs
        self.maxneighs = self.neigh_list.maxneighs

    def reverse_bind_views(self) -> None:
        self.neigh_list.neighs = self.neighs
        self.neigh_list.num_neighs = self.num_neighs
        self.neigh_list.maxneighs = self.maxneighs

    def get_neigh_list(self) -> NeighList2D:
        return self.neigh_list

    def name(self) -> str:
        return "Neighbor2D"

    @pk.main
    def run(self) -> None:
        if self.half_neigh:
            pk.parallel_for("Neighbor2D::fill_neigh_list_half",
                pk.TeamPolicy(self.nbins, "auto", 8), self.fill_neigh_list_half)
        else:
            pk.parallel_for("Neighbor2D::fill_neigh_list_full",
                pk.TeamPolicy(self.nbins, "auto", 8), self.fill_neigh_list_full)

    @pk.workunit
    def fill_neigh_list_full(self, team: pk.TeamMember) -> None:
        bx: int = team.league_rank() // (self.nbiny * self.nbinz) + self.nhalo
        by: int = (team.league_rank() // self.nbinz) % self.nbiny + self.nhalo
        bz: int = team.league_rank() % self.nbinz + self.nhalo

        i_offset: int = self.bin_offsets[bx][by][bz]

        def first_for_full(bi: int):
            i: int = self.permute_vector[i_offset + bi]
            if i >= self.N_local:
                return

            x_i: float = self.x[i][0]
            y_i: float = self.x[i][1]
            z_i: float = self.x[i][2]

            type_i: int = self.type[i]

            for bx_j in range(bx - 1, bx + 2):
                for by_j in range(by - 1, by + 2):
                    for bz_j in range(bz - 1, bz + 2):
                        j_offset: int = self.bin_offsets[bx_j][by_j][bz_j]

                        def second_for_full(bj: int):
                            j: int = self.permute_vector[j_offset + bj]

                            dx: float = x_i - self.x[j][0]
                            dy: float = y_i - self.x[j][1]
                            dz: float = z_i - self.x[j][2]
                            type_j: int = self.type[j]
                            rsq: float = dx * dx + dy * dy + dz * dz

                            if rsq <= (self.neigh_cut * self.neigh_cut) and i != j:
                                n: int = pk.atomic_fetch_add(
                                    self.num_neighs, [i], 1)
                                if n < self.maxneighs:
                                    self.neighs[i][n] = j

                        thread_vector_count: int = self.bin_count[bx_j][by_j][bz_j]
                        pk.parallel_for(pk.ThreadVectorRange(
                            team, thread_vector_count), second_for_full)

            def single_full():
                num_neighs_i: int = self.num_neighs[i]
                if num_neighs_i > self.maxneighs:
                    self.resize[0] = 1
                    self.new_maxneighs[0] = num_neighs_i

            pk.single(pk.PerThread(team), single_full)

        team_thread_count: int = self.bin_count[bx][by][bz]
        pk.parallel_for(pk.TeamThreadRange(
            team, team_thread_count), first_for_full)

    @pk.workunit
    def fill_neigh_list_half(self, team: pk.TeamMember) -> None:
        bx: int = team.league_rank() // (self.nbiny * self.nbinz) + self.nhalo
        by: int = (team.league_rank() // self.nbinz) % self.nbiny + self.nhalo
        bz: int = team.league_rank() % self.nbinz + self.nhalo

        i_offset: int = self.bin_offsets[bx][by][bz]
        def first_for_half(bi: int):
            i: int = self.permute_vector[i_offset + bi]
            if i >= self.N_local:
                return

            x_i: float = self.x[i][0]
            y_i: float = self.x[i][1]
            z_i: float = self.x[i][2]

            type_i: int = self.type[i]

            for bx_j in range(bx - 1, bx + 2):
                for by_j in range(by - 1, by + 2):
                    for bz_j in range(bz - 1, bz + 2):
                        j_offset: int = self.bin_offsets[bx_j][by_j][bz_j]

                        def second_for_half(bj: int):
                            j: int = self.permute_vector[j_offset + bj]

                            x_j: float = self.x[j][0]
                            y_j: float = self.x[j][1]
                            z_j: float = self.x[j][2]
                            if (
                                (j == i or j < self.N_local or self.comm_newton)
                                and not (x_j > x_i or
                                         (x_j == x_i and
                                          (y_j > y_i or
                                           (y_j == y_i and z_j == z_i))))
                            ):
                                return

                            dx: float = x_i - x_j
                            dy: float = y_i - y_j
                            dz: float = z_i - z_j

                            type_j: int = self.type[j]
                            rsq: float = dx * dx + dy * dy + dz * dz

                            if rsq <= (self.neigh_cut * self.neigh_cut):
                                n: int = pk.atomic_fetch_add(
                                    self.num_neighs, [i], 1)
                                if n < self.maxneighs:
                                    self.neighs[i][n] = j

                        thread_vector_count: int = self.bin_count[bx_j][by_j][bz_j]
                        pk.parallel_for(pk.ThreadVectorRange(
                            team, thread_vector_count), second_for_half)

            def single_half():
                num_neighs_i: int = self.num_neighs[i]
                if num_neighs_i > self.maxneighs:
                    self.resize[0] = 1
                    self.new_maxneighs[0] = num_neighs_i

            pk.single(pk.PerThread(team), single_half)

        team_thread_count: int = self.bin_count[bx][by][bz]
        pk.parallel_for(pk.TeamThreadRange(
            team, team_thread_count), first_for_half)
