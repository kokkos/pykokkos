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


@pk.workunit
def fill_neigh_list_full(
    team: pk.TeamMember, neigh_cut: float, resize: pk.View1D[int], new_maxneighs: pk.View1D[int],
    N_local: int, nhalo: int, nbiny: int, nbinz: int,
    maxneighs: int, num_neighs: pk.View1D[int], neighs: pk.View2D[int],
    bin_offsets: pk.View3D[int], bin_count: pk.View3D[int], permute_vector: pk.View1D[int],
    x: pk.View2D[float]
) -> None:

    bx: int = team.league_rank() // (nbiny * nbinz) + nhalo
    by: int = (team.league_rank() // nbinz) % nbiny + nhalo
    bz: int = team.league_rank() % nbinz + nhalo

    i_offset: int = bin_offsets[bx][by][bz]

    def first_for_full(bi: int):
        i: int = permute_vector[i_offset + bi]
        if i >= N_local:
            return

        x_i: float = x[i][0]
        y_i: float = x[i][1]
        z_i: float = x[i][2]

        for bx_j in range(bx - 1, bx + 2):
            for by_j in range(by - 1, by + 2):
                for bz_j in range(bz - 1, bz + 2):
                    j_offset: int = bin_offsets[bx_j][by_j][bz_j]

                    def second_for_full(bj: int):
                        j: int = permute_vector[j_offset + bj]

                        dx: float = x_i - x[j][0]
                        dy: float = y_i - x[j][1]
                        dz: float = z_i - x[j][2]
                        rsq: float = dx * dx + dy * dy + dz * dz

                        if rsq <= (neigh_cut * neigh_cut) and i != j:
                            n: int = pk.atomic_fetch_add(
                                num_neighs, [i], 1)
                            if n < maxneighs:
                                neighs[i][n] = j

                    thread_vector_count: int = bin_count[bx_j][by_j][bz_j]
                    pk.parallel_for(pk.ThreadVectorRange(
                        team, thread_vector_count), second_for_full)

        def single_full():
            num_neighs_i: int = num_neighs[i]
            if num_neighs_i > maxneighs:
                resize[0] = 1
                new_maxneighs[0] = num_neighs_i

        pk.single(pk.PerThread(team), single_full)

    team_thread_count: int = bin_count[bx][by][bz]
    pk.parallel_for(pk.TeamThreadRange(
        team, team_thread_count), first_for_full)

@pk.workunit
def fill_neigh_list_half(
    team: pk.TeamMember, neigh_cut: float, resize: pk.View1D[int], new_maxneighs: pk.View1D[int],
    comm_newton: bool, N_local: int, nhalo: int, nbiny: int, nbinz: int,
    maxneighs: int, num_neighs: pk.View1D[int], neighs: pk.View2D[int],
    bin_offsets: pk.View3D[int], bin_count: pk.View3D[int], permute_vector: pk.View1D[int],
    x: pk.View2D[float]
) -> None:

    bx: int = team.league_rank() // (nbiny * nbinz) + nhalo
    by: int = (team.league_rank() // nbinz) % nbiny + nhalo
    bz: int = team.league_rank() % nbinz + nhalo

    i_offset: int = bin_offsets[bx][by][bz]
    def first_for_half(bi: int):
        i: int = permute_vector[i_offset + bi]
        if i >= N_local:
            return

        x_i: float = x[i][0]
        y_i: float = x[i][1]
        z_i: float = x[i][2]

        for bx_j in range(bx - 1, bx + 2):
            for by_j in range(by - 1, by + 2):
                for bz_j in range(bz - 1, bz + 2):
                    j_offset: int = bin_offsets[bx_j][by_j][bz_j]

                    def second_for_half(bj: int):
                        j: int = permute_vector[j_offset + bj]

                        x_j: float = x[j][0]
                        y_j: float = x[j][1]
                        z_j: float = x[j][2]
                        if (
                            (j == i or j < N_local or comm_newton)
                            and not (x_j > x_i or
                                        (x_j == x_i and
                                        (y_j > y_i or
                                        (y_j == y_i and z_j == z_i))))
                        ):
                            return

                        dx: float = x_i - x_j
                        dy: float = y_i - y_j
                        dz: float = z_i - z_j

                        rsq: float = dx * dx + dy * dy + dz * dz

                        if rsq <= (neigh_cut * neigh_cut):
                            n: int = pk.atomic_fetch_add(
                                num_neighs, [i], 1)
                            if n < maxneighs:
                                neighs[i][n] = j

                    thread_vector_count: int = bin_count[bx_j][by_j][bz_j]
                    pk.parallel_for(pk.ThreadVectorRange(
                        team, thread_vector_count), second_for_half)

        def single_half():
            num_neighs_i: int = num_neighs[i]
            if num_neighs_i > maxneighs:
                resize[0] = 1
                new_maxneighs[0] = num_neighs_i

        pk.single(pk.PerThread(team), single_half)

    team_thread_count: int = bin_count[bx][by][bz]
    pk.parallel_for(pk.TeamThreadRange(
        team, team_thread_count), first_for_half)

class Neighbor2D(Neighbor):
    def __init__(self):
        self.neigh_cut: float = 0.0

        self.resize: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.new_maxneighs: pk.View1D[pk.int32] = pk.View([1], pk.int32)

    def init(self, neigh_cut: float) -> None:
        self.neigh_cut = neigh_cut

        self.neigh_list = NeighList2D()

    def create_neigh_list(self, system: System, binning: Binning, half_neigh: bool, b: bool, fill: bool) -> None:
        if self.neigh_list.num_neighs.extent(0) < system.N_local + 1:
            self.neigh_list.num_neighs = pk.View([system.N_local + 1], pk.int32)

        nhalo: int = binning.nhalo
        nbinx: int = binning.nbinx - 2 * nhalo
        nbiny: int = binning.nbiny - 2 * nhalo
        nbinz: int = binning.nbinz - 2 * nhalo

        nbins: int = nbinx * nbiny * nbinz

        condition: bool = True
        while condition:
            if self.neigh_list.neighs.extent(0) < system.N_local + 1 or self.neigh_list.neighs.extent(1) < self.neigh_list.maxneighs:
                self.neigh_list.neighs = pk.View([system.N_local + 1, self.neigh_list.maxneighs], pk.int32)

            if fill:
                self.neigh_list.num_neighs.fill(0)
            else:
                for i in range(len(self.neigh_list.num_neighs)):
                    self.neigh_list.num_neighs[i] = 0

            self.resize[0] = 0

            if half_neigh:
                pk.parallel_for("Neighbor2D::fill_neigh_list_half",
                    pk.TeamPolicy(nbins, pk.AUTO, 8), fill_neigh_list_half, neigh_cut=self.neigh_cut,
                    resize=self.resize, new_maxneighs=self.new_maxneighs, comm_newton=self.comm_newton, N_local=system.N_local,
                    nhalo=nhalo, nbiny=nbiny, nbinz=nbinz,
                    maxneighs=self.neigh_list.maxneighs, num_neighs=self.neigh_list.num_neighs, neighs=self.neigh_list.neighs,
                    bin_offsets=binning.binoffsets, bin_count=binning.bincount, permute_vector=binning.permute_vector,
                    x=system.x)
            else:
                pk.parallel_for("Neighbor2D::fill_neigh_list_full",
                    pk.TeamPolicy(nbins, pk.AUTO, 8), fill_neigh_list_full, neigh_cut=self.neigh_cut,
                    resize=self.resize, new_maxneighs=self.new_maxneighs, N_local=system.N_local,
                    nhalo=nhalo, nbiny=nbiny, nbinz=nbinz,
                    maxneighs=self.neigh_list.maxneighs, num_neighs=self.neigh_list.num_neighs, neighs=self.neigh_list.neighs,
                    bin_offsets=binning.binoffsets, bin_count=binning.bincount, permute_vector=binning.permute_vector,
                    x=system.x)

            if self.resize[0] != 0:
                self.neigh_list.maxneighs = int(self.new_maxneighs[0] * 1.2)

            condition = self.resize[0] != 0

    def reverse_bind_views(self) -> None:
        self.neigh_list.neighs = self.neighs
        self.neigh_list.num_neighs = self.num_neighs
        self.neigh_list.maxneighs = self.maxneighs

    def get_neigh_list(self) -> NeighList2D:
        return self.neigh_list

    def name(self) -> str:
        return "Neighbor2D"
