from typing import List

import pykokkos as pk

from binning import Binning
from system import System
from types_h import t_x, t_v, t_f, t_id, t_type, t_q, t_mass


@pk.workload(
    x=pk.ViewTypeInfo(layout=pk.LayoutRight))
class BinningKKSort(Binning):
    def __init__(self, s: System):
        super().__init__(s)

        # copied from Binning
        self.nbinx: int = 0
        self.nbiny: int = 0
        self.nbinz: int = 0
        self.nhalo: int = 0
        self.minx: float = 0.0
        self.maxx: float = 0.0
        self.miny: float = 0.0
        self.maxy: float = 0.0
        self.minz: float = 0.0
        self.maxz: float = 0.0

        # copied from self.create_binning()
        self.bincount: pk.View3D[pk.int32] = self.t_bincount(self.nbinx, self.nbiny, self.nbinz)
        self.binoffsets: pk.View3D[pk.int32] = self.t_binoffsets(self.nbinx, self.nbiny, self.nbinz)

        self.x: pk.View2D[pk.double] = s.x
        self.v: pk.View2D[pk.double] = s.v
        self.f: pk.View2D[pk.double] = s.f
        self.type: pk.View1D[pk.int32] = s.type
        self.id: pk.View1D[pk.int32] = s.id
        self.q: pk.View1D[pk.double] = s.q

        self.range_min: int = 0
        self.range_max: int = 0

        self.permute_vector: pk.View1D[pk.int32] = pk.View([self.x.extent(0)], pk.int32)
        self.bin_count_1d: pk.View1D[pk.int32] = pk.View([3], pk.int32)
        self.bin_offsets_1d: pk.View1D[pk.int32] = pk.View([3], pk.int32)

        self.sort: bool = False

    def create_binning(
            self, dx_in: float, dy_in: float, dz_in: float, halo_depth: int,
            do_local: bool, do_ghost: bool, sort: bool) -> None:
        if do_local or do_ghost:
            self.nhalo = halo_depth
            range_min: int = 0 if do_local else self.system.N_local
            range_max: int = int(
                ((self.system.N_local + self.system.N_ghost) if do_ghost else self.system.N_local))

            self.range_min = range_min
            self.range_max = range_max

            self.nbinx = int(self.system.sub_domain_x / dx_in)
            self.nbiny = int(self.system.sub_domain_y / dy_in)
            self.nbinz = int(self.system.sub_domain_z / dz_in)

            if self.nbinx == 0:
                self.nbinx = 1
            if self.nbiny == 0:
                self.nbiny = 1
            if self.nbinz == 0:
                self.nbinz = 1

            dx: float = self.system.sub_domain_x / self.nbinx
            dy: float = self.system.sub_domain_y / self.nbiny
            dz: float = self.system.sub_domain_z / self.nbinz

            self.nbinx += 2 * halo_depth
            self.nbiny += 2 * halo_depth
            self.nbinz += 2 * halo_depth

            eps: float = dx / 1000
            self.minx = -dx * halo_depth - eps + self.system.sub_domain_lo_x
            self.maxx = dx * halo_depth + eps + self.system.sub_domain_hi_x
            self.miny = -dy * halo_depth - eps + self.system.sub_domain_lo_y
            self.maxy = dy * halo_depth + eps + self.system.sub_domain_hi_y
            self.minz = -dz * halo_depth - eps + self.system.sub_domain_lo_z
            self.maxz = dz * halo_depth + eps + self.system.sub_domain_hi_z

            # Bind views
            self.x = self.system.x
            self.v = self.system.v
            self.f = self.system.f
            self.type = self.system.type
            self.id = self.system.id
            self.q = self.system.q

            # Views
            self.bincount: pk.View3D = self.t_bincount(
                self.nbinx, self.nbiny, self.nbinz, pk.int32)
            self.binoffsets: pk.View3D = self.t_binoffsets(
                self.nbinx, self.nbiny, self.nbinz, pk.int32)

            self.sort = sort
            self.permute_vector.resize(0, range_max - range_min)
            pk.execute(pk.ExecutionSpace.Default, self)


    @pk.main
    def run(self) -> None:
        nbin: List[int] = [self.nbinx, self.nbiny, self.nbinz]
        min_values: List[float] = [self.minx, self.miny, self.miny]
        max_values: List[float] = [self.maxx, self.maxy, self.maxz]

        x_sub = self.x[self.range_min:self.range_max, :]
        binop = pk.BinOp3D(x_sub, nbin, min_values, max_values)
        sorter = pk.BinSort(x_sub, binop)
        sorter.create_permute_vector()
        self.permute_vector = sorter.get_permute_vector()

        self.bin_count_1d = sorter.get_bin_count()
        self.bin_offsets_1d = sorter.get_bin_offsets()

        pk.parallel_for("Binning::AssignOffsets",
            self.nbinx * self.nbiny * self.nbinz, self.assign_offsets)

        if self.sort:
            sorter.sort(x_sub)
            v_sub = self.v[self.range_min:self.range_max, :]
            sorter.sort(v_sub)
            f_sub = self.f[self.range_min:self.range_max, :]
            sorter.sort(f_sub)
            sorter.sort(self.type)
            sorter.sort(self.id)
            sorter.sort(self.q)

    @pk.workunit
    def assign_offsets(self, i: int) -> None:
        ix: int = i // (self.nbiny * self.nbinz)
        iy: int = (i // self.nbinz) % self.nbiny
        iz: int = i % self.nbinz

        self.binoffsets[ix][iy][iz] = self.bin_offsets_1d[i]
        self.bincount[ix][iy][iz] = self.bin_count_1d[i]

    def name(self) -> str:
        return "BinningKKSort"
