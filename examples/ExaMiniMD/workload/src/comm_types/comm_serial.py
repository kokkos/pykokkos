import copy
from typing import List

import pykokkos as pk

from comm import Comm
from system import System
from types_h import t_f, t_id, t_mass, t_q, t_type, t_v, t_x


@pk.classtype
class Particle:
    def __init__(self):
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vz: float = 0.0
        self.mass: float = 0.0
        self.q: float = 0.0

        self.id: int = 0
        self.type: int = 0


@pk.workload(
    x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    pack_indicies_all=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    pack_indicies=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class CommSerial(Comm):
    def __init__(self, s: System, comm_depth: float):
        super().__init__(s, comm_depth)

        # copied from System
        self.domain_x: float = 0.0
        self.domain_y: float = 0.0
        self.domain_z: float = 0.0

        self.sub_domain_x: float = 0.0
        self.sub_domain_y: float = 0.0
        self.sub_domain_z: float = 0.0

        self.sub_domain_hi_x: float = 0.0
        self.sub_domain_hi_y: float = 0.0
        self.sub_domain_hi_z: float = 0.0

        self.sub_domain_lo_x: float = 0.0
        self.sub_domain_lo_y: float = 0.0
        self.sub_domain_lo_z: float = 0.0

        self.x: pk.View2D[pk.double] = t_x(0, 3)
        self.v: pk.View2D[pk.double] = t_v(0, 3)
        self.f: pk.View2D[pk.double] = t_f(0, 3)
        self.id: pk.View1D[pk.int32] = t_id(0)
        self.type: pk.View1D[pk.int32] = t_type(0)
        self.q: pk.View1D[pk.double] = t_q(0)
        self.mass: pk.View1D[pk.double] = t_mass(0)

        # copied from Comm
        self.comm_depth: float = comm_depth

        print("CommSerial")
        self.pack_count: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.pack_indicies_all: pk.View2D[pk.int32] = pk.View([6, 0], pk.int32, layout=pk.Layout.LayoutRight)

        self.num_ghost: List[int] = [0] * 6
        self.ghost_offsets: List[int] = [0] * 6

        self.phase: int = 0

        # Assign
        self.workunit_id: int = 0

        # Needed for translation to succeed
        self.N_local: int = 0
        self.nparticles: int = 0
        self.update_threads: int = 0
        self.force_threads: int = 0
        self.N_ghost: int = 0

        self.pack_indicies: pk.View1D[pk.int32] = self.pack_indicies_all[0, :]
        self.ghost_offsets: pk.View1D[pk.int32] = pk.View([6], pk.int32)

    def exchange(self) -> None:
        self.s = copy.copy(self.system)
        self.N_local = self.system.N_local

        self.bind_views()
        self.workunit_id = 0
        pk.execute(pk.ExecutionSpace.Default, self)

    def exchange_halo(self) -> None:
        self.N_local = self.system.N_local
        self.N_ghost = 0

        self.s = copy.copy(self.system)
        for self.phase in range(6):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]
            count: int = 0
            self.pack_count[0] = 0

            sub: int = 0
            if self.phase % 2 == 1:
                sub = self.num_ghost[self.phase - 1]

            self.nparticles = self.N_local + self.N_ghost - sub
            self.bind_views()
            self.workunit_id = 1
            pk.execute(pk.ExecutionSpace.Default, self)

            count = self.pack_count[0]

            redo: bool = False

            if self.N_local + self.N_ghost + count > self.s.x.extent(0):
                self.system.grow(self.N_local + int(self.N_ghost) + int(count))
                self.s = copy.copy(self.system)
                redo = True

            if count > self.pack_indicies.extent(0):
                self.pack_indicies_all.resize(0, 6)
                self.pack_indicies_all.resize(1, int(count * 1.1))
                self.pack_indicies = self.pack_indicies_all[self.phase, :]
                redo = True

            if redo:
                self.pack_count[0] = 0
                self.workunit_id = 1
                self.bind_views()
                pk.execute(pk.ExecutionSpace.Default, self)

            self.num_ghost[self.phase] = count

            self.N_ghost += count

        self.system.N_ghost = self.N_ghost

    def update_halo(self) -> None:
        self.N_ghost = 0
        self.s = copy.copy(self.system)
        for self.phase in range(0, 6):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]

            self.bind_views()
            self.workunit_id = 2
            self.update_threads = self.num_ghost[self.phase]
            pk.execute(pk.ExecutionSpace.Default, self)
            self.N_ghost += self.num_ghost[self.phase]

    def update_force(self) -> None:
        self.s = copy.copy(self.system)
        self.ghost_offsets[0] = self.s.N_local
        for self.phase in range(1, 6):
            self.ghost_offsets[self.phase] = self.ghost_offsets[self.phase -
                                                                1] + self.num_ghost[self.phase - 1]

        for self.phase in range(5, -1, -1):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]

            self.bind_views()
            self.workunit_id = 3
            self.force_threads = self.num_ghost[self.phase]
            pk.execute(pk.ExecutionSpace.Default, self)

    def name(self) -> str:
        return "CommSerial"

    def num_processes(self) -> int:
        return 1

    def bind_views(self) -> None:
        self.x = self.s.x
        self.v = self.s.v
        self.f = self.s.f
        self.q = self.s.q
        self.id = self.s.id
        self.type = self.s.type

        self.domain_x = self.s.domain_x
        self.domain_y = self.s.domain_y
        self.domain_z = self.s.domain_z

        self.sub_domain_hi_x = self.s.sub_domain_hi_x
        self.sub_domain_hi_y = self.s.sub_domain_hi_y
        self.sub_domain_hi_z = self.s.sub_domain_hi_z

        self.sub_domain_lo_x = self.s.sub_domain_lo_x
        self.sub_domain_lo_y = self.s.sub_domain_lo_y
        self.sub_domain_lo_z = self.s.sub_domain_lo_z

    @pk.main
    def run(self) -> None:
        if self.workunit_id == 0:
            pk.parallel_for("CommSerial::exchange_self", self.N_local, self.tag_exchange_self)
        elif self.workunit_id == 1:
            pk.parallel_for("CommSerial::halo_exchange_self", self.nparticles, self.tag_halo_self)
        elif self.workunit_id == 2:
            pk.parallel_for("CommSerial::halo_update_self", self.update_threads, self.tag_halo_update_self)
        elif self.workunit_id == 3:
            pk.parallel_for("CommSerial::halo_force_self", self.force_threads, self.tag_halo_force_self)

    @pk.workunit
    def tag_exchange_self(self, i: int) -> None:
        x_: float = self.x[i][0]
        if x_ > self.domain_x:
            self.x[i][0] -= self.domain_x
        if x_ < 0:
            self.x[i][0] += self.domain_x

        y: float = self.x[i][1]
        if y > self.domain_y:
            self.x[i][1] -= self.domain_y
        if y < 0:
            self.x[i][1] += self.domain_y

        z: float = self.x[i][2]
        if z > self.domain_z:
            self.x[i][2] -= self.domain_z
        if z < 0:
            self.x[i][2] += self.domain_z

    @pk.workunit
    def tag_halo_self(self, i: int) -> None:
        if self.phase == 0:
            if self.x[i][0] >= self.sub_domain_hi_x - self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.x -= self.domain_x
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

        if self.phase == 1:
            if self.x[i][0] <= self.sub_domain_lo_x + self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.x += self.domain_x
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

        if self.phase == 2:
            if self.x[i][1] >= self.sub_domain_hi_y - self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.y -= self.domain_y
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

        if self.phase == 3:
            if self.x[i][1] <= self.sub_domain_lo_y + self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.y += self.domain_y
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

        if self.phase == 4:
            if self.x[i][2] >= self.sub_domain_hi_z - self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.z -= self.domain_z
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

        if self.phase == 5:
            if self.x[i][2] <= self.sub_domain_lo_z + self.comm_depth:
                pack_idx: int = pk.atomic_fetch_add(
                    self.pack_count, [0], 1)
                if (
                    pack_idx < self.pack_indicies.extent(0)
                    and self.N_local + self.N_ghost + pack_idx < self.x.extent(0)
                ):
                    self.pack_indicies[pack_idx] = i
                    p: Particle = self.get_particle(i)
                    p.z += self.domain_z
                    self.set_particle(
                        self.N_local + self.N_ghost + pack_idx, p)

    @pk.workunit
    def tag_halo_update_self(self, i: int) -> None:
        p: Particle = self.get_particle(self.pack_indicies[i])
        if self.phase == 0:
            p.x -= self.domain_x
        elif self.phase == 1:
            p.x += self.domain_x
        elif self.phase == 2:
            p.y -= self.domain_y
        elif self.phase == 3:
            p.y += self.domain_y
        elif self.phase == 4:
            p.z -= self.domain_z
        elif self.phase == 5:
            p.z += self.domain_z

        self.set_particle(self.N_local + self.N_ghost + i, p)

    @pk.workunit
    def tag_halo_force_self(self, ii: int) -> None:
        i: int = self.pack_indicies[ii]
        ghost_offsets_index: int = self.ghost_offsets[self.phase]
        fx_i: float = self.f[ghost_offsets_index + ii][0]
        fy_i: float = self.f[ghost_offsets_index + ii][1]
        fz_i: float = self.f[ghost_offsets_index + ii][2]

        self.f[i][0] += fx_i
        self.f[i][1] += fy_i
        self.f[i][2] += fz_i

    @pk.function
    def get_particle(self, i: int) -> Particle:
        p: Particle = Particle()

        p.x = self.x[i][0]
        p.y = self.x[i][1]
        p.z = self.x[i][2]

        p.vx = self.v[i][0]
        p.vy = self.v[i][1]
        p.vz = self.v[i][2]

        p.q = self.q[i]
        p.id = self.id[i]
        p.type = self.type[i]

        return p

    @pk.function
    def set_particle(self, i: int, p: Particle) -> None:
        self.x[i][0] = p.x
        self.x[i][1] = p.y
        self.x[i][2] = p.z

        self.v[i][0] = p.vx
        self.v[i][1] = p.vy
        self.v[i][2] = p.vz

        self.q[i] = p.q
        self.id[i] = p.id
        self.type[i] = p.type
