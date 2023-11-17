from typing import List

import pykokkos as pk

from comm import Comm
from system import System


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


@pk.function
def get_particle(
    i: int, x: pk.View2D[float], v: pk.View2D[float],
    q: pk.View1D[float], id: pk.View1D[int], type: pk.View1D[int]
) -> Particle:
    p: Particle = Particle()

    p.x = x[i][0]
    p.y = x[i][1]
    p.z = x[i][2]

    p.vx = v[i][0]
    p.vy = v[i][1]
    p.vz = v[i][2]

    p.q = q[i]
    p.id = id[i]
    p.type = type[i]

    return p


@pk.function
def set_particle(
    i: int, p: Particle, x: pk.View2D[float], v: pk.View2D[float],
    q: pk.View1D[float], id: pk.View1D[int], type: pk.View1D[int]
) -> None:
    x[i][0] = p.x
    x[i][1] = p.y
    x[i][2] = p.z

    v[i][0] = p.vx
    v[i][1] = p.vy
    v[i][2] = p.vz

    q[i] = p.q
    id[i] = p.id
    type[i] = p.type


@pk.workunit
def tag_exchange_self(i: int, domain_x: float, domain_y: float, domain_z: float, x: pk.View2D[float]) -> None:
    x_: float = x[i][0]
    if x_ > domain_x:
        x[i][0] -= domain_x
    if x_ < 0:
        x[i][0] += domain_x

    y: float = x[i][1]
    if y > domain_y:
        x[i][1] -= domain_y
    if y < 0:
        x[i][1] += domain_y

    z: float = x[i][2]
    if z > domain_z:
        x[i][2] -= domain_z
    if z < 0:
        x[i][2] += domain_z


@pk.workunit
def tag_halo_self(
    i: int, domain_x: float, domain_y: float, domain_z: float,
    sub_domain_hi_x: float, sub_domain_hi_y: float, sub_domain_hi_z: float,
    sub_domain_lo_x: float, sub_domain_lo_y: float, sub_domain_lo_z: float,
    x: pk.View2D[float], v: pk.View2D[float], q: pk.View1D[float], id: pk.View1D[int],
    type: pk.View1D[int], comm_depth: float, pack_count: pk.View1D[int],
    phase: int, N_local: int, N_ghost: int, pack_indicies: pk.View1D[int]
) -> None:
    if phase == 0:
        if x[i][0] >= sub_domain_hi_x - comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.x -= domain_x
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)

    if phase == 1:
        if x[i][0] <= sub_domain_lo_x + comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.x += domain_x
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)

    if phase == 2:
        if x[i][1] >= sub_domain_hi_y - comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.y -= domain_y
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)

    if phase == 3:
        if x[i][1] <= sub_domain_lo_y + comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.y += domain_y
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)

    if phase == 4:
        if x[i][2] >= sub_domain_hi_z - comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.z -= domain_z
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)

    if phase == 5:
        if x[i][2] <= sub_domain_lo_z + comm_depth:
            pack_idx: int = pk.atomic_fetch_add(
                pack_count, [0], 1)
            if (
                pack_idx < pack_indicies.extent(0)
                and N_local + N_ghost + pack_idx < x.extent(0)
            ):
                pack_indicies[pack_idx] = i
                p: Particle = get_particle(i, x, v, q, id, type)
                p.z += domain_z
                set_particle(N_local + N_ghost + pack_idx, p, x, v, q, id, type)


@pk.workunit
def tag_halo_update_self(
    i: int, domain_x: float, domain_y: float, domain_z: float,
    x: pk.View2D[float], v: pk.View2D[float], q: pk.View1D[float],
    id: pk.View1D[int], type: pk.View1D[int], phase: int,
    N_local: int, N_ghost: int, pack_indicies: pk.View1D[int]
) -> None:
    p: Particle = get_particle(pack_indicies[i], x, v, q, id, type)
    if phase == 0:
        p.x -= domain_x
    elif phase == 1:
        p.x += domain_x
    elif phase == 2:
        p.y -= domain_y
    elif phase == 3:
        p.y += domain_y
    elif phase == 4:
        p.z -= domain_z
    elif phase == 5:
        p.z += domain_z

    set_particle(N_local + N_ghost + i, p, x, v, q, id, type)


@pk.workunit
def tag_halo_force_self(ii: int, f: pk.View2D[float], phase: int, pack_indicies: pk.View1D[int], ghost_offsets: pk.View1D[int]) -> None:
    i: int = pack_indicies[ii]
    ghost_offsets_index: int = ghost_offsets[phase]
    fx_i: float = f[ghost_offsets_index + ii][0]
    fy_i: float = f[ghost_offsets_index + ii][1]
    fz_i: float = f[ghost_offsets_index + ii][2]

    f[i][0] += fx_i
    f[i][1] += fy_i
    f[i][2] += fz_i


class CommSerial(Comm):
    def __init__(self, s: System, comm_depth: float):
        super().__init__(s, comm_depth)

        print("CommSerial")
        self.pack_count: pk.View1D[int] = pk.View([1], int)
        self.pack_indicies_all: pk.View2D[int] = pk.View([6, 0], int, layout=pk.Layout.LayoutRight)

        self.num_ghost: List[int] = [0] * 6
        self.ghost_offsets: List[int] = [0] * 6

        self.phase: int = 0

        self.pack_indicies: pk.View1D[int] = self.pack_indicies_all[0, :]
        self.ghost_offsets: pk.View1D[int] = pk.View([6], int)

    def exchange(self) -> None:
        pk.parallel_for("CommSerial::exchange_self", self.system.N_local, tag_exchange_self,
            domain_x=self.system.domain_x, domain_y=self.system.domain_y, domain_z=self.system.domain_z, x=self.system.x)

    def exchange_halo(self) -> None:
        N_ghost = 0

        for self.phase in range(6):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]
            count: int = 0
            self.pack_count[0] = 0

            sub: int = 0
            if self.phase % 2 == 1:
                sub = self.num_ghost[self.phase - 1]

            nparticles = self.system.N_local + N_ghost - sub
            pk.parallel_for("CommSerial::halo_exchange_self", nparticles, tag_halo_self,
                domain_x=self.system.domain_x, domain_y=self.system.domain_y, domain_z=self.system.domain_z,
                sub_domain_hi_x=self.system.sub_domain_hi_x, sub_domain_hi_y=self.system.sub_domain_hi_y,
                sub_domain_hi_z=self.system.sub_domain_hi_z, sub_domain_lo_x=self.system.sub_domain_lo_x,
                sub_domain_lo_y=self.system.sub_domain_lo_y, sub_domain_lo_z=self.system.sub_domain_lo_z, x=self.system.x,
                v=self.system.v, q=self.system.q, id=self.system.id, type=self.system.type, comm_depth=self.comm_depth,
                pack_count=self.pack_count, phase=self.phase, N_local=self.system.N_local, N_ghost=N_ghost, pack_indicies=self.pack_indicies)

            count = self.pack_count[0]

            redo: bool = False

            if self.system.N_local + N_ghost + count > self.system.x.extent(0):
                self.system.grow(self.system.N_local + int(N_ghost) + int(count))
                redo = True

            if count > self.pack_indicies.extent(0):
                self.pack_indicies_all.resize(0, 6)
                self.pack_indicies_all.resize(1, int(count * 1.1))
                self.pack_indicies = self.pack_indicies_all[self.phase, :]
                redo = True

            if redo:
                self.pack_count[0] = 0
                pk.parallel_for("CommSerial::halo_exchange_self", nparticles, tag_halo_self,
                    domain_x=self.system.domain_x, domain_y=self.system.domain_y, domain_z=self.system.domain_z,
                    sub_domain_hi_x=self.system.sub_domain_hi_x, sub_domain_hi_y=self.system.sub_domain_hi_y,
                    sub_domain_hi_z=self.system.sub_domain_hi_z, sub_domain_lo_x=self.system.sub_domain_lo_x,
                    sub_domain_lo_y=self.system.sub_domain_lo_y, sub_domain_lo_z=self.system.sub_domain_lo_z, x=self.system.x,
                    v=self.system.v, q=self.system.q, id=self.system.id, type=self.system.type, comm_depth=self.comm_depth,
                    pack_count=self.pack_count, phase=self.phase, N_local=self.system.N_local, N_ghost=N_ghost, pack_indicies=self.pack_indicies)

            self.num_ghost[self.phase] = count

            N_ghost += count

        self.system.N_ghost = N_ghost

    def update_halo(self) -> None:
        N_ghost = 0
        for self.phase in range(0, 6):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]

            pk.parallel_for("CommSerial::halo_update_self", self.num_ghost[self.phase], tag_halo_update_self,
                domain_x=self.system.domain_x, domain_y=self.system.domain_y, domain_z=self.system.domain_z,
                x=self.system.x, v=self.system.v, q=self.system.q, id=self.system.id, type=self.system.type, phase=self.phase,
                N_local=self.system.N_local, N_ghost=N_ghost, pack_indicies=self.pack_indicies)

            N_ghost += self.num_ghost[self.phase]

    def update_force(self) -> None:
        self.ghost_offsets[0] = self.system.N_local
        for self.phase in range(1, 6):
            self.ghost_offsets[self.phase] = self.ghost_offsets[self.phase -
                                                                1] + self.num_ghost[self.phase - 1]

        for self.phase in range(5, -1, -1):
            self.pack_indicies = self.pack_indicies_all[self.phase, :]

            pk.parallel_for("CommSerial::halo_force_self", self.num_ghost[self.phase], tag_halo_force_self,
                f=self.system.f, phase=self.phase, pack_indicies=self.pack_indicies, ghost_offsets=self.ghost_offsets)

    def name(self) -> str:
        return "CommSerial"

    def num_processes(self) -> int:
        return 1
