import pykokkos as pk

from types_h import t_f, t_id, t_mass, t_q, t_type, t_v, t_x


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


class System:
    def __init__(self):
        self.N: int = 0
        self.N_max: int = 0
        self.N_local: int = 0
        self.N_ghost: int = 0
        self.ntypes: int = 1

        self.x = t_x(0, 3, pk.double)
        self.v = t_v(0, 3, pk.double)
        self.f = t_f(0, 3, pk.double)
        self.id = t_id(0, pk.int32)
        self.type = t_type(0, pk.int32)
        self.q = t_q(0, pk.double)
        self.mass = t_mass(0, pk.double)

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

        self.mvv2e: float = 0.0
        self.boltz: float = 0.0
        self.dt: float = 0.0

        self.do_print: bool = True
        self.print_lammps: bool = False

    def init(self) -> None:
        self.x.resize(0, self.N_max)
        self.v.resize(0, self.N_max)
        self.f.resize(0, self.N_max)
        self.id.resize(0, self.N_max)
        self.type.resize(0, self.N_max)
        self.q.resize(0, self.N_max)
        self.mass.resize(0, self.N_max)

    def grow(self, N_new: int) -> None:
        if N_new > self.N_max:
            self.N_max = N_new

            self.x.resize(0, self.N_max)
            self.v.resize(0, self.N_max)
            self.f.resize(0, self.N_max)

            self.id.resize(0, self.N_max)

            self.type.resize(0, self.N_max)

            self.q.resize(0, self.N_max)

    def print_particles(self) -> None:
        print("Print all particles:")
        print(f"  Owned: {self.N_local}")
        for i in range(self.N_local):
            print(f"    {i} {self.x[i][0]} {self.x[i][1]} {self.x[i][0]} |"
                  f"{self.v[i][0]} {self.v[i][1]} {self.v[i][0]} |"
                  f"{self.f[i][0]} {self.f[i][1]} {self.f[i][0]} |"
                  f"{self.type[i]} {self.q[i]}")

        print(f"  Ghost: {self.N_ghost}")
        for i in range(self.N_local + self.N_ghost):
            print(f"    {i} {self.x[i][0]} {self.x[i][1]} {self.x[i][0]} |"
                  f"{self.v[i][0]} {self.v[i][1]} {self.v[i][0]} |"
                  f"{self.f[i][0]} {self.f[i][1]} {self.f[i][0]} |"
                  f"{self.type[i]} {self.q[i]}")

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

    @pk.function
    def copy(self, dest: int, src: int, nx: int, ny: int, nz: int) -> None:
        self.x[dest][0] = self.x[src][0] + self.domain_x * nx
        self.x[dest][1] = self.x[src][1] + self.domain_y * ny
        self.x[dest][2] = self.x[src][2] + self.domain_z * nz

        self.v[dest][0] = self.v[src][0]
        self.v[dest][1] = self.v[src][1]
        self.v[dest][2] = self.v[src][2]

        self.type[dest] = self.type[src]
        self.id[dest] = self.id[src]
        self.q[dest] = self.q[src]

    @pk.function
    def copy_halo_update(self, dest: int, src: int, nx: int, ny: int, nz: int) -> None:
        self.x[dest][0] = self.x[src][0] + self.domain_x * nx
        self.x[dest][1] = self.x[src][1] + self.domain_y * ny
        self.x[dest][2] = self.x[src][2] + self.domain_z * nz
