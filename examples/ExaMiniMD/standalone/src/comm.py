import sys

from system import System


class Comm:
    def __init__(self, s: System, comm_depth: float):
        self.system = s
        self.comm_depth = comm_depth

    def exchange(self) -> None:
        pass

    def exchange_halo(self) -> None:
        pass

    def update_halo(self) -> None:
        pass

    def update_force(self) -> None:
        pass

    def reduce_float(self, values, N: int) -> None:
        pass

    def reduce_int(self, values, N: int) -> None:
        pass

    def scan_int(self, values, N: int) -> None:
        pass

    def create_domain_decomposition(self) -> None:
        self.system.sub_domain_lo_x = 0.0
        self.system.sub_domain_lo_y = 0.0
        self.system.sub_domain_lo_z = 0.0

        self.system.sub_domain_x = self.system.domain_x
        self.system.sub_domain_hi_x = self.system.domain_x

        self.system.sub_domain_y = self.system.domain_y
        self.system.sub_domain_hi_y = self.system.domain_y

        self.system.sub_domain_z = self.system.domain_z
        self.system.sub_domain_hi_z = self.system.domain_z

    def num_processes(self) -> None:
        pass

    def error(self, errormsg: str) -> None:
        print(errormsg)
        sys.exit(1)
