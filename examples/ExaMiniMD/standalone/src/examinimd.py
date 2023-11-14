import os
from typing import Optional

import pykokkos as pk

from binning import Binning
from binning_types.binning_kksort import BinningKKSort
from comm import Comm
from force import Force
from input import Input
from integrator import Integrator
from integrator_nve import IntegratorNVE
from modules_comm import comm_modules_instantiation
from modules_force import force_modules_instantiation
from modules_neighbor import neighbor_modules_instantiation
from neighbor import Neighbor
from property_kine import KinE
from property_pote import PotE
from property_temperature import Temperature
from system import System
from types_h import BinningType, IntegratorType


class ExaMiniMD:
    def __init__(self):
        space: Optional[str] = os.environ.get("PK_EXA_SPACE")
        if space is not None:
            pk.set_default_space(pk.ExecutionSpace(space))
            if space in {"Cuda", "HIP"}:
                pk.enable_uvm()

        self.system = System()
        self.system.init()

        self.input = Input(self.system)

        self.integrator: Integrator = None
        self.binning: Binning = None
        self.force: Force = None
        self.neighbor: Neighbor = None
        self.comm: Comm = None

    def init(self) -> None:
        self.input.read_command_line_args()
        self.input.read_file()

        if self.input.integrator_type == IntegratorType.INTEGRATOR_NVE.value:
            self.integrator = IntegratorNVE(self.system)

        if self.input.binning_type == BinningType.BINNING_KKSORT.value:
            self.binning = BinningKKSort(self.system)

        self.force = force_modules_instantiation(
            self.input, self.system, self.binning)

        for line in range(self.input.force_coeff_lines.extent(0)):
            self.force.init_coeff(
                self.input.input_data.words_in_line(
                    int(self.input.force_coeff_lines[line])),
                self.input.input_data.words[int(self.input.force_coeff_lines[line])])

        self.neighbor = neighbor_modules_instantiation(self.input)

        self.comm = comm_modules_instantiation(self.input, self.system)

        self.force.comm_newton = self.input.comm_newton
        if self.neighbor is not None:
            self.neighbor.comm_newton = self.input.comm_newton

        if self.system.do_print:
            print(f"Using: {self.force.name()} {self.neighbor.name()}"
                  f" {self.comm.name()} {self.binning.name()}")

        if self.system.N == 0:
            self.input.create_lattice(self.comm)

        self.comm.exchange()

        neigh_cutoff: float = self.input.force_cutoff + self.input.neighbor_skin

        self.binning.create_binning(
            neigh_cutoff, neigh_cutoff, neigh_cutoff, 1, True, False, True)

        self.comm.exchange_halo()

        self.binning.create_binning(
            neigh_cutoff, neigh_cutoff, neigh_cutoff, 1, True, True, False)

        if self.neighbor is not None:
            self.neighbor.create_neigh_list(
                self.system, self.binning, self.force.half_neigh, False, self.input.fill)

        if self.input.fill:
            self.system.f.fill(0)
        else:
            for i in range(len(self.system.f)):
                self.system.f[i] = 0

        self.force.compute(self.system, self.binning, self.neighbor)

        if self.input.comm_newton:
            self.comm.update_force()

        step: int = 0
        if self.input.thermo_rate > 0:
            temp = Temperature(self.comm)
            pote = PotE(self.comm)
            kine = KinE(self.comm)

            T: float = temp.compute(self.system)
            PE: float = pote.compute(
                self.system, self.binning, self.neighbor, self.force) / self.system.N
            KE: float = kine.compute(self.system) / self.system.N

            if self.system.do_print:
                if not self.system.print_lammps:
                    print()
                    print("#Timestep Temperature PotE ETot Time Atomsteps/s")
                    print(
                        f"{step} {T:.6f} {PE:.6f} {PE + KE:.6f} {0.0:.6f} {0.0:e}")
                else:
                    print()
                    print("Step Temp E_pair TotEng CPU")
                    print(f"     {step} {T:.6f} {PE:.6f} {PE + KE:.6f} 0.0)")

        if self.input.dumpbinaryflag:
            self.dump_binary(step)

        if self.input.correctnessflag:
            self.check_correctness(step)

    def run(self, nsteps: int) -> None:
        neigh_cutoff: float = self.input.force_cutoff + self.input.neighbor_skin

        temp = Temperature(self.comm)
        pote = PotE(self.comm)
        kine = KinE(self.comm)

        force_time: float = 0
        comm_time: float = 0
        neigh_time: float = 0
        other_time: float = 0

        last_time: float = 0

        timer = pk.Timer()
        force_timer = pk.Timer()
        comm_timer = pk.Timer()
        neigh_timer = pk.Timer()
        other_timer = pk.Timer()

        for step in range(1, nsteps + 1):
            other_timer.reset()
            self.integrator.initial_integrate()
            other_time += other_timer.seconds()

            if step % self.input.comm_exchange_rate == 0 and step > 0:
                comm_timer.reset()
                self.comm.exchange()
                comm_time += comm_timer.seconds()

                other_timer.reset()
                self.binning.create_binning(
                    neigh_cutoff, neigh_cutoff, neigh_cutoff, 1, True, False, True)
                other_time += other_timer.seconds()

                comm_timer.reset()
                self.comm.exchange_halo()
                comm_time += comm_timer.seconds()

                neigh_timer.reset()
                self.binning.create_binning(
                    neigh_cutoff, neigh_cutoff, neigh_cutoff, 1, True, True, False)

                if self.neighbor is not None:
                    self.neighbor.create_neigh_list(
                        self.system, self.binning, self.force.half_neigh, False, self.input.fill)
                neigh_time += neigh_timer.seconds()

            else:
                comm_timer.reset()
                self.comm.update_halo()
                comm_time += comm_timer.seconds()

            force_timer.reset()

            if self.input.fill:
                self.system.f.fill(0)
            else:
                for i in range(self.system.f.extent(0)):
                    for j in range(self.system.f.extent(1)):
                        self.system.f[i][j] = 0.0

            self.force.compute(self.system, self.binning, self.neighbor)
            force_time += force_timer.seconds()

            if self.input.comm_newton:
                comm_timer.reset()
                self.comm.update_force()
                comm_time += comm_timer.seconds()

            other_timer.reset()
            self.integrator.final_integrate()

            if step % self.input.thermo_rate == 0:
                T: float = temp.compute(self.system)
                PE: float = pote.compute(
                    self.system, self.binning, self.neighbor, self.force) / self.system.N
                KE: float = kine.compute(self.system) / self.system.N

                if self.system.do_print:
                    if not self.system.print_lammps:
                        time: float = timer.seconds()
                        print(
                            f"{step} {T:.6f} {PE:.6f} {PE + KE:.6f} {timer.seconds():.6f}"
                            f" {1.0 * self.system.N * self.input.thermo_rate / (time - last_time):e}")
                        last_time = time
                    else:
                        time: float = timer.seconds()
                        print(
                            f"     {step} {T:.6f} {PE:.6f} {PE + KE:.6f} {timer.seconds():.6f}")
                        last_time = time

            if self.input.dumpbinaryflag:
                self.dump_binary(step)

            if self.input.correctnessflag:
                self.check_correctness(step)

            other_time += other_timer.seconds()

        time: float = timer.seconds()
        T: float = temp.compute(self.system)
        PE: float = pote.compute(
            self.system, self.binning, self.neighbor, self.force) / self.system.N
        KE: float = kine.compute(self.system) / self.system.N

        if self.system.do_print:
            if not self.system.print_lammps:
                print()
                print(
                    "#Procs Particles |"
                    " Time T_Force T_Neigh T_Comm T_Other |"
                    " Steps/s Atomsteps/s Atomsteps/(proc*s)"
                )
                print(f"{self.comm.num_processes()} {self.system.N} |"
                      f" {time:.6f} {force_time:.6f} {neigh_time:.6f} {comm_time:.6f} {other_time:.6f} |"
                      f" {1.0 * nsteps / time:.6f} {1.0 * self.system.N * nsteps / time:e}"
                      f" {1.0 * self.system.N * nsteps / time / self.comm.num_processes():e} PERFORMANCE"
                      )
            else:
                print(
                    f"Loop time of {time} on {self.comm.num_processes()}"
                    f" procs for {nsteps} with {self.system.N} atoms")

    def dump_binary(self, step: int) -> None:
        # TODO: Unused
        pass

    def check_correctness(self, step: int) -> None:
        # TODO: Unused
        pass

    def print_performance(self) -> None:
        # TODO: Unused
        pass

    def shutdown(self) -> None:
        # TODO: Unused
        pass
