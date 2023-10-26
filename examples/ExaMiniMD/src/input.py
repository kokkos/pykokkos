import argparse
import copy
import math
import os
import random
import struct
import sys
from typing import List, Tuple

NUMBA_ENABLED: bool = False
if "PK_EXA_NUMBA" in os.environ:
    NUMBA_ENABLED = True
    print("NUMBA ENABLED")

    from numba import jit, prange

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

import numpy as np
import pykokkos as pk

from comm import Comm
from system import System
from property_temperature import Temperature
from types_h import (
    Units, LatticeType, IntegratorType, CommType, ForceType,
    NeighborType, ForceIterationType, BinningType, InputFileType, t_mass
)


class ItemizedFile:
    def __init__(self):
        self.nlines: int = 0
        self.max_nlines: int = 0
        self.words: List[List[str]] = []
        self.words_per_line: int = 32
        self.max_word_size: int = 32

    def allocate_words(self, num_lines: int) -> None:
        self.nlines = 0

        if self.max_nlines >= num_lines:
            for i in range(self.max_nlines):
                for j in range(self.words_per_line):
                    self.words[i][j] = ""

            return

        self.max_nlines = num_lines
        self.words = [[]] * self.max_nlines
        for i in range(self.max_nlines):
            self.words[i] = [""] * self.words_per_line

    def free_words(self) -> None:
        self.words = []

    def print_line(self, i: int) -> None:
        for j in range(self.words_per_line):
            if self.words[i][j] != "":
                print(f"{self.words[i][j]} ", end="")

        print()

    def words_in_line(self, i: int) -> int:
        count: int = 0
        for j in range(self.words_per_line):
            if self.words[i][j] != "":
                count += 1

        return count

    def print(self) -> None:
        for l in range(self.nlines):
            self.print_line(l)

    def add_line(self, line: str) -> None:
        pos: int = 0
        if self.nlines < self.max_nlines:
            j: int = 0
            while pos < len(line) and j < self.words_per_line:
                while ((line[pos] == " " or line[pos] == "\t")
                        and pos < len(line)):
                    pos += 1

                k: int = 0
                while (pos < len(line) and line[pos] != " "
                        and line[pos] != "\t" and k < self.max_word_size):
                    self.words[self.nlines][j] += line[pos]
                    k += 1
                    pos += 1

                j += 1

        self.nlines += 1


class LAMMPS_RandomVelocityGeom:
    def __init__(self):
        self.seed: int = 0

    def uniform(self) -> float:
        IA: int = 16807
        IM: int = 2147483647
        AM: float = 1.0 / IM
        IQ: int = 127773
        IR: int = 2836
        k: int = self.seed // IQ

        self.seed = IA * (self.seed - k * IQ) - IR * k
        if self.seed < 0:
            self.seed += IM
        ans: float = AM * self.seed
        return ans

    def gaussian(self) -> float:
        v1: float = 2.0 * self.uniform() - 1.0
        v2: float = 2.0 * self.uniform() - 1.0
        rsq: float = (v1 * v1) + (v2 * v2)

        while rsq >= 1.0 or rsq == 0.0:
            v1 = 2.0 * self.uniform() - 1.0
            v2 = 2.0 * self.uniform() - 1.0
            rsq = (v1 * v1) + (v2 * v2)

        fac: float = math.sqrt(-2.0 * math.log(rsq) / rsq)
        return v2 * fac

    def reset(self, ibase: int, coord: List[float]) -> None:
        # Potential complete numpy implementation at:
        # https://loicpefferkorn.net/2013/09/python-force-c-integer-overflow-behavior/

        # An explanation of the C++ version:
        # By casting ibase to a char array, the data itself (ibase)
        # is unchanged. What changes is the way we access this data.
        # char *str is a char array of size 4, and each element is
        # the corresponds to 8 bits of ibase. Accessing an element
        # through str (str[i]) returns the 2's complement int value
        # of those 8 bits.

        # Convert ibase to a char array. We need & 0xffffffff to get
        # the 2's complement representation. We check the sign bit
        # since we want signed chars
        ibase &= 0xffffffff
        ibase_bin = list(ibase.to_bytes(4, "little"))
        for i in range(4):
            if (ibase_bin[i] & (1 << 7)):
                ibase_bin[i] -= 1 << 8

        hash_uint: int = 0

        for i in ibase_bin:
            hash_uint += i
            hash_uint &= 0xFFFFFFFF
            hash_uint += hash_uint << 10
            hash_uint &= 0xFFFFFFFF
            hash_uint ^= hash_uint >> 6
            hash_uint &= 0xFFFFFFFF

        # Need struct.pack since to_bytes is not implemented for float
        coord_bin: List[List[int]] = [[0], [0], [0]]
        for i in range(len(coord)):
            x = list(struct.pack("d", coord[i]))
            for j in range(len(x)):
                if (x[j] & (1 << 7)):
                    x[j] -= 1 << 8
            coord_bin[i] = x

        for index in range(3):
            for i in coord_bin[index]:
                hash_uint += i
                hash_uint &= 0xFFFFFFFF
                hash_uint += hash_uint << 10
                hash_uint &= 0xFFFFFFFF
                hash_uint ^= hash_uint >> 6
                hash_uint &= 0xFFFFFFFF

        hash_uint += hash_uint << 3
        hash_uint &= 0xFFFFFFFF
        hash_uint ^= hash_uint >> 11
        hash_uint &= 0xFFFFFFFF
        hash_uint += hash_uint << 15
        hash_uint &= 0xFFFFFFFF

        self.seed = int(hash_uint & 0x7ffffff)
        if self.seed == 0:
            self.seed = 1

        for i in range(5):
            self.uniform()

if NUMBA_ENABLED:
    @jit(nopython=True, cache=True)
    def uniform(seed) -> Tuple[int, int]:
        IA: int = 16807
        IM: int = 2147483647
        AM: float = 1.0 / IM
        IQ: int = 127773
        IR: int = 2836
        k: int = seed // IQ

        seed = IA * (seed - k * IQ) - IR * k
        if seed < 0:
            seed += IM
        ans: float = AM * seed
        return ans, seed

    @jit(nopython=True, cache=True)
    def reset(hash_uint, coord_bin: np.ndarray) -> int:
        # Potential complete numpy implementation at:
        # https://loicpefferkorn.net/2013/09/python-force-c-integer-overflow-behavior/

        # An explanation of the C++ version:
        # By casting ibase to a char array, the data itself (ibase)
        # is unchanged. What changes is the way we access this data.
        # char *str is a char array of size 4, and each element is
        # the corresponds to 8 bits of ibase. Accessing an element
        # through str (str[i]) returns the 2's complement int value
        # of those 8 bits.

        for index in range(3):
            for i in coord_bin[index]:
                hash_uint += i
                hash_uint &= 0xFFFFFFFF
                hash_uint += hash_uint << 10
                hash_uint &= 0xFFFFFFFF
                hash_uint ^= hash_uint >> 6
                hash_uint &= 0xFFFFFFFF

        hash_uint += hash_uint << 3
        hash_uint &= 0xFFFFFFFF
        hash_uint ^= hash_uint >> 11
        hash_uint &= 0xFFFFFFFF
        hash_uint += hash_uint << 15
        hash_uint &= 0xFFFFFFFF

        seed = int(hash_uint & 0x7ffffff)
        if seed == 0:
            seed = 1

        for i in range(5):
            IA: int = 16807
            IM: int = 2147483647
            IQ: int = 127773
            IR: int = 2836
            k: int = seed // IQ

            seed = IA * (seed - k * IQ) - IR * k
            if seed < 0:
                seed += IM

        return seed

    @jit(nopython=True, cache=True)
    def init_v(hash_uint: int, coord_bin: np.ndarray, v: np.ndarray, mass: np.ndarray, type: np.ndarray, N_local: int) -> Tuple[int, int, int, int]:
        total_mass: float = 0.0
        total_momentum_x: float = 0.0
        total_momentum_y: float = 0.0
        total_momentum_z: float = 0.0

        for i in range(N_local):
            seed = reset(hash_uint, coord_bin[i, :, :])

            mass_i: float = mass[int(type[i])]
            ans, seed = uniform(seed)
            vx: float = ans - 0.5
            ans, seed = uniform(seed)
            vy: float = ans - 0.5
            ans, seed = uniform(seed)
            vz: float = ans - 0.5

            mass_i_sqrt = math.sqrt(mass_i)
            v[i][0] = vx / mass_i_sqrt
            v[i][1] = vy / mass_i_sqrt
            v[i][2] = vz / mass_i_sqrt

            total_mass += mass_i
            total_momentum_x += mass_i * v[i][0]
            total_momentum_y += mass_i * v[i][1]
            total_momentum_z += mass_i * v[i][2]

        return total_mass, total_momentum_x, total_momentum_y, total_momentum_z

    @jit(nopython=True, cache=True, parallel=True)
    def calculate_n(ix_start, ix_end, iy_start, iy_end, iz_start, iz_end, lattice_constant, basis, sub_domain_lo_x, sub_domain_lo_y, sub_domain_lo_z, sub_domain_hi_x, sub_domain_hi_y, sub_domain_hi_z):
        n: int = 0
        for iz in prange(iz_start, iz_end + 1):
            for iy in range(iy_start, iy_end + 1):
                for ix in range(ix_start, ix_end + 1):
                    for k in range(4):
                        xtmp: float = (lattice_constant *
                                    (1.0 * ix + basis[k][0]))
                        ytmp: float = (lattice_constant *
                                    (1.0 * iy + basis[k][1]))
                        ztmp: float = (lattice_constant *
                                    (1.0 * iz + basis[k][2]))

                        if (
                            xtmp >= sub_domain_lo_x
                            and ytmp >= sub_domain_lo_y
                            and ztmp >= sub_domain_lo_z
                            and xtmp < sub_domain_hi_x
                            and ytmp < sub_domain_hi_y
                            and ztmp < sub_domain_hi_z
                        ):
                            n += 1
        return n

    @jit(nopython=True, cache=True)
    def init_x(ix_start, ix_end, iy_start, iy_end, iz_start, iz_end, lattice_constant, basis, sub_domain_lo_x, sub_domain_lo_y, sub_domain_lo_z, sub_domain_hi_x, sub_domain_hi_y, sub_domain_hi_z, x, type, id, ntypes):
        n: int = 0
        for iz in range(iz_start, iz_end + 1):
            for iy in range(iy_start, iy_end + 1):
                for ix in range(ix_start, ix_end + 1):
                    for k in range(4):
                        xtmp: float = (lattice_constant *
                                    (1.0 * ix + basis[k][0]))
                        ytmp: float = (lattice_constant *
                                    (1.0 * iy + basis[k][1]))
                        ztmp: float = (lattice_constant *
                                    (1.0 * iz + basis[k][2]))

                        if (
                            xtmp >= sub_domain_lo_x
                            and ytmp >= sub_domain_lo_y
                            and ztmp >= sub_domain_lo_z
                            and xtmp < sub_domain_hi_x
                            and ytmp < sub_domain_hi_y
                            and ztmp < sub_domain_hi_z
                        ):
                            x[n][0] = xtmp
                            x[n][1] = ytmp
                            x[n][2] = ztmp
                            type[n] = random.randint(0, ntypes - 1)
                            id[n] = n + 1
                            n += 1
        return n


@pk.functor(x=pk.ViewTypeInfo(layout=pk.LayoutRight))
class init_system:
    def __init__(self, s: System, ix_start: int, ix_end: int, iy_start: int, iy_end: int, iz_start: int, iz_end: int, lattice_constant: float, basis: pk.View2D[pk.double]):
        self.ix_start: int = ix_start
        self.ix_end: int = ix_end
        self.iy_start: int = iy_start
        self.iy_end: int = iy_end
        self.iz_start: int = iz_start
        self.iz_end: int = iz_end
        self.lattice_constant: float = lattice_constant
        self.basis: pk.View2D[pk.double] = basis

        self.x: pk.View2D[pk.double] = s.x
        self.type: pk.View1D[pk.int32] = s.type
        self.id: pk.View1D[pk.int32] = s.id

        self.sub_domain_hi_x: float = s.sub_domain_hi_x
        self.sub_domain_hi_y: float = s.sub_domain_hi_y
        self.sub_domain_hi_z: float = s.sub_domain_hi_z

        self.sub_domain_lo_x: float = s.sub_domain_lo_x
        self.sub_domain_lo_y: float = s.sub_domain_lo_y
        self.sub_domain_lo_z: float = s.sub_domain_lo_z

    @pk.workunit
    def get_n(self, iz: int, n: pk.Acc[int]):
        iz -= 1
        for iy in range(self.iy_start, self.iy_end + 1):
            for ix in range(self.ix_start, self.ix_end + 1):
                for k in range(4):
                    xtmp: float = (self.lattice_constant *
                                (1.0 * ix + self.basis[k][0]))
                    ytmp: float = (self.lattice_constant *
                                (1.0 * iy + self.basis[k][1]))
                    ztmp: float = (self.lattice_constant *
                                (1.0 * iz + self.basis[k][2]))

                    if (
                        xtmp >= self.sub_domain_lo_x
                        and ytmp >= self.sub_domain_lo_y
                        and ztmp >= self.sub_domain_lo_z
                        and xtmp < self.sub_domain_hi_x
                        and ytmp < self.sub_domain_hi_y
                        and ztmp < self.sub_domain_hi_z
                    ):
                        n += 1

    @pk.workunit
    def init_x(self, iz: int, n: pk.Acc[int]):
        iz -= 1
        for iy in range(self.iy_start, self.iy_end + 1):
            for ix in range(self.ix_start, self.ix_end + 1):
                for k in range(4):
                    xtmp: float = (self.lattice_constant *
                                (1.0 * ix + self.basis[k][0]))
                    ytmp: float = (self.lattice_constant *
                                (1.0 * iy + self.basis[k][1]))
                    ztmp: float = (self.lattice_constant *
                                (1.0 * iz + self.basis[k][2]))

                    if (
                        xtmp >= self.sub_domain_lo_x
                        and ytmp >= self.sub_domain_lo_y
                        and ztmp >= self.sub_domain_lo_z
                        and xtmp < self.sub_domain_hi_x
                        and ytmp < self.sub_domain_hi_y
                        and ztmp < self.sub_domain_hi_z
                    ):
                        self.x[n][0] = xtmp
                        self.x[n][1] = ytmp
                        self.x[n][2] = ztmp
                        self.type[n] = 0
                        self.id[n] = n + 1
                        n += 1

class Input:
    def __init__(self, p: System):
        self.system: System = p
        self.input_data = ItemizedFile()
        self.integrator_type: int = IntegratorType.INTEGRATOR_NVE.value

        self.nsteps: int = 0
        self.force_coeff_lines = pk.View([0], pk.int32)
        self.input_file: str = ""
        self.input_file_type: int = -1

        self.units: int = 0

        self.comm_type: int = CommType.COMM_SERIAL.value
        self.neighbor_type: int = NeighborType.NEIGH_2D.value
        self.neighbor_skin: float = 0.0
        self.force_type: int = 0
        self.force_iteration_type: int = ForceIterationType.FORCE_ITER_NEIGH_FULL.value
        self.force_line: int = 0
        self.force_cutoff: float = 0.0
        self.binning_type: int = BinningType.BINNING_KKSORT.value
        self.comm_exchange_rate: int = 20

        # set defaults

        self.thermo_rate: int = 0
        self.dumpbinary_rate: int = 0
        self.correctness_rate: int = 0
        self.dumpbinaryflag: bool = False
        self.correctnessflag: bool = False
        self.timestepflag: bool = False
        self.fill: bool = False

        self.lattice_style: int = 0
        self.lattice_constant: float = 0.0
        self.lattice_offset_x: float = 0.0
        self.lattice_offset_y: float = 0.0
        self.lattice_offset_z: float = 0.0
        self.lattice_nx: int = 0
        self.lattice_ny: int = 0
        self.lattice_nz: int = 0
        self.comm_newton: int = 0

        self.temperature_target: float = 0.0
        self.temperature_seed: int = 0

        self.dumpbinary_path: str = ""
        self.reference_path: str = ""
        self.correctness_file: str = ""

    def read_command_line_args(self) -> None:
        parser = argparse.ArgumentParser(
            description="ExaMiniMD 1.0 (Kokkos Reference Version)"
                        " (PyKokkos Implementation)")

        parser.add_argument("-il", "--inputlammps", type=str,
                            metavar="[FILE]",
                            help="Provide LAMMPS input file")
        parser.add_argument("--forceiteration", type=str,
                            metavar="[TYPE]",
                            help="Specify which iteration style to use for"
                                 " force calculations"
                                 " (CELL_FULL, NEIGH_FULL, NEIGH_HALF)")
        parser.add_argument("--commtype", type=str,
                            metavar="[TYPE]",
                            help="Specify Communication Routines"
                                 " implementation (MPI, SERIAL)")
        parser.add_argument("--dumpbinary", nargs=2,
                            metavar=("[N]", "[PATH]"),
                            help="Request that binary output files"
                                 " PATH/output* be generated every N steps"
                                 " (N = positive integer)"
                                 " (PATH = location of directory)")
        parser.add_argument("--correctness", nargs=3,
                            metavar=("[N]", "[PATH]", "[FILE]"),
                            help="Request that correctness check against"
                                 " files PATH/output* be performed every N"
                                 " steps, correctness data written to FILE"
                                 " (N = positive integer)"
                                 " (PATH = location of directory)")
        parser.add_argument("--neightype", type=str,
                            help="Specify Neighbor Routines implementation"
                                 " (2D, CSR, CSR_MAPCONSTR)")
        parser.add_argument("--fill", action="store_true",
                            help="Specify whether to use ViewType.fill() or"
                                 " to initialize with sequential for loop")

        args = parser.parse_args()

        if args.inputlammps:
            self.input_file = args.inputlammps
            self.input_file_type = InputFileType.INPUT_LAMMPS.value

        if args.forceiteration:
            if args.forceiteration == "CELL_FULL":
                self.force_iteration_type = ForceIterationType.FORCE_ITER_CELL_FULL.value
            elif args.forceiteration == "NEIGH_FULL":
                self.force_iteration_type = ForceIterationType.FORCE_ITER_NEIGH_FULL.value
            elif args.forceiteration == "NEIGH_HALF":
                self.force_iteration_type = ForceIterationType.FORCE_ITER_NEIGH_HALF.value

        if args.commtype:
            if args.commtype == "SERIAL":
                self.comm_type = CommType.COMM_SERIAL.value
            elif args.commtype == "MPI":
                self.comm_type = CommType.COMM_MPI.value

        if args.neightype:
            if args.neightype == "2D":
                self.neighbor_type = NeighborType.NEIGH_2D.value
            elif args.neightype == "CSR":
                self.neighbor_type = NeighborType.NEIGH_CSR.value
            elif args.neightype == "CSR_MAPCONSTR":
                self.neighbor_type = NeighborType.NEIGH_CSR_MAPCONSTR.value

        if args.dumpbinary:
            self.dumpbinary_rate = int(args.dumpbinary[0])
            self.dumpbinary_path = args.dumpbinary[1]
            self.dumpbinaryflag = True

        if args.correctness:
            self.correctness_rate = int(args.correctness[0])
            self.reference_path = args.correctness[1]
            self.correctness_file = args.correctness[2]
            self.correctnessflag = True

        # Added for pykokkos to use ViewType.fill() instead of for loops
        if args.fill:
            self.fill = True

    def read_file(self, filename: str = None) -> None:
        if filename is None:
            filename = self.input_file

        if self.input_file_type == InputFileType.INPUT_LAMMPS.value:
            self.read_lammps_file(filename)
            return

        if self.system.do_print:
            print("ERROR: Unknown input file type")

        sys.exit(1)

    def read_lammps_file(self, filename: str) -> None:
        self.input_data.allocate_words(100)

        with open(filename) as f:
            for line in f:
                self.input_data.add_line(line[:-1])

        if self.system.do_print:
            print("\n")
            print("#InputFile:")
            print("#=========================================================")

            self.input_data.print()

            print("#=========================================================")
            print("\n")

        for l in range(self.input_data.nlines):
            self.check_lammps_command(l)

    def check_lammps_command(self, line: int) -> None:
        known: bool = False

        if self.input_data.words[line][0] == "":
            known = True

        command: str = self.input_data.words[line][0]

        if "#" in command:
            known = True

        if command == "variable":
            if self.system.do_print:
                print("LAMMPS-Command:"
                      " 'variable' keyword is not supported in ExaMiniMD")

        if command == "units":
            unit: str = self.input_data.words[line][1]
            if unit == "metal":
                known = True
                self.units = Units.UNITS_METAL.value
                self.system.boltz = 8.617343e-5
                self.system.mvv2e = 1.0364269e-4
                self.system.dt = 0.001

            elif unit == "real":
                known = True
                self.units = Units.UNITS_REAL.value
                self.system.boltz = 0.0019872067
                self.system.mvv2e = 48.88821291 * 48.88821291

                if not self.timestepflag:
                    self.system.dt = 1.0

            elif unit == "lj":
                known = True
                self.units = Units.UNITS_LJ.value
                self.system.boltz = 1.0
                self.system.mvv2e = 1.0

                if not self.timestepflag:
                    self.system.dt = 0.005

            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'units' command only supports"
                          " 'real' and 'lj' in ExaMiniMD")

        if command == "atom_style":
            style: str = self.input_data.words[line][1]
            if style == "atomic":
                known = True
            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'atom_style' command only"
                          " supports 'atomic' in ExaMiniMD")

        if command == "lattice":
            lattice: str = self.input_data.words[line][1]
            if lattice == "sc":
                known = True
                self.lattice_style = LatticeType.LATTICE_SC.value
                self.lattice_constant = float(self.input_data.words[line][2])

            elif lattice == "fcc":
                known = True
                self.lattice_style = LatticeType.LATTICE_FCC.value
                self.lattice_constant = (
                    4.0 / float(self.input_data.words[line][2])) ** (1.0 / 3.0)

            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'lattice' command only supports"
                          " 'sc' and 'fcc' in ExaMiniMD")

            if self.input_data.words[line][3] == "origin":
                self.lattice_offset_x = float(self.input_data.words[line][4])
                self.lattice_offset_y = float(self.input_data.words[line][5])
                self.lattice_offset_z = float(self.input_data.words[line][6])

        if command == "region":
            region: str = self.input_data.words[line][2]
            if region == "block":
                known = True
                box: List[int] = [0] * 6
                box[0] = int(self.input_data.words[line][3])
                box[1] = int(self.input_data.words[line][4])
                box[2] = int(self.input_data.words[line][5])
                box[3] = int(self.input_data.words[line][6])
                box[4] = int(self.input_data.words[line][7])
                box[5] = int(self.input_data.words[line][8])

                if box[0] != 0 or box[2] != 0 or box[4] != 0:
                    if self.system.do_print:
                        print("Error: LAMMPS-Command: region only allows for"
                              " boxes with 0,0,0 offset")

                self.lattice_nx = box[1]
                self.lattice_ny = box[3]
                self.lattice_nz = box[5]

            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'region' command only supports"
                          " 'block' option in ExaMiniMD")

        if command == "create_box":
            known = True
            self.system.ntypes = int(self.input_data.words[line][1])
            self.system.mass = t_mass(self.system.ntypes)

        if command == "create_atoms":
            known = True

        if command == "mass":
            known = True
            mass_type: int = int(self.input_data.words[line][1]) - 1
            mass: float = float(self.input_data.words[line][2])
            self.system.mass[mass_type] = mass

        if command == "pair_style":
            pair_style: str = self.input_data.words[line][1]
            if pair_style == "lj/cut/idial":
                known = True
                self.force_type = ForceType.FORCE_LJ_IDIAL.value
                self.force_cutoff = float(self.input_data.words[line][2])
                self.force_line = line

            elif pair_style == "lj/cut":
                known = True
                self.force_type = ForceType.FORCE_LJ.value
                self.force_cutoff = float(self.input_data.words[line][2])
                self.force_line = line

            if pair_style == "snap":
                known = True
                self.force_type = ForceType.FORCE_SNAP.value
                self.force_cutoff = 4.73442
                self.force_line = line

                if self.system.do_print and not known:
                    print("LAMMPS-Command: 'pair_style' command only supports"
                          " 'lj/cut', 'lj/cut/idial', and 'snap' style"
                          " in ExaMiniMD")

        if command == "pair_coeff":
            known = True
            n_coeff_lines: int = self.force_coeff_lines.extent(0)
            self.force_coeff_lines.resize(0, n_coeff_lines + 1)
            self.force_coeff_lines[n_coeff_lines] = line
            n_coeff_lines += 1

        if command == "velocity":
            known = True
            if self.input_data.words[line][1] != "all":
                if self.system.do_print:
                    print("Error: LAMMPS-Command: 'velocity' command can only"
                          " be applied to 'all'")

            if self.input_data.words[line][2] != "create":
                if self.system.do_print:
                    print("Error: LAMMPS-Command: 'velocity' command can only"
                          " be used with option 'create'")

            self.temperature_target = float(self.input_data.words[line][3])
            self.temperature_seed = int(self.input_data.words[line][4])

        if command == "neighbor":
            known = True
            self.neighbor_skin = float(self.input_data.words[line][1])

        if command == "neigh_modify":
            known = True
            for i in range(1, self.input_data.words_per_line - 1):
                if self.input_data.words[line][i] == "every":
                    self.comm_exchange_rate = int(
                        self.input_data.words[line][i + 1])

        if command == "fix":
            if self.input_data.words[line][3] == "nve":
                known = True
                self.integrator_type = IntegratorType.INTEGRATOR_NVE.value

            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'fix' command only supports"
                          " 'nve' style in ExaMiniMD")

        if command == "run":
            known = True
            self.nsteps = int(self.input_data.words[line][1])

        if command == "thermo":
            known = True
            self.thermo_rate = int(self.input_data.words[line][1])

        if command == "timestep":
            known = True
            self.system.dt = float(self.input_data.words[line][1])
            self.timestepflag = True

        if command == "newton":
            known = True
            newton: str = self.input_data.words[line][1]
            if newton == "on":
                self.comm_newton = 1

            elif newton == "off":
                self.comm_newton = 0

            else:
                if self.system.do_print:
                    print("LAMMPS-Command: 'newton' must be followed by"
                          " 'on' or 'off'")

        if command == "":
            known = True

        if not known and self.system.do_print:
            print("ERROR: unknown keyword")
            self.input_data.print_line(line)

    def create_lattice(self, comm: Comm) -> None:
        s: System = copy.deepcopy(self.system)

        if self.lattice_style == LatticeType.LATTICE_SC.value:
            self.system.domain_x = self.lattice_constant * self.lattice_nx
            self.system.domain_y = self.lattice_constant * self.lattice_ny
            self.system.domain_z = self.lattice_constant * self.lattice_nz

            comm.create_domain_decomposition()
            s = copy.deepcopy(self.system)

            ix_start: int = math.floor(
                s.sub_domain_lo_x / s.domain_x * self.lattice_nx - 0.5)
            iy_start: int = math.floor(
                s.sub_domain_lo_y / s.domain_y * self.lattice_ny - 0.5)
            iz_start: int = math.floor(
                s.sub_domain_lo_z / s.domain_z * self.lattice_nz - 0.5)

            ix_end: int = math.floor(
                s.sub_domain_hi_x / s.domain_x * self.lattice_nx + 0.5)
            iy_end: int = math.floor(
                s.sub_domain_hi_y / s.domain_y * self.lattice_ny + 0.5)
            iz_end: int = math.floor(
                s.sub_domain_hi_z / s.domain_z * self.lattice_nz + 0.5)

            n: int = 0

            for iz in range(iz_start, iz_end + 1):
                ztmp: float = (self.lattice_constant *
                               (iz + self.lattice_offset_z))

                for iy in range(iy_start, iy_end + 1):
                    ytmp: float = (self.lattice_constant *
                                   (iy + self.lattice_offset_y))

                    for ix in range(ix_start, ix_end + 1):
                        xtmp: float = (self.lattice_constant *
                                       (ix + self.lattice_offset_x))

                        if (
                            xtmp >= s.sub_domain_lo_x
                            and ytmp >= s.sub_domain_lo_y
                            and ztmp >= s.sub_domain_lo_z
                            and xtmp < s.sub_domain_hi_x
                            and ytmp < s.sub_domain_hi_y
                            and ztmp < s.sub_domain_hi_z
                        ):
                            n += 1

            self.system.N_local = n
            self.system.N = n
            self.system.grow(n)

            s = copy.deepcopy(self.system)

            for iz in range(iz_start, iz_end + 1):
                ztmp: float = (self.lattice_constant *
                               (iz + self.lattice_offset_z))

                for iy in range(iy_start, iy_end + 1):
                    ytmp: float = (self.lattice_constant *
                                   (iy + self.lattice_offset_y))

                    for ix in range(ix_start, ix_end + 1):
                        xtmp: float = (self.lattice_constant *
                                       (ix + self.lattice_offset_x))

                        if (
                            xtmp >= s.sub_domain_lo_x
                            and ytmp >= s.sub_domain_lo_y
                            and ztmp >= s.sub_domain_lo_z
                            and xtmp < s.sub_domain_hi_x
                            and ytmp < s.sub_domain_hi_y
                            and ztmp < s.sub_domain_hi_z
                        ):
                            n += 1

            self.system.grow(n)
            s = copy.deepcopy(self.system)
            n = 0

            for iz in range(iz_start, iz_end + 1):
                ztmp: float = (self.lattice_constant *
                               (iz + self.lattice_offset_z))

                for iy in range(iy_start, iy_end + 1):
                    ytmp: float = (self.lattice_constant *
                                   (iy + self.lattice_offset_y))

                    for ix in range(ix_start, ix_end + 1):
                        xtmp: float = (self.lattice_constant *
                                       (ix + self.lattice_offset_x))

                        s.x[n][0] = xtmp
                        s.x[n][1] = ytmp
                        s.x[n][2] = ztmp
                        s.type[n] = random.randint(0, s.ntypes - 1)
                        s.id[n] = n + 1
                        n += 1

            comm.reduce_int(self.system.N, 1)

            N_local_offset: int = n
            comm.scan_int(N_local_offset, 1)
            for i in range(n):
                s.id[i] += N_local_offset - n

            if self.system.do_print:
                print(f"Atoms: {self.system.N} {self.system.N_local}")

        if self.lattice_style == LatticeType.LATTICE_FCC.value:
            self.system.domain_x = self.lattice_constant * self.lattice_nx
            self.system.domain_y = self.lattice_constant * self.lattice_ny
            self.system.domain_z = self.lattice_constant * self.lattice_nz

            comm.create_domain_decomposition()
            s = copy.deepcopy(self.system)

            basis: List[List[float]] = [[0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.0],
                                        [0.5, 0.0, 0.5],
                                        [0.0, 0.5, 0.5]]
            basis_view = pk.View([4, 3], pk.double)
            for i in range(4):
                basis_view[i][0] = basis[i][0]
                basis_view[i][1] = basis[i][1]
                basis_view[i][2] = basis[i][2]

            for i in range(4):
                basis_view[i][0] += self.lattice_offset_x
                basis_view[i][1] += self.lattice_offset_y
                basis_view[i][2] += self.lattice_offset_z

            print(f"{s.sub_domain_lo_x} {s.domain_x} {self.lattice_nx} - 0.5")
            ix_start: int = math.floor(
                s.sub_domain_lo_x / s.domain_x * self.lattice_nx - 0.5)
            iy_start: int = math.floor(
                s.sub_domain_lo_y / s.domain_y * self.lattice_ny - 0.5)
            iz_start: int = math.floor(
                s.sub_domain_lo_z / s.domain_z * self.lattice_nz - 0.5)

            ix_end: int = math.floor(
                s.sub_domain_hi_x / s.domain_x * self.lattice_nx + 0.5)
            iy_end: int = math.floor(
                s.sub_domain_hi_y / s.domain_y * self.lattice_ny + 0.5)
            iz_end: int = math.floor(
                s.sub_domain_hi_z / s.domain_z * self.lattice_nz + 0.5)


            init_s = init_system(s, ix_start, ix_end, iy_start, iy_end, iz_start, iz_end,
                                 self.lattice_constant, basis_view)
            n: int = pk.parallel_reduce("init_s", pk.RangePolicy(iz_start + 1, iz_end + 1), init_s.get_n)

            # n: int = calculate_n(ix_start, ix_end, iy_start, iy_end, iz_start, iz_end,
            #                      self.lattice_constant, np.array(basis),
            #                      s.sub_domain_lo_x, s.sub_domain_lo_y, s.sub_domain_lo_z,
            #                      s.sub_domain_hi_x, s.sub_domain_hi_y, s.sub_domain_hi_z)

            self.system.N_local = n
            self.system.N = n

            # Instead of calling it get_n twice, multiply by 2 (unlike c++ version)
            n *= 2
            self.system.grow(n)
            s = self.system

            global NUMBA_ENABLED
            if NUMBA_ENABLED:
                n: int = init_x(ix_start, ix_end, iy_start, iy_end, iz_start, iz_end,
                                self.lattice_constant, basis_view.data,
                                s.sub_domain_lo_x, s.sub_domain_lo_y, s.sub_domain_lo_z,
                                s.sub_domain_hi_x, s.sub_domain_hi_y, s.sub_domain_hi_z,
                                s.x.data, s.type.data, s.id.data, s.ntypes)
            else:
                n: int = pk.parallel_reduce("init_x", pk.RangePolicy(pk.Serial, iz_start + 1, iz_end + 2), init_s.init_x)

            N_local_offset: int = n
            comm.scan_int(N_local_offset, 1)
            to_add: int = N_local_offset - n
            s.id.data += to_add

            comm.reduce_int(self.system.N, 1)

            if self.system.do_print:
                print(f"Atoms: {self.system.N} {self.system.N_local}")

        s = self.system
        total_mass: float = 0.0
        total_momentum_x: float = 0.0
        total_momentum_y: float = 0.0
        total_momentum_z: float = 0.0

        ibase: int = self.temperature_seed
        ibase &= 0xffffffff
        ibase_bin = list(ibase.to_bytes(4, "little"))
        for i in range(4):
            if (ibase_bin[i] & (1 << 7)):
                ibase_bin[i] -= 1 << 8

        ibase &= 0xffffffff
        ibase_bin = list(ibase.to_bytes(4, "little"))
        for i in range(4):
            if (ibase_bin[i] & (1 << 7)):
                ibase_bin[i] -= 1 << 8

        hash_uint: int = 0
        for i in ibase_bin:
            hash_uint += i
            hash_uint &= 0xFFFFFFFF
            hash_uint += hash_uint << 10
            hash_uint &= 0xFFFFFFFF
            hash_uint ^= hash_uint >> 6
            hash_uint &= 0xFFFFFFFF

        if NUMBA_ENABLED:
            x_bytes = np.reshape(np.frombuffer(s.x.data.tobytes(), dtype=np.byte), (s.x.shape[0], s.x.shape[1], 8)).astype(int)
            total_mass, total_momentum_x, total_momentum_y, total_momentum_z = init_v(hash_uint, x_bytes, s.v.data, s.mass.data, s.type.data, self.system.N_local)
        else:
            rand = LAMMPS_RandomVelocityGeom()
            for i in range(self.system.N_local):
                rand.seed = 0
                x: List[float] = [s.x[i][0], s.x[i][1], s.x[i][2]]
                rand.reset(self.temperature_seed, x)

                mass_i: float = s.mass[int(s.type[i])]
                vx: float = rand.uniform() - 0.5
                vy: float = rand.uniform() - 0.5
                vz: float = rand.uniform() - 0.5

                mass_i_sqrt = math.sqrt(mass_i)
                s.v[i][0] = vx / mass_i_sqrt
                s.v[i][1] = vy / mass_i_sqrt
                s.v[i][2] = vz / mass_i_sqrt

                total_mass += mass_i
                total_momentum_x += mass_i * s.v[i][0]
                total_momentum_y += mass_i * s.v[i][1]
                total_momentum_z += mass_i * s.v[i][2]

        s.q.fill(0.0)

        comm.reduce_float(total_momentum_x, 1)
        comm.reduce_float(total_momentum_y, 1)
        comm.reduce_float(total_momentum_z, 1)
        comm.reduce_float(total_mass, 1)

        system_vx: float = total_momentum_x / total_mass
        system_vy: float = total_momentum_y / total_mass
        system_vz: float = total_momentum_z / total_mass

        system_v = np.array([system_vx, system_vy, system_vz])
        s.v.data -= system_v

        temp = Temperature(comm)
        T: float = temp.compute(self.system)

        T_init_scale: float = math.sqrt(self.temperature_target / T)

        s.v.data *= T_init_scale
