from comm import Comm
from comm_types.comm_serial import CommSerial
from input import Input
from system import System
from types_h import CommType


def comm_modules_instantiation(exa_input: Input, system: System) -> Comm:
    if exa_input.comm_type == CommType.COMM_SERIAL.value:
        return CommSerial(system, exa_input.force_cutoff + exa_input.neighbor_skin)
