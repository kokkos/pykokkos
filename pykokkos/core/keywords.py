from enum import Enum

class Keywords(Enum):
    """
    A group of keywords reserved by PyKokkos
    """

    Instance = "pk_f"
    Accumulator = "pk_acc"
    LeagueSize = "pk_league_size"
    TeamSize = "pk_team_size"
    VectorLength = "pk_vector_length"
    ThreadsBegin = "pk_threads_begin"
    ThreadsEnd = "pk_threads_end"
    ArgMemSpace = "pk_arg_memspace"
    DefaultExecSpace = "pk_exec_space"
    DefaultExecSpaceInstance = "pk_exec_space_instance"
    KernelName = "pk_kernel_name"
    RealPrecision = "pk_real"
    RandPool = "pk_randpool"
    RandPoolState = "pk_rgen"
    RandPoolSeed = "pk_randpool_seed"
    RandPoolNumStates = "pk_randpool_num_states"
