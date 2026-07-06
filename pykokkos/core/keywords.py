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
    ScratchSizeLevel = "pk_scratch_size_level"
    ScratchSizeValue = "pk_scratch_size_value"
    ScratchSizeIsPerTeam = "pk_scratch_size_is_per_team"
    ScratchSizeLevel0Value = "pk_scratch_size_level_0_value"
    ScratchSizeLevel0IsPerTeam = "pk_scratch_size_level_0_is_per_team"
    ScratchSizeLevel0Enabled = "pk_scratch_size_level_0_enabled"
    ScratchSizeLevel1Value = "pk_scratch_size_level_1_value"
    ScratchSizeLevel1IsPerTeam = "pk_scratch_size_level_1_is_per_team"
    ScratchSizeLevel1Enabled = "pk_scratch_size_level_1_enabled"
