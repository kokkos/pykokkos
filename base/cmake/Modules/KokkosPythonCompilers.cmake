# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Compilers
#
##########################################################################################
#
#   sets (cached):
#
#       CMAKE_C_COMPILER_IS_<TYPE>
#       CMAKE_CXX_COMPILER_IS_<TYPE>
#
#   where TYPE is:
#       - GNU
#       - CLANG
#       - INTEL
#       - INTEL_ICC
#       - INTEL_ICPC
#       - PGI
#       - XLC
#       - HP_ACC
#       - MIPS
#       - MSVC
#

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

include(CMakeParseArguments)

#-----------------------------------------------------------------------
# Save a set of variables with the given prefix
#-----------------------------------------------------------------------
MACRO(PYKOKKOS_SAVE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(SAVE
        ""                # options
        "CONDITION"       # single value args
        "VARIABLES"       # multiple value args
        ${ARGN})
    if(DEFINED SAVE_CONDITION AND NOT "${SAVE_CONDITION}" STREQUAL "")
        if(${SAVE_CONDITION})
            foreach(_VAR ${ARGN})
                if(DEFINED ${_VAR})
                    set(${_PREFIX}_${_VAR} ${${_VAR}})
                else()
                    message(AUTHOR_WARNING "${_VAR} is not defined")
                endif()
            endforeach()
        endif()
    endif()
    unset(SAVE_CONDITION)
    unset(SAVE_VARIABLES)
ENDMACRO()

#-----------------------------------------------------------------------
# Restore a set of variables with the given prefix
#-----------------------------------------------------------------------
MACRO(PYKOKKOS_RESTORE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(RESTORE
        ""                # options
        "CONDITION"       # single value args
        "VARIABLES"       # multiple value args
        ${ARGN})
    if(DEFINED RESTORE_CONDITION AND NOT "${RESTORE_CONDITION}" STREQUAL "")
        if(${RESTORE_CONDITION})
            foreach(_VAR ${ARGN})
                if(DEFINED ${_PREFIX}_${_VAR})
                    set(${_VAR} ${${_PREFIX}_${_VAR}})
                    unset(${_PREFIX}_${_VAR})
                else()
                    message(AUTHOR_WARNING "${_PREFIX}_${_VAR} is not defined")
                endif()
            endforeach()
        endif()
    endif()
    unset(RESTORE_CONDITION)
    unset(RESTORE_VARIABLES)
ENDMACRO()

#----------------------------------------------------------------------------------------#
# call before running check_{c,cxx}_compiler_flag
#----------------------------------------------------------------------------------------#
macro(PYKOKKOS_BEGIN_FLAG_CHECK)
    if(NOT DEFINED CMAKE_REQUIRED_QUIET)
        set(CMAKE_REQUIRED_QUIET OFF)
    endif()
    pykokkos_save_variables(FLAG_CHECK
        VARIABLES CMAKE_REQUIRED_QUIET)
    set(CMAKE_REQUIRED_QUIET ON)
endmacro()

#----------------------------------------------------------------------------------------#
# call after running check_{c,cxx}_compiler_flag
#----------------------------------------------------------------------------------------#
macro(PYKOKKOS_END_FLAG_CHECK)
    pykokkos_restore_variables(FLAG_CHECK
        VARIABLES CMAKE_REQUIRED_QUIET)
endmacro()

#----------------------------------------------------------------------------------------#
# check flag
#----------------------------------------------------------------------------------------#
function(PYKOKKOS_TARGET_FLAG _TARG_TARGET)
    cmake_parse_arguments(_TARG "CHECK_FLAGS;BREAK_ON_SUCCESS" "MODE;OPTION_NAME" "FLAGS;LANGUAGES" ${ARGN})

    if(NOT _TARG_MODE)
        set(_TARG_MODE INTERFACE)
    endif()

    get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if(NOT _TARG_LANGUAGES)
        get_property(_TARG_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    endif()

    string(TOLOWER "_${_TARG_TARGET}" _LTARG)

    foreach(_LANG ${_TARG_LANGUAGES})
        if(NOT "${_TARG_OPTION_NAME}" STREQUAL "" AND DEFINED "${_TARG_OPTION_NAME}" AND NOT ${_TARG_OPTION_NAME})
            break()
        endif()

        foreach(_FLAG ${_TARG_FLAGS})
            if(NOT _TARG_CHECK_FLAGS)
                if(NOT ENABLE_QUIET)
                    message(STATUS "[${_LANG}] Enabling flag ${_FLAG} - Always")
                endif()
                target_compile_options(${_TARG_TARGET} ${_TARG_MODE} $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                continue()
            endif()

            if(NOT ENABLE_QUIET)
                message(STATUS "[${_LANG}] Checking flag ${_FLAG}")
            endif()

            if("${_LANG}" STREQUAL "C")
                string(REGEX REPLACE "^/" "c${_LTARG}_" FLAG_NAME "${_FLAG}")
                string(REGEX REPLACE "^-" "c${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
                pykokkos_begin_flag_check()
                check_c_compiler_flag("-Werror" c_werror)
                if(c_werror)
                    check_c_compiler_flag("${_FLAG} -Werror" ${FLAG_NAME})
                else()
                    check_c_compiler_flag("${_FLAG}" ${FLAG_NAME})
                endif()
                pykokkos_end_flag_check()
                if(${FLAG_NAME})
                    target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                        $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                else()
                    # disable the option
                    if(_TARG_OPTION_NAME)
                        set(${_TARG_OPTION_NAME} OFF)
                    endif()
                endif()
            elseif("${_LANG}" STREQUAL "CXX")
                string(REGEX REPLACE "^/" "cxx${_LTARG}_" FLAG_NAME "${_FLAG}")
                string(REGEX REPLACE "^-" "cxx${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
                pykokkos_begin_flag_check()
                check_cxx_compiler_flag("-Werror" cxx_werror)
                if(cxx_werror)
                    check_cxx_compiler_flag("${_FLAG} -Werror" ${FLAG_NAME})
                else()
                    check_cxx_compiler_flag("${_FLAG}" ${FLAG_NAME})
                endif()
                pykokkos_end_flag_check()
                if(${FLAG_NAME})
                    target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                        $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                    if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
                        target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_FLAG}>)
                    elseif(CMAKE_CUDA_COMPILER_IS_CLANG)
                        target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                            $<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>)
                    endif()
                else()
                    # disable the option
                    if(_TARG_OPTION_NAME)
                        set(${_TARG_OPTION_NAME} OFF)
                    endif()
                endif()
            endif()

            if(NOT ENABLE_QUIET)
                if(${FLAG_NAME})
                    message(STATUS "[${_LANG}] Checking flag ${_FLAG} - Success")
                else()
                    message(STATUS "[${_LANG}] Checking flag ${_FLAG} - Failed")
                endif()
            endif()

            # if successful, exit out of loop
            if(_TARG_BREAK_ON_SUCCESS AND ${FLAG_NAME})
                break()
            endif()
        endforeach()
    endforeach()
endfunction()

#----------------------------------------------------------------------------------------#
# add compiler definition
#----------------------------------------------------------------------------------------#
function(PYKOKKOS_TARGET_COMPILE_DEFINITIONS _TARG _VIS)
    foreach(_DEF ${ARGN})
        target_compile_definitions(${_TARG} ${_VIS}
            $<$<COMPILE_LANGUAGE:CXX>:${_DEF}>)
        if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
            target_compile_definitions(${_TARG} ${_VIS}
                $<$<COMPILE_LANGUAGE:CUDA>:${_DEF}>)
        elseif(CMAKE_CUDA_COMPILER_IS_CLANG)
            target_compile_definitions(${_TARG} ${_VIS}
                $<$<COMPILE_LANGUAGE:CUDA>:${_DEF}>)
        endif()
    endforeach()
endfunction()

#----------------------------------------------------------------------------------------#
# determine compiler types for each language
#----------------------------------------------------------------------------------------#
get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(LANG C CXX CUDA)

    if(NOT DEFINED CMAKE_${LANG}_COMPILER)
        set(CMAKE_${LANG}_COMPILER "")
    endif()

    if(NOT DEFINED CMAKE_${LANG}_COMPILER_ID)
        set(CMAKE_${LANG}_COMPILER_ID "")
    endif()

    function(PYKOKKOS_SET_COMPILER_VAR VAR _BOOL)
        set(CMAKE_${LANG}_COMPILER_IS_${VAR} ${_BOOL} CACHE INTERNAL
            "CMake ${LANG} compiler identification (${VAR})" FORCE)
        mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
    endfunction()

    if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
        OR
       ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

        # GNU compiler
        PYKOKKOS_SET_COMPILER_VAR(       GNU                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

        # Intel icc compiler
        PYKOKKOS_SET_COMPILER_VAR(       INTEL               1)
        PYKOKKOS_SET_COMPILER_VAR(       INTEL_ICC           1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

        # Intel icpc compiler
        PYKOKKOS_SET_COMPILER_VAR(       INTEL               1)
        PYKOKKOS_SET_COMPILER_VAR(       INTEL_ICPC          1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "AppleClang")

        # Clang/LLVM compiler
        PYKOKKOS_SET_COMPILER_VAR(       CLANG               1)
        PYKOKKOS_SET_COMPILER_VAR( APPLE_CLANG               1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang")

        # Clang/LLVM compiler
        PYKOKKOS_SET_COMPILER_VAR(       CLANG               1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

        # PGI compiler
        PYKOKKOS_SET_COMPILER_VAR(       PGI                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

        # IBM xlC compiler
        PYKOKKOS_SET_COMPILER_VAR(       XLC                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

        # HP aC++ compiler
        PYKOKKOS_SET_COMPILER_VAR(       HP_ACC              1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "CC" AND
           CMAKE_SYSTEM_NAME MATCHES "IRIX" AND UNIX)

        # IRIX MIPSpro CC Compiler
        PYKOKKOS_SET_COMPILER_VAR(       MIPS                1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

        PYKOKKOS_SET_COMPILER_VAR(       INTEL               1)

        set(CTYPE ICC)
        if("${LANG}" STREQUAL "CXX")
            set(CTYPE ICPC)
        endif()

        PYKOKKOS_SET_COMPILER_VAR(       INTEL_${CTYPE}      1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "MSVC")

        # Windows Visual Studio compiler
        PYKOKKOS_SET_COMPILER_VAR(       MSVC                1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "NVIDIA")

        PYKOKKOS_SET_COMPILER_VAR(       NVIDIA              1)

    endif()

    if("${LANG}" STREQUAL "CXX" AND
       (CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper"
        OR
        "CUDA" IN_LIST Kokkos_DEVICES
        OR
        (DEFINED Kokkos_NVCC_WRAPPER AND EXISTS "${Kokkos_NVCC_WRAPPER}"
         AND
         DEFINED Kokkos_COMPILE_LAUNCHER AND EXISTS "${Kokkos_COMPILE_LAUNCHER}")))

        PYKOKKOS_SET_COMPILER_VAR(       NVCC_WRAPPER        1)

    endif()

    # set other to no
    foreach(TYPE GNU INTEL INTEL_ICC INTEL_ICPC APPLE_CLANG CLANG PGI XLC HP_ACC MIPS MSVC NVIDIA NVCC_WRAPPER)
        if(NOT DEFINED CMAKE_${LANG}_COMPILER_IS_${TYPE})
            PYKOKKOS_SET_COMPILER_VAR(${TYPE} 0)
        endif()
    endforeach()

endforeach()
