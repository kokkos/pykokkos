#
#   Configures CMake Options and Build Settings
#
INCLUDE(KokkosPythonUtilities)

# backwards compat
IF(NOT DEFINED ENABLE_EXAMPLES)
    SET(BUILD_EXAMPLES OFF CACHE BOOL "(deprecated) Use ENABLE_EXAMPLES")
    MARK_AS_ADVANCED(BUILD_EXAMPLES)
ELSE()
    SET(BUILD_EXAMPLES ${ENABLE_EXAMPLES})
ENDIF()

# default UNITY_BUILD to ON except when compiling CUDA
SET(_UNITY_BUILD ON)
IF("CUDA" IN_LIST Kokkos_DEVICES OR ENABLE_CUDA OR Kokkos_ENABLE_CUDA)
    SET(_UNITY_BUILD OFF)
ENDIF()

SET(_ENABLE_MEM_DEFAULT ON)
SET(_ENABLE_LAY_DEFAULT ON)
# unless ENABLE_LAYOUTS or ENABLE_MEMORY_TRAITS were set
# or, NVCC is really, really slow so never default memory traits to ON
IF("CUDA" IN_LIST Kokkos_DEVICES OR ENABLE_CUDA OR Kokkos_ENABLE_CUDA)
    # one or both of these will be ignored bc of existing cache values
    SET(_ENABLE_MEM_DEFAULT OFF)
    SET(_ENABLE_LAY_DEFAULT ON)
ENDIF()
SET(_VIEW_RANK_MSG "Set this value to the max number of ranks needed for Kokkos::View<...>. E.g. value of 4 means Kokkos::View<int*****, Kokkos::HostSpace> cannot be returned to python")

ADD_FEATURE(CMAKE_BUILD_TYPE "Build type")
ADD_FEATURE(CMAKE_INSTALL_PREFIX "Installation prefix")
ADD_FEATURE(CMAKE_CXX_FLAGS "C++ compiler flags")
ADD_FEATURE(Kokkos_CXX_STANDARD "Kokkos C++ Standard")
ADD_FEATURE(Kokkos_DIR "Kokkos installation")

ADD_OPTION(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Install with rpath to linked libraries" ON)
ADD_OPTION(ENABLE_INTERNAL_PYBIND11 "Build with pybind11 submodule" ON)
ADD_OPTION(ENABLE_INTERNAL_KOKKOS "Build with kokkos submodule" ${_INTERNAL_KOKKOS})
ADD_OPTION(ENABLE_WERROR "Build with -Werror" OFF)
ADD_OPTION(ENABLE_TIMING "Build with -ftime-trace (Clang) or -ftime-report (GNU) or -time (NVCC)" OFF)
ADD_OPTION(ENABLE_QUIET "Suppress non-essential information" OFF)
ADD_OPTION(ENABLE_EXAMPLES "Build the examples" ${BUILD_EXAMPLES})
ADD_OPTION(ENABLE_EXPERIMENTAL "Build the experimental code" OFF)
ADD_OPTION(ENABLE_THIN_LTO "Pass THIN_LTO to pybind11_add_module instead of NO_EXTRAS" OFF)
# these affect which Kokkos::View and Kokkos::DynRankView templates that are instantiated
ADD_OPTION(ENABLE_LAYOUTS "Build support for layouts (long NVCC compile times)"
    ${_ENABLE_LAY_DEFAULT})
ADD_OPTION(ENABLE_MEMORY_TRAITS "Build support for memory traits (long NVCC compile times)"
    ${_ENABLE_MEM_DEFAULT})
ADD_OPTION(ENABLE_PRECOMPILED_HEADERS "Enable precompiling kokkos and pybind11 headers" OFF)

IF(PYKOKKOS_BASE_MAIN_PROJECT)
    ADD_OPTION(CMAKE_UNITY_BUILD "Enable unity build" ${_UNITY_BUILD})

    IF(CMAKE_UNITY_BUILD)
        SET(CMAKE_UNITY_BUILD_BATCH_SIZE 8 CACHE STRING "Unity build batch size")
        ADD_FEATURE(CMAKE_UNITY_BUILD_BATCH_SIZE "Unity build batch size")
    ENDIF()
ENDIF()

SET(ENABLE_VIEW_RANKS "4" CACHE STRING "${_VIEW_RANK_MSG}")
IF(ENABLE_VIEW_RANKS LESS 0 OR ENABLE_VIEW_RANKS GREATER 7)
    MESSAGE(FATAL_ERROR "ENABLE_VIEW_RANKS (=${ENABLE_VIEW_RANKS}) must be in range: [0, 7]")
ENDIF()
ADD_FEATURE(ENABLE_VIEW_RANKS "${_VIEW_RANK_MSG}")

IF(Kokkos_DIR)
    STRING(REGEX REPLACE "/lib(|..)/cmake/.*" "" Kokkos_INSTALL_DIR "${Kokkos_DIR}")
    IF("${Kokkos_INSTALL_DIR}" STREQUAL "${Kokkos_DIR}")
        GET_FILENAME_COMPONENT(Kokkos_INSTALL_DIR "${Kokkos_DIR}" PATH)
        GET_FILENAME_COMPONENT(Kokkos_INSTALL_DIR "${Kokkos_INSTALL_DIR}" PATH)
        GET_FILENAME_COMPONENT(Kokkos_INSTALL_DIR "${Kokkos_INSTALL_DIR}" PATH)
    ENDIF()

    IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        MESSAGE(STATUS "Setting CMAKE_INSTALL_PREFIX to install prefix for Kokkos: ${Kokkos_INSTALL_DIR}")
        SET(CMAKE_INSTALL_PREFIX "${Kokkos_INSTALL_DIR}" CACHE PATH "Installation directory" FORCE)
    ENDIF()
ENDIF()

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Add compile flag to hide symbols of inline functions")
set(CMAKE_C_VISIBILITY_PRESET "default" CACHE STRING "Default visibility")
set(CMAKE_CXX_VISIBILITY_PRESET "default" CACHE STRING "Default visibility")

# pybind11 has not migrated to CMAKE_CXX_STANDARD and neither has kokkos
SET(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ language standard")
SET(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require standard")
SET(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Extensions")
IF(NOT Kokkos_CXX_STANDARD)
    SET(Kokkos_CXX_STANDARD ${CMAKE_CXX_STANDARD})
ENDIF()
UNSET(PYBIND11_CPP_STANDARD CACHE)

# Stops lookup as soon as a version satisfying version constraints is found.
SET(Python3_FIND_STRATEGY "LOCATION" CACHE STRING "Stops lookup as soon as a version satisfying version constraints is found")

# virtual environment is used before any other standard paths to look-up for the interpreter
SET(Python3_FIND_VIRTUALENV "FIRST" CACHE STRING "Virtual environment is used before any other standard paths")
SET_PROPERTY(CACHE Python3_FIND_VIRTUALENV PROPERTY STRINGS "FIRST;LAST;NEVER")

IF(APPLE)
    SET(Python3_FIND_FRAMEWORK "LAST" CACHE STRING "Order of preference between Apple-style and unix-style package components")
    SET_PROPERTY(CACHE Python3_FIND_FRAMEWORK PROPERTY STRINGS "FIRST;LAST;NEVER")
ENDIF()

# NOTE: PyPy does not support embedding the interpreter
SET(Python3_FIND_IMPLEMENTATIONS "CPython;PyPy" CACHE STRING "Different implementations which will be searched.")
SET_PROPERTY(CACHE Python3_FIND_IMPLEMENTATIONS PROPERTY STRINGS "CPython;IronPython;PyPy")

# create cache entries
SET(Python3_ARTIFACTS_INTERACTIVE ON CACHE BOOL "Create CMake cache entries so that users can edit them interactively" FORCE)

IF(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    SET(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
ENDIF()

OPTION(ENABLE_CTP "Enable compile-time-perf" OFF)
MARK_AS_ADVANCED(ENABLE_CTP)
IF(ENABLE_CTP)
    FIND_PACKAGE(compile-time-perf)
    IF(compile-time-perf_FOUND)
        enable_compile_time_perf(pykokkos-base-compile-time
            LINK
            ANALYZER_OPTIONS
                -s "${PROJECT_BINARY_DIR}/" "${PROJECT_SOURCE_DIR}/"
                -f "lang-all" "so" "a" "dylib" "dll"
                -i ".*(_tests)$" "^(ex_).*"
                -e "^(@rpath).*" "^(/usr)" "^(/opt)")
        SET(ENABLE_CTP ON)
    ELSE()
        SET(ENABLE_CTP OFF)
    ENDIF()
ENDIF()
