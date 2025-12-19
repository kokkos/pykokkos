
#----------------------------------------------------------------------------------------#
#   Kokkos submodule
#----------------------------------------------------------------------------------------#

INCLUDE_GUARD(GLOBAL)

INCLUDE(KokkosPythonUtilities)  # miscellaneous macros and functions

# if first time cmake is run and no external/internal preference is specified,
# try to find already installed kokkos unless (A) the Kokkos targets already
# exist or (B) pykokkos-base is being build via scikit-build. In the case
# of scikit-build, we want to prefer the internal kokkos because it is
# unlikely the user will see or kokkos which kokkos is found
IF(NOT DEFINED ENABLE_INTERNAL_KOKKOS AND NOT TARGET Kokkos::kokkoscore AND NOT SKBUILD)
    FIND_PACKAGE(Kokkos)
    # set the default cache value
    IF(Kokkos_FOUND)
        SET(_INTERNAL_KOKKOS OFF)
        # force using same compiler as kokkos
        kokkos_compilation(GLOBAL)
    ELSE()
        SET(_INTERNAL_KOKKOS ON)
    ENDIF()
ELSEIF(TARGET Kokkos::kokkoscore)
    SET(_INTERNAL_KOKKOS OFF)
ELSEIF(NOT DEFINED ENABLE_INTERNAL_KOKKOS AND SKBUILD)
    set(_INTERNAL_KOKKOS ON)
ELSE()
    # make sure ADD_OPTION in KokkosPythonOptions has a value
    SET(_INTERNAL_KOKKOS ${ENABLE_INTERNAL_KOKKOS})
ENDIF()

# force an error
IF(NOT _INTERNAL_KOKKOS AND NOT TARGET Kokkos::kokkoscore)
    FIND_PACKAGE(Kokkos REQUIRED COMPONENTS launch_compiler)

    kokkos_compilation(GLOBAL)

    IF(NOT Kokkos_INCLUDE_DIR)
        GET_TARGET_PROPERTY(Kokkos_INCLUDE_DIR Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
    ENDIF()

    FIND_FILE(Kokkos_InterOp_Header
        NO_DEFAULT_PATH
        NAMES           Kokkos_InterOp.hpp KokkosExp_InterOp.hpp
        PATHS           ${Kokkos_INCLUDE_DIR} ${Kokkos_ROOT}
        HINTS           ${Kokkos_INCLUDE_DIR} ${Kokkos_ROOT}
        DOC             "Path to Kokkos InterOp header"
        PATH_SUFFIXES   include ../../../include)

    ADD_FEATURE(Kokkos_CXX_COMPILER "Compiler used to build Kokkos")
    ADD_FEATURE(Kokkos_CXX_COMPILER_ID "Compiler ID used to build Kokkos")
ELSEIF(TARGET Kokkos::kokkoscore)

    IF(NOT Kokkos_INCLUDE_DIR)
        GET_TARGET_PROPERTY(Kokkos_INCLUDE_DIR Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
    ENDIF()

    FIND_FILE(Kokkos_InterOp_Header
        NO_DEFAULT_PATH
        NAMES           Kokkos_InterOp.hpp KokkosExp_InterOp.hpp
        PATHS           ${Kokkos_INCLUDE_DIR} ${Kokkos_ROOT}
        HINTS           ${Kokkos_INCLUDE_DIR} ${Kokkos_ROOT}
        DOC             "Path to Kokkos InterOp header"
        PATH_SUFFIXES   include ../../../include)

    ADD_FEATURE(Kokkos_CXX_COMPILER "Compiler used to build Kokkos")
    ADD_FEATURE(Kokkos_CXX_COMPILER_ID "Compiler ID used to build Kokkos")
ELSE()
    FIND_FILE(Kokkos_InterOp_Header
        NO_DEFAULT_PATH
        NAMES           Kokkos_InterOp.hpp KokkosExp_InterOp.hpp
        PATHS           ${PROJECT_SOURCE_DIR}/external/kokkos/core/src
        HINTS           ${PROJECT_SOURCE_DIR}/external/kokkos/core/src
        DOC             "Path to Kokkos InterOp header")
ENDIF()

#
IF(_INTERNAL_KOKKOS)

    # try to find some packages quietly in order to set some defaults
    SET(OpenMP_FOUND OFF)
    SET(Threads_FOUND OFF)
    SET(CUDA_FOUND OFF)

    IF(NOT Kokkos_ENABLE_THREADS)
        FIND_PACKAGE(OpenMP QUIET)
    ENDIF()

    IF(NOT DEFINED Kokkos_ENABLE_THREADS AND NOT OpenMP_FOUND)
        FIND_PACKAGE(Threads QUIET)
    ENDIF()

    IF(NOT DEFINED Kokkos_ENABLE_CUDA)
        FIND_PACKAGE(CUDA QUIET)
    ENDIF()

    ADD_OPTION(ENABLE_SERIAL "Enable Serial backend when building Kokkos submodule" ON)
    ADD_OPTION(ENABLE_OPENMP "Enable OpenMP when building Kokkos submodule" ${OpenMP_FOUND})
    ADD_OPTION(ENABLE_THREADS "Enable Pthreads when building Kokkos submodule" ${Threads_FOUND})
    ADD_OPTION(ENABLE_CUDA "Enable CUDA when building Kokkos submodule" ${CUDA_FOUND})

    # if OpenMP defaulted to ON but Kokkos_ENABLE_THREADS was explicitly set,
    # disable OpenMP defaulting to ON
    IF(ENABLE_OPENMP AND Kokkos_ENABLE_THREADS)
        SET(ENABLE_OPENMP OFF)
        SET(Kokkos_ENABLE_OPENMP OFF)
    ENDIF()

    # always disable pthread backend since pthreads are not supported on Windows
    IF(WIN32)
        SET(ENABLE_THREADS OFF)
        SET(Kokkos_ENABLE_THREADS OFF)
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_SERIAL)
        SET(ENABLE_SERIAL ${Kokkos_ENABLE_SERIAL})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_OPENMP)
        SET(ENABLE_OPENMP ${Kokkos_ENABLE_OPENMP})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_THREADS)
        SET(ENABLE_THREADS ${Kokkos_ENABLE_THREADS})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_CUDA)
        SET(ENABLE_CUDA ${Kokkos_ENABLE_CUDA})
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_SERIAL)
        ADD_OPTION(Kokkos_ENABLE_SERIAL "Build Kokkos submodule with serial support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_OPENMP)
        ADD_OPTION(Kokkos_ENABLE_OPENMP "Build Kokkos submodule with OpenMP support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_THREADS)
        ADD_OPTION(Kokkos_ENABLE_THREADS "Build Kokkos submodule with Pthread support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_CUDA)
        ADD_OPTION(Kokkos_ENABLE_CUDA "Build Kokkos submodule with CUDA support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_UVM "Build Kokkos submodule with CUDA UVM support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_LAMBDA "Build Kokkos submodule with CUDA lambda support" ON)
    ENDIF()

    # Check if we should use submodule or FetchContent
    IF(EXISTS ${PROJECT_SOURCE_DIR}/external/kokkos/CMakeLists.txt)
        # Use git submodule
        ADD_SUBDIRECTORY(external)
        SET(Kokkos_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/kokkos/core/src)
    ELSE()
        # Use FetchContent to download Kokkos
        INCLUDE(FetchContent)
        MESSAGE(STATUS "Fetching Kokkos via FetchContent")
        FETCHCONTENT_DECLARE(
          Kokkos
          URL https://github.com/kokkos/kokkos/archive/refs/heads/release-candidate-4.7.01.zip
          URL_HASH SHA256=e256f111716259ef0cec0339ddf44d716b1f495e5514ca0806fcf80635f5b4cc
        )
        FETCHCONTENT_MAKEAVAILABLE(Kokkos)
        FETCHCONTENT_GETPROPERTIES(Kokkos SOURCE_DIR Kokkos_SOURCE_DIR)
        SET(Kokkos_INCLUDE_DIR ${Kokkos_SOURCE_DIR}/core/src)
    ENDIF()
ENDIF()
