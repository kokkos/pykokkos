
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
    SET(HIP_FOUND OFF)

    # Enable OpenMP if explicitly requested
    IF((DEFINED ENABLE_OPENMP AND ENABLE_OPENMP) OR (DEFINED Kokkos_ENABLE_OPENMP AND Kokkos_ENABLE_OPENMP))
        FIND_PACKAGE(OpenMP QUIET)
    # If Threads is explicitly requested, find it
    ELSEIF((DEFINED ENABLE_THREADS AND ENABLE_THREADS) OR (DEFINED Kokkos_ENABLE_THREADS AND Kokkos_ENABLE_THREADS))
        FIND_PACKAGE(Threads QUIET)
    ELSE()
        # if none requested - check what's available and prefer OpenMP
        FIND_PACKAGE(OpenMP QUIET)
        IF(NOT OpenMP_FOUND)
            FIND_PACKAGE(Threads QUIET)
        ENDIF()
    ENDIF()

    # Only search for CUDA if explicitly enabled
    IF((DEFINED ENABLE_CUDA AND ENABLE_CUDA) OR (DEFINED Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_CUDA))
        FIND_PACKAGE(CUDAToolkit QUIET)
        IF(CUDAToolkit_FOUND)
            FOREACH(INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${INCLUDE_DIR}")
                SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${INCLUDE_DIR}")
            ENDFOREACH()
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "Flags used by the C++ compiler" FORCE)
            SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "Flags used by the CUDA compiler" FORCE)
        ENDIF()

        ENABLE_LANGUAGE(CUDA)
        IF(CUDAToolkit_FOUND)
            INCLUDE_DIRECTORIES(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
            GET_FILENAME_COMPONENT(CUDA_TOOLKIT_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
            SET(Kokkos_CUDA_DIR "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA installation directory" FORCE)
        ENDIF()
        SET(CUDA_FOUND ON)
    ENDIF()

    # search for HIP if explicitly enabled
    IF((DEFINED ENABLE_HIP AND ENABLE_HIP) OR (DEFINED Kokkos_ENABLE_HIP AND Kokkos_ENABLE_HIP))
        INCLUDE(CheckLanguage)
        CHECK_LANGUAGE(HIP)
        
        IF(CMAKE_HIP_COMPILER)
            ENABLE_LANGUAGE(HIP)
            SET(HIP_FOUND ON)
        ENDIF()
    ENDIF()

    ADD_OPTION(ENABLE_SERIAL "Enable Serial backend when building Kokkos submodule" ON)
    # If OpenMP was explicitly requested, use that; otherwise use auto-detection result
    IF((DEFINED ENABLE_OPENMP AND ENABLE_OPENMP) OR (DEFINED Kokkos_ENABLE_OPENMP AND Kokkos_ENABLE_OPENMP))
        ADD_OPTION(ENABLE_OPENMP "Enable OpenMP when building Kokkos submodule" ON)
    ELSE()
        ADD_OPTION(ENABLE_OPENMP "Enable OpenMP when building Kokkos submodule" ${OpenMP_FOUND})
    ENDIF()
    # Only enable Threads if OpenMP is not available
    IF(OpenMP_FOUND)
        ADD_OPTION(ENABLE_THREADS "Enable Pthreads when building Kokkos submodule" OFF)
    ELSE()
        ADD_OPTION(ENABLE_THREADS "Enable Pthreads when building Kokkos submodule" ${Threads_FOUND})
    ENDIF()
    # CUDA must be explicitly enabled - default to OFF
    ADD_OPTION(ENABLE_CUDA "Enable CUDA when building Kokkos submodule" OFF)
    # HIP must be explicitly enabled - default to OFF
    ADD_OPTION(ENABLE_HIP "Enable HIP when building Kokkos submodule" OFF)

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

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_HIP)
        SET(ENABLE_HIP ${Kokkos_ENABLE_HIP})
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_SERIAL)
        ADD_OPTION(Kokkos_ENABLE_SERIAL "Build Kokkos submodule with serial support" ON)
    ENDIF()

    IF(ENABLE_OPENMP)
        ADD_OPTION(Kokkos_ENABLE_OPENMP "Build Kokkos submodule with OpenMP support" ON)
    ENDIF()

    IF(ENABLE_THREADS)
        ADD_OPTION(Kokkos_ENABLE_THREADS "Build Kokkos submodule with Pthread support" ON)
    ENDIF()

    IF(ENABLE_CUDA)
        ADD_OPTION(Kokkos_ENABLE_CUDA "Build Kokkos submodule with CUDA support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_UVM "Build Kokkos submodule with CUDA UVM support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_LAMBDA "Build Kokkos submodule with CUDA lambda support" ON)
    ENDIF()

    IF(ENABLE_HIP)
        ADD_OPTION(Kokkos_ENABLE_HIP "Build Kokkos submodule with HIP support" ON)
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
          URL https://github.com/kokkos/kokkos/releases/download/4.7.01/kokkos-4.7.01.zip
          URL_HASH SHA256=2b7c9964ace4245dec0b952932873d4b1235933dbb7d8d1d69e17b4368784503
        )
        FETCHCONTENT_MAKEAVAILABLE(Kokkos)
        FETCHCONTENT_GETPROPERTIES(Kokkos SOURCE_DIR Kokkos_SOURCE_DIR)
        SET(Kokkos_INCLUDE_DIR ${Kokkos_SOURCE_DIR}/core/src)
    ENDIF()
ENDIF()
