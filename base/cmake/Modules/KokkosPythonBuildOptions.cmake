
#----------------------------------------------------------------------------------------#
#   build-options interface library
#----------------------------------------------------------------------------------------#

INCLUDE_GUARD(GLOBAL)

INCLUDE(KokkosPythonUtilities)

# add extra build properties to this target
ADD_LIBRARY(libpykokkos-build-options INTERFACE)
ADD_LIBRARY(libpykokkos::build-options ALIAS libpykokkos-build-options)

TARGET_INCLUDE_DIRECTORIES(libpykokkos-build-options INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

if(WIN32)
  PYKOKKOS_TARGET_FLAG(
      libpykokkos-build-options
      CHECK_FLAGS
      MODE        INTERFACE
      FLAGS       /bigobj
      LANGUAGES   CXX)
endif()

PYKOKKOS_TARGET_FLAG(
    libpykokkos-build-options
    CHECK_FLAGS
    MODE        INTERFACE
    FLAGS       -W -Wall -Wextra -Wno-deprecated-declarations -Wno-attributes -fvisibility=default
    LANGUAGES   CXX)

IF(NOT Kokkos_InterOp_Header)
    CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/cmake/Templates/KokkosExp_InterOp.hpp
        ${PROJECT_BINARY_DIR}/include/KokkosExp_InterOp.hpp COPYONLY)
    CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/cmake/Templates/KokkosExp_InterOp.hpp
        ${PROJECT_BINARY_DIR}/examples/KokkosExp_InterOp.hpp COPYONLY)
ENDIF()

TRY_COMPILE(ENABLE_DEMANGLE
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/cmake/Templates/demangle.cpp
    OUTPUT_VARIABLE ENABLE_DEMANGLE_OUTPUT)

IF(ENABLE_DEMANGLE)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_DEMANGLE)
ENDIF()

IF(ENABLE_EXPERIMENTAL)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_EXPERIMENTAL)
ENDIF()

IF(ENABLE_LAYOUTS)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_LAYOUTS)
ENDIF()

IF(ENABLE_MEMORY_TRAITS)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_MEMORY_TRAITS)
ENDIF()

TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_VIEW_RANKS=${ENABLE_VIEW_RANKS})

PYKOKKOS_TARGET_FLAG(
    libpykokkos-build-options
    CHECK_FLAGS
    MODE        INTERFACE
    FLAGS       -Werror
    LANGUAGES   CXX
    OPTION_NAME ENABLE_WERROR)

PYKOKKOS_TARGET_FLAG(
    libpykokkos-build-options
    CHECK_FLAGS
    BREAK_ON_SUCCESS
    MODE        INTERFACE
    FLAGS       -flto=thin -flto
    LANGUAGES   CXX
    OPTION_NAME ENABLE_THIN_LTO)

IF(CMAKE_CXX_COMPILER_IS_GNU AND NOT CMAKE_CXX_COMPILER_IS_NVCC_WRAPPER)
    PYKOKKOS_TARGET_FLAG(
        libpykokkos-build-options
        CHECK_FLAGS
        MODE        INTERFACE
        FLAGS       -fno-fat-lto-objects
        LANGUAGES   CXX
        OPTION_NAME ENABLE_THIN_LTO)
ENDIF()

IF(ENABLE_TIMING AND CMAKE_CXX_COMPILER_IS_NVCC_WRAPPER)
    PYKOKKOS_TARGET_FLAG(
        libpykokkos-build-options
        MODE        INTERFACE
        FLAGS       -time=${PROJECT_BINARY_DIR}/nvcc-compile-time.csv
        LANGUAGES   CXX)
ELSE()
    PYKOKKOS_TARGET_FLAG(
        libpykokkos-build-options
        CHECK_FLAGS
        BREAK_ON_SUCCESS
        MODE        INTERFACE
        FLAGS       -ftime-trace -ftime-report
        LANGUAGES   CXX
        OPTION_NAME ENABLE_TIMING)
ENDIF()
