#-----------------------------------------------------------------------
#
#   Finds the Packages
#
#-----------------------------------------------------------------------

INCLUDE(KokkosPythonUtilities)

# synchronize Python3_EXECUTABLE and PYTHON_EXECUTABLE
IF(Python3_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ELSEIF(PYTHON_EXECUTABLE)
    SET(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
ENDIF()

# cache the include directory if provided via old python find-package
IF(PYTHON_INCLUDE_DIR AND NOT Python3_INCLUDE_DIR)
    SET(Python3_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}" CACHE PATH "PYTHON_INCLUDE_DIR")
ENDIF()

# cache the library if provided via old python find-package
IF(PYTHON_LIBRARY AND NOT Python3_LIBRARY)
    SET(Python3_LIBRARY "${PYTHON_LIBRARY}" CACHE FILEPATH "PYTHON_LIBRARY")
ENDIF()

# basically just used to get Python3_SITEARCH for installation
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development)

FOREACH(_VAR MAJOR MINOR PATCH STRING)
    IF(Python3_VERSION_${_VAR})
        SET(PYTHON_VERSION_${_VAR} ${Python3_VERSION_${_VAR}})
    ENDIF()
ENDFOREACH()

IF(NOT PYTHON_VERSION_STRING)
    IF(PYTHON_VERSION_MAJOR AND PYTHON_VERSION_MINOR AND PYTHON_VERSION_PATCH)
        SET(PYTHON_VERSION_STRING "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_PATCH}" CACHE STRING "Python version" FORCE)
    ELSEIF(PYTHON_VERSION_MAJOR AND PYTHON_VERSION_MINOR)
        SET(PYTHON_VERSION_STRING "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" CACHE STRING "Python version" FORCE)
    ENDIF()
ENDIF()
SET(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_STRING}" CACHE STRING "Python version" FORCE)
ADD_FEATURE(PYBIND11_PYTHON_VERSION "Python version used by PyBind11")
ADD_FEATURE(PYTHON_VERSION_STRING "Python version found")
ADD_FEATURE(Python3_EXECUTABLE "Python interpreter")
ADD_FEATURE(Python3_INCLUDE_DIR "Python include directory")
ADD_FEATURE(Python3_LIBRARY "Python library")

# python binding library
IF(ENABLE_INTERNAL_PYBIND11 AND NOT TARGET pybind11::pybind11)
    CHECKOUT_GIT_SUBMODULE(
        RECURSIVE
        RELATIVE_PATH     external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE         CMakeLists.txt
        REPO_URL          https://github.com/pybind/pybind11.git
        REPO_BRANCH       master)
    ADD_SUBDIRECTORY(external/pybind11)
    SET(pybind11_INCLUDE_DIR external/pybind11/include)
ELSEIF(NOT TARGET pybind11::pybind11)
    FIND_PACKAGE(pybind11 REQUIRED)
ENDIF()

IF(TARGET pybind11 AND NOT TARGET pybind11::pybind11)
    ADD_LIBRARY(pybind11::pybind11 ALIAS pybind11)
ENDIF()

#-----------------------------------------------------------------------
#           precompiled headers
#-----------------------------------------------------------------------

ADD_LIBRARY(libpykokkos-precompiled-headers INTERFACE)
ADD_LIBRARY(libpykokkos::precompiled-headers ALIAS libpykokkos-precompiled-headers)

IF(ENABLE_PRECOMPILED_HEADERS)
    # STL headers
    TARGET_PRECOMPILE_HEADERS(libpykokkos-precompiled-headers
        INTERFACE
            <string>
            <sstream>
            <iostream>
            <regex>
            <type_traits>
            <map>
            <set>
            <vector>
            <unordered_map>
            <cstdlib>
            <cstdint>
            <cstdio>
    )

    # Kokkos headers
    TARGET_PRECOMPILE_HEADERS(libpykokkos-precompiled-headers
        INTERFACE
            <Kokkos_Core.hpp>
            <Kokkos_DynRankView.hpp>
    )

    # pybind11 headers
    TARGET_PRECOMPILE_HEADERS(libpykokkos-precompiled-headers
        INTERFACE
            <pybind11/pybind11.h>
            <pybind11/operators.h>
            <pybind11/pytypes.h>
            <pybind11/stl.h>
    )

    TARGET_LINK_LIBRARIES(libpykokkos-precompiled-headers INTERFACE
        Kokkos::kokkos
        pybind11::pybind11)
ENDIF()
