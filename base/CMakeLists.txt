CMAKE_MINIMUM_REQUIRED(VERSION 3.16 FATAL_ERROR)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})


INCLUDE(KokkosPythonSetup)
INCLUDE(FetchContent)

PROJECT(
    pykokkos-base
    LANGUAGES   C CXX
    VERSION     ${pykokkos-base_VERSION})

IF("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    SET(PYKOKKOS_BASE_MAIN_PROJECT ON)
ELSE()
    SET(PYKOKKOS_BASE_MAIN_PROJECT OFF)
ENDIF()

INCLUDE(KokkosPythonUtilities)  # miscellaneous macros and functions

# ensure always PIC
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

ADD_OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
# force to release if not specified
IF("${CMAKE_BUILD_TYPE}" STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
ENDIF()

# Check if pybind11 target already exists (e.g., when used as a subproject)
if(NOT TARGET pybind11::module)
    FIND_PACKAGE(pybind11 CONFIG)
    if (pybind11_FOUND)
        MESSAGE(STATUS "Found pybind11: ${pybind11_DIR} (version \"${pybind11_VERSION}\")")
    else()
        MESSAGE(STATUS "Fetching pybind11")
        FETCHCONTENT_DECLARE(
          pybind11
          URL https://github.com/pybind/pybind11/archive/refs/heads/v3.0.zip
          URL_HASH SHA256=1816ea10bf7de362fb7864b131cc90bed6fc393feaf7e27b0ed5a03987bedd10
        )
        FETCHCONTENT_MAKEAVAILABLE(pybind11)
    endif()
else()
    MESSAGE(STATUS "Using existing pybind11 target")
endif()

INCLUDE(KokkosPythonKokkos)         # find external Kokkos or add submodule
INCLUDE(KokkosPythonFormat)         # format target
INCLUDE(KokkosPythonCompilers)      # compiler identification
INCLUDE(KokkosPythonOptions)        # cache options and various variable settings
INCLUDE(KokkosPythonPackages)       # Python Interp
INCLUDE(KokkosPythonBuildOptions)   # build-options interface library

IF(ENABLE_THIN_LTO)
    SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
ENDIF()

SET(libpykokkos_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/libpykokkos.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/backend_version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/enumeration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/available.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/tools.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/execution_spaces.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/complex_dtypes.cpp)

SET(libpykokkos_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/include/libpykokkos.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/deep_copy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/concepts.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/defines.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/traits.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/views.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/execution_spaces.hpp)

ADD_LIBRARY(libpykokkos-core OBJECT
    ${libpykokkos_SOURCES}
    ${libpykokkos_HEADERS})

TARGET_LINK_LIBRARIES(libpykokkos-core PUBLIC
    pybind11::pybind11
    Kokkos::kokkos
    libpykokkos::precompiled-headers
    libpykokkos::build-options)

PYBIND11_ADD_MODULE(libpykokkos MODULE NO_EXTRAS
    $<TARGET_OBJECTS:libpykokkos-core>)

ADD_SUBDIRECTORY(src)

# link to kokkos and the custom build properties
TARGET_LINK_LIBRARIES(libpykokkos PRIVATE
    pybind11::pybind11
    Kokkos::kokkos
    libpykokkos::precompiled-headers
    libpykokkos::build-options)

IF(SKBUILD)
    SET(Kokkos_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX})
    SET(Kokkos_INSTALL_LIBDIR    ${CMAKE_INSTALL_PREFIX}/kokkos)
ELSE()
    SET(Kokkos_INSTALL_PYTHONDIR ${Python3_SITEARCH}/kokkos)
    SET(Kokkos_INSTALL_LIBDIR    ${Python3_SITEARCH}/kokkos)
ENDIF()

# figure out if we can install to Python3_SITEARCH
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E touch ${Python3_SITEARCH}/.__kokkos__init__.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    ERROR_VARIABLE ERR_MSG
    RESULT_VARIABLE ERR_CODE)

ADD_FEATURE(Python3_SITEARCH "Python site-packages directory")

IF(ERR_CODE AND NOT SKBUILD)
    # get the python directory name, e.g. 'python3.6' from
    # '/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6'
    get_filename_component(PYDIR "${Python3_STDLIB}" NAME)
    # Should not be CMAKE_INSTALL_LIBDIR! Python won't look in a lib64 folder
    set(Kokkos_INSTALL_PYTHONDIR lib/${PYDIR}/site-packages/kokkos)
    set(Kokkos_INSTALL_LIBDIR    lib/${PYDIR}/site-packages/kokkos)
ENDIF()

EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E rm -f ${Python3_SITEARCH}/.__kokkos__init__.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

# location where kokkos libraries are/will be installed
IF(ENABLE_INTERNAL_KOKKOS OR NOT Kokkos_ROOT)
    SET(_Kokkos_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
ELSEIF(Kokkos_ROOT)
    STRING(REGEX REPLACE "/cmake/.*" "" _Kokkos_LIBDIR "${Kokkos_ROOT}")
ENDIF()

# absolute path to libpykokkos install
SET(Kokkos_INSTALL_FULL_PYTHONDIR ${Kokkos_INSTALL_LIBDIR})
IF(NOT IS_ABSOLUTE "${Kokkos_INSTALL_FULL_PYTHONDIR}")
    SET(Kokkos_INSTALL_FULL_PYTHONDIR ${CMAKE_INSTALL_PREFIX}/${Kokkos_INSTALL_FULL_PYTHONDIR})
ENDIF()

# relative path from libpykokkos install directory to library install directory
FILE(RELATIVE_PATH LIB_RELPATH "${Kokkos_INSTALL_FULL_PYTHONDIR}" "${_Kokkos_LIBDIR}")

# set the output path to <BINARY_DIR>/kokkos so one
# can test the python import from the build directory
# Really, only LIBRARY_* is needed for Unix but Windows
# builds are weird so just setting all of them
SET_TARGET_PROPERTIES(libpykokkos PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    PDB_OUTPUT_DIRECTORY     ${PROJECT_BINARY_DIR}/kokkos)

# configure the rpath: <RELATIVE>:<CWD>:<FULL>
IF(APPLE)
    SET_TARGET_PROPERTIES(libpykokkos PROPERTIES
        MACOSX_RPATH "@loader_path/${LIB_RELPATH}:@loader_path:${_Kokkos_LIBDIR}:${CMAKE_INSTALL_RPATH}")
ELSEIF(UNIX)
    SET_TARGET_PROPERTIES(libpykokkos PROPERTIES
        INSTALL_RPATH "\$ORIGIN/${LIB_RELPATH}:\$ORIGIN:${_Kokkos_LIBDIR}:${CMAKE_INSTALL_RPATH}")
ENDIF()

CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/pytest.ini
    ${PROJECT_BINARY_DIR}/pytest.ini COPYONLY)

IF(NOT SKBUILD)
    CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/setup.cfg
        ${PROJECT_BINARY_DIR}/setup.cfg COPYONLY)
ENDIF()

INSTALL(TARGETS libpykokkos
    DESTINATION ${Kokkos_INSTALL_LIBDIR})

INSTALL(FILES ${PROJECT_BINARY_DIR}/kokkos/__init__.py
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})

INSTALL(FILES ${PROJECT_BINARY_DIR}/pytest.ini
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})

# glob any python package files
FILE(GLOB_RECURSE PYPACKAGE_FILES ${CMAKE_CURRENT_LIST_DIR}/kokkos/*.py*)
FOREACH(_FILE ${PYPACKAGE_FILES})
    # make it a relative path
    STRING(REPLACE "${CMAKE_CURRENT_LIST_DIR}/" "" _OUT_NAME "${_FILE}")
    # get the directory of the relative path
    GET_FILENAME_COMPONENT(_OUT_PATH "${_OUT_NAME}" DIRECTORY)
    # get the name without the extension
    GET_FILENAME_COMPONENT(_OUT_NAME "${_OUT_NAME}" NAME_WE)
    # target file for configure
    SET(_OUT_FILE ${PROJECT_BINARY_DIR}/${_OUT_PATH}/${_OUT_NAME}.py)
    # put version, python interpreter, etc. in the file for reference
    CONFIGURE_FILE(${_FILE} ${_OUT_FILE} @ONLY)
    # patch duplicated subfolder
    STRING(REPLACE "kokkos/kokkos" "kokkos" _OUT_PATH "${Kokkos_INSTALL_PYTHONDIR}/${_OUT_PATH}")
    # install to the correct folder structure
    INSTALL(FILES ${_OUT_FILE} DESTINATION ${_OUT_PATH})
ENDFOREACH()

# build the examples, not designed to be built stand-alone
IF(ENABLE_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()

IF(NOT ENABLE_QUIET)
    PRINT_FEATURES()
ENDIF()

IF("CUDA" IN_LIST Kokkos_DEVICES AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0 AND
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        MESSAGE(AUTHOR_WARNING "\nNVCC + GCC 8.x + PyBind11 has known compiler errors related to pybind11::detail::collect_arguments overloads\n")
    ENDIF()
ENDIF()
