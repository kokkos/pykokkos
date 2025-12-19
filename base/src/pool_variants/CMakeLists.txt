
SET(_functions)

IF(CMAKE_UNITY_BUILD)
    SET(CMAKE_UNITY_BUILD_MODE GROUP)
ENDIF()

ADD_LIBRARY(libpykokkos-pool-variants OBJECT)

TARGET_LINK_LIBRARIES(libpykokkos-pool-variants PUBLIC
    pybind11::pybind11
    Kokkos::kokkos
    libpykokkos::precompiled-headers
    libpykokkos::build-options)

SET(_types              XorShift64 XorShift1024)

MACRO(ADD_VARIANT TYPE_VARIANT)
    STRING(TOLOWER "${TYPE_VARIANT}_pool" _TAG)
    SET(FUNC "generate_${_TAG}")
    IF(NOT ENABLE_QUIET)
        MESSAGE(STATUS "Generating '${_TAG}.cpp'...")
    ENDIF()
    CONFIGURE_FILE(
        ${CMAKE_CURRENT_SOURCE_DIR}/variant.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
        @ONLY)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
        PROPERTIES
        UNITY_GROUP "${TYPE_VARIANT}")
    TARGET_SOURCES(libpykokkos-pool-variants PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp)
    LIST(APPEND _functions "${FUNC}")
ENDMACRO()

# slightly different iterations schemes to improve grouping for unity builds
FOREACH(TYPE_VARIANT ${_types})
    ADD_VARIANT(${TYPE_VARIANT})
ENDFOREACH()

FOREACH(_FUNC ${_functions})
    SET(PROTOTYPES "${PROTOTYPES}void ${_FUNC}(py::module&);\n")
    SET(INVOCATIONS "${INVOCATIONS}  ${_FUNC}(kokkos);\n")
ENDFOREACH()

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/pool.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/pools.cpp
    @ONLY)

FILE(GLOB EXISTING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/include/pool_variants/*.hpp)

TARGET_SOURCES(libpykokkos-core PUBLIC
    ${EXISTING_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/pools.cpp)

TARGET_SOURCES(libpykokkos PUBLIC
    $<TARGET_OBJECTS:libpykokkos-pool-variants>)
