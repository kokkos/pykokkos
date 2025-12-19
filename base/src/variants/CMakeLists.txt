
SET(_functions)

#
#   TODO:
#       Determine whether we want to remove expansion in code and replace with
#       all explicit instantiations.
#
#   TODO:
#       Reduce (or make it configurable) the number of dimensions supported by
#       concrete views. Currently, we instantiate up to 8 dimensions but we could
#       theoretically just say that python only supports up to 3 dimensions and
#       if higher than that, user bindings must convert View to DynRankView.
#
IF(CMAKE_UNITY_BUILD)
    SET(CMAKE_UNITY_BUILD_MODE GROUP)
ENDIF()

ADD_LIBRARY(libpykokkos-variants OBJECT)

TARGET_LINK_LIBRARIES(libpykokkos-variants PUBLIC
    pybind11::pybind11
    Kokkos::kokkos
    libpykokkos::precompiled-headers
    libpykokkos::build-options)

SET(_types              concrete dynamic)
SET(_variants           layout memory_trait)
SET(_data_types         Int8 Int16 Int32 Int64 Uint8 Uint16 Uint32 Uint64 Float32 Float64 ComplexFloat32 ComplexFloat64)

SET(layout_enums        Right)
SET(memory_trait_enums  Managed)

IF(ENABLE_LAYOUTS)
    LIST(APPEND layout_enums        Left)
ENDIF()

#
#   TODO:
#       Are there any combinations of memory traits that are commonly used?
#       E.g. RandomAccess + Restrict
#
IF(ENABLE_MEMORY_TRAITS)
    LIST(APPEND memory_trait_enums  Aligned Atomic RandomAccess Restrict)
ENDIF()

MACRO(ADD_VARIANT TYPE_VARIANT DATA_VARIANT LAYOUT_VARIANT MEMORY_TRAIT_VARIANT)
    STRING(TOLOWER "${DATA_VARIANT}_${LAYOUT_VARIANT}_${MEMORY_TRAIT_VARIANT}" VARIANT)
    STRING(TOLOWER "${TYPE_VARIANT}_view_${VARIANT}" _TAG)
    SET(ENUM "${DATA_VARIANT}, ${LAYOUT_VARIANT}, ${MEMORY_TRAIT_VARIANT}")
    SET(FUNC "generate_${_TAG}")
    IF(NOT ENABLE_QUIET)
        MESSAGE(STATUS "Generating '${_TAG}.cpp'...")
    ENDIF()
    CONFIGURE_FILE(
        ${CMAKE_CURRENT_SOURCE_DIR}/variant.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
        @ONLY)
    IF(ENABLE_MEMORY_TRAITS)
        SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
            PROPERTIES
            UNITY_GROUP "${TYPE_VARIANT}_${DATA_VARIANT}_${LAYOUT_VARIANT}")
    ELSEIF(ENABLE_LAYOUTS)
        SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
            PROPERTIES
            UNITY_GROUP "${TYPE_VARIANT}_${LAYOUT_VARIANT}")
    ELSE()
        SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
            PROPERTIES
            UNITY_GROUP "${TYPE_VARIANT}")
    ENDIF()
    TARGET_SOURCES(libpykokkos-variants PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp)
    LIST(APPEND _functions "${FUNC}")
ENDMACRO()

# slightly different iterations schemes to improve grouping for unity builds
FOREACH(TYPE_VARIANT ${_types})
    FOREACH(DATA_VARIANT ${_data_types})
        FOREACH(LAYOUT_VARIANT ${layout_enums})
            FOREACH(MEMORY_TRAIT_VARIANT ${memory_trait_enums})
                ADD_VARIANT(${TYPE_VARIANT} ${DATA_VARIANT} ${LAYOUT_VARIANT} ${MEMORY_TRAIT_VARIANT})
            ENDFOREACH()
        ENDFOREACH()
    ENDFOREACH()
ENDFOREACH()

FOREACH(_FUNC ${_functions})
    SET(PROTOTYPES "${PROTOTYPES}void ${_FUNC}(py::module&);\n")
    SET(INVOCATIONS "${INVOCATIONS}  ${_FUNC}(kokkos);\n")
ENDFOREACH()

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/view.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/views.cpp
    @ONLY)

FILE(GLOB EXISTING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/include/variants/*.hpp)

TARGET_SOURCES(libpykokkos-core PUBLIC
    ${EXISTING_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/views.cpp)

TARGET_SOURCES(libpykokkos PUBLIC
    $<TARGET_OBJECTS:libpykokkos-variants>)
