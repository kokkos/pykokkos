cmake_minimum_required(VERSION 3.18)

# Setup
if(ENABLE_CUDA)
    project(PyKokkosModule CXX CUDA)
elseif(ENABLE_HIP)
    project(PyKokkosModule CXX HIP)
else()
    project(PyKokkosModule CXX)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(pybind11 2.11 CONFIG REQUIRED)
find_package(Kokkos 4.7.1 EXACT REQUIRED)

# Sources and targets
file(GLOB SOURCE_FILES "*.cpp")
add_library(${MODULE_NAME} SHARED ${SOURCE_FILES})

set_target_properties(${MODULE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Includes
target_include_directories(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compiler definitions and flags
target_compile_definitions(${MODULE_NAME} PRIVATE
    SPACE=${PK_EXEC_SPACE}
    pk_arg_memspace=${PK_ARG_MEMSPACE}
    pk_arg_layout=${PK_ARG_LAYOUT}
    pk_exec_space=Kokkos::${PK_EXEC_SPACE}
    pk_real=${PK_REAL}
)

# Link libraries
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::module Kokkos::kokkos)
target_link_libraries(${MODULE_NAME} PRIVATE Kokkos::kokkoscore${LIB_SUFFIX} Kokkos::kokkoscontainers${LIB_SUFFIX})

# Install
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
