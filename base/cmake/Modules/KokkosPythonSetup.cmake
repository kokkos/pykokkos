
#----------------------------------------------------------------------------------------#
#   versioning
#----------------------------------------------------------------------------------------#

INCLUDE_GUARD(DIRECTORY)

FILE(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
STRING(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
STRING(REGEX REPLACE
    "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)"
    "\\1.\\2.\\3"
    VERSION_STRING "${FULL_VERSION_STRING}")
SET(pykokkos-base_VERSION "${VERSION_STRING}")
IF(NOT "${pykokkos-base_VERSION}" STREQUAL "${FULL_VERSION_STRING}")
    MESSAGE(STATUS "pykokkos-base version ${pykokkos-base_VERSION} (${FULL_VERSION_STRING})")
ELSE()
    MESSAGE(STATUS "pykokkos-base version ${pykokkos-base_VERSION}")
ENDIF()
SET(pykokkos-base_VERSION_STRING "${FULL_VERSION_STRING}")
STRING(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")
LIST(GET VERSION_LIST 0 pykokkos-base_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 pykokkos-base_VERSION_MINOR)
LIST(GET VERSION_LIST 2 pykokkos-base_VERSION_PATCH)
SET(pykokkos-base_VERSION
    "${pykokkos-base_VERSION_MAJOR}.${pykokkos-base_VERSION_MINOR}.${pykokkos-base_VERSION_PATCH}")

MATH(EXPR pykokkos-base_VERSION_CODE
    "${pykokkos-base_VERSION_MAJOR} * 10000 + ${pykokkos-base_VERSION_MINOR} * 100 + ${pykokkos-base_VERSION_PATCH}")

SET(CMAKE_MACOSX_RPATH ON)
