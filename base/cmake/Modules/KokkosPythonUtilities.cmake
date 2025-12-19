#
# useful macros and functions for generic tasks
#

# include guard
include_guard(DIRECTORY)

include(CMakeDependentOption)
include(CMakeParseArguments)


#-----------------------------------------------------------------------
# function - capitalize - make a string capitalized (first letter is capital)
#   usage:
#       capitalize("SHARED" CShared)
#   MESSAGE(STATUS "-- CShared is \"${CShared}\"")
#   $ -- CShared is "Shared"
FUNCTION(CAPITALIZE str var)
    # make string lower
    STRING(TOLOWER "${str}" str)
    STRING(SUBSTRING "${str}" 0 1 _first)
    STRING(TOUPPER "${_first}" _first)
    STRING(SUBSTRING "${str}" 1 -1 _remainder)
    STRING(CONCAT str "${_first}" "${_remainder}")
    SET(${var} "${str}" PARENT_SCOPE)
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# macro CHECKOUT_GIT_SUBMODULE()
#
#   Run "git submodule update" if a file in a submodule does not exist
#
#   ARGS:
#       RECURSIVE (option) -- add "--recursive" flag
#       RELATIVE_PATH (one value) -- typically the relative path to submodule
#                                    from PROJECT_SOURCE_DIR
#       WORKING_DIRECTORY (one value) -- (default: PROJECT_SOURCE_DIR)
#       TEST_FILE (one value) -- file to check for (default: CMakeLists.txt)
#       ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
FUNCTION(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS"
        ${ARGN})

    IF(NOT CHECKOUT_WORKING_DIRECTORY)
        SET(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    ENDIF()

    IF(NOT CHECKOUT_TEST_FILE)
        SET(CHECKOUT_TEST_FILE "CMakeLists.txt")
    ENDIF()

    # default assumption
    IF(NOT CHECKOUT_REPO_BRANCH)
        SET(CHECKOUT_REPO_BRANCH "master")
    ENDIF()

    FIND_PACKAGE(Git)
    SET(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    IF(NOT EXISTS "${_DIR}")
        IF(NOT CHECKOUT_REPO_URL)
            MESSAGE(FATAL_ERROR "submodule directory does not exist")
        ENDIF()
    ENDIF()

    # if this file exists --> project has been checked out
    # if not exists --> not been checked out
    SET(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")
    # assuming a .gitmodules file exists
    SET(_SUBMODULE "${PROJECT_SOURCE_DIR}/.gitmodules")

    SET(_TEST_FILE_EXISTS OFF)
    IF(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        SET(_TEST_FILE_EXISTS ON)
    ENDIF()

    IF(_TEST_FILE_EXISTS)
        return()
    ENDIF()

    FIND_PACKAGE(Git REQUIRED)

    SET(_SUBMODULE_EXISTS OFF)
    IF(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        SET(_SUBMODULE_EXISTS ON)
    ENDIF()

    SET(_HAS_REPO_URL OFF)
    IF(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        SET(_HAS_REPO_URL ON)
    ENDIF()

    # if the module has not been checked out
    IF(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        # perform the checkout
        EXECUTE_PROCESS(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        IF(RET GREATER 0)
            SET(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            MESSAGE(STATUS "FUNCTION(CHECKOUT_GIT_SUBMODULE) failed.")
            MESSAGE(FATAL_ERROR "Command: \"${_CMD}\"")
        ELSE()
            SET(_TEST_FILE_EXISTS ON)
        ENDIF()
    ENDIF()

    IF(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        MESSAGE(STATUS "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'...")

        # remove the existing directory
        IF(EXISTS "${_DIR}")
            EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        ENDIF()

        # perform the checkout
        EXECUTE_PROCESS(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                    ${CHECKOUT_ADDITIONAL_CMDS}
                    ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # perform the submodule update
        IF(CHECKOUT_RECURSIVE AND EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
            EXECUTE_PROCESS(
                COMMAND
                    ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY
                    ${_DIR}
                RESULT_VARIABLE RET)
        ENDIF()

        # check the return code
        IF(RET GREATER 0)
            SET(_CMD "${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}")
            MESSAGE(STATUS "FUNCTION(CHECKOUT_GIT_SUBMODULE) failed.")
            MESSAGE(FATAL_ERROR "Command: \"${_CMD}\"")
        ELSE()
            SET(_TEST_FILE_EXISTS ON)
        ENDIF()
    ENDIF()

    IF(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        MESSAGE(STATUS "Test file '${_TEST_FILE}' does not exist: ${_TEST_FILE_EXISTS}")
        MESSAGE(FATAL_ERROR "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'.")
    ENDIF()

ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# require variable
#
FUNCTION(CHECK_REQUIRED VAR)
    IF(NOT DEFINED ${VAR} OR "${${VAR}}" STREQUAL "")
        MESSAGE(FATAL_ERROR "Variable '${VAR}' must be defined and not empty")
    ENDIF()
ENDFUNCTION()


#-----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a project feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
FUNCTION(ADD_FEATURE _var _description)
  SET(EXTRA_DESC "")
  FOREACH(currentArg ${ARGN})
      IF(NOT "${currentArg}" STREQUAL "${_var}" AND
         NOT "${currentArg}" STREQUAL "${_description}")
          SET(EXTRA_DESC "${EXTA_DESC}${currentArg}")
      ENDIF()
  ENDFOREACH()

  SET_PROPERTY(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_FEATURES ${_var})
  SET_PROPERTY(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")

  IF("CMAKE_DEFINE" IN_LIST ARGN)
      SET_PROPERTY(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES "${_var} @${_var}@")
  ENDIF()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    GET_PROPERTY(_ADDED GLOBAL PROPERTY ${PROJECT_NAME}_ADDED_OPTIONS)
    IF("${_NAME}" IN_LIST _ADDED)
        RETURN()
    ENDIF()

    SET_PROPERTY(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_ADDED_OPTIONS ${_NAME})
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF("NO_FEATURE" IN_LIST ARGN)
        MARK_AS_ADVANCED(${_NAME})
    ELSE()
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ENDIF()
    IF("ADVANCED" IN_LIST ARGN)
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
FUNCTION(PRINT_ENABLED_FEATURES)
    SET(_basemsg "The following features are defined/enabled (+):")
    SET(_currentFeatureText "${_basemsg}")
    GET_PROPERTY(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    IF(NOT "${_features}" STREQUAL "")
        LIST(REMOVE_DUPLICATES _features)
        LIST(SORT _features)
    ENDIF()
    FOREACH(_feature ${_features})
        IF(${_feature})
            # add feature to text
            SET(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            GET_PROPERTY(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            IF(_desc)
                IF(NOT "${${_feature}}" STREQUAL "ON" AND
                   NOT "${${_feature}}" STREQUAL "TRUE")
                    SET(_currentFeatureText "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                ELSE()
                    STRING(REGEX REPLACE "^${PROJECT_NAME}_USE_" "" _feature_tmp "${_feature}")
                    STRING(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    capitalize("${_feature_tmp}" _feature_tmp_c)
                    FOREACH(_var _feature _feature_tmp _feature_tmp_l _feature_tmp_c)
                        SET(_ver "${${${_var}}_VERSION}")
                        IF(NOT "${_ver}" STREQUAL "")
                            SET(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        ENDIF()
                        UNSET(_ver)
                    ENDFOREACH()
                    SET(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                ENDIF()
                SET(_desc NOTFOUND)
            ENDIF()
        ENDIF()
    ENDFOREACH()

    IF(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        MESSAGE(STATUS "${_currentFeatureText}\n")
    ENDIF()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
FUNCTION(PRINT_DISABLED_FEATURES)
    SET(_basemsg "The following features are NOT defined/enabled (-):")
    SET(_currentFeatureText "${_basemsg}")
    GET_PROPERTY(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    IF(NOT "${_features}" STREQUAL "")
        LIST(REMOVE_DUPLICATES _features)
        LIST(SORT _features)
    ENDIF()
    FOREACH(_feature ${_features})
        IF(NOT ${_feature})
            SET(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            GET_PROPERTY(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            IF(_desc)
              SET(_currentFeatureText "${_currentFeatureText}: ${_desc}")
              SET(_desc NOTFOUND)
            ENDIF(_desc)
        ENDIF()
    ENDFOREACH(_feature)

    IF(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        MESSAGE(STATUS "${_currentFeatureText}\n")
    ENDIF()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
FUNCTION(PRINT_FEATURES)
    MESSAGE(STATUS "")
    print_enabled_features()
    print_disabled_features()
ENDFUNCTION()
