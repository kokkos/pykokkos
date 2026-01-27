# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Creates a 'format' target that runs clang-format
#
##########################################################################################

set(_FMT ON)
# Visual Studio GUI reports "errors" occasionally
if(WIN32)
    set(_FMT OFF)
endif()

option(ENABLE_FORMAT "Enable a format target" ${_FMT})
mark_as_advanced(ENABLE_FORMAT)

if(NOT ENABLE_FORMAT)
    return()
endif()

# prefer clang-format 8.0
find_program(CLANG_FORMATTER
    NAMES
        clang-format-8
        clang-format-8.0
        clang-format-mp-8.0)

# python formatting
find_program(BLACK_FORMATTER
    NAMES black)

if(CLANG_FORMATTER OR BLACK_FORMATTER)
    # name of the format target
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-${PROJECT_NAME})
    endif()

    if(CLANG_FORMATTER AND BLACK_FORMATTER)
        set(_COMMENT "[${PROJECT_NAME}] Running '${CLANG_FORMATTER}' and '${BLACK_FORMATTER}'...")
    elseif(CLANG_FORMATTER)
        set(_COMMENT "[${PROJECT_NAME}] Running '${CLANG_FORMATTER}'...")
    elseif(BLACK_FORMATTER)
        set(_COMMENT "[${PROJECT_NAME}] Running '${BLACK_FORMATTER}'...")
    endif()

    # files and command for clang-format
    if(CLANG_FORMATTER)
        file(GLOB_RECURSE headers
            ${PROJECT_SOURCE_DIR}/include/*.hpp
            ${PROJECT_SOURCE_DIR}/examples/*.hpp)

        file(GLOB_RECURSE sources
            ${PROJECT_SOURCE_DIR}/src/*.cpp
            ${PROJECT_SOURCE_DIR}/src/*.cpp.in
            ${PROJECT_SOURCE_DIR}/examples/*.cpp)

        set(_COMMAND
            COMMAND ${CLANG_FORMATTER} -i ${headers}
            COMMAND ${CLANG_FORMATTER} -i ${sources})
    endif()

    # command for black
    if(BLACK_FORMATTER)
        set(_COMMAND ${_COMMAND}
            COMMAND ${BLACK_FORMATTER} -q ${PROJECT_SOURCE_DIR})
    endif()

    # create the format target
    add_custom_target(${FORMAT_NAME}
        ${_COMMAND}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "${_COMMENT}"
        SOURCES ${headers} ${sources})
endif()
