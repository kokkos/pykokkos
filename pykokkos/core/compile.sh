#!/bin/bash

COMPILER="${1}"
MODULE="${2}"
EXEC_SPACE="${3}"
PK_ARG_MEMSPACE="${4}"
PK_ARG_LAYOUT="${5}"
PK_REAL="${6}"
KOKKOS_LIB_PATH="${7}"
KOKKOS_INCLUDE_PATH="${8}"
COMPUTE_CAPABILITY="${9}"
LIB_SUFFIX="${10}"
COMPILER_PATH="${11}"
SRC=$(find -name "*.cpp")

# set CXX standard to be the same as in KokkosCore_config.h
CXX_STANDARD=$(g++ -dM -E -DKOKKOS_MACROS_HPP ${KOKKOS_INCLUDE_PATH}/KokkosCore_config.h | grep KOKKOS_ENABLE_CXX | tr -d ' ' | sed -e 's/.*\(..\)$/\1/')

if [ "${COMPILER}" == "g++" ]; then
        g++ \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -march=native -mtune=native \
        -isystem "${KOKKOS_INCLUDE_PATH}" \
        -fPIC \
        -fopenmp -std=c++${CXX_STANDARD} \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        g++ \
        -I.. \
        -O3 \
        -shared \
        -fopenmp \
        "${SRC}".o -o "${MODULE}" \
        "${KOKKOS_LIB_PATH}/libkokkoscontainers${LIB_SUFFIX}.so" \
        "${KOKKOS_LIB_PATH}/libkokkoscore${LIB_SUFFIX}.so"

elif [ "${COMPILER}" == "nvcc" ]; then
        "${COMPILER_PATH}" \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -Xcompiler -march=native -Xcompiler -mtune=native \
        -isystem "${KOKKOS_INCLUDE_PATH}" \
        -arch="${COMPUTE_CAPABILITY}" \
        --expt-extended-lambda -fPIC \
        -Xcompiler -fopenmp -std=c++${CXX_STANDARD} \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        "${COMPILER_PATH}" \
        -I.. \
        -O3 \
        -shared \
        -arch="${COMPUTE_CAPABILITY}" \
        --expt-extended-lambda \
        -fopenmp \
        "${SRC}".o -o "${MODULE}" \
        "${KOKKOS_LIB_PATH}/libkokkoscontainers${LIB_SUFFIX}.so" \
        "${KOKKOS_LIB_PATH}/libkokkoscore${LIB_SUFFIX}.so"

elif [ "${COMPILER}" == "hipcc" ]; then
        hipcc \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -isystem "${KOKKOS_INCLUDE_PATH}" \
        -fPIC -fno-gpu-rdc \
        -fopenmp -std=c++${CXX_STANDARD} \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        hipcc \
        -I.. \
        -O3 \
        -shared \
        -fopenmp -fno-gpu-rdc \
        "${SRC}".o -o "${MODULE}" \
        "${KOKKOS_LIB_PATH}/libkokkoscontainers${LIB_SUFFIX}.so" \
        "${KOKKOS_LIB_PATH}/libkokkoscore${LIB_SUFFIX}.so"
fi
