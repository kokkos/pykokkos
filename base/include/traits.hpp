/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#pragma once

#include <Kokkos_Core.hpp>

#include "common.hpp"
#include "concepts.hpp"
#include "defines.hpp"
#include "fwd.hpp"

//----------------------------------------------------------------------------//

DISABLE_TYPE(type_list<>)

//----------------------------------------------------------------------------//
/// \todo Reduce (or make it configurable) the number of dimensions supported by
/// concrete views. Currently, we instantiate up to 8 dimensions but we could
/// theoretically just say that python only supports up to 3 dimensions and if
/// higher than that, user bindings must convert View to DynRankView.
///
VIEW_DATA_DIMS(1, T *)
VIEW_DATA_DIMS(2, T **)
VIEW_DATA_DIMS(3, T ***)
VIEW_DATA_DIMS(4, T ****)
VIEW_DATA_DIMS(5, T *****)
VIEW_DATA_DIMS(6, T ******)
VIEW_DATA_DIMS(7, T *******)
VIEW_DATA_DIMS(8, T ********)

//----------------------------------------------------------------------------//
// <data-type> <enum> <string identifiers>
//  the first string identifier is the "canonical name" (i.e. what gets encoded)
//  and the remaining string entries are used to generate aliases
//
VIEW_DATA_TYPE(int8_t, Int8, "int8", "signed_char")
VIEW_DATA_TYPE(int16_t, Int16, "int16", "short")
VIEW_DATA_TYPE(int32_t, Int32, "int32", "int")
VIEW_DATA_TYPE(int64_t, Int64, "int64", "long")
VIEW_DATA_TYPE(uint8_t, Uint8, "uint8", "unsigned_char")
VIEW_DATA_TYPE(uint16_t, Uint16, "uint16", "unsigned_short")
VIEW_DATA_TYPE(uint32_t, Uint32, "uint32", "unsigned", "unsigned_int")
VIEW_DATA_TYPE(uint64_t, Uint64, "uint64", "unsigned_long")
VIEW_DATA_TYPE(float, Float32, "float32", "float")
VIEW_DATA_TYPE(double, Float64, "float64", "double")
VIEW_DATA_TYPE(Kokkos::complex<float>, ComplexFloat32, "complex_float32_dtype",
               "complex_float_dtype")
VIEW_DATA_TYPE(Kokkos::complex<double>, ComplexFloat64, "complex_float64_dtype",
               "complex_double_dtype")

//----------------------------------------------------------------------------//
// <data-type> <enum> <string identifiers>
//  the first string identifier is the "canonical name" (i.e. what gets encoded)
//  and the remaining string entries are used to generate aliases
//
MEMORY_LAYOUT(Kokkos::LayoutLeft, Left, "LayoutLeft")
MEMORY_LAYOUT(Kokkos::LayoutRight, Right, "LayoutRight")
MEMORY_LAYOUT(Kokkos::LayoutStride, Stride, "LayoutStride")

#if !defined(ENABLE_LAYOUTS)
DISABLE_TYPE(Kokkos::LayoutLeft)
#endif
DISABLE_TYPE(Kokkos::LayoutStride)

//----------------------------------------------------------------------------//
// <data-type> <enum> <string identifiers>
//  the first string identifier is the "canonical name" (i.e. what gets encoded)
//  and the remaining string entries are used to generate aliases
//
/// \todo Determine if there any combinations of memory traits that are commonly
/// used and should be supported, e.g. RandomAccess + Restrict
///
MEMORY_TRAIT(Kokkos::MemoryTraits<0>, Managed, "Managed")
MEMORY_TRAIT(Kokkos::MemoryTraits<Kokkos::Unmanaged>, Unmanaged, "Unmanaged")
MEMORY_TRAIT(Kokkos::MemoryTraits<Kokkos::Aligned>, Aligned, "Aligned")
MEMORY_TRAIT(Kokkos::MemoryTraits<Kokkos::Atomic>, Atomic, "Atomic")
MEMORY_TRAIT(Kokkos::MemoryTraits<Kokkos::RandomAccess>, RandomAccess,
             "RandomAccess")
MEMORY_TRAIT(Kokkos::MemoryTraits<Kokkos::Restrict>, Restrict, "Restrict")

#if !defined(ENABLE_MEMORY_TRAITS)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::Aligned>)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::Atomic>)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::RandomAccess>)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::Restrict>)
#endif

ENABLE_IMPLICIT(Kokkos::MemoryTraits<0>)

//----------------------------------------------------------------------------//
// provides a mapping from an execution space to the memory space enumeration
//
EXECUTION_SPACE(Kokkos::Serial, Serial_Backend, "Serial")
EXECUTION_SPACE(Kokkos::Threads, Threads_Backend, "Threads")
EXECUTION_SPACE(Kokkos::OpenMP, OpenMP_Backend, "OpenMP")
EXECUTION_SPACE(Kokkos::Cuda, Cuda_Backend, "Cuda")
EXECUTION_SPACE(Kokkos::Experimental::HPX, HPX_Backend, "HPX")
EXECUTION_SPACE(Kokkos::Experimental::HIP, HIP_Backend, "HIP")
EXECUTION_SPACE(Kokkos::Experimental::SYCL, SYCL_Backend, "SYCL")
EXECUTION_SPACE(Kokkos::Experimental::OpenMPTarget, OpenMPTarget_Backend,
                "OpenMPTarget")

MEMORY_SPACE_IDX(Kokkos::Serial, HostSpace)
MEMORY_SPACE_IDX(Kokkos::Threads, HostSpace)
MEMORY_SPACE_IDX(Kokkos::OpenMP, HostSpace)
MEMORY_SPACE_IDX(Kokkos::Cuda, CudaSpace)
MEMORY_SPACE_IDX(Kokkos::Experimental::HPX, HostSpace)
MEMORY_SPACE_IDX(Kokkos::Experimental::HIP, HIPSpace)
MEMORY_SPACE_IDX(Kokkos::Experimental::SYCL, SYCLSharedUSMSpace)
MEMORY_SPACE_IDX(Kokkos::Experimental::OpenMPTarget, OpenMPTargetSpace)

#if !defined(KOKKOS_ENABLE_SERIAL)
DISABLE_TYPE(Kokkos::Serial)
#endif

#if !defined(KOKKOS_ENABLE_THREADS)
DISABLE_TYPE(Kokkos::Threads)
#endif

#if !defined(KOKKOS_ENABLE_OPENMP)
DISABLE_TYPE(Kokkos::OpenMP)
#endif

#if !defined(KOKKOS_ENABLE_CUDA)
DISABLE_TYPE(Kokkos::Cuda)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
DISABLE_TYPE(Kokkos::Experimental::HIP)
#endif

#if !defined(KOKKOS_ENABLE_HPX)
DISABLE_TYPE(Kokkos::Experimental::HPX)
#endif

#if !defined(KOKKOS_ENABLE_SYCL)
DISABLE_TYPE(Kokkos::Experimental::SYCL)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
DISABLE_TYPE(Kokkos::Experimental::OpenMPTarget)
#endif

//----------------------------------------------------------------------------//
//  declare any spaces that might not be available and mark them as unavailable
//  we declare these so that we can map the enum value to the type along with a
//  label
//
#if !defined(KOKKOS_ENABLE_CUDA)
DISABLE_TYPE(Kokkos::CudaSpace)
DISABLE_TYPE(Kokkos::CudaHostPinnedSpace)
#endif

#if !defined(KOKKOS_ENABLE_CUDA_UVM)
DISABLE_TYPE(Kokkos::CudaUVMSpace)
#endif

#if !defined(KOKKOS_ENABLE_HBWSPACE)
DISABLE_TYPE(Kokkos::Experimental::HBWSpace)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
DISABLE_TYPE(Kokkos::Experimental::HIPSpace)
DISABLE_TYPE(Kokkos::Experimental::HIPHostPinnedSpace)
DISABLE_TYPE(Kokkos::Experimental::HIPManagedSpace)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
DISABLE_TYPE(Kokkos::Experimental::OpenMPTargetSpace)
#endif

#if !defined(KOKKOS_ENABLE_SYCL)
DISABLE_TYPE(Kokkos::Experimental::SYCLSharedUSMSpace)
DISABLE_TYPE(Kokkos::Experimental::SYCLDeviceUSMSpace)
#endif

MEMORY_SPACE(Kokkos::HostSpace, HostSpace, "HostSpace", "Host", "Serial",
             "Threads", "OpenMP", "HPX")
MEMORY_SPACE(Kokkos::CudaSpace, CudaSpace, "CudaSpace", "Cuda")
MEMORY_SPACE(Kokkos::CudaUVMSpace, CudaUVMSpace, "CudaUVMSpace")
MEMORY_SPACE(Kokkos::CudaHostPinnedSpace, CudaHostPinnedSpace,
             "CudaHostPinnedSpace")
MEMORY_SPACE(Kokkos::Experimental::HBWSpace, HBWSpace, "HBWSpace")
MEMORY_SPACE(Kokkos::Experimental::HIPSpace, HIPSpace, "HIPSpace", "HIP")
MEMORY_SPACE(Kokkos::Experimental::HIPHostPinnedSpace, HIPHostPinnedSpace,
             "HIPHostPinnedSpace")
MEMORY_SPACE(Kokkos::Experimental::HIPManagedSpace, HIPManagedSpace,
             "HIPManagedSpace")
MEMORY_SPACE(Kokkos::Experimental::OpenMPTargetSpace, OpenMPTargetSpace,
             "OpenMPTargetSpace", "OpenMPTarget")
MEMORY_SPACE(Kokkos::Experimental::SYCLSharedUSMSpace, SYCLSharedUSMSpace,
             "SYCLSharedUSMSpace", "SYCL")
MEMORY_SPACE(Kokkos::Experimental::SYCLDeviceUSMSpace, SYCLDeviceUSMSpace,
             "SYCLDeviceUSMSpace")
