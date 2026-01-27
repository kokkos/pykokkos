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
#include <set>
#include <string>

#include "common.hpp"
#include "defines.hpp"

//----------------------------------------------------------------------------//
//
//                                   Execution Spaces
//
//----------------------------------------------------------------------------//

namespace Kokkos {
class Serial;
class Threads;
class OpenMP;
class Cuda;
namespace Experimental {
class HIP;
class HPX;
class OpenMPTarget;
class SYCL;
}  // namespace Experimental
}  // namespace Kokkos

//----------------------------------------------------------------------------//
//
//                                   Memory spaces
//
//----------------------------------------------------------------------------//
//
// NOTE: do not support AnonymousSpace
//
namespace Kokkos {
class HostSpace;
class CudaSpace;
class CudaUVMSpace;
class CudaHostPinnedSpace;
// experimental spaces
namespace Experimental {
class HBWSpace;
class HIPSpace;
class HIPHostPinnedSpace;
class HIPManagedSpace;
class OpenMPTargetSpace;
class SYCLSharedUSMSpace;
class SYCLDeviceUSMSpace;
}  // namespace Experimental
}  // namespace Kokkos

//----------------------------------------------------------------------------//
//
//                                   Miscellaneous
//
//----------------------------------------------------------------------------//

namespace Kokkos {
//
template <unsigned T>
struct MemoryTraits;
//
template <typename DataType, class... Properties>
class DynRankView;  // forward declare
//
template <typename DataType, class... Properties>
class View;  // forward declare
//
template <class ExecutionSpace, class MemorySpace>
struct Device;
}  // namespace Kokkos

//----------------------------------------------------------------------------//
//
//                                  Enumerations
//
//----------------------------------------------------------------------------//

enum KokkosViewDataType {
  Int8 = 0,
  Int16,
  Int32,
  Int64,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Float32,
  Float64,
  ComplexFloat32,
  ComplexFloat64,
  ViewDataTypesEnd
};

/// \enum KokkosExecutionSpace
/// \brief An enumeration identifying all the device execution spaces
enum KokkosExecutionSpace {
  Serial_Backend = 0,
  Threads_Backend,
  OpenMP_Backend,
  Cuda_Backend,
  HPX_Backend,
  HIP_Backend,
  SYCL_Backend,
  OpenMPTarget_Backend,
  ExecutionSpacesEnd
};

/// \enum KokkosMemorySpace
/// \brief An enumeration identifying all the memory spaces for a view
enum KokkosMemorySpace {
  HostSpace = 0,
  // AnonymousSpace,
  CudaSpace,
  CudaUVMSpace,
  CudaHostPinnedSpace,
  HBWSpace,
  HIPSpace,
  HIPHostPinnedSpace,
  HIPManagedSpace,
  OpenMPTargetSpace,
  SYCLSharedUSMSpace,
  SYCLDeviceUSMSpace,
  MemorySpacesEnd
};

/// \enum KokkosMemoryLayoutType
/// \brief An enumeration identifying all the layouts for a view
enum KokkosMemoryLayoutType { Right = 0, Left, Stride, MemoryLayoutEnd };

/// \enum KokkosMemoryTrait
/// \brief An enumeration identifying all the memory traits for a view
enum KokkosMemoryTrait {
  Managed = 0,
  Unmanaged,
  RandomAccess,
  Atomic,
  Restrict,
  Aligned,
  MemoryTraitEnd
};

//----------------------------------------------------------------------------//
//
//                              Trait specialization types
//
//----------------------------------------------------------------------------//

static constexpr size_t ViewDataMaxDimensions = ENABLE_VIEW_RANKS;

static_assert(ViewDataMaxDimensions < 8,
              "Error! ViewDataMaxDimensions must be in range [0, 8)");

template <typename T, size_t Dim>
struct ViewDataTypeRepr;

template <size_t DataT>
struct ViewDataTypeSpecialization;

template <size_t SpaceT>
struct ExecutionSpaceSpecialization;

template <typename Tp>
struct ExecutionSpaceIndex;

template <size_t SpaceT>
struct MemorySpaceSpecialization;

template <typename Tp>
struct MemorySpaceIndex;

template <size_t DataT>
struct MemoryLayoutSpecialization;

template <typename Tp>
struct MemoryLayoutIndex;

template <size_t DataT>
struct MemoryTraitSpecialization;

template <typename Tp>
struct MemoryTraitIndex;

template <size_t Idx>
using execution_space_t = typename ExecutionSpaceSpecialization<Idx>::type;

template <size_t Idx>
using memory_space_t = typename MemorySpaceSpecialization<Idx>::type;

template <size_t Idx>
using memory_layout_t = typename MemoryLayoutSpecialization<Idx>::type;

template <size_t Idx>
using memory_trait_t = typename MemoryTraitSpecialization<Idx>::type;
