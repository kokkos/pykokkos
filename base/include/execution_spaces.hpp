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
#include "pools.hpp"
#include "traits.hpp"

namespace Common {
template <typename Sp, size_t SpaceIdx>
void generate_execution_space_init(
    [[maybe_unused]] py::class_<Sp> &,
    enable_if_t<SpaceIdx == Serial_Backend, int> = 0) {}

template <typename Sp, size_t SpaceIdx>
void generate_execution_space_init(
    [[maybe_unused]] py::class_<Sp> &,
    enable_if_t<SpaceIdx == Threads_Backend, int> = 0) {}

template <typename Sp, size_t SpaceIdx>
void generate_execution_space_init(
    [[maybe_unused]] py::class_<Sp> &,
    enable_if_t<SpaceIdx == OpenMP_Backend, int> = 0) {}

template <typename Sp, size_t SpaceIdx>
void generate_execution_space_init(
    [[maybe_unused]] py::class_<Sp> &_space,
    enable_if_t<SpaceIdx == Cuda_Backend, int> = 0) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__)
  _space.def(py::init([](uint64_t stream_ptr, bool manage_stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    return new Sp{s, manage_stream};
  }));
#endif
}

template <typename Sp, size_t SpaceIdx>
void generate_execution_space_init(
    [[maybe_unused]] py::class_<Sp> &_space,
    enable_if_t<SpaceIdx == HIP_Backend, int> = 0) {
#if defined(KOKKOS_ENABLE_HIP) && defined(__HIPCC__)
  _space.def(py::init([](uint64_t stream_ptr, bool manage_stream) {
    hipStream_t s = reinterpret_cast<hipStream_t>(stream_ptr);
    return new Sp{s, manage_stream};
  }));
#endif
}

template <typename Sp, size_t SpaceIdx>
void generate_execution_space(py::module &_mod, const std::string &_name,
                              const std::string &_msg) {
  if (debug_output())
    std::cerr << "Registering " << _msg << " as python class '" << _name
              << "'..." << std::endl;

  py::class_<Sp> _space(_mod, _name.c_str());
  _space.def(py::init([]() { return new Sp{}; }));

  // Add other constructors with arguments if they exist
  generate_execution_space_init<Sp, SpaceIdx>(_space);
}
}  // namespace Common

namespace Space {
namespace SpaceDim {

template <size_t SpaceIdx>
void generate_execution_space(py::module &_mod) {
  using space_spec_t = ExecutionSpaceSpecialization<SpaceIdx>;
  using Sp           = typename space_spec_t::type;

  auto name = join("_", "KokkosExecutionSpace", space_spec_t::label());

  Common::generate_execution_space<Sp, SpaceIdx>(_mod, name, demangle<Sp>());
}
}  // namespace SpaceDim

template <size_t SpaceIdx>
void generate_execution_space(
    py::module &,
    std::enable_if_t<!is_available<execution_space_t<SpaceIdx>>::value, int> =
        0) {}

template <size_t SpaceIdx>
void generate_execution_space(
    py::module &_mod,
    std::enable_if_t<is_available<execution_space_t<SpaceIdx>>::value, int> =
        0) {
  SpaceDim::generate_execution_space<SpaceIdx>(_mod);
}
}  // namespace Space

template <size_t... SpaceIdx>
void generate_execution_spaces(py::module &_mod,
                               std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(Space::generate_execution_space<SpaceIdx>(_mod));
}
