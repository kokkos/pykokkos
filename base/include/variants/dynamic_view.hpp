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

#include "common.hpp"
#include "concepts.hpp"
#include "traits.hpp"
#include "views.hpp"

#pragma once

namespace Space {

// this generates a binding for a dynamic rank view (i.e.
// Kokkos::DynRankView<...>)
template <size_t DataIdx, size_t SpaceIdx, size_t LayoutIdx, size_t TraitIdx>
void generate_dynamic_view_variant(
    py::module &,
    std::enable_if_t<!is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
}

template <size_t DataIdx, size_t SpaceIdx, size_t LayoutIdx, size_t TraitIdx>
void generate_dynamic_view_variant(
    py::module &_mod,
    std::enable_if_t<is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
  constexpr size_t DimIdx = 7;
  using data_spec_t       = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t      = MemorySpaceSpecialization<SpaceIdx>;
  using layout_spec_t     = MemoryLayoutSpecialization<LayoutIdx>;
  using trait_spec_t      = MemoryTraitSpecialization<TraitIdx>;
  using Tp                = typename data_spec_t::type;
  using Vp                = Tp;
  using Sp                = typename space_spec_t::type;
  using Lp                = typename layout_spec_t::type;
  using Mp                = typename trait_spec_t::type;
  using ViewT             = view_type_t<Kokkos::DynRankView<Vp>, Lp, Sp, Mp>;
  using UniformT          = kokkos_python_view_type_t<ViewT>;

  constexpr bool explicit_trait = !is_implicit<Mp>::value;

  auto name = join("_", "KokkosDynRankView", data_spec_t::label(),
                   space_spec_t::label(), layout_spec_t::label(),
                   (explicit_trait) ? trait_spec_t::label() : std::string{});

  // the Common::generate_view adds 1 so subtract one in order
  // generate initializers and accessors for all rank dimensions
  constexpr auto nIdx = DimIdx - 1;
  Common::generate_view<UniformT, Sp, Tp, Lp, Mp, nIdx>(
      _mod, name, demangle<UniformT>(), DimIdx,
      std::make_index_sequence<nIdx>{});

#if !defined(ENABLE_LAYOUTS)
  using MirrorT = kokkos_python_view_type_t<typename UniformT::HostMirror>;

  IF_CONSTEXPR(!std::is_same<UniformT, MirrorT>::value) {
    Common::generate_view<MirrorT, Sp, Tp, Lp, Mp, nIdx>(
        _mod, name + "_mirror", demangle<MirrorT>(), DimIdx,
        std::make_index_sequence<nIdx>{});
  }
#endif
}
}  // namespace Space

namespace {
// generate data-type, memory-space buffers for dynamic dimension
template <size_t DataIdx, size_t LayoutIdx, size_t TraitIdx, size_t... SpaceIdx>
void generate_dynamic_view_variant(py::module &_mod,
                                   std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(
      Space::generate_dynamic_view_variant<DataIdx, SpaceIdx, LayoutIdx,
                                           TraitIdx>(_mod));
}
}  // namespace
