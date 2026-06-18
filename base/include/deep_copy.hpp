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
#include <Kokkos_DynRankView.hpp>

#include "common.hpp"
#include "concepts.hpp"
#include "fwd.hpp"
#include "traits.hpp"

//----------------------------------------------------------------------------//

template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t LayoutIdx,
          size_t TraitIdx>
struct concrete_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                               index_val<DimIdx>, index_val<LayoutIdx>,
                               index_val<TraitIdx>> {
  static constexpr bool value =
      is_available<typename MemoryTraitSpecialization<TraitIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      type_list<typename view_type<
          Kokkos::View<typename ViewDataTypeRepr<
              typename ViewDataTypeSpecialization<DataIdx>::type,
              DimIdx>::type>,
          typename MemoryLayoutSpecialization<LayoutIdx>::type,
          typename MemorySpaceSpecialization<SpaceIdx>::type,
          typename MemoryTraitSpecialization<TraitIdx>::type>::type>>;
};

template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t LayoutIdx,
          size_t... TraitIdx>
struct concrete_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                               index_val<DimIdx>, index_val<LayoutIdx>,
                               std::index_sequence<TraitIdx...>> {
  static constexpr bool value =
      is_available<typename MemoryLayoutSpecialization<LayoutIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      concat_t<type_list<typename concrete_view_type_list<
          index_val<DataIdx>, index_val<SpaceIdx>, index_val<DimIdx>,
          index_val<LayoutIdx>, index_val<TraitIdx>>::type...>>>;
};

template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t... LayoutIdx>
struct concrete_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                               index_val<DimIdx>,
                               std::index_sequence<LayoutIdx...>> {
  using type = concat_t<type_list<typename concrete_view_type_list<
      index_val<DataIdx>, index_val<SpaceIdx>, index_val<DimIdx>,
      index_val<LayoutIdx>,
      decltype(std::make_index_sequence<MemoryTraitEnd>())>::type...>>;
};

template <size_t DataIdx, size_t SpaceIdx, size_t... DimIdx>
struct concrete_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                               std::index_sequence<DimIdx...>> {
  static constexpr bool value =
      is_available<typename MemorySpaceSpecialization<SpaceIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      concat_t<type_list<typename concrete_view_type_list<
          index_val<DataIdx>, index_val<SpaceIdx>, index_val<DimIdx>,
          decltype(std::make_index_sequence<MemoryLayoutEnd>())>::type...>>>;
};

template <size_t DataIdx, size_t... SpaceIdx>
struct concrete_view_type_list<index_val<DataIdx>,
                               std::index_sequence<SpaceIdx...>> {
  using type = concat_t<type_list<typename concrete_view_type_list<
      index_val<DataIdx>, index_val<SpaceIdx>,
      decltype(std::make_index_sequence<ViewDataMaxDimensions>())>::type...>>;
};

template <size_t... DataIdx>
struct concrete_view_type_list<std::index_sequence<DataIdx...>> {
  using type = concat_t<type_list<typename concrete_view_type_list<
      index_val<DataIdx>,
      decltype(std::make_index_sequence<MemorySpacesEnd>())>::type...>>;
};

template <>
struct concrete_view_type_list<> {
  using type = concat_t<type_list<typename concrete_view_type_list<
      decltype(std::make_index_sequence<ViewDataTypesEnd>())>::type>>;
};

//----------------------------------------------------------------------------//

template <size_t DataIdx, size_t SpaceIdx, size_t LayoutIdx, size_t TraitIdx>
struct dynamic_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                              index_val<LayoutIdx>, index_val<TraitIdx>> {
  static constexpr bool value =
      is_available<typename MemoryTraitSpecialization<TraitIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      type_list<typename view_type<
          Kokkos::DynRankView<
              typename ViewDataTypeSpecialization<DataIdx>::type>,
          typename MemoryLayoutSpecialization<LayoutIdx>::type,
          typename MemorySpaceSpecialization<SpaceIdx>::type,
          typename MemoryTraitSpecialization<TraitIdx>::type>::type>>;
};

template <size_t DataIdx, size_t SpaceIdx, size_t LayoutIdx, size_t... TraitIdx>
struct dynamic_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                              index_val<LayoutIdx>,
                              std::index_sequence<TraitIdx...>> {
  static constexpr bool value =
      is_available<typename MemoryLayoutSpecialization<LayoutIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      concat_t<type_list<typename dynamic_view_type_list<
          index_val<DataIdx>, index_val<SpaceIdx>, index_val<LayoutIdx>,
          index_val<TraitIdx>>::type...>>>;
};

template <size_t DataIdx, size_t SpaceIdx, size_t... LayoutIdx>
struct dynamic_view_type_list<index_val<DataIdx>, index_val<SpaceIdx>,
                              std::index_sequence<LayoutIdx...>> {
  static constexpr bool value =
      is_available<typename MemorySpaceSpecialization<SpaceIdx>::type>::value;

  using type = std::conditional_t<
      !value, type_list<>,
      concat_t<type_list<typename dynamic_view_type_list<
          index_val<DataIdx>, index_val<SpaceIdx>, index_val<LayoutIdx>,
          decltype(std::make_index_sequence<MemoryTraitEnd>())>::type...>>>;
};

template <size_t DataIdx, size_t... SpaceIdx>
struct dynamic_view_type_list<index_val<DataIdx>,
                              std::index_sequence<SpaceIdx...>> {
  using type = concat_t<type_list<typename dynamic_view_type_list<
      index_val<DataIdx>, index_val<SpaceIdx>,
      decltype(std::make_index_sequence<MemoryLayoutEnd>())>::type...>>;
};

template <size_t... DataIdx>
struct dynamic_view_type_list<std::index_sequence<DataIdx...>> {
  using type = concat_t<type_list<typename dynamic_view_type_list<
      index_val<DataIdx>,
      decltype(std::make_index_sequence<MemorySpacesEnd>())>::type...>>;
};

template <>
struct dynamic_view_type_list<> {
  using type = concat_t<type_list<typename dynamic_view_type_list<
      decltype(std::make_index_sequence<ViewDataTypesEnd>())>::type>>;
};

//----------------------------------------------------------------------------//

using concrete_view_type_list_t = typename concrete_view_type_list<>::type;
using dynamic_view_type_list_t  = typename dynamic_view_type_list<>::type;

//----------------------------------------------------------------------------//

template <typename LhsT, typename RhsT>
struct deep_copy_compatible {
 private:
  template <typename Tp, typename Up>
  static constexpr bool available() {
    return is_available<Tp>::value && is_available<Up>::value;
  }

  template <typename Tp, typename Up>
  static constexpr bool same_type() {
    return std::is_same<std::decay_t<typename Tp::value_type>,
                        std::decay_t<typename Up::value_type>>::value;
  }

  template <typename Tp, size_t TpR = Tp::rank>
  static constexpr size_t rank(int) {
    return TpR;
  }

  template <typename Tp>
  static constexpr bool rank(long) {
    return 0;
  }

  template <typename Tp, size_t TpR = Tp::rank>
  static constexpr bool is_dynamic(int) {
    return false;
  }

  template <typename Tp>
  static constexpr bool is_dynamic(long) {
    return true;
  }

  template <typename Tp, typename Up>
  static constexpr bool is_managed() {
    return !Tp::traits::memory_traits::is_unmanaged &&
           !Up::traits::memory_traits::is_unmanaged;
  }

  template <typename Tp, bool TpA = Tp::traits::memory_traits::is_atomic>
  static constexpr bool is_atomic(int) {
    return TpA;
  }

  template <typename Tp>
  static constexpr bool is_atomic(long) {
    return false;
  }

 public:
  static constexpr bool value =
      (available<LhsT, RhsT>() && same_type<LhsT, RhsT>() &&
       is_managed<LhsT, RhsT>() && !is_atomic<LhsT>(0) &&
       (rank<LhsT>(0) == rank<RhsT>(0)) &&
       !(is_dynamic<LhsT>(0) && !is_dynamic<RhsT>(0)));
};

//----------------------------------------------------------------------------//

template <typename LhsT>
struct PYKOKKOS_HIDDEN deep_copy {
  static_assert(!std::is_same<LhsT, type_list<>>::value,
                "Error! Empty type list");

  explicit inline deep_copy(py::class_<LhsT>& _module) : m_module{_module} {}

  template <typename... RhsT>
  inline auto operator()(type_list<RhsT...>&&) {
    FOLD_EXPRESSION(sfinae<kokkos_python_view_type_t<LhsT>,
                           kokkos_python_view_type_t<RhsT>>(0));
  }

 private:
  py::class_<LhsT>& m_module;

  template <typename Tp, typename Up>
  inline auto sfinae(
      int, enable_if_t<deep_copy_compatible<Tp, Up>::value, int> = 0) const
      -> decltype(Kokkos::deep_copy(std::declval<Tp&>(),
                                    std::declval<const Up&>()),
                  void()) {
    m_module.def("deep_copy", [](Tp& _lhs, const Up& _rhs) {
      return Kokkos::deep_copy(_lhs, _rhs);
    });
  }

  template <typename Tp, typename Up>
  inline auto sfinae(long) const {}
};
