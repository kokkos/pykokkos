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

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <type_traits>

#include "common.hpp"
#include "traits.hpp"
#include "views.hpp"

namespace Space {
namespace SpaceDim {

// Helper to convert values to string, with specialization for complex types
template <typename T>
std::string value_to_string(const T &val) {
  return std::to_string(val);
}

template <typename T>
std::string value_to_string(const Kokkos::complex<T> &val) {
  return "(" + std::to_string(val.real()) + "+" + std::to_string(val.imag()) +
         "i)";
}

// Helper to detect if a type is complex
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<Kokkos::complex<T>> : std::true_type {};

// this function creates bindings for the atomic type returned to python from
// views with MemoryTrait<Kokkos::Atomic | ...>
template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t LayoutIdx>
void generate_atomic_variant(py::module &_mod) {
  using data_spec_t   = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t  = MemorySpaceSpecialization<SpaceIdx>;
  using layout_spec_t = MemoryLayoutSpecialization<LayoutIdx>;
  using trait_spec_t  = MemoryTraitSpecialization<Atomic>;
  using Tp            = typename data_spec_t::type;
  using Vp            = typename ViewDataTypeRepr<Tp, DimIdx>::type;
  using Sp            = typename space_spec_t::type;
  using Lp            = typename layout_spec_t::type;
  using Mp            = typename trait_spec_t::type;
  using ViewT         = view_type_t<Kokkos::View<Vp>, Lp, Sp, Mp>;
  using atomic_type   = typename ViewT::reference_type;
  using value_type    = typename atomic_type::value_type;

  auto name = join("_", "KokkosAtomicDataElement", data_spec_t::label(),
                   space_spec_t::label(), layout_spec_t::label(),
                   trait_spec_t::label(), DimIdx + 1);

  auto desc =
      std::string{"Kokkos::Impl::AtomicDataElement<Kokkos::ViewTraits<"} +
      join(", ", demangle<Vp>(), demangle<Lp>(), demangle<Sp>(),
           demangle<Mp>()) +
      ">>";

  if (debug_output())
    std::cerr << "Registering " << desc << " as python class '" << name
              << "'..." << std::endl;

  // class decl
  py::class_<atomic_type> _atomic{_mod, name.c_str()};
  _atomic.def(py::init([](value_type *_v) {
    return new atomic_type{_v, Kokkos::Impl::AtomicViewConstTag{}};
  }));

  _atomic.def(
      "inc", [](atomic_type &_obj) { return _obj.inc(); },
      "Increment the atomic");
  _atomic.def(
      "dec", [](atomic_type &_obj) { return _obj.dec(); },
      "Decrement the atomic");
  _atomic.def(
      "__str__",
      [](atomic_type &_obj) {
        return value_to_string(static_cast<value_type>(_obj));
      },
      "String repr");

  _atomic.def(
      "__eq__",
      [](atomic_type &_obj, value_type _v) {
        return (static_cast<value_type>(_obj) == _v);
      },
      py::is_operator());
  _atomic.def(
      "__ne__",
      [](atomic_type &_obj, value_type _v) {
        return (static_cast<value_type>(_obj) != _v);
      },
      py::is_operator());

  // Only bind ordering operators for non-complex types
  if constexpr (!is_complex<value_type>::value) {
    _atomic.def(
        "__lt__",
        [](atomic_type &_obj, value_type _v) {
          return (static_cast<value_type>(_obj) < _v);
        },
        py::is_operator());
    _atomic.def(
        "__gt__",
        [](atomic_type &_obj, value_type _v) {
          return (static_cast<value_type>(_obj) > _v);
        },
        py::is_operator());
    _atomic.def(
        "__le__",
        [](atomic_type &_obj, value_type _v) {
          return (static_cast<value_type>(_obj) <= _v);
        },
        py::is_operator());
    _atomic.def(
        "__ge__",
        [](atomic_type &_obj, value_type _v) {
          return (static_cast<value_type>(_obj) >= _v);
        },
        py::is_operator());
  }

  // self type
  _atomic.def(py::self + py::self);
  _atomic.def(py::self - py::self);
  _atomic.def(py::self += py::self);
  _atomic.def(
      "__isub__",
      [](atomic_type &lhs, const atomic_type &rhs) { return (lhs -= rhs); },
      py::is_operator());

  // value type
  _atomic.def(
      "__add__", [](atomic_type _obj, value_type _v) { return (_obj += _v); },
      py::is_operator());
  _atomic.def(
      "__sub__", [](atomic_type _obj, value_type _v) { return (_obj -= _v); },
      py::is_operator());
  _atomic.def(
      "__mul__", [](atomic_type _obj, value_type _v) { return (_obj * _v); },
      py::is_operator());
  _atomic.def(
      "__truediv__",
      [](atomic_type _obj, value_type _v) { return (_obj / _v); },
      py::is_operator());

  _atomic.def(
      "__iadd__", [](atomic_type &_obj, value_type _v) { return (_obj += _v); },
      py::is_operator());
  _atomic.def(
      "__isub__", [](atomic_type &_obj, value_type _v) { return (_obj -= _v); },
      py::is_operator());
  _atomic.def(
      "__imul__",
      [](atomic_type &_obj, value_type _v) { return _obj = (_obj * _v); },
      py::is_operator());
  _atomic.def(
      "__itruediv__",
      [](atomic_type &_obj, value_type _v) { return _obj = (_obj / _v); },
      py::is_operator());
}
}  // namespace SpaceDim

// if the space is not available do nothing
template <size_t LayoutIdx, size_t DataIdx, size_t SpaceIdx, size_t... DimIdx>
void generate_atomic_variant(
    py::module &, std::index_sequence<DimIdx...>,
    std::enable_if_t<!is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
}

// if the space is available expand for every dimension
template <size_t LayoutIdx, size_t DataIdx, size_t SpaceIdx, size_t... DimIdx>
void generate_atomic_variant(
    py::module &_mod, std::index_sequence<DimIdx...>,
    std::enable_if_t<is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
  FOLD_EXPRESSION(
      SpaceDim::generate_atomic_variant<DataIdx, SpaceIdx, DimIdx, LayoutIdx>(
          _mod));
}
}  // namespace Space

namespace variants {

// expand for all the spaces with a given layout and data-type
template <size_t LayoutIdx, size_t DataIdx, size_t... SpaceIdx>
void generate_atomic_variant(py::module &_mod,
                             std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(Space::generate_atomic_variant<LayoutIdx, DataIdx, SpaceIdx>(
      _mod, std::make_index_sequence<ViewDataMaxDimensions>{}));
  // ensure atomic type for DynRankView is created
#if ENABLE_VIEW_RANKS < 6
  FOLD_EXPRESSION(Space::generate_atomic_variant<LayoutIdx, DataIdx, SpaceIdx>(
      _mod, std::index_sequence<6>{}));
#endif
}

}  // namespace variants

namespace {
// generate data type buffers for each memory space
template <size_t LayoutIdx, size_t... DataIdx>
void generate_atomic_variant(py::module &_mod,
                             std::index_sequence<DataIdx...>) {
  FOLD_EXPRESSION(variants::generate_atomic_variant<LayoutIdx, DataIdx>(
      _mod, std::make_index_sequence<MemorySpacesEnd>{}));
}
}  // namespace
