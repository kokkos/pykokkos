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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cassert>
#include <iostream>

#include "common.hpp"
#include "defines.hpp"
#include "fwd.hpp"
#include "traits.hpp"

//----------------------------------------------------------------------------//

template <template <size_t> class SpecT, typename Tp, size_t... Idx>
auto generate_enumeration(py::enum_<Tp> &_enum, std::index_sequence<Idx...>) {
  auto _generate = [&_enum](const auto &_labels, Tp _idx) {
    for (const auto &itr : _labels) {
      if (debug_output())
        std::cerr << "Registering " << demangle<Tp>() << " enumeration entry "
                  << itr << " to index " << _idx << "..." << std::endl;

      assert(!itr.empty());
      _enum.value(itr.c_str(), _idx);
    }
  };
  FOLD_EXPRESSION(_generate(SpecT<Idx>::labels(), static_cast<Tp>(Idx)));
}

//----------------------------------------------------------------------------//

template <typename Tp, size_t... Idx>
auto generate_enumeration_tuple(std::index_sequence<Idx...>) {
  return std::make_tuple(static_cast<Tp>(Idx)...);
}

//----------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_enumeration(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx) return SpecT<Idx>::label();
  std::stringstream ss;
  ss << "Error! Index " << i << " does not match any known enumeration type";
  throw py::value_error{ss.str()};
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_enumeration(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx)
    return SpecT<Idx>::label();
  else
    return get_enumeration<SpecT>(i, std::index_sequence<Tail...>{});
}

//----------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_enumeration(const std::string &str,
                     std::index_sequence<Idx, Tail...>) {
  if (str == SpecT<Idx>::label() || SpecT<Idx>::labels().count(str) > 0)
    return Idx;
  std::stringstream ss;
  ss << "Error! Identifier " << str
     << " does not match any known enumeration type";
  throw py::value_error{ss.str()};
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_enumeration(const std::string &str,
                     std::index_sequence<Idx, Tail...>) {
  assert(!str.empty());
  if (str == SpecT<Idx>::label() || SpecT<Idx>::labels().count(str) > 0)
    return Idx;
  else
    return get_enumeration<SpecT>(str, std::index_sequence<Tail...>{});
}

//----------------------------------------------------------------------------//

void generate_enumeration(py::module &kokkos) {
  //----------------------------------------------------------------------------//
  //
  //                                execution spaces
  //
  //----------------------------------------------------------------------------//
  py::enum_<KokkosExecutionSpace> _device(kokkos, "device",
                                          "Device execution spaces");
  generate_enumeration<ExecutionSpaceSpecialization>(
      _device, std::make_index_sequence<ExecutionSpacesEnd>{});

  _device.value("DefaultExecutionSpace",
                ExecutionSpaceIndex<Kokkos::DefaultExecutionSpace>::value);
  _device.value("DefaultHostExecutionSpace",
                ExecutionSpaceIndex<Kokkos::DefaultHostExecutionSpace>::value);
  _device.export_values();

  kokkos.attr("devices") = []() {
    return generate_enumeration_tuple<KokkosExecutionSpace>(
        std::make_index_sequence<ExecutionSpacesEnd>{});
  }();

  auto _get_execspace_name = [](int idx) {
    return get_enumeration<ExecutionSpaceSpecialization>(
        idx, std::make_index_sequence<ExecutionSpacesEnd>{});
  };
  auto _get_execspace_idx = [](std::string str) {
    return get_enumeration<ViewDataTypeSpecialization>(
        str, std::make_index_sequence<ExecutionSpacesEnd>{});
  };
  kokkos.def("get_execution_space", _get_execspace_name,
             "Get the execution space");
  kokkos.def("get_execution_space", _get_execspace_idx,
             "Get the execution space");

  //----------------------------------------------------------------------------//
  //
  //                                data types
  //
  //----------------------------------------------------------------------------//
  // an enumeration of the data types for views
  py::enum_<KokkosViewDataType> _dtype(kokkos, "dtype", "View data types");
  generate_enumeration<ViewDataTypeSpecialization>(
      _dtype, std::make_index_sequence<ViewDataTypesEnd>{});
  _dtype.export_values();

  kokkos.attr("dtypes") = []() {
    return generate_enumeration_tuple<KokkosViewDataType>(
        std::make_index_sequence<ViewDataTypesEnd>{});
  }();

  auto _get_dtype_name = [](int idx) {
    return get_enumeration<ViewDataTypeSpecialization>(
        idx, std::make_index_sequence<ViewDataTypesEnd>{});
  };
  auto _get_dtype_idx = [](std::string str) {
    return get_enumeration<ViewDataTypeSpecialization>(
        str, std::make_index_sequence<ViewDataTypesEnd>{});
  };
  kokkos.def("get_dtype", _get_dtype_name, "Get the data type");
  kokkos.def("get_dtype", _get_dtype_idx, "Get the data type");

  //----------------------------------------------------------------------------//
  //
  //                                memory spaces
  //
  //----------------------------------------------------------------------------//
  // an enumeration of the memory spaces for views
  py::enum_<KokkosMemorySpace> _memspace(kokkos, "memory_space",
                                         "View memory spaces");
  generate_enumeration<MemorySpaceSpecialization>(
      _memspace, std::make_index_sequence<MemorySpacesEnd>{});
  _memspace.value(
      "DefaultMemorySpace",
      MemorySpaceIndex<
          typename Kokkos::DefaultExecutionSpace::memory_space>::value);
  _memspace.value(
      "DefaultHostMemorySpace",
      MemorySpaceIndex<
          typename Kokkos::DefaultHostExecutionSpace::memory_space>::value);
  _memspace.export_values();

  kokkos.attr("memory_spaces") = []() {
    return generate_enumeration_tuple<KokkosMemorySpace>(
        std::make_index_sequence<MemorySpacesEnd>{});
  }();

  auto _get_memspace_name = [](int idx) {
    return get_enumeration<MemorySpaceSpecialization>(
        idx, std::make_index_sequence<MemorySpacesEnd>{});
  };
  auto _get_memspace_idx = [](std::string str) {
    return get_enumeration<MemorySpaceSpecialization>(
        str, std::make_index_sequence<MemorySpacesEnd>{});
  };
  kokkos.def("get_memory_space", _get_memspace_name, "Get the memory space");
  kokkos.def("get_memory_space", _get_memspace_idx, "Get the memory space");

  //----------------------------------------------------------------------------//
  //
  //                                  layouts
  //
  //----------------------------------------------------------------------------//
  // an enumeration of the layout types for views
  py::enum_<KokkosMemoryLayoutType> _ltype(kokkos, "layout",
                                           "View layout types");
  generate_enumeration<MemoryLayoutSpecialization>(
      _ltype, std::make_index_sequence<MemoryLayoutEnd>{});
  _ltype.export_values();

  kokkos.attr("layouts") = []() {
    return generate_enumeration_tuple<KokkosMemoryLayoutType>(
        std::make_index_sequence<MemoryLayoutEnd>{});
  }();

  auto _get_ltype_name = [](int idx) {
    return get_enumeration<MemoryLayoutSpecialization>(
        idx, std::make_index_sequence<MemoryLayoutEnd>{});
  };
  auto _get_ltype_idx = [](std::string str) {
    return get_enumeration<MemoryLayoutSpecialization>(
        str, std::make_index_sequence<MemoryLayoutEnd>{});
  };
  kokkos.def("get_layout", _get_ltype_name, "Get the layout type");
  kokkos.def("get_layout", _get_ltype_idx, "Get the layout type");

  //----------------------------------------------------------------------------//
  //
  //                                memory traits
  //
  //----------------------------------------------------------------------------//
  // an enumeration of the memory traits for views
  py::enum_<KokkosMemoryTrait> _memtrait(kokkos, "memory_trait",
                                         "View memory traits");
  generate_enumeration<MemoryTraitSpecialization>(
      _memtrait, std::make_index_sequence<MemoryTraitEnd>{});
  _memtrait.export_values();

  kokkos.attr("memory_traits") = []() {
    return generate_enumeration_tuple<KokkosMemoryTrait>(
        std::make_index_sequence<MemoryTraitEnd>{});
  }();

  auto _get_memtrait_name = [](int idx) {
    return get_enumeration<MemoryTraitSpecialization>(
        idx, std::make_index_sequence<MemoryTraitEnd>{});
  };
  auto _get_memtype_idx = [](std::string str) {
    return get_enumeration<MemoryTraitSpecialization>(
        str, std::make_index_sequence<MemoryTraitEnd>{});
  };
  kokkos.def("get_memory_trait", _get_memtrait_name, "Get the memory trait");
  kokkos.def("get_memory_trait", _get_memtype_idx, "Get the memory trait");
}
