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

#include <pybind11/numpy.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>
#include <iostream>

#include "common.hpp"
#include "concepts.hpp"
#include "deep_copy.hpp"
#include "defines.hpp"
#include "fwd.hpp"
#include "traits.hpp"

//----------------------------------------------------------------------------//

template <typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_extents(Tp &m, std::index_sequence<Idx...>) {
  return RetT{m.extent(Idx)...};
}

template <typename Up, size_t Idx, typename Tp>
constexpr auto get_stride(Tp &m);

template <typename Tp>
inline std::string get_format() {
  return py::format_descriptor<Tp>::format();
}

template <>
inline std::string get_format<Kokkos::complex<float>>() {
  return py::format_descriptor<std::complex<float>>::format();
}

template <>
inline std::string get_format<Kokkos::complex<double>>() {
  return py::format_descriptor<std::complex<double>>::format();
}

template <typename Up, typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_strides(Tp &m, std::index_sequence<Idx...>) {
  return RetT{(sizeof(Up) * m.stride(Idx))...};
}

GET_STRIDE(0)
GET_STRIDE(1)
GET_STRIDE(2)
GET_STRIDE(3)
GET_STRIDE(4)
GET_STRIDE(5)
GET_STRIDE(6)
GET_STRIDE(7)

template <typename Up, typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_stride(Tp &m, std::index_sequence<Idx...>) {
  return RetT{(sizeof(Up) * get_stride<Idx>(m))...};
}

//----------------------------------------------------------------------------//

namespace Impl {
template <typename Tp, typename... Args>
struct get_item {
  static constexpr auto sequence = std::make_index_sequence<sizeof...(Args)>{};
  using tuple_type               = std::tuple<Args...>;

  template <size_t... Idx>
  static decltype(auto) get(Tp &_obj, tuple_type &&_args,
                            std::index_sequence<Idx...>) {
    return _obj.access(std::get<Idx>(std::forward<tuple_type>(_args))...);
  }

  static auto get() {
    return [](Tp &_obj, tuple_type _args) {
      return get(_obj, std::move(_args), sequence);
    };
  }
  template <typename Vp>
  static auto set() {
    return [](Tp &_obj, tuple_type _args, Vp _val) {
      get(_obj, std::move(_args), sequence) = _val;
    };
  }
};

template <typename Tp>
struct get_item<Tp, size_t> {
  static auto get() {
    return [](Tp &_obj, size_t _arg) { return _obj.access(_arg); };
  }
  template <typename Vp>
  static auto set() {
    return [](Tp &_obj, size_t _arg, Vp _val) { _obj.access(_arg) = _val; };
  }
};

template <typename Tp, size_t Idx>
struct get_type;

template <typename Tp>
struct get_type<Tp, 1> {
  using type = Impl::get_item<Tp, size_t>;
};

template <typename Tp>
struct get_type<Tp, 2> {
  using type = Impl::get_item<Tp, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 3> {
  using type = Impl::get_item<Tp, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 4> {
  using type = Impl::get_item<Tp, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 5> {
  using type = Impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 6> {
  using type =
      Impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 7> {
  using type = Impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t,
                              size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 8> {
  using type = Impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t,
                              size_t, size_t, size_t>;
};

}  // namespace Impl

template <typename Tp, size_t Idx>
struct get_item {
  using impl_type                = typename Impl::get_type<Tp, Idx>::type;
  using array_type               = std::array<Tp, Idx>;
  static constexpr auto sequence = std::make_index_sequence<Idx>{};

  static auto get() { return impl_type::get(); }

  template <typename Vp>
  static auto set() {
    return impl_type::template set<Vp>();
  }
};

//----------------------------------------------------------------------------//

namespace Impl {
//
template <typename ViewT, typename Up, size_t... Idx>
auto get_init(const std::string &lbl, const Up &arr,
              std::index_sequence<Idx...>) {
  return new ViewT{lbl, static_cast<size_t>(std::get<Idx>(arr))...};
}
//
template <typename ViewT, typename Up, typename Tp, size_t... Idx>
auto get_unmanaged_init(const Up &arr, const Tp data,
                        std::index_sequence<Idx...>) {
  return new ViewT{data, static_cast<size_t>(std::get<Idx>(arr))...};
}
//
}  // namespace Impl

template <typename ViewT, size_t Idx>
auto get_init() {
  return [](std::string lbl, std::array<size_t, Idx> arr) {
    return Impl::get_init<ViewT>(lbl, arr, std::make_index_sequence<Idx>{});
  };
}

template <typename ViewT, size_t Idx, typename Tp>
auto get_unmanaged_init() {
  return [](py::buffer buf, std::array<size_t, Idx> arr) {
    return Impl::get_unmanaged_init<ViewT>(arr,
                                           static_cast<Tp *>(buf.request().ptr),
                                           std::make_index_sequence<Idx>{});
  };
}

template <typename ViewT, size_t Idx, typename Tp, typename Vp>
auto get_init(
    Vp &_view,
    enable_if_t<!ViewT::traits::memory_traits::is_unmanaged, int> = 0) {
  // define managed init
  _view.def(py::init(get_init<ViewT, Idx>()));
  // define unmanaged init
  _view.def(py::init(get_unmanaged_init<ViewT, Idx, Tp>()));
}

template <typename ViewT, size_t Idx, typename Tp, typename Vp>
auto get_init(
    Vp &_view,
    enable_if_t<ViewT::traits::memory_traits::is_unmanaged, int> = 0) {
  // define unmanaged init
  _view.def(py::init(get_unmanaged_init<ViewT, Idx, Tp>()));
}

//----------------------------------------------------------------------------//

namespace Common {
// creates overloads for data access from python
template <typename Tp, typename ViewT, size_t... Idx>
void generate_view_access(py::class_<ViewT> &_view, std::index_sequence<Idx...>,
                          enable_if_t<(sizeof...(Idx) == 1), int> = 0) {
  FOLD_EXPRESSION(_view.def("__getitem__", get_item<ViewT, Idx + 1>::get(),
                            "Get the element"));
  FOLD_EXPRESSION(_view.def("__setitem__",
                            get_item<ViewT, Idx + 1>::template set<Tp>(),
                            "Set the element"));
}

template <typename Tp, typename ViewT>
void generate_view_access(py::class_<ViewT> &_view, std::index_sequence<0>) {
  _view.def("__getitem__", get_item<ViewT, 1>::get(), "Get the element");
  _view.def("__setitem__", get_item<ViewT, 1>::template set<Tp>(),
            "Set the element");
  _view.def(
      "__getitem__",
      [](ViewT &_obj, std::tuple<size_t> _arg) {
        return _obj.access(std::get<0>(_arg));
      },
      "Get the element");
  _view.def(
      "__setitem__",
      [](ViewT &_obj, std::tuple<size_t> _arg, Tp _val) {
        _obj.access(std::get<0>(_arg)) = _val;
      },
      "Set the element");
}

template <typename Tp, typename ViewT, size_t... Idx>
void generate_view_access(py::class_<ViewT> &_view, std::index_sequence<Idx...>,
                          enable_if_t<(sizeof...(Idx) > 1), int> = 0) {
  FOLD_EXPRESSION(_view.def("__getitem__", get_item<ViewT, Idx + 1>::get(),
                            "Get the element"));
  FOLD_EXPRESSION(_view.def("__setitem__",
                            get_item<ViewT, Idx + 1>::template set<Tp>(),
                            "Set the element"));
  _view.def(
      "__getitem__",
      [](ViewT &_obj, std::tuple<size_t> _arg) {
        return _obj.access(std::get<0>(_arg));
      },
      "Get the element");
  _view.def(
      "__setitem__",
      [](ViewT &_obj, std::tuple<size_t> _arg, Tp _val) {
        _obj.access(std::get<0>(_arg)) = _val;
      },
      "Set the element");
}

//----------------------------------------------------------------------------//
//
//                          Primary View generation function
//
//----------------------------------------------------------------------------//

// generic function to generate a view once the view type has been specified
template <typename ViewT, typename Sp, typename Tp, typename Lp, typename Mp,
          size_t DimIdx, size_t... Idx>
void generate_view(py::module &_mod, const std::string &_name,
                   const std::string &_msg, size_t _ndim = DimIdx + 1) {
  // some mirror views will instantiate types that were added already
  if (!add_pyclass<ViewT>()) return;

  if (debug_output())
    std::cerr << "Registering " << _msg << " as python class '" << _name
              << "'..." << std::endl;

  // class decl
  py::class_<ViewT> _view(_mod, _name.c_str(), py::buffer_protocol());

  // default initializer
  _view.def(py::init([]() { return new ViewT{}; }));

  // initializer with extents
  FOLD_EXPRESSION(get_init<ViewT, Idx + 1, Tp>(_view));

  // conversion to/from numpy
  _view.def_buffer([_ndim](ViewT &m) -> py::buffer_info {
    auto _extents = get_extents(m, std::make_index_sequence<DimIdx + 1>{});
    auto _strides = get_stride<Tp>(m, std::make_index_sequence<DimIdx + 1>{});
    auto _format  = get_format<Tp>();
    return py::buffer_info(m.data(),    // Pointer to buffer
                           sizeof(Tp),  // Size of one scalar
                           _format,     // Descriptor
                           _ndim,       // Number of dimensions
                           _extents,    // Buffer dimensions
                           _strides     // Strides (in bytes) for each index
    );
  });

  using mirror_type = typename ViewT::host_mirror_type;
  using mirror_cast = kokkos_python_view_type_t<mirror_type>;

  // if (!std::is_same<mirror_type, mirror_cast>::value) {
  //  py::implicitly_convertible<mirror_type, mirror_cast>();
  // }

  _view.def(
      "create_mirror",
      [](ViewT &_v, bool _copy) {
        auto _m = Kokkos::create_mirror(_v);
        if (_copy) Kokkos::deep_copy(_m, _v);
        return static_cast<mirror_cast>(_m);
      },
      "Create a host mirror (always creates a new view)",
      py::arg("copy") = true);

  _view.def(
      "create_mirror_view",
      [](ViewT &_v, bool _copy) {
        auto _m = Kokkos::create_mirror_view(_v);
        if (_copy) Kokkos::deep_copy(_m, _v);
        return static_cast<mirror_cast>(_m);
      },
      "Create a host mirror view (only creates new view if this is not on "
      "host)",
      py::arg("copy") = true);

  using view_type_list_t =
      std::conditional_t<Kokkos::is_dyn_rank_view<ViewT>::value,
                         dynamic_view_type_list_t, concrete_view_type_list_t>;

  deep_copy<ViewT>{_view}(view_type_list_t{});

  // shape property
  _view.def_property_readonly(
      "shape",
      [](ViewT &m) {
        return get_extents(m, std::make_index_sequence<DimIdx + 1>{});
      },
      "Get the shape of the array (extents)");

  _view.def_property_readonly(
      "ndim",
      [](ViewT &m) {
        auto &&_extents =
            get_extents(m, std::make_index_sequence<DimIdx + 1>{});
        if (Kokkos::is_dyn_rank_view<ViewT>::value) {
          size_t _ndim = 0;
          for (auto &&itr : _extents) {
            if (itr > 1) ++_ndim;
          }
          return _ndim;
        }
        return _extents.size();
      },
      "Get the number of allocated ranks of the array");

  _view.def_property_readonly(
      "space", [](ViewT &) { return MemorySpaceIndex<Sp>::value; },
      "Memory space of the view (alias for 'memory_space')");

  _view.def_property_readonly(
      "layout", [](ViewT &) { return MemoryLayoutIndex<Lp>::value; },
      "Memory layout of the view");

  _view.def_property_readonly(
      "trait", [](ViewT &) { return MemoryTraitIndex<Mp>::value; },
      "Memory trait of the view (alias for 'memory_trait')");

  _view.def_property_readonly(
      "memory_space", [](ViewT &) { return MemorySpaceIndex<Sp>::value; },
      "Memory space of the view (alias for 'space')");

  _view.def_property_readonly(
      "memory_trait", [](ViewT &) { return MemoryTraitIndex<Mp>::value; },
      "Memory trait of the view (alias for 'trait')");

  _view.def_property_readonly(
      "dynamic", [](ViewT &) { return Kokkos::is_dyn_rank_view<ViewT>::value; },
      "Whether the rank is dynamic");

  _view.def_property_readonly(
      "cpp_type", [=](ViewT &) { return _msg; },
      "Underlying C++ type as string");

  // support []
  generate_view_access<Tp>(_view, std::index_sequence<Idx...>{});
}

template <typename ViewT, typename Sp, typename Tp, typename Lp, typename Mp,
          size_t DimIdx, size_t... Idx>
void generate_view(py::module &_mod, const std::string &_name,
                   const std::string &_msg, size_t _ndim,
                   std::index_sequence<Idx...>) {
  generate_view<ViewT, Sp, Tp, Lp, Mp, DimIdx, Idx...>(_mod, _name, _msg,
                                                       _ndim);
}
}  // namespace Common
//
