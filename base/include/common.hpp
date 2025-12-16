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

#include "defines.hpp"

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace py = pybind11;

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>

#if defined(ENABLE_DEMANGLE)
#  include <cxxabi.h>
#endif

//----------------------------------------------------------------------------//

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename Tp>
using decay_t = typename std::decay<Tp>::type;

//----------------------------------------------------------------------------//

template <typename...>
struct type_list {};

//----------------------------------------------------------------------------//

template <typename... Args>
void consume_parameters(Args &&...) {}

//----------------------------------------------------------------------------//

namespace Impl {
// replaces type_list<CONTENTS> with CONTENTS
std::string remove_type_list_wrapper(std::string);
}  // namespace Impl

//----------------------------------------------------------------------------//

std::string demangle(const char *_cstr);

//----------------------------------------------------------------------------//

std::string demangle(const std::string &_str);

//----------------------------------------------------------------------------//

template <typename Tp>
inline std::string demangle() {
  // static because a type demangle will always be the same
  static auto _val =
      Impl::remove_type_list_wrapper(demangle(typeid(type_list<Tp>).name()));
  return _val;
}

//----------------------------------------------------------------------------//

template <typename... Args>
std::string join(const std::string &delim, Args &&...args) {
  auto _construct = [&](auto &&_arg) {
    std::ostringstream _ss;
    _ss << std::forward<decltype(_arg)>(_arg);
    if (_ss.str().length() > 0) return delim + _ss.str();
    return std::string{};
  };
  std::ostringstream ss;
  FOLD_EXPRESSION(ss << _construct(std::forward<Args>(args)));
  return ss.str().substr(delim.length());
}

//----------------------------------------------------------------------------//

std::set<std::string> &get_existing_pyclass_names();

//----------------------------------------------------------------------------//

template <typename Tp>
inline auto add_pyclass() {
  if (get_existing_pyclass_names().count(demangle<Tp>()) > 0) return false;
  get_existing_pyclass_names().insert(demangle<Tp>());
  return true;
}

//----------------------------------------------------------------------------//

bool debug_output();
