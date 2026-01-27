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

#include <regex>

//----------------------------------------------------------------------------//

std::string demangle(const char *_cstr) {
#if defined(ENABLE_DEMANGLE)
  // demangling a string when delimiting
  int _ret      = 0;
  char *_demang = abi::__cxa_demangle(_cstr, 0, 0, &_ret);
  if (_demang && _ret == 0)
    return std::string{const_cast<const char *>(_demang)};
  else
    return _cstr;
#else
  return _cstr;
#endif
}

//----------------------------------------------------------------------------//

std::string demangle(const std::string &_str) { return demangle(_str.c_str()); }

//----------------------------------------------------------------------------//

std::string Impl::remove_type_list_wrapper(std::string _tmp) {
  auto _key = std::string{"type_list"};
  auto _idx = _tmp.find(_key);
  _idx      = _tmp.find("<", _idx);
  _tmp      = _tmp.substr(_idx + 1);
  _idx      = _tmp.find_last_of(">");
  _tmp      = _tmp.substr(0, _idx);
  // strip trailing whitespaces
  while ((_idx = _tmp.find_last_of(" ")) == _tmp.length() - 1)
    _tmp = _tmp.substr(0, _idx);
  return _tmp;
}

//----------------------------------------------------------------------------//

std::set<std::string> &get_existing_pyclass_names() {
  static std::set<std::string> _instance{};
  return _instance;
}

//----------------------------------------------------------------------------//

bool debug_output() {
  static auto _value = []() {
#if !defined(NDEBUG) || defined(DEBUG)
    bool val = true;
#else
    bool val = false;
#endif
    auto _env_value = std::getenv("DEBUG_OUTPUT");
    if (_env_value) {
      auto var = std::string{_env_value};
      if (var.find_first_not_of("0123456789") == std::string::npos) {
        val = (std::stoi(var) == 0) ? false : true;
      } else if (std::regex_match(var,
                                  std::regex("^(off|false|no|n|f)$",
                                             std::regex_constants::icase))) {
        val = false;
      } else if (std::regex_match(var, std::regex("^(on|true|yes|y|t)$",
                                                  std::regex_constants::icase)))
        val = true;
    }
    return val;
  }();
  return _value;
}
