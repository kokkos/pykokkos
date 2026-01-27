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

#include "KokkosExp_InterOp.hpp"
#include "user.hpp"

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <pybind11/pybind11.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <cstdlib>

namespace py = pybind11;

//----------------------------------------------------------------------------//
//
//        The python module
//
//----------------------------------------------------------------------------//

PYBIND11_MODULE(ex_generate, ex) {
  // Ensure the core Kokkos bindings are loaded so the view types are
  // registered before we expose functions that return them.
  (void)py::module::import("kokkos");

  ///
  /// This is a python binding to the user-defined generate_view function
  /// declared in user.hpp which returns a Kokkos::View. This function is called
  /// from ex-numpy.py
  ///
  ex.def(
      "generate_view",
      [](size_t n) {
        return Kokkos::Experimental::as_python_type(generate_view(n));
      },
      "Generate view");

  ex.def(
      "modify_view",
      [](Kokkos::Experimental::python_view_type_t<view_type> _v) {
        modify_view(_v);
      },
      "Generate view");

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}
