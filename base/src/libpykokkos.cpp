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

#include "libpykokkos.hpp"

#include <Kokkos_Core.hpp>
#include <iostream>

#include "common.hpp"
#include "defines.hpp"
#include "fwd.hpp"
#include "traits.hpp"

//----------------------------------------------------------------------------//
//
//        The python module
//
//----------------------------------------------------------------------------//

PYBIND11_MODULE(libpykokkos, kokkos) {
  kokkos.doc() =
      "Python bindings to critical Kokkos functions, Kokkos data strucures, "
      "and tools";

  // Initialize kokkos
  auto _initialize = [&]() {
    if (Kokkos::is_initialized()) return false;
    if (debug_output()) std::cerr << "Initializing Kokkos..." << std::endl;
    // python system module
    py::module sys = py::module::import("sys");
    // get the arguments for python system module
    py::object args = sys.attr("argv");
    auto argv       = args.cast<py::list>();
    int _argc       = argv.size();
    char** _argv    = new char*[argv.size()];
    for (int i = 0; i < _argc; ++i) {
      auto _args = argv[i].cast<std::string>();
      if (_args == "--") {
        for (int j = i; j < _argc; ++j) _argv[i] = nullptr;
        _argc = i;
        break;
      }
      _argv[i] = strdup(_args.c_str());
    }
    Kokkos::initialize(_argc, _argv);
    for (int i = 0; i < _argc; ++i) free(_argv[i]);
    delete[] _argv;
    return true;
  };

  // Finalize kokkos
  auto _finalize = []() {
    if (!Kokkos::is_initialized()) return false;
    if (debug_output()) std::cerr << "Finalizing Kokkos..." << std::endl;
    destroy_callbacks();
    Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
    py::module gc = py::module::import("gc");
    gc.attr("collect")();
    Kokkos::finalize();
    return true;
  };

  std::atexit(destroy_callbacks);

  kokkos.def("is_initialized", &Kokkos::is_initialized,
             "Query initialization state");
  kokkos.def("is_finalized", &Kokkos::is_finalized, "Query finalization state");
  kokkos.def("initialize", _initialize, "Initialize Kokkos");
  kokkos.def("finalize", _finalize, "Finalize Kokkos");

  generate_tools(kokkos);
  generate_available(kokkos);
  generate_enumeration(kokkos);
  generate_view_variants(kokkos);
  generate_atomic_variants(kokkos);
  generate_backend_versions(kokkos);
  generate_pool_variants(kokkos);
  generate_execution_spaces(kokkos);
  generate_complex_dtypes(kokkos);
}
