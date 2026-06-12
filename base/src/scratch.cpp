#include "scratch.hpp"

#include <Kokkos_Core.hpp>

void generate_scratch(py::module& kokkos) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;

  kokkos.def(
      "scratch_size_max",
      [](int level) {
        return static_cast<int>(team_policy::scratch_size_max(level));
      },
      py::arg("level") = 0,
      "Returns the maximum total scratch size in bytes for the given level "
      "using Kokkos::TeamPolicy<DefaultExecutionSpace>::scratch_size_max. "
      "Note: If a kernel performs team-level reductions or scan operations, "
      "not all of this memory is available for dynamic user requests.");
}
