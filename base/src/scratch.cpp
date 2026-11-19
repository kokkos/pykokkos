#include "scratch.hpp"

#include <Kokkos_Core.hpp>

#include "traits.hpp"

namespace {
template <size_t SpaceIdx>
void bind_scratch_size_max(py::module& kokkos) {
  using space = execution_space_t<SpaceIdx>;
  if constexpr (is_available<space>::value) {
    kokkos.def(
        "scratch_size_max_for_space",
        [](const space&, int level) {
          return static_cast<int>(
              Kokkos::TeamPolicy<space>::scratch_size_max(level));
        },
        py::arg("space"), py::arg("level") = 0);
  }
}

template <size_t... SpaceIdx>
void bind_scratch_sizes(py::module& kokkos, std::index_sequence<SpaceIdx...>) {
  (bind_scratch_size_max<SpaceIdx>(kokkos), ...);
}
}  // namespace

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
  bind_scratch_sizes(kokkos, std::make_index_sequence<ExecutionSpacesEnd>{});
}
