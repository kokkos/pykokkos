#pragma once

#include "common.hpp"

namespace py = pybind11;

void generate_scratch(py::module& kokkos);
