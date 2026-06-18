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

#include <Kokkos_Core.hpp>

#include "common.hpp"
#include "defines.hpp"
#include "fwd.hpp"
#include "traits.hpp"

//----------------------------------------------------------------------------//
//
//        Kokkos Tools submodule
//
//----------------------------------------------------------------------------//

using device_info_t  = struct Kokkos_Profiling_KokkosPDeviceInfo;
using space_handle_t = struct Kokkos_Profiling_SpaceHandle;

using initFunction     = std::function<void(const int, const uint64_t,
                                        const uint32_t, device_info_t*)>;
using finalizeFunction = std::function<void()>;
using parseArgsFunction =
    std::function<void(int, const std::vector<const char*>&)>;
using printHelpFunction = std::function<void(char*)>;
using beginFunction     = std::function<uint64_t(const char*, const uint32_t)>;
using endFunction       = std::function<void(uint64_t)>;
using pushFunction      = std::function<void(const char*)>;
using popFunction       = std::function<void()>;
using allocateDataFunction          = std::function<void(
    const space_handle_t, const char*, const void*, const uint64_t)>;
using deallocateDataFunction        = std::function<void(
    const space_handle_t, const char*, const void*, const uint64_t)>;
using createProfileSectionFunction  = std::function<uint32_t(const char*)>;
using startProfileSectionFunction   = std::function<void(const uint32_t)>;
using stopProfileSectionFunction    = std::function<void(const uint32_t)>;
using destroyProfileSectionFunction = std::function<void(const uint32_t)>;
using profileEventFunction          = std::function<void(const char*)>;
using beginDeepCopyFunction =
    std::function<void(space_handle_t, const char*, const void*, space_handle_t,
                       const char*, const void*, uint64_t)>;
using endDeepCopyFunction = std::function<void()>;
using beginFenceFunction = std::function<uint64_t(const char*, const uint32_t)>;
using endFenceFunction   = std::function<void(uint64_t)>;

template <typename Ret, typename... Args>
auto pyfunction_wrapper(std::function<Ret(Args...)>& _cb, py::object _func,
                        std::enable_if_t<!std::is_void<Ret>::value, int> = 0) {
  if (_func.is_none()) {
    _cb = [](Args...) -> Ret { return Ret{}; };
  } else {
    _cb = [_func](Args... args) -> Ret {
      return _func(args...).template cast<Ret>();
    };
  }
}

template <typename Ret, typename... Args>
auto pyfunction_wrapper(std::function<Ret(Args...)>& _cb, py::object _func,
                        std::enable_if_t<std::is_void<Ret>::value, long> = 0) {
  if (_func.is_none()) {
    _cb = [](Args...) {};
  } else {
    _cb = [_func](Args... args) { _func(args...); };
  }
}

template <typename Ret, typename... Args>
auto cppfunction_wrapper(std::function<Ret(Args...)>& _cb,
                         std::function<Ret(Args...)> _func,
                         std::enable_if_t<!std::is_void<Ret>::value, int> = 0) {
  if (_func)
    _cb = std::move(_func);
  else
    _cb = [](Args...) -> Ret { return Ret{}; };
}

template <typename Ret, typename... Args>
auto cppfunction_wrapper(std::function<Ret(Args...)>& _cb,
                         std::function<Ret(Args...)> _func,
                         std::enable_if_t<std::is_void<Ret>::value, long> = 0) {
  if (_func)
    _cb = std::move(_func);
  else
    _cb = [](Args...) {};
}

namespace {
struct internal_callbacks {
  internal_callbacks() = default;
  ~internal_callbacks() {}

  internal_callbacks(const internal_callbacks&)            = delete;
  internal_callbacks& operator=(const internal_callbacks&) = delete;

  internal_callbacks(internal_callbacks&&)            = default;
  internal_callbacks& operator=(internal_callbacks&&) = default;

  initFunction init;
  finalizeFunction fini;
  parseArgsFunction parse_args;
  printHelpFunction print_help;
  std::array<beginFunction, 3> begin_parallel;
  std::array<endFunction, 3> end_parallel;
  pushFunction push;
  popFunction pop;
  allocateDataFunction alloc_data;
  deallocateDataFunction dealloc_data;
  createProfileSectionFunction create_prof;
  destroyProfileSectionFunction destroy_prof;
  startProfileSectionFunction start_prof;
  stopProfileSectionFunction stop_prof;
  profileEventFunction prof_event;
  beginDeepCopyFunction begin_deep_copy;
  endDeepCopyFunction end_deep_copy;
  beginFenceFunction begin_fence;
  endFenceFunction end_fence;
  std::function<void(const char*, const char*)> metadata;
};
}  // namespace
//
static auto callbacks = std::make_unique<internal_callbacks>();
//
namespace pykokkos_tools {
void print_help(char* argv) {
  if (callbacks && callbacks->print_help) callbacks->print_help(argv);
}

void parse_args(int argc, char** argv) {
  if (callbacks && callbacks->parse_args) {
    std::vector<const char*> _argv{};
    _argv.reserve(argc);
    for (int i = 0; i < argc; ++i) _argv.emplace_back(argv[i]);
    callbacks->parse_args(argc, _argv);
  }
}

void declare_metadata(const char* key, const char* value) {
  if (callbacks && callbacks->metadata) callbacks->metadata(key, value);
}

void init_library(const int loadSeq, const uint64_t interfaceVer,
                  const uint32_t devInfoCount, device_info_t* deviceInfo) {
  if (callbacks && callbacks->init)
    callbacks->init(loadSeq, interfaceVer, devInfoCount, deviceInfo);
}

void finalize_library() {
  if (callbacks && callbacks->fini) {
    Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
    callbacks->fini();
  }
}

//----------------------------------------------------------------------------------//

void begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid) {
  if (callbacks && callbacks->begin_parallel[0])
    *kernid = callbacks->begin_parallel[0](name, devid);
}

void end_parallel_for(uint64_t kernid) {
  if (callbacks && callbacks->end_parallel[0])
    callbacks->end_parallel[0](kernid);
}

//----------------------------------------------------------------------------------//

void begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid) {
  if (callbacks && callbacks->begin_parallel[1])
    *kernid = callbacks->begin_parallel[1](name, devid);
}

void end_parallel_reduce(uint64_t kernid) {
  if (callbacks && callbacks->end_parallel[1])
    callbacks->end_parallel[1](kernid);
}

//----------------------------------------------------------------------------------//

void begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid) {
  if (callbacks && callbacks->begin_parallel[2])
    *kernid = callbacks->begin_parallel[2](name, devid);
}

void end_parallel_scan(uint64_t kernid) {
  if (callbacks && callbacks->end_parallel[2])
    callbacks->end_parallel[2](kernid);
}

//----------------------------------------------------------------------------------//

void begin_fence(const char* name, uint32_t devid, uint64_t* kernid) {
  if (callbacks && callbacks->begin_fence)
    *kernid = callbacks->begin_fence(name, devid);
}

void end_fence(uint64_t kernid) {
  if (callbacks && callbacks->end_fence) callbacks->end_fence(kernid);
}

//----------------------------------------------------------------------------------//

void push_profile_region(const char* name) {
  if (callbacks && callbacks->push) callbacks->push(name);
}

void pop_profile_region() {
  if (callbacks && callbacks->pop) callbacks->pop();
}

//----------------------------------------------------------------------------------//

void create_profile_section(const char* name, uint32_t* secid) {
  if (callbacks && callbacks->create_prof)
    *secid = callbacks->create_prof(name);
}

void destroy_profile_section(uint32_t secid) {
  if (callbacks && callbacks->destroy_prof) callbacks->destroy_prof(secid);
}

//----------------------------------------------------------------------------------//

void start_profile_section(uint32_t secid) {
  if (callbacks && callbacks->start_prof) callbacks->start_prof(secid);
}

void stop_profile_section(uint32_t secid) {
  if (callbacks && callbacks->stop_prof) callbacks->stop_prof(secid);
}

//----------------------------------------------------------------------------------//

void allocate_data(const space_handle_t space, const char* label,
                   const void* const ptr, const uint64_t size) {
  if (callbacks && callbacks->alloc_data)
    callbacks->alloc_data(space, label, ptr, size);
}

void deallocate_data(const space_handle_t space, const char* label,
                     const void* const ptr, const uint64_t size) {
  if (callbacks && callbacks->dealloc_data)
    callbacks->dealloc_data(space, label, ptr, size);
}

//----------------------------------------------------------------------------------//

void begin_deep_copy(space_handle_t dst_handle, const char* dst_name,
                     const void* dst_ptr, space_handle_t src_handle,
                     const char* src_name, const void* src_ptr, uint64_t size) {
  if (callbacks && callbacks->begin_deep_copy)
    callbacks->begin_deep_copy(dst_handle, dst_name, dst_ptr, src_handle,
                               src_name, src_ptr, size);
}

void end_deep_copy() {
  if (callbacks && callbacks->end_deep_copy) callbacks->end_deep_copy();
}

//----------------------------------------------------------------------------------//

void profile_event(const char* name) {
  if (callbacks) callbacks->prof_event(name);
}
}  // namespace pykokkos_tools

void internal_test();
void internal_setup();

//----------------------------------------------------------------------------------//
//
void destroy_callbacks() { callbacks.reset(); }

void generate_tools(py::module& kokkos) {
  //--------------------------------------------------------------------//
  //
  //                        Lambdas
  //
  //--------------------------------------------------------------------//
  auto _devinfo_init = []() { return new device_info_t{}; };
  auto _devinfo_read = [](device_info_t& _obj) { return _obj.deviceID; };

  auto _space_init0 = []() { return new space_handle_t{}; };
  auto _space_init1 = [](const std::string& _lbl) {
    auto _handle = new space_handle_t{};
    memcpy(_handle->name, _lbl.c_str(), std::min<size_t>(_lbl.length(), 64));
    _handle->name[63] = '\0';
    return _handle;
  };
  auto _space_read  = [](space_handle_t& _obj) { return _obj.name; };
  auto _space_write = [](space_handle_t& _obj, const std::string& _lbl) {
    memcpy(_obj.name, _lbl.c_str(), std::min<size_t>(_lbl.length(), 64));
    _obj.name[63] = '\0';
  };

  //--------------------------------------------------------------------//
  //
  //                          Submodules
  //
  //--------------------------------------------------------------------//
  auto _tools = kokkos.def_submodule("tools", "Kokkos Tools interface");

  auto _internal =
      _tools.def_submodule("_internal", "Internal module for unittest");

  _internal.def("setup", &internal_setup, "Setup the internal testing suite");
  _internal.def("test", &internal_test,
                "Equivalent to core/unit_test/tools/TestAllCalls.cpp");

  //--------------------------------------------------------------------//
  //
  //                           Classes
  //
  //--------------------------------------------------------------------//
  py::class_<device_info_t> _devInfo(_tools, "DeviceInfo");
  _devInfo.def(py::init(_devinfo_init), "Initialize");
  _devInfo.def_property_readonly("deviceID", _devinfo_read, "device ID");

  py::class_<space_handle_t> _spaceHandle(_tools, "SpaceHandle");
  _spaceHandle.def(py::init(_space_init0), "Initialize");
  _spaceHandle.def(py::init(_space_init1), "Initialize");
  _spaceHandle.def_property("name", _space_read, _space_write, "name");

  //--------------------------------------------------------------------//
  //
  //                           Functions
  //
  //--------------------------------------------------------------------//
  _tools.def("profile_library_loaded", &Kokkos::Tools::profileLibraryLoaded,
             "Query whether a profiling library is loaded");

  _tools.def("push_region", &Kokkos::Tools::pushRegion,
             "Generate a region in tools");

  _tools.def("pop_region", &Kokkos::Tools::popRegion,
             "Terminate last region in tools");

  _tools.def(
      "create_profile_section",
      [](std::string key) {
        uint32_t _idx = 0;
        return (Kokkos::Tools::createProfileSection(key.c_str(), &_idx), _idx);
      },
      "Create a profiling section");

  _tools.def("destroy_profile_section", &Kokkos::Tools::destroyProfileSection,
             "Destroy a profiling section");

  _tools.def("start_section", &Kokkos::Tools::startSection,
             "Start a profiling section");

  _tools.def("stop_section", &Kokkos::Tools::stopSection,
             "Stop a profiling section");

  _tools.def("mark_event", &Kokkos::Tools::markEvent, "Mark an event");

  _tools.def("declare_metadata", &Kokkos::Tools::declareMetadata,
             "Declare some metadata");

  //--------------------------------------------------------------------//
  //
  //                           Set functions
  //
  //--------------------------------------------------------------------//

#define TOOL_SET_CALLBACK(NAME, FUNC, REF)                         \
  _tools.def(                                                      \
      #NAME,                                                       \
      [](py::object _func) {                                       \
        if (!callbacks) return;                                    \
        pyfunction_wrapper(callbacks->REF, _func);                 \
        if (_func.is_none()) {                                     \
          Kokkos::Tools::Experimental::NAME(nullptr);              \
        } else {                                                   \
          Kokkos::Tools::Experimental::NAME(pykokkos_tools::FUNC); \
        }                                                          \
      },                                                           \
      "");                                                         \
  _tools.def(                                                      \
      #NAME,                                                       \
      [](decltype(callbacks->REF) _func) {                         \
        if (!callbacks) return;                                    \
        cppfunction_wrapper(callbacks->REF, _func);                \
        if (!_func) {                                              \
          Kokkos::Tools::Experimental::NAME(nullptr);              \
        } else {                                                   \
          Kokkos::Tools::Experimental::NAME(pykokkos_tools::FUNC); \
        }                                                          \
      },                                                           \
      "");

  TOOL_SET_CALLBACK(set_init_callback, init_library, init)
  TOOL_SET_CALLBACK(set_finalize_callback, finalize_library, fini)
  TOOL_SET_CALLBACK(set_parse_args_callback, parse_args, parse_args)
  TOOL_SET_CALLBACK(set_print_help_callback, print_help, print_help)
  TOOL_SET_CALLBACK(set_begin_parallel_for_callback, begin_parallel_for,
                    begin_parallel[0])
  TOOL_SET_CALLBACK(set_end_parallel_for_callback, end_parallel_for,
                    end_parallel[0])
  TOOL_SET_CALLBACK(set_begin_parallel_reduce_callback, begin_parallel_reduce,
                    begin_parallel[1])
  TOOL_SET_CALLBACK(set_end_parallel_reduce_callback, end_parallel_reduce,
                    end_parallel[1])
  TOOL_SET_CALLBACK(set_begin_parallel_scan_callback, begin_parallel_scan,
                    begin_parallel[2])
  TOOL_SET_CALLBACK(set_end_parallel_scan_callback, end_parallel_scan,
                    end_parallel[2])
  TOOL_SET_CALLBACK(set_push_region_callback, push_profile_region, push)
  TOOL_SET_CALLBACK(set_pop_region_callback, pop_profile_region, pop)
  TOOL_SET_CALLBACK(set_allocate_data_callback, allocate_data, alloc_data)
  TOOL_SET_CALLBACK(set_deallocate_data_callback, deallocate_data, dealloc_data)
  TOOL_SET_CALLBACK(set_create_profile_section_callback, create_profile_section,
                    create_prof)
  TOOL_SET_CALLBACK(set_start_profile_section_callback, start_profile_section,
                    start_prof)
  TOOL_SET_CALLBACK(set_stop_profile_section_callback, stop_profile_section,
                    stop_prof)
  TOOL_SET_CALLBACK(set_destroy_profile_section_callback,
                    destroy_profile_section, destroy_prof)
  TOOL_SET_CALLBACK(set_profile_event_callback, profile_event, prof_event)
  TOOL_SET_CALLBACK(set_begin_deep_copy_callback, begin_deep_copy,
                    begin_deep_copy)
  TOOL_SET_CALLBACK(set_end_deep_copy_callback, end_deep_copy, end_deep_copy)
  TOOL_SET_CALLBACK(set_begin_fence_callback, begin_fence, begin_fence)
  TOOL_SET_CALLBACK(set_end_fence_callback, end_fence, end_fence)
}

using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space    = typename execution_space::memory_space;

namespace {
//
auto& get_src_view() {
  static auto _v =
      std::make_unique<Kokkos::View<int32_t*, memory_space>>("cxx_source", 10);
  return _v;
}

auto& get_dst_view() {
  static auto _v =
      std::make_unique<Kokkos::View<int32_t*, memory_space>>("cxx_target", 10);
  return _v;
}
}  // namespace

void internal_setup() {
  // initialize before we attach parallel callbacks
  (void)get_src_view();
  (void)get_dst_view();
}

void internal_test() {
  // This test only uses host kernel launch mechanisms. This is to allow for
  // the test to run on platforms where CUDA lambda launch isn't supported.
  // This is safe because this test only seeks to test that the dlsym-based
  // tool loading mechanisms work, all of which happens completely
  // independently of the enabled backends
  auto& src_view = *get_src_view();
  auto& dst_view = *get_dst_view();
  Kokkos::deep_copy(dst_view, src_view);
  Kokkos::parallel_for("cxx_parallel_for",
                       Kokkos::RangePolicy<execution_space>(0, 1),
                       [=](int i) { (void)i; });
  int result;
  Kokkos::parallel_reduce(
      "cxx_parallel_reduce", Kokkos::RangePolicy<execution_space>(0, 1),
      [=](int i, int& hold_result) { hold_result += i; }, result);
  Kokkos::parallel_scan("cxx_parallel_scan",
                        Kokkos::RangePolicy<execution_space>(0, 1),
                        [=](const int i, int& hold_result, const bool final) {
                          if (final) {
                            hold_result += i;
                          }
                        });
  Kokkos::Tools::pushRegion("cxx_push_region");
  Kokkos::fence();
  Kokkos::Tools::popRegion();
  uint32_t sectionId;
  Kokkos::Tools::createProfileSection("cxx_created_section", &sectionId);
  Kokkos::Tools::startSection(sectionId);
  Kokkos::Tools::stopSection(sectionId);
  Kokkos::Tools::destroyProfileSection(sectionId);
  Kokkos::Tools::markEvent("cxx_profiling_event");
  Kokkos::Tools::declareMetadata("cats", "good");

  // reset unique pointers
  get_src_view().reset();
  get_dst_view().reset();
}
