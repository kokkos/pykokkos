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

#include <cstdlib>
#include <initializer_list>
#include <set>
#include <string>
#include <type_traits>

//----------------------------------------------------------------------------//

#define FOLD_EXPRESSION(...) \
  ::consume_parameters(::std::initializer_list<int>{(__VA_ARGS__, 0)...})

//----------------------------------------------------------------------------//

#if !defined(EXPAND)
#  define EXPAND(...) __VA_ARGS__
#endif

//----------------------------------------------------------------------------//

#if defined(__GNUC__) || defined(__clang__)
#  define PYKOKKOS_HIDDEN __attribute__((visibility("hidden")))
#elif defined(__has_attribute)
#  if __has_attribute(visibility)
#    define PYKOKKOS_HIDDEN __attribute__((visibility("hidden")))
#  else
#    define PYKOKKOS_HIDDEN
#  endif
#else
#  define PYKOKKOS_HIDDEN
#endif

//----------------------------------------------------------------------------//

#if !defined(IF_CONSTEXPR)
#  if __cplusplus >= 201703L  // C++17
#    define IF_CONSTEXPR(...) if constexpr (__VA_ARGS__)
#  else
#    define IF_CONSTEXPR(...) if (__VA_ARGS__)
#  endif
#endif

//----------------------------------------------------------------------------//
//
//                          Number of views to instantiate
//
//----------------------------------------------------------------------------//

#if !defined(ENABLE_VIEW_RANKS)
#  define ENABLE_VIEW_RANKS 4
#endif

//----------------------------------------------------------------------------//
//
//                          type specialization macros
//
//----------------------------------------------------------------------------//

#define ENABLE_IMPLICIT(TYPE) \
  template <>                 \
  struct is_implicit<TYPE> : std::true_type {};

#define DISABLE_TYPE(TYPE) \
  template <>              \
  struct is_available<TYPE> : std::false_type {};

#define GET_FIRST_STRING(...)                         \
  static std::string _value = []() {                  \
    return std::get<0>(std::make_tuple(__VA_ARGS__)); \
  }();                                                \
  return _value

#define GET_STRING_SET(...)                               \
  static auto _value = []() {                             \
    auto _ret = std::set<std::string>{};                  \
    for (auto itr : std::set<std::string>{__VA_ARGS__}) { \
      if (!itr.empty()) {                                 \
        _ret.insert(itr);                                 \
      }                                                   \
    }                                                     \
    return _ret;                                          \
  }();                                                    \
  return _value

#define VIEW_DATA_DIMS(NUM, ...)        \
  template <typename T>                 \
  struct ViewDataTypeRepr<T, NUM - 1> { \
    using type = __VA_ARGS__;           \
  };

#define VIEW_DATA_TYPE(TYPE, ENUM_ID, ...)                        \
  template <>                                                     \
  struct ViewDataTypeSpecialization<ENUM_ID> {                    \
    using type = TYPE;                                            \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };

#define EXECUTION_SPACE_IDX(EXECUTION_SPACE, ENUM_ID) \
  template <>                                         \
  struct ExecutionSpaceIndex<EXECUTION_SPACE> {       \
    static constexpr auto value = ENUM_ID;            \
  };

#define MEMORY_SPACE_IDX(SPACE, ENUM_ID)   \
  template <>                              \
  struct MemorySpaceIndex<SPACE> {         \
    static constexpr auto value = ENUM_ID; \
  };

#define MEMORY_LAYOUT_IDX(LAYOUT, ENUM_ID) \
  template <>                              \
  struct MemoryLayoutIndex<LAYOUT> {       \
    static constexpr auto value = ENUM_ID; \
  };

#define MEMORY_TRAIT_IDX(TRAIT, ENUM_ID)   \
  template <>                              \
  struct MemoryTraitIndex<TRAIT> {         \
    static constexpr auto value = ENUM_ID; \
  };

#define EXECUTION_SPACE(SPACE, ENUM_ID, ...)                      \
  template <>                                                     \
  struct ExecutionSpaceSpecialization<ENUM_ID> {                  \
    using type = SPACE;                                           \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };                                                              \
  EXECUTION_SPACE_IDX(SPACE, ENUM_ID)

#define MEMORY_SPACE(SPACE, ENUM_ID, ...)                         \
  template <>                                                     \
  struct MemorySpaceSpecialization<ENUM_ID> {                     \
    using type = SPACE;                                           \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };                                                              \
  MEMORY_SPACE_IDX(SPACE, ENUM_ID)

#define MEMORY_LAYOUT(LAYOUT, ENUM_ID, LABEL)                 \
  template <>                                                 \
  struct MemoryLayoutSpecialization<ENUM_ID> {                \
    using type = LAYOUT;                                      \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  MEMORY_LAYOUT_IDX(LAYOUT, ENUM_ID)

#define MEMORY_TRAIT(TRAIT, ENUM_ID, LABEL)                   \
  template <>                                                 \
  struct MemoryTraitSpecialization<ENUM_ID> {                 \
    using type = TRAIT;                                       \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  MEMORY_TRAIT_IDX(TRAIT, ENUM_ID)

//----------------------------------------------------------------------------//
//
//                              miscellaneous
//
//----------------------------------------------------------------------------//

#define GET_STRIDE(IDX_NUM)                                             \
  template <size_t Idx, typename Tp, enable_if_t<(Idx == IDX_NUM)> = 0> \
  constexpr auto get_stride(Tp& m) {                                    \
    return m.stride(IDX_NUM);                                           \
  }
