
#include <cxxabi.h>
#include <string>
#include <iostream>

inline std::string demangle(const char* _cstr) {
  // demangling a string when delimiting
  int _ret      = 0;
  char* _demang = abi::__cxa_demangle(_cstr, 0, 0, &_ret);
  if (_demang && _ret == 0)
    return std::string(const_cast<const char*>(_demang));
  else
    return _cstr;
}

template <typename Tp>
inline std::string demangle() {
  return demangle(typeid(Tp).name());
}

int main()
{
    auto str = demangle<double>();
    std::cout << str << std::endl;
}
