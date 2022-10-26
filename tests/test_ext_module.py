import unittest

import pykokkos as pk


# Tests if pk is able to compile an external module

class TestExternalModule(unittest.TestCase):
    def setUp(self):
        self.path: Path = Path('./')
        self.module_name = "test_ext_module"
        self.source: List[str] = []

        source.append("#include <pybind11/pybind11.h>\n")
        source.append(f"PYBIND11_MODULE({module_name},m){")
        source.append(f"m.attr(\"__name__\") = \"{module_name}\";")
        source.append("m.def(\"get_five\",[](){return 5;});")
        source.append("}") 

    def test_compile(self):
        self.ext_module = pk.runtime_singleton.runtime.compile_into_module(path,source,module_name,pk.ExecutionSpace.Default)

    def test_call(self):
        self.assertEqual(5, self.ext_module.get_five())

if __name__ == '__main__':
    unittest.main()
