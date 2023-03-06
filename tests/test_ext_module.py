import unittest

from pathlib import Path
import pykokkos as pk


# Tests if pk is able to compile an external module

class TestExternalModule(unittest.TestCase):
    def setUp(self):
        self.path: Path = Path('./')
        self.module_name = "test_ext_module"
        self.source: List[str] = []

        self.source.append("#include <pybind11/pybind11.h>\n")
        self.source.append(f"PYBIND11_MODULE({self.module_name},m){{")
        self.source.append(f"m.attr(\"__name__\") = \"{self.module_name}\";")
        self.source.append("m.def(\"get_five\",[](){return 5;});")
        self.source.append("}") 
        self.ext_module = pk.compile_into_module(self.path,self.source,self.module_name)

    def test_call(self):
        self.assertEqual(5, self.ext_module.get_five())

if __name__ == '__main__':
    unittest.main()
