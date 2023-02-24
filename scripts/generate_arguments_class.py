from code_generation.cpp_class import CppClass
from code_generation.cpp_variable import CppVariable
from code_generation.cpp_function import CppFunction
from code_generation.code_generator import CppFile
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import sys

description = load(open(sys.argv[1]), Loader=Loader)

class_o = CppClass(name=description['name'])

for argument in description['arguments']:
    class_o.add_variable(CppVariable(name=argument['name'], type=argument['type']))
    class_o.add_method(CppClass.CppMethod(name="get_"+argument['name'], ret_type=argument['type'], implementation_handle=lambda self, cpp: cpp("return " + argument['name'] + ";")))
h = CppFile("test.h")

print(class_o.render_to_string_declaration(h))
c = CppFile("test.cpp")
print(class_o.render_to_string_implementation(c))