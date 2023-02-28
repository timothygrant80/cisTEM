from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import sys

description = load(open(sys.argv[1]), Loader=Loader)

rec_function_dict = {
    "std::string": "ReturnStringArgument",
}

def user_input_function(argument):
    if argument['type'] == "std::string":
        return f"GetFilenameFromUser(\"{argument['description']}\", \"{argument['description']}\", \"{argument['default']}\", false)"
    else:
        raise Exception(f"Unknown type {argument['type']}")

def generate_type_string(arguments):
    type_string = "\""
    for argument in arguments:
        if argument['type'] == "std::string":
            type_string += "t"
        else:
            raise Exception(f"Unknown type {argument['type']}")
    type_string += "\""
    return type_string

print(f"class {description['name']} {{\n public:\n")

for argument in description['arguments']:
    print(f"  {argument['type']} {argument['name']};\n")


print(f"{description['name']}() {{\n")
for argument in description['arguments']:
    print(f"  {argument['name']} = \"{argument['default']}\";\n")
print("};")

print(f"void recieve(RunArgument* arguments) {{\n")
for i, argument in enumerate(description['arguments']):
    print(f"  {argument['name']} = arguments[{i}].{rec_function_dict[argument['type']]}( );\n")
print("};")

print(f"void userinput() {{\n")
print(f"UserInput* my_input = new UserInput(\"{description['name']}\", 1.00);\n")
for i, argument in enumerate(description['arguments']):
    
    print(f"  {argument['name']} = my_input->{user_input_function(argument)};\n")
print("delete my_input;\n};")

print(f"void setargument(RunJob& my_current_job) {{\n")
print("my_current_job.ManualSetArguments(",end='')
print(generate_type_string(description['arguments']))
print
for i, argument in enumerate(description['arguments']):
    print(f",{argument['name']}.c_str()")
print(");\n};")


print("};")