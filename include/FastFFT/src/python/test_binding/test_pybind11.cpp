#include <pybind11/pybind11.h>

namespace py = pybind11;

template<class TypeOne, class TypeTwo>
class TestClass {

    public:
        TestClass(TypeOne one, TypeTwo two) : one_(one), two_(two) {}

        TypeOne getOne() { return one_; }
        TypeTwo getTwo() { return two_; }

        int add(TypeOne i, TypeTwo j) {
            return int(i + j);
        }

    private:
        TypeOne one_;
        TypeTwo two_;
};

template<typename typeOne, typename typeTwo>
void declare_array(py::module &m, const std::string &typestr) {
    using Class = TestClass<typeOne, typeTwo>;
    std::string pyclass_name = std::string("TestClass") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
    .def(py::init<typeOne, typeTwo>())
    .def("getOne", &TestClass<typeOne, typeTwo>::getOne) 
    .def("getTwo", &TestClass<typeOne, typeTwo>::getTwo) 
    .def("add", &TestClass<typeOne, typeTwo>::add);
}

PYBIND11_MODULE(fastfft_test, m) {
    
    declare_array<int, float>(m, "_int_float");
    declare_array<float, int>(m, "_float_int");
}
