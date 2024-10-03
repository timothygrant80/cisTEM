#include <pybind11/pybind11.h>

#include <iostream>
#include <chrono>
#include <string>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftdx.hpp>

#include "../../../include/FastFFT.h"
#include "../../../include/FastFFT.cuh"

namespace py = pybind11;

template <class TypeOne, class TypeTwo>
__global__ void add_kernel(TypeOne one, TypeTwo two, TypeOne& retval) {

    // add then numbers and 1 to be sure it ran in this device code.
    retval = one + TypeOne(two) + TypeOne(1.0f);
}

template <class TypeOne>
__global__ void sum_array(TypeOne* array_ptr, int n_elem) {

    for ( int i = 1; i < n_elem; i++ ) {
        array_ptr[0] += array_ptr[i];
    }
}

template <class TypeOne, class TypeTwo>
class TestClass {

  public:
    TestClass(TypeOne one, TypeTwo two) : one_(one), two_(two) {}

    TypeOne getOne( ) { return one_; }

    TypeTwo getTwo( ) { return two_; }

    TypeOne add(TypeOne i, TypeTwo j) {

        TypeOne  retval = TypeOne(4);
        TypeOne* d_retval;
        cudaErr(cudaMallocManaged(&d_retval, sizeof(TypeOne)));

        std::cout << "Value pre kernel is " << retval << std::endl;
        precheck;
        add_kernel<<<1, 1, 0, cudaStreamPerThread>>>(i, j, *d_retval);
        postcheck;
        cudaStreamSynchronize(cudaStreamPerThread);

        retval = *d_retval;
        std::cout << "Value post kernel is " << retval << std::endl;

        cudaErr(cudaFree(d_retval));
        return retval;
    }

    void sum_cupy_array(long cupy_ptr, int cupy_size) {

        // Simple test to take the pointer from a cupy array and work on it in the gpu.
        TypeOne* d_array = reinterpret_cast<TypeOne*>(cupy_ptr);
        precheck;
        sum_array<<<1, 1, 0, cudaStreamPerThread>>>(d_array, cupy_size);
        postcheck;
        cudaStreamSynchronize(cudaStreamPerThread);
    }

  private:
    TypeOne one_;
    TypeTwo two_;
};

template <typename typeOne, typename typeTwo>
void declare_array(py::module& m, const std::string& typestr) {
    using Class              = TestClass<typeOne, typeTwo>;
    std::string pyclass_name = std::string("TestClass") + typestr;
    py::class_<Class>(m, pyclass_name.c_str( ))
            .def(py::init<typeOne, typeTwo>( ))
            .def("getOne", &TestClass<typeOne, typeTwo>::getOne)
            .def("getTwo", &TestClass<typeOne, typeTwo>::getTwo)
            .def("add", &TestClass<typeOne, typeTwo>::add)
            .def("sum_cupy_array", &TestClass<typeOne, typeTwo>::sum_cupy_array);
}

PYBIND11_MODULE(fastfft_test, m) {

    declare_array<int, float>(m, "_int_float");
    declare_array<float, int>(m, "_float_int");
    declare_array<float, float>(m, "_float_float");
}
