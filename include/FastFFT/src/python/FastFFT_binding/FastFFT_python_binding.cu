#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// py::bind_vector<std::vector<int>>(m, "VectorInt");

#include <iostream>
#include <chrono>
#include <string>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>
#include "../../../include/cufftdx/include/cufftdx.hpp"
#include "../../../include/FastFFT.h"
#include "../../../include/FastFFT.cuh"
#include "../fastfft/FastFFT.cu"

namespace py = pybind11;

template <class ComputeType, class InputType, class OtherImageType>
void declare_array(py::module& m, const std::string& typestr) {

    using FT_t = FastFFT::FourierTransformer<ComputeType, InputType, OtherImageType>;

    std::string pyclass_name = std::string("FourierTransformer") + typestr;
    py::class_<FT_t>(m, pyclass_name.c_str( ))
            // Constructor and initialization functions
            .def(py::init<>( ))
            // TODO: I have a virtual destructor, check what should be done here.
            .def("SetForwardFFTPlan", &FT_t::SetForwardFFTPlan)
            .def("SetInverseFFTPlan", &FT_t::SetInverseFFTPlan)
            .def("SetInputPointerFromPython", py::overload_cast<long>(&FT_t::SetInputPointerFromPython))

            // Memory operations: For now, assume user has cupy or torch to handle getting the data to and from the GPU.
            // TODO: may need to check on whether the data block is reliably contiguous.
            // CopyHostToDevice, CopyDeviceToHost, CopyDeviceToHost

            // FFT operations
            .def("FwdFFT", &FT_t::FwdFFT)
            .def("InvFFT", &FT_t::InvFFT)

            // Cross-correlation operations
            // TODO: confirm overload resolution is working (r/n float2* and __half2* are the first args)
            // I think there is a float2 built in to  numpy  but how to get pybind11 to recognize it?
            // .def("CrossCorrelate", &FT_t::CrossCorrelate)

            // Getters
            .def("ReturnInputMemorySize", &FT_t::ReturnInputMemorySize)
            .def("ReturnFwdOutputMemorySize", &FT_t::ReturnFwdOutputMemorySize)
            .def("ReturnInvOutputMemorySize", &FT_t::ReturnInvOutputMemorySize)

            .def("ReturnFwdInputDimensions", &FT_t::ReturnFwdInputDimensions)
            .def("ReturnInvInputDimensions", &FT_t::ReturnInvInputDimensions)
            .def("ReturnFwdOutputDimensions", &FT_t::ReturnFwdOutputDimensions)
            .def("ReturnInvOutputDimensions", &FT_t::ReturnInvOutputDimensions)

            // Debugging info
            .def("PrintState", &FT_t::PrintState)
            .def("Wait", &FT_t::Wait);

    // py::enum_<FastFFT::FourierTransformer::OriginType>(FourierTransformer, "OriginType")
    // .value("natural", FastFFT::FourierTransformer::OriginType::natural)
    // .value("centered", FastFFT::FourierTransformer::OriginType::centered)
    // .value("quadrant_swapped", FastFFT::FourierTransformer::OriginType::quadrant_swapped)
    // .export_values();
}

PYBIND11_MODULE(FastFFT, m) {

    declare_array<float, float, float>(m, "_float_float_float");
}
