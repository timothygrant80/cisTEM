#include "../../../gpu/gpu_core_headers.h"
#include "test_gpu_core_headers.cuh"

__global__ void
test_complex_add_kernel(Complex* a, Complex* b, Complex* output) {
    *output = ComplexAdd(*a, *b);
}

void test_complex_add(Complex* a, Complex* b, Complex* output) {
    test_complex_add_kernel<<<1, 1, 0, cudaStreamPerThread>>>(a, b, output);
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread))
}

__global__ void
test_complex_scale_kernel(Complex* a, float output) {
    ComplexScale(a, output);
}

void test_complex_scale(Complex* a, float scalar) {
    test_complex_scale_kernel<<<1, 1, 0, cudaStreamPerThread>>>(a, scalar);
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread))
}

__global__ void
test_complex_scale_kernel(Complex& a, Complex& output, float scalar) {
    output = ComplexScale(a, scalar);
}

Complex test_complex_scale(Complex& a, float scalar) {
    Complex* output;
    Complex  ret_val;
    cudaErr(cudaMalloc(&output, sizeof(Complex)));

    test_complex_scale_kernel<<<1, 1, 0, cudaStreamPerThread>>>(a, *output, scalar);
    cudaErr(cudaMemcpyAsync(&ret_val, output, sizeof(Complex), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    return ret_val;
}