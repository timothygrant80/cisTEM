/*
 * gpu_core_headers.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#ifndef GPU_CORE_HEADERS_H_
#define GPU_CORE_HEADERS_H_

#include "../core/core_headers.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// #include <cutensor.h>

const int MAX_GPU_COUNT = 32;

// The following block is
#define gMin(a, b) (((a) < (b)) ? (a) : (b))
#define gMax(a, b) (((a) > (b)) ? (a) : (b))

// clang-format off

#ifndef ENABLE_GPU_DEBUG
#define cudaErr(err, ...) { err; }
#define nppErr(err, ...) { err; }
#define cuTensorErr(err, ...) { err; }
#define cufftErr(err, ...) { err; }
#define postcheck 
#define precheck 
#else
// The static path to the error code definitions is brittle, but better than the internet. At least you can click in VSCODE to get there.
#define nppErr(npp_stat)  {if (npp_stat != NPP_SUCCESS) { std::cerr << "NPP_CHECK_NPP - npp_stat = " << npp_stat ; wxPrintf(" at %s:(%d)\nFind error codes at /usr/local/cuda-11.7/targets/x86_64-linux/include/nppdefs.h:(170)\n\n",__FILE__,__LINE__); DEBUG_ABORT} }
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define cufftErr(error) { auto status = static_cast<cufftResult>(error); if (status != CUFFT_SUCCESS) { std::cerr << cistem::gpu::cufft_error_types[status] << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define cuTensorErr(error) { auto status = static_cast<cutensorStatus_t>(error); if (status != CUTENSOR_STATUS_SUCCESS) { std::cerr << cutensorGetErrorString(status) << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define postcheck { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error); }
#define precheck { cudaErr(cudaGetLastError()) }
#endif

// //s
// // REVERTME
// #undef postcheck
// #undef precheck
// #define postcheck
// #define precheck
// #define mcheck { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error); }

// clang-format on
template <typename T>
void* print_pointer_atrributes(T ptr, const char* ptr_name = nullptr) {

    cudaPointerAttributes attr;
    cudaErr(cudaPointerGetAttributes(&attr, ptr));
    std::cerr << "\n";
    if ( ptr_name ) {
        std::cerr << "Pointer " << ptr_name << std::endl;
    }
    std::cerr << "Device: " << attr.device << std::endl;
    std::cerr << "Your pointer is for: ";
    switch ( attr.type ) {
        case 0:
            std::cerr << "Unmanaged memory" << std::endl;
            break;
        case 1:
            std::cerr << "Host memory" << std::endl;
            break;
        case 2:
            std::cerr << "Device memory" << std::endl;
            break;
        case 3:
            std::cerr << "Managed memory" << std::endl;
            break;
        default:
            std::cerr << "Unknown memory" << std::endl;
            break;
    }
    std::cerr << "\n";
    std::cerr << "with possible device address () " << attr.devicePointer << std::endl;
    std::cerr << "with possible host address () " << attr.hostPointer << std::endl;
    return attr.hostPointer;
}

// Limits for specific kernels
constexpr int ntds_x_WhitenPS = 32;
constexpr int ntds_y_WhitenPS = 32;

// Complex data type
typedef float2 Complex;

// static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
// static __device__ __host__ inline void    ComplexScale(Complex*, float);
// static __device__ __host__ inline Complex ComplexScale(Complex&, float);
// static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
// static __device__ __host__ inline Complex ComplexMul(Complex*, const Complex&);
// static __device__ __host__ inline Complex ComplexConjMul(Complex, Complex);
// static __device__ __host__ inline Complex ComplexConjMulAndScale(Complex a, Complex b, float s);

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
template <typename T>
static __device__ __host__ inline T ComplexAdd(T a, T b) {
    T c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// // Complex scale
// static __device__ __host__ inline Complex ComplexScale(Complex& a, float& s) {
//     Complex c;
//     c.x = s * a.x;
//     c.y = s * a.y;
//     return c;
// }

// Complex scale
template <typename T>
static __device__ __host__ inline T ComplexScale(T& a, float s) {
    T c;
    if constexpr ( std::is_same_v<T, float2> ) {
        c.x = s * a.x;
        c.y = s * a.y;
    }
    else {
        c.x = __float2half_rn(s * __half2float(a.x));
        c.y = __float2half_rn(s * __half2float(a.y));
    }

    return c;
}

// Complex scale
template <typename T>
static __device__ __host__ inline void ComplexScale(T* a, float s) {
    if constexpr ( std::is_same_v<T, float2> ) {
        a->x *= s;
        a->y *= s;
    }
    else {
        a->x = __float2half_rn(s * __half2float(a->x));
        a->y = __float2half_rn(s * __half2float(a->y));
    }
}

// Complex multiplication
template <typename T>
static __device__ __host__ inline T ComplexMul(const T& a, const T& b) {
    T c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

template <typename T, typename U>
static __device__ __host__ inline T ComplexMul(const T& a, const U& b) {
    T c;
    c.x = __float2half_rn(__half2float(a.x) * b.x - __half2float(a.y) * b.y);
    c.y = __float2half_rn(__half2float(a.x) * b.y + __half2float(a.y) * b.x);
    return c;
}

// Complex multiplication
template <typename T>
static __device__ __host__ inline T ComplexMulAndScale(T a, T b, float s) {
    T c;
    if constexpr ( std::is_same_v<T, float2> ) {
        c.x = s * (a.x * b.x - a.y * b.y);
        c.y = s * (a.x * b.y + a.y * b.x);
    }
    else {
        // TODO: not sure of all these conversions
        c.x = __float2half_rn(s * (__half2float(a.x) * __half2float(b.x) - __half2float(a.y) * __half2float(b.y)));
        c.y = __float2half_rn(s * (__half2float(a.x) * __half2float(b.y) + __half2float(a.y) * __half2float(b.x)));
    }
    return c;
}

// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline T ComplexConjMul(T a, T b) {
    T c;
    if constexpr ( std::is_same_v<T, float2> ) {
        c.x = a.x * b.x + a.y * b.y;
        c.y = a.y * b.x - a.x * b.y;
    }
    else {
        c.x = __float2half_rn(__half2float(a.x) * __half2float(b.x) + __half2float(a.y) * __half2float(b.y));
        c.y = __float2half_rn(__half2float(a.y) * __half2float(b.x) - __half2float(a.x) * __half2float(b.y));
    }
    return c;
}

// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline T ComplexConjMul(const T& a, T& b) {
    T c;
    if constexpr ( std::is_same_v<T, float2> ) {
        c.x = a.x * b.x + a.y * b.y;
        c.y = a.y * b.x - a.x * b.y;
    }
    else {
        c.x = __float2half_rn(__half2float(a.x) * __half2float(b.x) + __half2float(a.y) * __half2float(b.y));
        c.y = __float2half_rn(__half2float(a.y) * __half2float(b.x) - __half2float(a.x) * __half2float(b.y));
    }
    return c;
}

// Trying decltype(auto)
// may need auto RealPartOfComplexConjMul(T a, T b) ->decltype(a.x * b.x + a.y * b.y)
// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline decltype(auto) RealPartOfComplexConjMul(T a, T b) {
    if constexpr ( std::is_same_v<T, float2> ) {
        return a.x * b.x + a.y * b.y;
    }
    else {
        return __float2half_rn(__half2float(a.x) * __half2float(b.x) + __half2float(a.y) * __half2float(b.y));
    }
}

// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline decltype(auto) ComplexModulus(T a) {
    if constexpr ( std::is_same_v<T, float2> ) {
        return sqrtf(a.x * a.x + a.y * a.y);
    }
    else {
        return __float2half_rn(sqrtf(__half2float(a.x) * __half2float(a.x) + __half2float(a.y) * __half2float(a.y)));
    }
}

// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline decltype(auto) ComplexModulusSquared(T a) {
    if constexpr ( std::is_same_v<T, float2> ) {
        return a.x * a.x + a.y * a.y;
    }
    else {
        return __float2half_rn(__half2float(a.x) * __half2float(a.x) + __half2float(a.y) * __half2float(a.y));
    }
}

// Complex a * conj b multiplication
template <typename T>
static __device__ __host__ inline T ComplexConjMulAndScale(T a, T b, float s) {
    T c;
    if constexpr ( std::is_same_v<T, float2> ) {
        c.x = s * (a.x * b.x + a.y * b.y);
        c.y = s * (a.y * b.x - a.x * b.y);
    }
    else {
        c.x = __float2half_rn(s * (__half2float(a.x) * __half2float(b.x) + __half2float(a.y) * __half2float(b.y)));
        c.y = __float2half_rn(s * (__half2float(a.y) * __half2float(b.x) - __half2float(a.x) * __half2float(b.y)));
    }
    return c;
}

// static constexpr int warpSize = 32;

#endif /* GPU_CORE_HEADERS_H_ */
