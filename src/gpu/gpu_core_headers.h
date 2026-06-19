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

// From the cuda samples TODO: add license bit for this
#include "cuda_common/helper_math.h"

// #include <cutensor.h>

const int MAX_GPU_COUNT = 32;

// The following block is
#define gMin(a, b) (((a) < (b)) ? (a) : (b))
#define gMax(a, b) (((a) > (b)) ? (a) : (b))

// clang-format off

/**
 * @defgroup gpu_debug GPU Error Checking and Debug Levels
 * @brief Three-tier error checking system controlled by ENABLE_GPU_DEBUG preprocessor define
 *
 * @section debug_levels Debug Levels
 *
 * **Level 0 (ENABLE_GPU_DEBUG == 0): Release Mode**
 * - All error checking macros compile to no-ops (zero overhead)
 * - Use for production builds where maximum performance is required
 * - No error detection - GPU errors will silently corrupt data or crash later
 *
 * **Level 1 (ENABLE_GPU_DEBUG == 1): Fast Development Mode**
 * - cudaErr(), nppErr(), cufftErr(), cuTensorErr() check return codes and exit on failure
 * - postcheck and precheck are no-ops (no stream synchronization)
 * - Catches API errors but NOT asynchronous kernel execution errors
 * - Minimal performance overhead - use for day-to-day development
 * - Recommended for CI builds to balance speed and error detection
 *
 * **Level 2 (ENABLE_GPU_DEBUG >= 2): Full Synchronous Debugging**
 * - All API error checking active (same as Level 1)
 * - postcheck synchronizes streams to catch kernel execution errors
 * - precheck clears stale error state before kernel launches
 * - Significant performance impact due to forced synchronization after every kernel
 * - Use when debugging race conditions, memory corruption, or kernel crashes
 *
 * @section error_macros Error Checking Macros
 *
 * **cudaErr(err)** - Wraps CUDA runtime API calls, exits on error
 * @code
 * cudaErr(cudaMalloc(&ptr, size));
 * @endcode
 *
 * **nppErr(err)** - Wraps NPP (NVIDIA Performance Primitives) calls
 *
 * **cufftErr(err)** - Wraps cuFFT library calls
 *
 * **cuTensorErr(err)** - Wraps cuTensor library calls
 *
 * **precheck** - Clears lingering GPU error state before kernel launch
 * - Only active at Level 2
 * - Critical for isolating which kernel actually caused an error
 * - Without this, kernel B might report an error that kernel A caused
 * @code
 * precheck;
 * myKernel<<<grid, block, 0, stream>>>(args);
 * stream(stream);
 * @endcode
 *
 * **postcheck** - Checks for kernel launch and execution errors (implicit stream)
 * - Only active at Level 2
 * - Calls cudaPeekAtLastError() to catch invalid launch parameters
 * - Calls cudaStreamSynchronize() to wait for kernel completion and catch execution errors
 * - Uses implicit cudaStreamPerThread which can be fragile
 * - Prefer postcheck_withstream() for explicit stream control
 *
 * **postcheck_withstream(stream)** - Checks for kernel errors on explicit stream
 * - Only active at Level 2
 * - Same error checking as postcheck but requires explicit stream argument
 * - Preferred over postcheck - forces developers to be aware of stream context
 * - Prevents bugs where kernel uses different stream than error check
 * @code
 * myKernel<<<grid, block, 0, my_stream>>>(args);
 * postcheck_withstream(my_stream);  // Explicitly check the correct stream
 * @endcode
 *
 * @section performance_implications Performance Implications
 *
 * **Level 0**: No overhead (macros are empty)
 *
 * **Level 1**: ~1-5% overhead from API error checking
 * - Function call overhead from checking return codes
 * - Negligible compared to kernel execution time
 *
 * **Level 2**: 10-100x slowdown depending on kernel characteristics
 * - cudaStreamSynchronize() forces CPU to wait for GPU completion after every kernel
 * - Destroys pipelining and overlapping of kernels/transfers
 * - Short kernels suffer most (synchronization overhead >> kernel time)
 * - Long-running kernels less affected (synchronization overhead << kernel time)
 *
 * @section usage_guidelines Usage Guidelines
 *
 * **When to use each level:**
 * - Level 0: Final production builds, performance benchmarking
 * - Level 1: Daily development, CI automated testing, performance profiling with error detection
 * - Level 2: Debugging crashes, investigating race conditions, validating kernel correctness
 *
 * **Why postcheck_withstream requires explicit stream:**
 * - Kernel launch uses explicit stream: myKernel<<<grid, block, 0, stream>>>
 * - postcheck uses implicit cudaStreamPerThread which may differ from kernel's stream
 * - If streams don't match, postcheck synchronizes wrong stream and misses errors
 * - postcheck_withstream forces stream consistency and prevents this class of bugs
 * - FIXME note at line 65: Eventually postcheck should be removed in favor of postcheck_withstream
 *
 * @note At Level 2, every postcheck/postcheck_withstream synchronizes a stream. This means
 *       GPU parallelism is completely disabled - kernels execute serially. This is intentional
 *       for debugging but catastrophic for performance.
 *
 * @warning Level 0 will silently allow data corruption. Only use in production builds where
 *          code has been thoroughly validated at Level 1 or Level 2.
 */

#if !defined(ENABLE_GPU_DEBUG) || ENABLE_GPU_DEBUG == 0

// Level 0: No GPU error checking (release mode)
#define cudaErr(err) { err; }
#define nppErr(err) { err; }
#define cuTensorErr(err) { err; }
#define cufftErr(err) { err; }
#define postcheck(stream)
#define precheck

#elif ENABLE_GPU_DEBUG >= 1

// Level 1: Error checking without expensive synchronization
// This provides maximum debugging detail but is slow - syncs after every kernel launch
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess && status != cudaErrorNotReady) { std::cerr << "Failed Assert: " << cudaGetErrorString(status) << " :-> "; print_debug_to_cerr("");} }

#define nppErr(npp_stat) { if (npp_stat != NPP_SUCCESS) { std::cerr << "Failed Assert NPP_CHECK_NPP NPP_SUCCESS = (" << NPP_SUCCESS << ") - npp_stat = " << npp_stat << " Find error codes at /usr/local/cuda/targets/x86_64-linux/include/nppdefs.h:(170)\n\n"; print_debug_to_cerr("");} } 

#define cufftErr(error) { auto status = static_cast<cufftResult>(error); if (status != CUFFT_SUCCESS) { std::cerr << "Failed Assert: " << cistem::gpu::cufft_error_types[status] << " :-> \n"; print_debug_to_cerr("");} }

#define cuTensorErr(error) { auto status = static_cast<cutensorStatus_t>(error); if (status != CUTENSOR_STATUS_SUCCESS) { std::cerr << "Failed Assert " << cutensorGetErrorString(status) << " :-> \n"; print_debug_to_cerr("");} }
// #define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess && status != cudaErrorNotReady) { std::cerr << "Failed Assert: " << cudaGetErrorString(status) << " :-> "; MyPrintWithDetails(""); {StackDump dump(NULL); dump.MyWalk(1); abort();};} }

// #define nppErr(npp_stat) { if (npp_stat != NPP_SUCCESS) { std::cerr << "Failed Assert NPP_CHECK_NPP NPP_SUCCESS = (" << NPP_SUCCESS << ") - npp_stat = " << npp_stat; wxPrintf(" at %s:(%d)\nFind error codes at /usr/local/cuda/targets/x86_64-linux/include/nppdefs.h:(170)\n\n", __FILE__, __LINE__);  {StackDump dump(NULL); dump.MyWalk(1); abort();};} }

// #define cufftErr(error) { auto status = static_cast<cufftResult>(error); if (status != CUFFT_SUCCESS) { std::cerr << "Failed Assert: " << cistem::gpu::cufft_error_types[status] << " :-> "; MyPrintWithDetails("");  {StackDump dump(NULL); dump.MyWalk(1); abort();};} }

// #define cuTensorErr(error) { auto status = static_cast<cutensorStatus_t>(error); if (status != CUTENSOR_STATUS_SUCCESS) { std::cerr << "Failed Assert " << cutensorGetErrorString(status) << " :-> "; MyPrintWithDetails("");  {StackDump dump(NULL); dump.MyWalk(1); abort();};} }

#if ENABLE_GPU_DEBUG == 1

#define precheck   // No-op at level 1
#define postcheck(stream)  // No-op at level 1
#endif

#if ENABLE_GPU_DEBUG >=2 
#define precheck { cudaErr(cudaGetLastError()) }
// FIXME: We should just make postCheck require the stream
#define postcheck(stream) { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(stream); cudaErr(error); }

#endif

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

struct FlushKernelPrintF {
    FILE* tmpout;

    FlushKernelPrintF(std::string& message) {
        std::cerr << "Flushing kernel printfs " << message << "\n\n";
        std::cerr << "There can only be one instance of this created as it modifies the global stdout pointer\n";
        cudaErr(cudaDeviceSynchronize( ));
        tmpout = stdout;
        stdout = stderr;
    }

    ~FlushKernelPrintF( ) {
        cudaErr(cudaDeviceSynchronize( ));
        fflush(stdout);
        stdout = tmpout;
        std::cerr << "Stdout pointer reset" << std::endl;
    }
};

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
