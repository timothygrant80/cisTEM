/*
 * gpu_core_headers.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */


#ifndef GPU_CORE_HEADERS_H_
#define GPU_CORE_HEADERS_H_

#ifndef ENABLEGPU
#define ENABLEGPU true
#endif

#ifndef HEAVY_ERROR_CHECKING
#define HEAVY_ERROR_CHECKING true
#endif

#include "../core/core_headers.h"


const int MAX_GPU_COUNT = 32;

// The following block is
#define gMin(a,b) (((a) < (b)) ? (a) : (b))
#define gMax(a,b) (((a) > (b)) ? (a) : (b))

#define checkNppErrors(npp_stat, ...) {if (npp_stat != NPP_SUCCESS) { wxPrintf("NPP_CHECK_NPP - npp_stat = %s at line %d in file %s\n", _cudaGetErrorEnum(npp_stat), __LINE__,__FILE__); DEBUG_ABORT}}
#define checkCudaEvent(event_handle, ...) {if (cudaEventQuery(event_handle) != cudaSuccess) {return false; else return true;}}

#ifdef DEBUG_MODE
#define cudaErr(err, ...) { checkCudaErrors(err);}
#define nppErr(err, ...) { checkNppErrors(err);}
#else
#define cudaErr(err, ...) { err; }
#define nppErr(err, ...) { err; }
#endif

#if HEAVY_ERROR_CHECKING
#define postcheck { cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr2(error); };  //if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << std::endl; MyPrintWithDetails(""); exit(-1);} 
#define cudaErr2(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << std::endl; MyPrintWithDetails(""); exit(-1); } }; 
#define precheck { cudaErr2(cudaGetLastError()); } 
#define checkErrorsAndTimingWithSynchronization(input_stream) { cudaError_t cuda_error = cudaStreamSynchronize(input_stream); if (cuda_error != cudaSuccess) {wxPrintf("Sync Check error = %s at line %d in file %s\n", _cudaGetErrorEnum(cuda_error), __LINE__, __FILE__);} };
#define pre_checkErrorsAndTimingWithSynchronization(input_sream) { checkCudaErrors(cudaGetLastError()); }
#else
#define checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
#define pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
#endif

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex&, float&);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __device__ __host__ inline Complex ComplexConjMul(Complex, Complex);
static __device__ __host__ inline Complex ComplexConjMulAndScale(Complex a, Complex b, float s);


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__  inline Complex ComplexScale(Complex &a, float &s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex a * conj b multiplication
static __device__ __host__ inline Complex ComplexConjMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x + a.y * b.y;
    c.y = a.y * b.x - a.x * b.y  ;
    return c;
}

// Complex a * conj b multiplication
static __device__ __host__ inline Complex ComplexConjMulAndScale(Complex a, Complex b, float s)
{
    Complex c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y) ;
    return c;
}

#endif /* GPU_CORE_HEADERS_H_ */
