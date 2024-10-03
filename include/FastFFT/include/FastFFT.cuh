// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)
#include "FastFFT.h"

#ifndef __INCLUDE_FAST_FFT_CUH__
#define __INCLUDE_FAST_FFT_CUH__

#include "detail/detail.cuh"
#include <cuda_fp16.h>

// “This software contains source code provided by NVIDIA Corporation.”
// This is located in include/cufftdx*
// Please review the license in the cufftdx directory.

// #define forceforce( type )  __nv_is_extended_device_lambda_closure_type( type )
//FIXME: change to constexpr func

namespace FastFFT {

using namespace cufftdx;

// constexpr const std::map<unsigned int, unsigned int> elements_per_thread = {
//     {16, 4}, {"GPU", 15}, {"RAM", 20},
// };
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FFT kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
// BLOCK FFT based Kernel definitions
////////////////////////////////////////

/* 

transpose definitions in the kernel names refer to the physical axes in memory, which may not match the logical axes if following a previous transpose.
    2 letters indicate a swap of the axes specified
    3 letters indicate a permutation. E.g./ XZY, x -> Z, z -> Y, y -> X
R2C and C2R kernels are named as:
<cufftdx transform method>_fft_kernel_< fft type >_< size change >_< transpose axes >

C2C additionally specify direction and may specify an operation.
<cufftdx transform method>_fft_kernel_< fft type >_< direction >_< size change >_< transpose axes >_< operation in between round trip kernels >

*/

/////////////
// R2C
/////////////

/*
  For these kernels the XY transpose is intended for 2d transforms, while the XZ is for 3d transforms.
*/

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XZ(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XZ(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_R2C_DECREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

/////////////
// C2C
/////////////

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_DECREASE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t, class PreOpType, class IntraOpType, class PostOpType>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values,
                                                           Offsets mem_offsets, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv,
                                                           PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values,
                                                                   Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XYZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_XYZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

/////////////
// C2R
/////////////

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_C2R_DECREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, const float twiddle_in, const unsigned int Q, typename FFT::workspace_type workspace);

//////////////////////////////
// Thread FFT based Kernel definitions
//////////////////////////////

/////////////
// R2C
/////////////

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_R2C_decomposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_R2C_decomposed_transposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

/////////////
// C2C
/////////////

template <class FFT, class ComplexData_t>
__global__ void thread_fft_kernel_C2C_decomposed(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void thread_fft_kernel_C2C_decomposed_ConjMul(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

/////////////
// C2R
/////////////

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_C2R_decomposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_C2R_decomposed_transposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// End FFT Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class InputType, class OutputBaseType>
__global__ void clip_into_top_left_kernel(InputType* input_values, OutputBaseType* output_values, const short4 dims);

// Modified from GpuImage::ClipIntoRealKernel
template <typename InputType, typename OutputBaseType>
__global__ void clip_into_real_kernel(InputType*      real_values_gpu,
                                      OutputBaseType* other_image_real_values_gpu,
                                      short4          dims,
                                      short4          other_dims,
                                      int3            wanted_coordinate_of_box_center,
                                      OutputBaseType  wanted_padding_value);

// TODO: This would be much cleaner if we could first go from complex_compute_t -> float 2 then do conversions
// I think since this would be a compile time decision, it would be fine, but it would be good to confirm.
template <class FFT, typename SetTo_t, typename GetFrom_t>
inline __device__ SetTo_t convert_if_needed(const GetFrom_t* __restrict__ ptr, const int idx) {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // For now, we are assuming (as everywhere else) that compute precision is never double
    // and may in the future be _half. But is currently only float.
    // FIXME: THis should be caught earlier I think.
    if constexpr ( std::is_same_v<complex_compute_t, double> ) {
        static_no_doubles( );
    }
    if constexpr ( std::is_same_v<complex_compute_t, __half> ) {
        static_no_half_support_yet( );
    }

    if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> || std::is_same_v<std::decay_t<GetFrom_t>, float2> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(reinterpret_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(__float2half_rn(reinterpret_cast<const float*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Note: we will eventually need a similar hase for __nv_bfloat16
            // I think I may need to strip the const news for this to work
            if constexpr ( std::is_same_v<GetFrom_t, complex_compute_t> ) {
                return std::move(__floats2half2_rn(ptr[idx].real( ), 0.f));
            }
            else {
                return std::move(__floats2half2_rn(static_cast<const float*>(ptr)[idx], 0.f));
            }
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> && std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move(ptr[idx]);
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> && std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move(SetTo_t{ptr[idx].real( ), ptr[idx].imag( )});
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, float2> && std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move(SetTo_t{ptr[idx].x, ptr[idx].y});
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, float2> && std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move(ptr[idx]);
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, scalar_compute_t> || std::is_same_v<std::decay_t<GetFrom_t>, float> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(__float2half_rn(static_cast<const float*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(__floats2half2_rn(static_cast<const float*>(ptr)[idx], 0.f));
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(SetTo_t{static_cast<const float*>(ptr)[idx], 0.f});
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, __half> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            return std::move(__half2float(ptr[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            // FIXME: For some reason CUDART_ZERO_FP16 is not defined even with cuda_fp16.h included
            return std::move(__halves2half2(static_cast<const __half*>(ptr)[idx], __ushort_as_half((unsigned short)0x0000U)));
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(SetTo_t{__half2float(static_cast<const __half*>(ptr)[idx]), 0.f});
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, __half2> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> || std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(reinterpret_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            // FIXME: For some reason CUDART_ZERO_FP16 is not defined even with cuda_fp16.h included
            return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(SetTo_t{__low2float(static_cast<const __half2*>(ptr)[idx]), __high2float(static_cast<const __half2*>(ptr)[idx])});
        }
        else {
            static_no_match( );
        }
    }
    else {
        static_no_match( );
    }
}

//////////////////////////////////////////////
// IO functions adapted from the cufftdx examples
///////////////////////////////

template <class FFT>
struct io {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    static inline __device__ unsigned int
    stride_size( ) {

        return cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    }

    template <typename data_io_t>
    static inline __device__ void load_r2c(const data_io_t*   input,
                                           complex_compute_t* thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = convert_if_needed<FFT, scalar_compute_t>(input, index);
            thread_data[i].y = 0.0f;
            index += stride;
        }
    }

    static inline __device__ void store_r2c(const complex_compute_t* __restrict__ thread_data,
                                            complex_compute_t* __restrict__ output,
                                            int offset) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[index] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
        constexpr unsigned int values_left_to_store =
                threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[index] = thread_data[FFT::elements_per_thread / 2];
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    static inline __device__ void load_shared(const complex_compute_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float twiddle_in,
                                              int*  input_map,
                                              int* __restrict__ output_map,
                                              int Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i]         = input[index];
            shared_input[index]    = thread_data[i];
            index += stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    template <typename data_io_t>
    static inline __device__ void load_shared(const data_io_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float twiddle_in) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i]         = convert_if_needed<FFT, scalar_compute_t>(input, index);
            shared_input[index]    = thread_data[i];
            index += stride;
        }
    }

    static inline __device__ void load_shared(const complex_compute_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float twiddle_in,
                                              int* __restrict__ input_map,
                                              int* __restrict__ output_map,
                                              int Q,
                                              int number_of_elements) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < number_of_elements ) {
                input_map[i]           = index;
                output_map[i]          = Q * index;
                twiddle_factor_args[i] = twiddle_in * index;
                thread_data[i]         = input[index];
                shared_input[index]    = thread_data[i];
                index += stride;
            }
            else {
                input_map[i] = -9999; // ignore this in subsequent ops
            }
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    template <typename data_io_t>
    static inline __device__ void load_r2c_shared(const data_io_t* __restrict__ input,
                                                  scalar_compute_t* __restrict__ shared_input,
                                                  complex_compute_t* __restrict__ thread_data,
                                                  float* __restrict__ twiddle_factor_args,
                                                  float twiddle_in,
                                                  int* __restrict__ input_map,
                                                  int* __restrict__ output_map,
                                                  int Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // if (blockIdx.y == 0) ("blck %i index %i \n", Q*index, index);
            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i].x       = convert_if_needed<FFT, scalar_compute_t>(input, index);
            thread_data[i].y       = 0.0f;
            shared_input[index]    = thread_data[i].x;
            index += stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    template <typename data_io_t>
    static inline __device__ void load_r2c_shared(const data_io_t* __restrict__ input,
                                                  scalar_compute_t* __restrict__ shared_input,
                                                  complex_compute_t* __restrict__ thread_data,
                                                  float* __restrict__ twiddle_factor_args,
                                                  float twiddle_in) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i].x       = convert_if_needed<FFT, scalar_compute_t>(input, index);
            thread_data[i].y       = 0.0f;
            shared_input[index]    = thread_data[i].x;
            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_r2c_shared_and_pad(const data_io_t* __restrict__ input,
                                                          complex_compute_t* __restrict__ shared_mem) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = complex_compute_t(convert_if_needed<FFT, scalar_compute_t>(input, index), 0.f);
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_mem,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   const unsigned int Q) {
        const unsigned int stride = stride_size( ) * Q; // I think the Q is needed, but double check me TODO
        unsigned int       index  = (threadIdx.x * Q) + threadIdx.y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_mem[GetSharedMemPaddedIndex(index)];
            index += stride;
        }
        __syncthreads( ); // FFT().execute is setup to reuse the shared mem, so we need to sync here. Optionally, we could allocate more shared mem and remove this sync
    }

    // Note that unlike most functions in this file, this one does not have a
    // const decorator on the thread mem, as we want to modify it with the twiddle factors
    // before reducing the full shared mem space.
    static inline __device__ void reduce_block_fft(complex_compute_t* __restrict__ thread_data,
                                                   complex_compute_t* __restrict__ shared_mem,
                                                   const float        twiddle_in,
                                                   const unsigned int Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        complex_compute_t  twiddle;
        // In the first loop, all threads participate and write back to natural order in shared memory
        // while also updating with the full size twiddle factor.
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // ( index * threadIdx.y) == ( k % P * n2 )
            SINCOS(twiddle_in * (index * threadIdx.y), &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;

            shared_mem[GetSharedMemPaddedIndex(index)] = thread_data[i];
            index += stride;
        }
        __syncthreads( );

        // Now we reduce the shared memory into the first block of size P
        // Reuse index
        for ( index = 2; index <= Q; index *= 2 ) {
            // Some threads drop out each loop
            if ( threadIdx.y % index == 0 ) {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    thread_data[i] += shared_mem[GetSharedMemPaddedIndex(threadIdx.x + (i * stride) + (index / 2 * size_of<FFT>::value))];
                }
            } // end if condition
            // All threads can reach this point
            __syncthreads( );
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output,
                                                    const unsigned int pixel_pitch,
                                                    const unsigned int memory_limit) {
        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x;
            for ( unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++ ) {
                if ( index < memory_limit ) {
                    // transposed index.
                    output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += stride;
            }
        }
    }

    // when using load_shared || load_r2c_shared, we need then copy from shared mem into the registers.
    // notice we still need the packed complex values for the xform.
    static inline __device__ void copy_from_shared(const scalar_compute_t* __restrict__ shared_input,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   int* input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[input_map[i]];
            thread_data[i].y = 0.0f;
        }
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_input_complex,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   int* __restrict__ input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[input_map[i]];
        }
    }

    static inline __device__ void copy_from_shared(const scalar_compute_t* __restrict__ shared_input,
                                                   complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[index];
            thread_data[i].y = 0.0f;
            index += stride;
        }
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_input_complex,
                                                   complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[index];
            index += stride;
        }
    }

    template <class ExternalImage_t>
    static inline __device__ void load_shared_and_conj_multiply(ExternalImage_t* __restrict__ image_to_search,
                                                                complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        complex_compute_t  c;
        if constexpr ( std::is_same_v<ExternalImage_t, __half2> ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (thread_data[i].x * __low2float(image_to_search[index]) + thread_data[i].y * __high2float(image_to_search[index].y));
                c.y            = (thread_data[i].y * __low2float(image_to_search[index]) - thread_data[i].x * __high2float(image_to_search[index].y));
                thread_data[i] = c;
                index += stride;
            }
        }
        else if constexpr ( std::is_same_v<ExternalImage_t, float2> ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (thread_data[i].x * image_to_search[index].x + thread_data[i].y * image_to_search[index].y);
                c.y            = (thread_data[i].y * image_to_search[index].x - thread_data[i].x * image_to_search[index].y);
                thread_data[i] = c;
                index += stride;
            }
        }
        else {
            static_assert_type_name(image_to_search);
        }
    }

    // TODO: set user lambda to default = false, then get rid of other load_shared
    template <typename ExternalImage_t, class FunctionType = std::nullptr_t>
    static inline __device__ void load_shared(const ExternalImage_t* __restrict__ image_to_search,
                                              complex_compute_t* __restrict__ thread_data,
                                              FunctionType intra_op_functor = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            if constexpr ( std::is_same_v<ExternalImage_t, __half2*> ) {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    intra_op_functor(thread_data[i].x, thread_data[i].y, __low2float(image_to_search[index]), __high2float(image_to_search[index]));
                    index += stride;
                }
            }
            else {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    intra_op_functor(thread_data[i].x, thread_data[i].y, image_to_search[index].x, image_to_search[index].y);
                    index += stride;
                }
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                // a * conj b
                thread_data[i] = thread_data[i], image_to_search[index];
                index += stride;
            }
        }
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_r2c_in_shared_XZ(complex_compute_t* __restrict__ shared_mem,
                                                             complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_in_shared_XZ(complex_compute_t* __restrict__ shared_mem,
                                                         complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = io<FFT>::stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // return (XZ_STRIDE*blockIdx.z + threadIdx.y) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + X * gridDim.y );
            // XZ_STRIDE == XZ_STRIDE
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store_r2c_transposed_xz(const complex_compute_t* __restrict__ thread_data,
                                                          complex_compute_t* __restrict__ output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of contiguous (full) FFTS
    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                    data_io_t* __restrict__ output) {
        const unsigned int     stride                 = stride_size( );
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.y * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += stride;
        }
        constexpr unsigned int threads_per_fft      = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of non-contiguous (strided partial) FFTS
    //
    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                    data_io_t* __restrict__ output,
                                                                    const unsigned int Q,
                                                                    const unsigned int sub_fft) {
        const unsigned int     stride                 = stride_size( );
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.y * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += stride;
        }
        constexpr unsigned int threads_per_fft      = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                data_io_t* __restrict__ output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.y * cufftdx::size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += stride;
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ thread_data,
                                                          data_io_t* __restrict__ output,
                                                          int pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, FFT::elements_per_thread / 2);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ thread_data,
                                                          data_io_t* __restrict__ output,
                                                          int* output_MAP,
                                                          int  pixel_pitch) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[output_MAP[i] * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[output_MAP[FFT::elements_per_thread / 2] * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, FFT::elements_per_thread / 2);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ thread_data,
                                                          data_io_t* __restrict__ output,
                                                          int* __restrict__ output_MAP,
                                                          int pixel_pitch,
                                                          int memory_limit) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            // if (blockIdx.y == 1) printf("index, pitch, blcok, address %i, %i, %i, %i\n", output_MAP[i], pixel_pitch, memory_limit, output_MAP[i]*pixel_pitch + blockIdx.y);

            if ( output_MAP[i] < memory_limit )
                output[output_MAP[i] * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);
        }
        // constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        // constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        // constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        // if (threadIdx.x < values_left_to_store)
        // {
        //   printf("index, pitch, blcok, address %i, %i, %i, %i\n", output_MAP[FFT::elements_per_thread / 2], pixel_pitch, blockIdx.y, output_MAP[FFT::elements_per_thread / 2]*pixel_pitch + blockIdx.y);
        //   if (output_MAP[FFT::elements_per_thread / 2] < memory_limit) output[output_MAP[FFT::elements_per_thread / 2]*pixel_pitch + blockIdx.y] =  thread_data[FFT::elements_per_thread / 2];
        // }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2r(const data_io_t* __restrict__ input,
                                           complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            thread_data[FFT::elements_per_thread / 2] = convert_if_needed<FFT, complex_compute_t>(input, index);
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2r_transposed(const data_io_t* __restrict__ input,
                                                      complex_compute_t* __restrict__ thread_data,
                                                      unsigned int pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, (pixel_pitch * index) + blockIdx.y);
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            thread_data[FFT::elements_per_thread / 2] = convert_if_needed<FFT, complex_compute_t>(input, (pixel_pitch * index) + blockIdx.y);
        }
    }

    static inline __device__ void load_c2r_shared_and_pad(const complex_compute_t* __restrict__ input,
                                                          complex_compute_t* __restrict__ shared_mem,
                                                          const unsigned int pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
        }
        __syncthreads( );
    }

    // this may benefit from asynchronous execution
    template <typename data_io_t>
    static inline __device__ void load(const data_io_t* __restrict__ input,
                                       complex_compute_t* __restrict__ thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += stride;
        }
    }

    //  TODO: set pre_op_functor to default=false and get rid of other load
    template <typename data_io_t, class FunctionType = std::nullptr_t>
    static inline __device__ void load(const data_io_t* __restrict__ input,
                                       complex_compute_t* __restrict__ thread_data,
                                       int          last_index_to_load,
                                       FunctionType pre_op_functor = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        // FIXME: working out how to use these functors and this is NOT what is intended
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            float2 temp;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load ) {
                    temp           = pre_op_functor(convert_if_needed<FFT, float2>(input, index));
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(&temp, 0);
                }
                else {
                    // thread_data[i] = complex_compute_t{0.0f, 0.0f};
                    temp           = pre_op_functor(float2{0.0f, 0.0f});
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(&temp, 0);
                }

                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load )
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
                else
                    thread_data[i] = complex_compute_t{0.0f, 0.0f};
                index += stride;
            }
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_compute_t* __restrict__ thread_data,
                                                           complex_compute_t* __restrict__ output,
                                                           int first_negative_index) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        complex_compute_t  phase_shift;
        int                logical_y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            phase_shift = thread_data[i];
            logical_y   = index;
            if ( logical_y >= first_negative_index )
                logical_y -= 2 * first_negative_index;
            if ( (int(blockIdx.y) + logical_y) % 2 != 0 )
                phase_shift *= -1.f;
            output[index] = phase_shift;
            index += stride;
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_compute_t* __restrict__ thread_data,
                                                           complex_compute_t* __restrict__ output,
                                                           int* __restrict__ source_idx,
                                                           int first_negative_index) {
        const unsigned int stride = stride_size( );
        complex_compute_t  phase_shift;
        int                logical_y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            phase_shift = thread_data[i];
            logical_y   = source_idx[i];
            if ( logical_y >= first_negative_index )
                logical_y -= 2 * first_negative_index;
            if ( (int(blockIdx.y) + logical_y) % 2 != 0 )
                phase_shift *= -1.f;
            output[source_idx[i]] = phase_shift;
        }
    }

    template <typename data_io_t, class FunctionType = std::nullptr_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        FunctionType post_op_functor = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = post_op_functor(convert_if_needed<FFT, data_io_t>(thread_data, i));
                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                index += stride;
            }
        }
    }

    template <typename data_io_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        const unsigned int Q,
                                        const unsigned int sub_fft) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[index * Q + sub_fft] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_Z(const complex_compute_t* __restrict__ shared_mem,
                                          data_io_t* __restrict__ output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.y * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);

            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_Z(const complex_compute_t* __restrict__ shared_mem,
                                          data_io_t* __restrict__ output,
                                          const unsigned int Q,
                                          const unsigned int sub_fft) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.y * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += stride;
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        unsigned int memory_limit) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < memory_limit )
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        int* __restrict__ source_idx) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            output[source_idx[i]] = convert_if_needed<FFT, data_io_t>(thread_data, i);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_subset(const complex_compute_t* __restrict__ thread_data,
                                               data_io_t* __restrict__ output,
                                               int* __restrict__ source_idx) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( source_idx[i] >= 0 )
                output[source_idx[i]] = convert_if_needed<FFT, data_io_t>(thread_data, i);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_coalesced(const complex_compute_t* __restrict__ shared_output,
                                                  data_io_t* __restrict__ global_output,
                                                  int offset) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            global_output[index] = convert_if_needed<FFT, data_io_t>(shared_output, index);
            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2c_shared_and_pad(const data_io_t* __restrict__ input,
                                                          complex_compute_t* __restrict__ shared_mem) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += stride;
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_c2c_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output) {
        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += stride;
            }
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_c2r_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output) {
        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x + (threadIdx.y * size_of<FFT>::value);

            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += stride;
            }
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_transposed(const complex_compute_t* __restrict__ thread_data,
                                                   data_io_t* __restrict__ output,
                                                   int* __restrict__ output_map,
                                                   int* __restrict__ rotated_offset,
                                                   int memory_limit) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( output_map[i] < memory_limit )
                output[rotated_offset[1] * output_map[i] + rotated_offset[0]] = convert_if_needed<FFT, data_io_t>(thread_data, i);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_c2r(const complex_compute_t* __restrict__ thread_data,
                                            data_io_t* __restrict__ output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;

        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_c2r(const complex_compute_t* __restrict__ thread_data,
                                            data_io_t* __restrict__ output,
                                            unsigned int memory_limit) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;

        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // TODO: does reinterpret_cast<const scalar_compute_t*>(thread_data)[i] make more sense than just thread_data[i].x??
            if ( index < memory_limit )
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += stride;
        }
    }
};

template <class FFT>
struct io_thread {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    template <typename data_io_t>
    static inline __device__ void load_r2c(const data_io_t* __restrict__ input,
                                           complex_compute_t* __restrict__ thread_data,
                                           const int stride) {
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            thread_data[i].x = convert_if_needed<FFT, scalar_compute_t>(input, index);
            thread_data[i].y = scalar_compute_t{ };
            index += stride;
        }
    }

    static inline __device__ void store_r2c(const complex_compute_t* __restrict__ shared_output,
                                            complex_compute_t* __restrict__ output,
                                            const int stride,
                                            const int memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value / 2; i++ ) {
            output[index] = shared_output[index];
            index += stride;
        }
        if ( index < memory_limit ) {
            output[index] = shared_output[index];
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ shared_output,
                                                          data_io_t* __restrict__ output,
                                                          int stride,
                                                          int pixel_pitch,
                                                          int memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value / 2; i++ ) {
            output[index * pixel_pitch] = convert_if_needed<FFT, data_io_t>(shared_output, index);
            index += stride;
        }
        if ( index < memory_limit ) {
            output[index * pixel_pitch] = convert_if_needed<FFT, data_io_t>(shared_output, index);
        }
    }

    static inline __device__ void remap_decomposed_segments(const complex_compute_t* __restrict__ thread_data,
                                                            complex_compute_t* __restrict__ shared_output,
                                                            float twiddle_in,
                                                            int   Q,
                                                            int   memory_limit) {
        // Unroll the first loop and initialize the shared mem.
        complex_compute_t twiddle;
        int               index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            twiddle *= thread_data[i];
            if ( index + i < memory_limit )
                shared_output[index + i] = twiddle;
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                twiddle *= thread_data[i];
                if ( index + i < memory_limit ) {
                    atomicAdd_block(&shared_output[index + i].x, twiddle.x);
                    atomicAdd_block(&shared_output[index + i].y, twiddle.y);
                }
            }
        }
        __syncthreads( );
    }

    static inline __device__ void load_c2c(const complex_compute_t* __restrict__ input,
                                           complex_compute_t* __restrict__ thread_data,
                                           const int stride) {
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            thread_data[i] = input[index];
            index += stride;
        }
    }

    static inline __device__ void store_c2c(const complex_compute_t* __restrict__ shared_output,
                                            complex_compute_t* __restrict__ output,
                                            const int stride) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            output[index] = shared_output[index];
            index += stride;
        }
    }

    static inline __device__ void remap_decomposed_segments(const complex_compute_t* __restrict__ thread_data,
                                                            complex_compute_t* __restrict__ shared_output,
                                                            float twiddle_in,
                                                            int   Q) {
        // Unroll the first loop and initialize the shared mem.
        complex_compute_t twiddle;
        int               index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            twiddle *= thread_data[i];
            shared_output[index + i] = twiddle;
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                twiddle *= thread_data[i];
                atomicAdd_block(&shared_output[index + i].x, twiddle.x);
                atomicAdd_block(&shared_output[index + i].y, twiddle.y);
            }
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void load_c2r(const data_io_t* __restrict__ input,
                                           complex_compute_t* __restrict__ thread_data,
                                           const int stride,
                                           const int memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index  = threadIdx.x;
        unsigned int offset = 2 * memory_limit - 2;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            if ( index < memory_limit ) {
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            }
            else {
                // assuming even dimension
                // FIXME shouldn't need to read in from global for an even stride
                thread_data[i]   = convert_if_needed<FFT, complex_compute_t>(input, offset - index);
                thread_data[i].y = -thread_data[i].y; // conjugate
            }
            index += stride;
        }
    }

    // FIXME as above
    static inline __device__ void load_c2r_transposed(const complex_compute_t* __restrict__ input,
                                                      complex_compute_t* __restrict__ thread_data,
                                                      int stride,
                                                      int pixel_pitch,
                                                      int memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        // unsigned int offset = 2*memory_limit - 2;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            if ( index < memory_limit ) {
                thread_data[i] = input[index * pixel_pitch];
            }
            else {
                // input[2*memory_limit - index - 2];
                // assuming even dimension
                // FIXME shouldn't need to read in from global for an even stride
                thread_data[i]   = input[(2 * memory_limit - index) * pixel_pitch];
                thread_data[i].y = -thread_data[i].y; // conjugate
            }
            index += stride;
        }
    }

    static inline __device__ void remap_decomposed_segments_c2r(const complex_compute_t* __restrict__ thread_data,
                                                                scalar_compute_t* __restrict__ shared_output,
                                                                scalar_compute_t twiddle_in,
                                                                int              Q) {
        // Unroll the first loop and initialize the shared mem.
        complex_compute_t twiddle;
        int               index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            shared_output[index + i] = (twiddle.x * thread_data[i].x - twiddle.y * thread_data[i].y); // assuming the output is real, only the real parts add, so don't bother with the complex
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;

            for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
                // if (threadIdx.x == 32) printf("remap tid, subfft, q, index + i %i %i %i %i\n", threadIdx.x,sub_fft, Q, index+i);
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                atomicAdd_block(&shared_output[index + i], twiddle.x * thread_data[i].x - twiddle.y * thread_data[i].y);
            }
        }
        __syncthreads( );
    }

    static inline __device__ void store_c2r(const scalar_compute_t* __restrict__ shared_output,
                                            scalar_compute_t* __restrict__ output,
                                            const int stride) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;

        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            output[index] = convert_if_needed(shared_output, index);
            index += stride;
        }
    }

    template <typename ExternalImage_t>
    static inline __device__ void load_shared_and_conj_multiply(const ExternalImage_t* __restrict__ image_to_search,
                                                                const complex_compute_t* __restrict__ shared_mem,
                                                                complex_compute_t* __restrict__ thread_data,
                                                                const int stride) {
        unsigned int      index = threadIdx.x;
        complex_compute_t c;
        if constexpr ( std::is_same_v<ExternalImage_t, __half2*> ) {

            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (shared_mem[index].x * __low2float(image_to_search[index]) + shared_mem[index].y * __high2float(image_to_search[index]));
                c.y            = (shared_mem[index].y * __low2float(image_to_search[index]) - shared_mem[index].x * __high2float(image_to_search[index]));
                thread_data[i] = c;
                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (shared_mem[index].x * image_to_search[index].x + shared_mem[index].y * image_to_search[index].y);
                c.y            = (shared_mem[index].y * image_to_search[index].x - shared_mem[index].x * image_to_search[index].y);
                thread_data[i] = c;
                index += stride;
            }
        }

        __syncthreads( );
    }
}; // struct thread_io

} // namespace FastFFT

#endif // Fast_FFT_cuh_
