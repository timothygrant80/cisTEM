
#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "GpuImage.h"
#include "template_matching_empirical_distribution.h"
#include "../constants/constants.h"

namespace TM = cistem::match_template;

template <typename T>
inline __device__ __host__ bool test_gt_zero(T value) {
    if constexpr ( std::is_same_v<T, __half> )
        return value > CUDART_ZERO_FP16;
    else if constexpr ( std::is_same_v<T, __nv_bfloat16> )
        return value > CUDART_ZERO_BF16;
    else if constexpr ( std::is_same_v<T, histogram_storage_t> )
        return value > 0.0f;
    else
        MyDebugAssertTrue(false, "input_type must be either __half __nv_bfloat16, or histogram_storage_t");
}

/**
 * @brief Construct a new TM_EmpiricalDistribution
 * Note: both histogram_min and histogram step must be > 0 or no histogram will be created
 * Note: the number of histogram bins is fixed by TM::histogram_number_of_points
 * 
 * @param reference_image - used to determine the size of the input images and set gpu launch configurations
 * @param histogram_min - the minimum value of the histogram
 * @param histogram_step - the step size of the histogram
 * @param n_images_to_accumulate_concurrently - the number of images to accumulate concurrently
 * 
 */
template <typename ccfType, typename mipType, bool per_image>
TM_EmpiricalDistribution<ccfType, mipType, per_image>::TM_EmpiricalDistribution(GpuImage&           reference_image,
                                                                                histogram_storage_t histogram_min,
                                                                                histogram_storage_t histogram_step,
                                                                                int                 n_border_pixels_to_ignore_for_histogram,
                                                                                const int           n_images_to_accumulate_concurrently,
                                                                                cudaStream_t        calc_stream) : n_images_to_accumulate_concurrently_{n_images_to_accumulate_concurrently},
                                                                                                            n_border_pixels_to_ignore_for_histogram_{n_border_pixels_to_ignore_for_histogram},
                                                                                                            calc_stream_{calc_stream},
                                                                                                            higher_order_moments_{false} {

    static_assert(per_image == false, "This class does not support per image accumulation yet");

    MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");
    // I suspect we'll move to bfloat16 for the input data, as it was not available at the time the
    // original code was implemented. The extended dynamic range, and ease of conversion to/from histogram_storage_t
    // are likely a benefit, while the further reduced precision is unlikely to be a problem in the raw data values.
    // If anything, given that the output of the matched filter is ~ Gaussian, all the numbers closer to zero are less
    // likely to be flushed to zero when denormal, so in that respect, bflaot16 may actually maintain higher precision.
    if constexpr ( std::is_same_v<ccfType, __half> ) {
        histogram_min_  = __float2half_rn(histogram_min);
        histogram_step_ = __float2half_rn(histogram_step);
    }
    else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
        histogram_min_  = __float2bfloat16_rn(histogram_min);
        histogram_step_ = __float2bfloat16_rn(histogram_step);
    }
    else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
        histogram_min_  = histogram_min;
        histogram_step_ = histogram_step;
    }
    else {
        MyDebugAssertTrue(false, "input_type must be either __half __nv_bfloat16, or histogram_storage_t");
    }

    // FIXME: this should probably be a bool rather than testing for a default zero value. Hacky habits die hard
    if ( test_gt_zero(histogram_step_) ) {
        MyDebugAssertTrue(TM::histogram_number_of_points <= 1024, "The histogram kernel assumes <= 1024 threads per block");
        MyDebugAssertTrue(TM::histogram_number_of_points % cistem::gpu::warp_size == 0, "The histogram kernel assumes a multiple of 32 threads per block");
        histogram_n_bins_ = TM::histogram_number_of_points;
    }
    else {
        // will be used as check on which kernels to call
        histogram_n_bins_ = 0;
    }

    image_dims_.x = reference_image.dims.x;
    image_dims_.y = reference_image.dims.y;
    image_dims_.z = reference_image.dims.z;
    image_dims_.w = reference_image.dims.w;

    MyDebugAssertTrue(image_dims_.x > 0 && image_dims_.y > 0 && image_dims_.z > 0 && image_dims_.w > 0, "Image dimensions must be > 0");

    // Set-up the launch configuration - assumed to be a real space image.
    // WARNING: this is up to the developer to ensure, as we'll use pointers for the input arrays
    // Note: we prefer the "1d" grid as a NxN patch is more likely to have similar values than a N^2x1 line, and so more atomic collisions in the histogram kernel.
    reference_image.ReturnLaunchParameters<TM::histogram_number_of_points, 1>(image_dims_, true);
    gridDims_        = reference_image.gridDims;
    threadsPerBlock_ = reference_image.threadsPerBlock;

    // Every block will have a shared memory array of the size of the number of bins and aggregate those into their own
    // temp arrays. Only at the end of the search will these be added together'

    // Array of temporary storage to accumulate the shared mem to
    cudaErr(cudaMallocAsync(&histogram_, gridDims_.x * gridDims_.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), calc_stream_));
    cudaErr(cudaMemsetAsync(histogram_, 0, gridDims_.x * gridDims_.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), calc_stream_));
};

template <typename ccfType, typename mipType, bool per_image>
TM_EmpiricalDistribution<ccfType, mipType, per_image>::~TM_EmpiricalDistribution( ) {
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    cudaErr(cudaFreeAsync(histogram_, calc_stream_));
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels and inline helper functions called from EmpiricalDistribution::AccumulateDistribution
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ void convert_input(T* input_ptr, int x, int y, int z, int NY, int pitch_in_pixels, const T bin_min, const T bin_inc, int& pixel_idx, T& val, int& address) {
    address = ((z * NY + y) * pitch_in_pixels) + x;
    if constexpr ( std::is_same_v<T, __half> ) {
        val       = input_ptr[address];
        pixel_idx = __half2int_rd((val - bin_min) / bin_inc);
    }
    if constexpr ( std::is_same_v<T, __nv_bfloat16> ) {
        val       = input_ptr[address];
        pixel_idx = __bfloat162int_rd((val - bin_min) / bin_inc);
    }
    if constexpr ( std::is_same_v<T, histogram_storage_t> ) {
        val       = input_ptr[address];
        pixel_idx = __float2int_rd((val - bin_min) / bin_inc);
    }
}

template <bool evalType, typename ccfType>
inline __device__ void sum_squares_and_check_max(ccfType& val, float& sum, float& sum_sq, ccfType& max_val, int& max_idx, const int idx) {
    if constexpr ( evalType > 0 ) {
        if ( val > max_val ) {
            max_val = val;
            max_idx = idx;
        }
        if constexpr ( std::is_same_v<ccfType, __half> ) {
            sum += __half2float(val);
            sum_sq += __half2float(val) * __half2float(val);
        }
        else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
            sum += __bfloat162float(val);
            sum_sq += __bfloat162float(val) * __bfloat162float(val);
        }
        else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
            sum += val;
            sum_sq += val * val;
        }
    }
}

template <bool evalType, typename ccfType, typename mipType>
inline __device__ void write_mip_and_stats(float* sum_array, float* sum_sq_array,
                                           mipType* mip_psi, mipType* theta_phi,
                                           float& sum, float& sum_sq,
                                           ccfType* psi, ccfType* theta, ccfType* phi,
                                           ccfType& max_val, int max_idx, const int address) {
    if constexpr ( evalType > 0 ) {
        sum_array[address] += sum;
        sum_sq_array[address] += sum_sq;
        sum    = 0.f;
        sum_sq = 0.f;

        if constexpr ( std::is_same_v<ccfType, __half> ) {
            if ( max_val > ccfType{-10.0} && max_val > __low2half(mip_psi[address]) ) {
                mip_psi[address]   = __halves2half2(max_val, psi[max_idx]);
                theta_phi[address] = __halves2half2(theta[max_idx], phi[max_idx]);
            }
        }
        else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
            if ( max_val > ccfType{-10.0} && max_val > __low2bfloat16(mip_psi[address]) ) {
                mip_psi[address]   = __halves2bfloat162(max_val, psi[max_idx]);
                theta_phi[address] = __halves2bfloat162(theta[max_idx], phi[max_idx]);
            }
        }
        else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
            if ( max_val > ccfType{-10.0} && max_val > mip_psi[address] ) {
                mip_psi[address].x   = max_val;
                mip_psi[address].y   = psi[max_idx];
                theta_phi[address].x = theta[max_idx];
                theta_phi[address].y = phi[max_idx];
            }
        }
    }
}

// TODO: __half2 atomicAdd(__half2 *address, __half2 val);
// TODO: __nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
// This would allow us to double the number of bins in the histogram, and still use atomicAdd reducing contention
template <int evalType, typename ccfType, typename mipType>
__global__ void __launch_bounds__(TM::histogram_number_of_points)
        AccumulateDistributionKernel(ccfType*             input_ptr,
                                     histogram_storage_t* output_ptr,
                                     int4                 dims,
                                     const ccfType        bin_min,
                                     const ccfType        bin_inc,
                                     const int            max_padding,
                                     const int            n_slices_to_process,
                                     float*               sum_array    = nullptr,
                                     float*               sum_sq_array = nullptr,
                                     mipType*             mip_psi      = nullptr,
                                     mipType*             mip_theta    = nullptr,
                                     ccfType*             psi          = nullptr,
                                     ccfType*             theta        = nullptr,
                                     ccfType*             phi          = nullptr) {

    // initialize temporary accumulation array input_ptr shared memory, this is equal to the number of bins input_ptr the histogram,
    // which may  be more or less than the number of threads input_ptr a block
    __shared__ int smem[TM::histogram_number_of_points];

    // Each block has it's own copy of the histogram stored input_ptr global memory, found at the linear block index
    histogram_storage_t* stored_array = &output_ptr[LinearBlockIdx_2dGrid( ) * TM::histogram_number_of_points];

    // Since the number of x-threads is enforced to be = to the number of bins, we can just copy the bins to shared memory
    // Otherwise, we would need a loop to copy the bins to shared memory e.g. ->
    //        smem[threadIdx.x] = __float2int_rn(stored_array[threadIdx.x]);
    // FIXME:     // smem[i] =
    for ( int i = threadIdx.x; i < TM::histogram_number_of_points; i += BlockDimension_2d( ) )
        smem[i] = int(stored_array[i]);

    __syncthreads( );

    int     address;
    int     pixel_idx;
    ccfType val;
    // updates our block's partial histogram input_ptr shared memory
    int     max_idx;
    ccfType max_val = ccfType{0.0};
    float   sum{0.f}, sum_sq{0.f};
    for ( int j = max_padding + physical_Y( ); j < dims.y - max_padding; j += blockDim.y * gridDim.y ) {
        for ( int i = max_padding + physical_X( ); i < dims.x - max_padding; i += blockDim.x * gridDim.x ) {
            for ( int k = 0; k < n_slices_to_process; k++ ) {
                // pixel_idx = __half2int_rd((input_ptr[j * dims.w + i] - bin_min) / bin_inc);
                convert_input(input_ptr, i, j, k, dims.y, dims.w, bin_min, bin_inc, pixel_idx, val, address);
                if ( pixel_idx >= 0 && pixel_idx < TM::histogram_number_of_points ) {
                    atomicAdd(&smem[pixel_idx], 1);
                }
                sum_squares_and_check_max<evalType>(val, sum, sum_sq, max_val, max_idx, k);
            } // loop over slices

            // Now we need to actually write out to global memory for the mip if we are doint it
            write_mip_and_stats<evalType>(sum_array, sum_sq_array, mip_psi, mip_theta, sum, sum_sq, psi, theta, phi, max_val, max_idx, address);
        }
    }

    __syncthreads( );

    // write partial histogram into the global memory
    // Converting to long was super slow. Given that I don't care about representing the number exactly, but do care about overflow, just switch the bins to histogram_storage_t
    // As in the read case, we would need a loop if the number of threads != number of bins e.g. ->
    // stored_array[threadIdx.x] = __int2float_rn(smem[threadIdx.x]);
    for ( int i = threadIdx.x; i < TM::histogram_number_of_points; i += BlockDimension_2d( ) )
        stored_array[i] = int(smem[i]);
}

__global__ void
FinalAccumulateKernel(histogram_storage_t* input_ptr, const int n_bins, const int n_blocks) {

    int lIDX = physical_X( );

    if ( lIDX < n_bins ) {
        histogram_storage_t total{0.0};
        for ( int j = 0; j < n_blocks; j++ ) {
            total += input_ptr[lIDX + n_bins * j];
        }
        // We accumulate all histograms into the first block
        input_ptr[lIDX] = total;
    }
}

/**
 * @brief Accumulate new values into the pixel wise distribution.
 * If set to record a histogram, a fused kernal will be called to accumulate the histogram and the pixel wise distribution
 * If set to track 3rd and 4th moments of the distribution, a fused kernel will be called to accumulate the moments and the pixel wise distribution
 * 
 * @param input_data - pointer to the input data to accumulate, a stack of images.
 * @param n_images_this_batch - number of slices to accumulate, must be <= n_images_to_accumulate_concurrently
 */

template <typename ccfType, typename mipType, bool per_image>
void TM_EmpiricalDistribution<ccfType, mipType, per_image>::AccumulateDistribution(ccfType* input_data, int n_images_this_batch) {
    MyDebugAssertTrue(input_data != nullptr, "The data to acmmulate is not input_ptr memory.");
    MyDebugAssertTrue(n_images_this_batch <= n_images_to_accumulate_concurrently_, "The number of images to accumulate is greater than the number of images to accumulate concurrently");
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    constexpr int n_threads_in_y_or_z = 1;
    const int     y_grid_divisor      = 32; // TODO: optimize this.
    dim3          threadsPerBlock_img = dim3(TM::histogram_number_of_points, n_threads_in_y_or_z, n_threads_in_y_or_z);

    dim3 gridDims_img = dim3((image_dims_.x + threadsPerBlock_img.x - 1) / threadsPerBlock_img.x,
                             (image_dims_.y + (y_grid_divisor + threadsPerBlock_img.y) - 1) / (y_grid_divisor - 1 + threadsPerBlock_img.y), 1);

    // Instead of calculating int((value - bin_min) / bin_inc), use a fused multiply add

    if ( histogram_n_bins_ != 0 ) {
        // TODO: move eval conditions to an enum
        constexpr int only_histogram = 0;
        precheck;
        AccumulateDistributionKernel<only_histogram><<<gridDims_img, threadsPerBlock_img, 0, calc_stream_>>>(
                input_data,
                histogram_,
                image_dims_,
                histogram_min_,
                histogram_step_,
                n_border_pixels_to_ignore_for_histogram_,
                n_images_this_batch,
                sum_array,
                sum_sq_array,
                mip_psi,
                mip_theta,
                psi,
                theta,
                phi);
        postcheck;
    }
    else if ( higher_order_moments_ ) {
        MyDebugAssertTrue(false, "Skew and kurtosis not implemented yet");
        // call the pixel wise kernel
    }
    else {
        MyDebugAssertFalse(true, "The fused kernel is not yet implemented.");
        constexpr int histogram_and_mip = 1;
        precheck;

        AccumulateDistributionKernel<histogram_and_mip><<<gridDims_img, threadsPerBlock_img, 0, calc_stream_>>>(
                input_data,
                histogram_,
                image_dims_,
                histogram_min_,
                histogram_step_,
                n_border_pixels_to_ignore_for_histogram_,
                n_images_this_batch,
                sum_array,
                sum_sq_array,
                mip_psi,
                mip_theta,
                psi,
                theta,
                phi);
        postcheck;
    }
};

template <typename ccfType, typename mipType, bool per_image>
void TM_EmpiricalDistribution<ccfType, mipType, per_image>::FinalAccumulate( ) {
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    const int n_blocks = gridDims_.x * gridDims_.y;
    const int n_bins   = TM::histogram_number_of_points;

    // FIXME: this is from histogram.cu, but is probably not optimal
    dim3 threadsPerBlock_accum_array = dim3(32, 1, 1);
    dim3 gridDims_accum_array        = dim3((TM::histogram_number_of_points + threadsPerBlock_accum_array.x - 1) / threadsPerBlock_accum_array.x, 1, 1);

    precheck;
    FinalAccumulateKernel<<<gridDims_accum_array, threadsPerBlock_accum_array, 0, calc_stream_>>>(histogram_, n_bins, n_blocks);
    postcheck;
}

template <typename ccfType, typename mipType, bool per_image>
void TM_EmpiricalDistribution<ccfType, mipType, per_image>::CopyToHostAndAdd(long* array_to_add_to) {

    // Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
    histogram_storage_t* tmp_array;
    cudaErr(cudaMallocHost(&tmp_array, TM::histogram_number_of_points * sizeof(histogram_storage_t)));
    cudaErr(cudaMemcpy(tmp_array, histogram_, TM::histogram_number_of_points * sizeof(histogram_storage_t), cudaMemcpyDeviceToHost));

    for ( int iBin = 0; iBin < TM::histogram_number_of_points; iBin++ ) {
        array_to_add_to[iBin] += long(tmp_array[iBin]);
    }

    cudaErr(cudaFreeHost(tmp_array));
}

template class TM_EmpiricalDistribution<__half, __half2, false>;
template class TM_EmpiricalDistribution<__nv_bfloat16, __nv_bfloat162, false>;

// Note: we allow for float in the constructor checking, however, we don't need this for our implementation, so we won't instantiate it.
// template class TM_EmpiricalDistribution<float, per_image>;