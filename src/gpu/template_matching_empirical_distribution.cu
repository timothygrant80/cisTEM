
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
    return false;
}

/**
 * @brief Construct a new TM_EmpiricalDistribution
 * Note: both histogram_min and histogram step must be > 0 or no histogram will be created
 * Note: the number of histogram bins is fixed by TM::histogram_number_of_points
 * 
 * @param reference_image - used to determine the size of the input images and set gpu launch configurations
 * @param histogram_min - the minimum value of the histogram
 * @param histogram_step - the step size of the histogram
 * @param n_imgs_to_process_at_once_ - the number of images to accumulate concurrently
 * 
 */

template <typename ccfType, typename mipType>
TM_EmpiricalDistribution<ccfType, mipType>::TM_EmpiricalDistribution(GpuImage& reference_image,
                                                                     int2      pre_padding,
                                                                     int2      roi) : pre_padding_{pre_padding},
                                                                                 roi_{roi},
                                                                                 higher_order_moments_{false},
                                                                                 image_plane_mem_allocated_{reference_image.real_memory_allocated} {

    std::cerr << "n_images" << n_imgs_to_process_at_once_ << std::endl;
    int least_priority, highest_priority;

    my_rng_ = std::make_unique<RandomNumberGenerator>(pi_v<float>);

    cudaErr(cudaDeviceGetStreamPriorityRange(&least_priority, &highest_priority));
    cudaErr(cudaStreamCreateWithPriority(&calc_stream_[0], cudaStreamNonBlocking, least_priority));
    cudaErr(cudaEventCreateWithFlags(&mip_stack_is_ready_event_[0], cudaEventBlockingSync)); // blocking sync makes the host wait if calling cudaEventSynchronize

    // I suspect we'll move to bfloat16 for the input data, as it was not available at the time the
    // original code was implemented. The extended dynamic range, and ease of conversion to/from histogram_storage_t
    // are likely a benefit, while the further reduced precision is unlikely to be a problem in the raw data values.
    // If anything, given that the output of the matched filter is ~ Gaussian, all the numbers closer to zero are less
    // likely to be flushed to zero when denormal, so in that respect, bflaot16 may actually maintain higher precision.
    if constexpr ( std::is_same_v<ccfType, __half> ) {
        histogram_min_  = __float2half_rn(TM::histogram_min);
        histogram_step_ = __float2half_rn(TM::histogram_step);
    }
    else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
        histogram_min_  = __float2bfloat16_rn(TM::histogram_min);
        histogram_step_ = __float2bfloat16_rn(TM::histogram_step);
    }
    else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
        histogram_min_  = TM::histogram_min;
        histogram_step_ = TM::histogram_step;
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
    // TODO: Given that we may be skipping large areas, we may want to consider either the ROI or using fewer threads and a grid stride loop
    reference_image.ReturnLaunchParametersNoFFTWPadding<TM::histogram_number_of_points, 1>(roi_.x / 4, roi_.y / 4, 1, gridDims_, threadsPerBlock_);

    // Every block will have a shared memory array of the size of the number of bins and aggregate those into their own
    // temp arrays. Only at the end of the search will these be added together'

    // For an GpuImage the following would be GridDimension_2d( ) * TM::histogram_number_of_points * sizeof(histogram_storage_t)
    cudaErr(cudaMallocAsync(&histogram_, gridDims_.x * gridDims_.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), calc_stream_[0]));
    ZeroHistogram( );

    AllocateAndZeroStatisticalArrays( );
    object_initialized_ = true;
};

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::AllocateAndZeroStatisticalArrays( ) {

    cudaErr(cudaMallocAsync(&sum_array, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&sum_sq_array, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&sum_counter, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&mip_psi, image_plane_mem_allocated_ * sizeof(mipType), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&theta_phi, image_plane_mem_allocated_ * sizeof(mipType), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&psi, image_plane_mem_allocated_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&theta, image_plane_mem_allocated_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&phi, image_plane_mem_allocated_ * sizeof(decltype(phi)), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&ccf_array_.at(0), image_plane_mem_allocated_ * n_imgs_to_process_at_once_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMallocAsync(&ccf_array_.at(1), image_plane_mem_allocated_ * n_imgs_to_process_at_once_ * sizeof(ccfType), calc_stream_[0]));

    cudaErr(cudaMemsetAsync(sum_array, 0, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(sum_sq_array, 0, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(sum_counter, 0, image_plane_mem_allocated_ * sizeof(float), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(mip_psi, 0, image_plane_mem_allocated_ * sizeof(mipType), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(theta_phi, 0, image_plane_mem_allocated_ * sizeof(mipType), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(psi, 0, image_plane_mem_allocated_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(theta, 0, image_plane_mem_allocated_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(phi, 0, image_plane_mem_allocated_ * sizeof(decltype(phi)), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(ccf_array_.at(0), 0, image_plane_mem_allocated_ * n_imgs_to_process_at_once_ * sizeof(ccfType), calc_stream_[0]));
    cudaErr(cudaMemsetAsync(ccf_array_.at(1), 0, image_plane_mem_allocated_ * n_imgs_to_process_at_once_ * sizeof(ccfType), calc_stream_[0]));

    for ( int i = 0; i < 2; i++ ) {

        host_angle_arrays_.at(i) = new ccfType[n_imgs_to_process_at_once_ * 3];
        std::memset(host_angle_arrays_.at(i), 0, n_imgs_to_process_at_once_ * 3 * sizeof(ccfType));

        cudaErr(cudaMallocAsync(&device_host_angle_arrays_.at(i), n_imgs_to_process_at_once_ * 3 * sizeof(ccfType), calc_stream_[0]));
        cudaErr(cudaMemcpyAsync(device_host_angle_arrays_.at(i), host_angle_arrays_.at(i), n_imgs_to_process_at_once_ * 3 * sizeof(ccfType), cudaMemcpyHostToDevice, calc_stream_[0]));
    }

    // TODO: higher_order_moments_
    // if ( n_global_search_images_to_save > 1 ) {
    //     cudaErr(cudaMallocAsync((void**)&secondary_peaks, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4, cudaStreamPerThread));
    //     cudaErr(cudaMemsetAsync(secondary_peaks, 0, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4, cudaStreamPerThread));
    // }
};

template <typename ccfType, typename mipType>
TM_EmpiricalDistribution<ccfType, mipType>::~TM_EmpiricalDistribution( ) {
    if ( object_initialized_ )
        Delete( );
};

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::Delete( ) {
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_[0]) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    cudaErr(cudaFreeAsync(histogram_, calc_stream_[0]));
    cudaErr(cudaFreeAsync(sum_array, calc_stream_[0]));
    cudaErr(cudaFreeAsync(sum_sq_array, calc_stream_[0]));
    cudaErr(cudaFreeAsync(sum_counter, calc_stream_[0]));
    cudaErr(cudaFreeAsync(mip_psi, calc_stream_[0]));
    cudaErr(cudaFreeAsync(theta_phi, calc_stream_[0]));
    cudaErr(cudaFreeAsync(psi, calc_stream_[0]));
    cudaErr(cudaFreeAsync(theta, calc_stream_[0]));
    cudaErr(cudaFreeAsync(phi, calc_stream_[0]));
    cudaErr(cudaFreeAsync(ccf_array_.at(0), calc_stream_[0]));
    cudaErr(cudaFreeAsync(ccf_array_.at(1), calc_stream_[0]));

    for ( int i = 0; i < 2; i++ ) {
        delete[] host_angle_arrays_.at(i);
        cudaErr(cudaFreeAsync(device_host_angle_arrays_.at(i), calc_stream_[0]));
    }

    cudaErr(cudaStreamDestroy(calc_stream_[0]));
    cudaErr(cudaEventDestroy(mip_stack_is_ready_event_[0]));

    object_initialized_ = false;
}

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::ZeroHistogram( ) {
    cudaErr(cudaMemsetAsync(histogram_, 0, gridDims_.x * gridDims_.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), calc_stream_[0]));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels and inline helper functions called from EmpiricalDistribution::AccumulateDistribution
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ void convert_input(T* input_ptr, const T bin_min, const T bin_inc, int& pixel_idx, T& val, const int address) {
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

template <typename ccfType>
inline __device__ void sum_squares_and_check_max(ccfType     val,
                                                 float&      sum,
                                                 float&      sum_sq,
                                                 float&      sum_counter_val,
                                                 float&      sum_err,
                                                 float&      sum_sq_err,
                                                 ccfType&    max_val,
                                                 int&        max_idx,
                                                 const int   idx,
                                                 const float min_counter_val,
                                                 const float threshold_val) {

    if ( val > max_val ) {
        max_val = val;
        max_idx = idx;
    }

    // if ( sum_counter_val == 0.f || fabsf(float_val - sum / sum_counter_val) < sqrtf(((sum_sq / sum_counter_val) - powf(sum / sum_counter_val, 2))) * 3.0f ) {
    float float_val;
    if constexpr ( std::is_same_v<ccfType, __half> ) {
        float_val = __half2float(val);
    }
    else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
        float_val = __bfloat162float(val);
    }
    else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
        float_val = val;
    }
    // for Welfords
    // For Kahan summation
    float mean_val = sum / sum_counter_val;

    if ( sum_counter_val < min_counter_val || fabsf((float_val - mean_val) * rsqrtf(sum_sq / sum_counter_val - mean_val * mean_val)) < threshold_val ) {
        sum_counter_val += 1.0f;

        // Kahan summation
        const float y = float_val - sum_err;
        const float t = sum + y;
        sum_err       = (t - sum) - y;
        sum           = t;

        const float y2 = __fmaf_ieee_rn(float_val, float_val, -sum_sq_err);
        const float t2 = sum_sq + y2;
        sum_sq_err     = (t2 - sum_sq) - y2;
        sum_sq         = t2;
    }
}

template <typename ccfType, typename mipType>
inline __device__ void write_mip_and_stats(float*      sum_array,
                                           float*      sum_sq_array,
                                           float*      sum_counter,
                                           mipType*    mip_psi,
                                           mipType*    theta_phi,
                                           const float sum,
                                           const float sum_sq,
                                           const float sum_counter_val,
                                           const ccfType* __restrict__ psi,
                                           const ccfType* __restrict__ theta,
                                           const ccfType* __restrict__ phi,
                                           ccfType&  max_val,
                                           int       max_idx,
                                           const int address) {

    // There may be rare cases where no stats have been evaluated, but then sum/sum_sq == 0. Rather than introduce extra branching logic, just do the extra io for those rare cases.
    sum_array[address]    = sum;
    sum_sq_array[address] = sum_sq;
    sum_counter[address]  = sum_counter_val;

    // TODO: I'm assuming we can avoid reading the mip value when <= histogram min based on short circuit logic, but
    // there may prefetching going on that might be prevented with a second nested if?
    if constexpr ( std::is_same_v<ccfType, __half> ) {
        // I though short circuit logic would be equivalent, but maybe the cuda driver is pre-fetching values? The nested conditional is ~3% faster on total run time
        // indicating we are skipping unnecessary reads.
        if ( max_val > ccfType{TM::MIN_VALUE_TO_MIP} ) {
            if ( max_val > __low2half(mip_psi[address]) ) {
                mip_psi[address]   = __halves2half2(max_val, psi[max_idx]);
                theta_phi[address] = __halves2half2(theta[max_idx], phi[max_idx]);
            }
        }
    }
    else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
        if ( max_val > ccfType{TM::MIN_VALUE_TO_MIP} ) {
            if ( max_val > __low2bfloat16(mip_psi[address]) ) {
                mip_psi[address]   = __halves2bfloat162(max_val, psi[max_idx]);
                theta_phi[address] = __halves2bfloat162(theta[max_idx], phi[max_idx]);
            }
        }
    }
    else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
        if ( max_val > ccfType{TM::MIN_VALUE_TO_MIP} ) {
            if ( max_val > mip_psi[address].x ) {
                mip_psi[address].x   = max_val;
                mip_psi[address].y   = psi[max_idx];
                theta_phi[address].x = theta[max_idx];
                theta_phi[address].y = phi[max_idx];
            }
        }
    }

    return;
}

// TODO: __half2 atomicAdd(__half2 *address, __half2 val);
// TODO: __nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
// This would allow us to double the number of bins in the histogram, and still use atomicAdd reducing contention
template <typename ccfType, typename mipType>
__global__ void __launch_bounds__(TM::histogram_number_of_points)
        AccumulateDistributionKernel(ccfType*             input_ptr,
                                     histogram_storage_t* output_ptr,
                                     const int            NY_img,
                                     const int            pitch_in_pixels_img,
                                     const ccfType        bin_min,
                                     const ccfType        bin_inc,
                                     const int2           pre_padding,
                                     const int2           roi,
                                     const int            n_slices_to_process,
                                     float*               sum_array,
                                     float*               sum_sq_array,
                                     float*               sum_counter,
                                     mipType*             mip_psi,
                                     mipType*             theta_phi,
                                     const ccfType* __restrict__ psi,
                                     const ccfType* __restrict__ theta,
                                     const ccfType* __restrict__ phi,
                                     const float min_counter_val,
                                     const float threshold_val) {

    // initialize temporary accumulation array input_ptr shared memory, this is equal to the number of bins input_ptr the histogram,
    // which may  be more or less than the number of threads in a block
    __shared__ int smem[TM::histogram_number_of_points];

    // Each block has it's own copy of the histogram stored in global memory, found at the linear block index
    // The arrays are initally zeroed in the constructor
    histogram_storage_t* stored_array = &output_ptr[LinearBlockIdx_2dGrid( ) * TM::histogram_number_of_points];

    // Since the number of x-threads is enforced to be = to the number of bins, we can just copy the bins to shared memory
    // Otherwise, we would need a loop to copy the bins to shared memory e.g. ->
    //        smem[threadIdx.x] = __float2int_rn(stored_array[threadIdx.x]);
    // FIXME:     // smem[i] =
    for ( int i = threadIdx.x; i < TM::histogram_number_of_points; i += BlockDimension_2d( ) ) {
        smem[i] = int(stored_array[i]);
    }

    __syncthreads( );

    int     address;
    int     pixel_idx;
    ccfType val;
    // updates our block's partial histogram input_ptr shared memory

    // Currently, threads_per_block x is generally < image_dims.x, but we should be launching enough blocks in X to cover the image, making the grid stride loop overkill.
    // If we were to launch fewer threads_per_grid in x/y then we would still be okay here. The pre_padding values shift the physical x/y indices from the ROI to
    // the physical x/y indicies in the image. The linear address is then calculated in the usual fashion by convert_input using the image coordinates.
    for ( int j = pre_padding.y + physical_Y( ); j < pre_padding.y + roi.y; j += GridStride_2dGrid_Y( ) ) {
        for ( int i = pre_padding.x + physical_X( ); i < pre_padding.x + roi.x; i += GridStride_2dGrid_X( ) ) {
            // address = ((z * NY + y) * pitch_in_pixels) + x;
            address         = j * pitch_in_pixels_img + i;
            ccfType max_val = ccfType{TM::histogram_min};
            int     max_idx = 0;
            // even though we only use kahan summation over ~ 20 numbers, the increase in accuracy is worth it.
            float sum    = sum_array[address];
            float sum_sq = sum_sq_array[address];
            float sum_err{0.f}, sum_sq_err{0.f};
            float sum_counter_val = sum_counter[address];
            for ( int k = 0; k < n_slices_to_process; k++ ) {
                // pixel_idx = __half2int_rd((input_ptr[j * dims.w + i] - bin_min) / bin_inc);
                convert_input(input_ptr, bin_min, bin_inc, pixel_idx, val, address + k * NY_img * pitch_in_pixels_img);

                if ( pixel_idx >= 0 && pixel_idx < TM::histogram_number_of_points )
                    atomicAdd(&smem[pixel_idx], 1);
                sum_squares_and_check_max(val, sum, sum_sq, sum_counter_val, sum_err, sum_sq_err, max_val, max_idx, k, min_counter_val, threshold_val);
            } // loop over slices

            // Now we need to actually write out to global memory for the mip if we are doing it
            write_mip_and_stats(sum_array, sum_sq_array, sum_counter, mip_psi, theta_phi, sum, sum_sq, sum_counter_val, psi, theta, phi, max_val, max_idx, address);
        }
    }

    // write partial histogram into the global memory
    // Converting to long was super slow. Given that I don't care about representing the number exactly, but do care about overflow, just switch the bins to histogram_storage_t
    // As in the read case, we would need a loop if the number of threads != number of bins e.g. ->
    // stored_array[threadIdx.x] = __int2float_rn(smem[threadIdx.x]);
    __syncthreads( );
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
 * @param n_images_this_batch - number of slices to accumulate, must be <= n_imgs_to_process_at_once_
 */

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::AccumulateDistribution(int n_images_this_batch) {
    MyDebugAssertTrue(n_images_this_batch <= n_imgs_to_process_at_once_, "The number of images to accumulate is greater than the number of images to accumulate concurrently");
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_[0]) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    // Copy the host angle arrays to the device (async in calc_stream_[0])
    UpdateDeviceAngleArrays( );

    precheck;
    AccumulateDistributionKernel<<<gridDims_, threadsPerBlock_, 0, calc_stream_[0]>>>(
            ccf_array_.at(active_idx_),
            histogram_,
            image_dims_.y,
            image_dims_.w,
            histogram_min_,
            histogram_step_,
            pre_padding_,
            roi_,
            n_images_this_batch,
            sum_array,
            sum_sq_array,
            sum_counter,
            mip_psi,
            theta_phi,
            (ccfType*)&device_host_angle_arrays_.at(active_idx_)[psi_idx],
            (ccfType*)&device_host_angle_arrays_.at(active_idx_)[theta_idx],
            (ccfType*)&device_host_angle_arrays_.at(active_idx_)[phi_idx],
            min_counter_val_,
            threshold_val_);
    postcheck;

    // Switch the active index
    SetActive_idx( );
};

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::FinalAccumulate( ) {
    MyDebugAssertFalse(cudaStreamQuery(calc_stream_[0]) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");

    const int n_blocks = gridDims_.x * gridDims_.y;
    const int n_bins   = TM::histogram_number_of_points;

    // FIXME: this is from histogram.cu, but is probably not optimal
    dim3 threadsPerBlock_accum_array = dim3(32, 1, 1);
    dim3 gridDims_accum_array        = dim3((TM::histogram_number_of_points + threadsPerBlock_accum_array.x - 1) / threadsPerBlock_accum_array.x, 1, 1);

    precheck;
    FinalAccumulateKernel<<<gridDims_accum_array, threadsPerBlock_accum_array, 0, calc_stream_[0]>>>(histogram_, n_bins, n_blocks);
    postcheck;
}

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::CopyToHostAndAdd(long* array_to_add_to) {

    // Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
    histogram_storage_t* tmp_array;
    cudaErr(cudaHostAlloc(&tmp_array, TM::histogram_number_of_points * sizeof(histogram_storage_t), cudaHostAllocDefault));
    cudaErr(cudaMemcpy(tmp_array, histogram_, TM::histogram_number_of_points * sizeof(histogram_storage_t), cudaMemcpyDeviceToHost));

    for ( int iBin = 0; iBin < TM::histogram_number_of_points; iBin++ ) {
        array_to_add_to[iBin] += long(tmp_array[iBin]);
    }

    cudaErr(cudaFreeHost(tmp_array));
}

// Note: we allow for float in the constructor checking, however, we don't need this for our implementation, so we won't instantiate it.
// template class TM_EmpiricalDistribution<float>;

// TODO: I'm not sure  __restrict__ can be applied to the sum image b/c the value is both read and written to, but this migh tbe okay.
__global__ void AccumulateSumsKernel(float* sum,
                                     float* sumsq,
                                     float* __restrict__ sum_img_array,
                                     float* __restrict__ sq_sum_img_array,
                                     float*    sum_counter,
                                     const int numel) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < numel ) {

        sum_img_array[x] += sum[x];
        sq_sum_img_array[x] += sumsq[x];

        sum[x]         = 0.0f;
        sumsq[x]       = 0.0f;
        sum_counter[x] = 0.0f;
    }
}

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::CopySumAndSumSqAndZero(GpuImage& sum_img, GpuImage& sq_sum_img) {
    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((image_plane_mem_allocated_ + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    AccumulateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(sum_array,
                                                                                sum_sq_array,
                                                                                sum_img.real_values,
                                                                                sq_sum_img.real_values,
                                                                                sum_counter,
                                                                                sq_sum_img.real_memory_allocated);
    postcheck;
}

//                                 const __half* __restrict__ secondary_peaks,
//const int n_peaks

template <typename mipType>
__global__ void MipToImageKernel(const mipType* __restrict__ mip_psi,
                                 const mipType* __restrict__ theta_phi,
                                 const int numel,
                                 float* __restrict__ mip,
                                 float* __restrict__ psi,
                                 float* __restrict__ theta,
                                 float* __restrict__ phi) {

    const int x = physical_X( );

    if ( x < numel ) {
        mip[x]   = __low2float(mip_psi[x]);
        psi[x]   = __high2float(mip_psi[x]);
        theta[x] = __low2float(theta_phi[x]);
        phi[x]   = __high2float(theta_phi[x]);
    }

    // if ( n_peaks == 1 ) {

    // }
    // else {
    //     int offset;
    //     for ( int iPeak = 0; iPeak < n_peaks; iPeak++ ) {
    //         offset = x + numel * iPeak; // out puts are NX * NY * NZ

    //         mip[offset]   = (cufftReal)secondary_peaks[offset];
    //         psi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks];
    //         theta[offset] = (cufftReal)secondary_peaks[offset + numel * n_peaks * 2];
    //         phi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks * 3];
    //     }
    // }
}

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::MipToImage(GpuImage& d_max_intensity_projection,
                                                            GpuImage& d_best_psi,
                                                            GpuImage& d_best_theta,
                                                            GpuImage& d_best_phi) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((image_plane_mem_allocated_ + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // third arg was secondary_peaks,
    // last arg was n_global_search_images_to_save
    MipToImageKernel<mipType><<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(mip_psi,
                                                                                     theta_phi,
                                                                                     image_plane_mem_allocated_,
                                                                                     d_max_intensity_projection.real_values,
                                                                                     d_best_psi.real_values,
                                                                                     d_best_theta.real_values,
                                                                                     d_best_phi.real_values);
    postcheck;
}

// Apparenty clang cares if this is not at the end of the file, and doesn't generate these instantiations for any methods defined after.
template class TM_EmpiricalDistribution<__half, __half2>;
template class TM_EmpiricalDistribution<__nv_bfloat16, __nv_bfloat162>;