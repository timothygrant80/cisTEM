/**
 * @file template_matching_empirical_distribution.cu
 * @brief CUDA implementation of the TM_EmpiricalDistribution class.
 *
 * This file contains the CUDA kernels and device management functions for the
 * TM_EmpiricalDistribution class, which is responsible for GPU-accelerated
 * accumulation of statistics (sum, sum of squares, histogram) from
 * batches of Cross-Correlation Function (CCF) images.
 *
 * Key functionalities include:
 * - Initialization of CUDA resources (streams, events, device memory).
 * - CUDA kernels for accumulating per-pixel statistics and histograms.
 * - Management of data transfers between host and device.
 * - Generation of Maximum Intensity Projections (MIPs) and corresponding angle maps.
 *
 * @note Thread Safety: The methods of TM_EmpiricalDistribution are intended to be called
 * by a single host thread. Internal operations are enqueued onto a dedicated CUDA stream
 * (`calc_stream_`) for asynchronous execution on the GPU. Synchronization primitives
 * like `cudaEventSynchronize` are used where necessary to coordinate host and device.
 * The `active_idx_` mechanism for double buffering CCF and angle data is managed
 * internally and does not make the class methods thread-safe for concurrent host calls.
 */

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
TM_EmpiricalDistribution<ccfType, mipType>::TM_EmpiricalDistribution(GpuImage* reference_image,
                                                                     int2      pre_padding,
                                                                     int2      roi) : pre_padding_{pre_padding},
                                                                                 roi_{roi},
                                                                                 higher_order_moments_{false},
                                                                                 image_plane_mem_allocated_{reference_image->real_memory_allocated} {

    // Design Note: This constructor initializes all necessary GPU resources.
    // - A dedicated CUDA stream (`calc_stream_`) is created for all operations within this class instance.
    //   This allows for asynchronous execution and potential overlap with CPU tasks or other GPU operations
    //   if the application is designed to support it (though this class itself is single-threaded from host).
    // - A CUDA event (`mip_stack_is_ready_event_`) is used for fine-grained synchronization,
    //   particularly for signaling the completion of MIP stack processing.
    // - GPU memory is allocated for statistical arrays, CCF image batches (double-buffered),
    //   angle data, and the histogram.
    // - Launch parameters for CUDA kernels are determined based on the reference image dimensions and ROI.

    std::cerr << "n_images" << n_imgs_to_process_at_once_ << std::endl;
    int least_priority, highest_priority;

    my_rng_ = std::make_unique<RandomNumberGenerator>(pi_v<float>);

    cudaErr(cudaDeviceGetStreamPriorityRange(&least_priority, &highest_priority));
    cudaErr(cudaStreamCreateWithPriority(&calc_stream_[0], cudaStreamNonBlocking, least_priority));
    cudaErr(cudaEventCreateWithFlags(&mip_stack_is_ready_event_[0], cudaEventBlockingSync)); // blocking sync makes the host wait if calling cudaEventSynchronize

    image_dims_.x = reference_image->dims.x;
    image_dims_.y = reference_image->dims.y;
    image_dims_.z = reference_image->dims.z;
    image_dims_.w = reference_image->dims.w;

    MyDebugAssertTrue(image_dims_.x > 0 && image_dims_.y > 0 && image_dims_.z > 0 && image_dims_.w > 0, "Image dimensions must be > 0");

    // Set-up the launch configuration - assumed to be a real space image.
    // WARNING: this is up to the developer to ensure, as we'll use pointers for the input arrays
    // Note: we prefer the "1d" grid as a NxN patch is more likely to have similar values than a N^2x1 line, and so more atomic collisions in the histogram kernel.
    // with grid_sub_division = 1 size of histogram storage == image size
    constexpr int grid_sub_division = 1;
    reference_image->ReturnLaunchParametersNoFFTWPadding<TM::histogram_number_of_points, 1>(roi_.x / grid_sub_division, roi_.y / grid_sub_division, 1, gridDims_, threadsPerBlock_);

    // reference_image->ReturnLaunchParametersLimitSMs(0.1, TM::histogram_number_of_points, gridDims_, threadsPerBlock_);
    // std::cerr << "gridDims: " << gridDims_.x << " " << gridDims_.y << " " << gridDims_.z << std::endl;
    // std::cerr << "threadsPerBlock: " << threadsPerBlock_.x << " " << threadsPerBlock_.y << " " << threadsPerBlock_.z << std::endl;
    // exit(1);

    // Every block will have a shared memory array of the size of the number of bins and aggregate those into their own
    // temp arrays. Only at the end of the search will these be added together'

    // Design Note: Histogram data is stored per-block in global memory initially,
    // then aggregated in a final step. Shared memory (`smem`) is used within
    // `AccumulateDistributionKernel` for efficient per-block histogram updates.
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
    // Thread Safety Note: All CUDA API calls here are enqueued onto `calc_stream_[0]`.
    // The host thread calling this function will continue execution after these calls are enqueued.
    // Synchronization, if needed before using these buffers, must be handled by the caller
    // or by subsequent operations within this class that use `cudaStreamSynchronize` or events.
};

template <typename ccfType, typename mipType>
TM_EmpiricalDistribution<ccfType, mipType>::~TM_EmpiricalDistribution( ) {
    if ( object_initialized_ )
        Delete( );
};

template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::Delete( ) {
    // Design Note: Releases all GPU resources associated with this instance.
    // - Frees all `cudaMallocAsync` allocated memory.
    // - Destroys the CUDA stream and event.
    // - Frees host-pinned memory.
    // All `cudaFreeAsync` calls are enqueued onto `calc_stream_[0]`.
    // A `cudaStreamDestroy` will implicitly synchronize the stream before destruction.
    // Thread Safety Note: This method should only be called when no other operations
    // are pending on `calc_stream_[0]`. The `cudaStreamDestroy` will wait for
    // all enqueued tasks in `calc_stream_[0]` to complete.
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
    // Design Note: Specifically zeros the global histogram buffer on the GPU.
    // This is used during initialization and potentially between distinct accumulation phases if needed.
    // Operation is asynchronous on `calc_stream_[0]`.
    cudaErr(cudaMemsetAsync(histogram_, 0, gridDims_.x * gridDims_.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), calc_stream_[0]));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels and inline helper functions called from EmpiricalDistribution::AccumulateDistribution
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Device function to convert input CCF value to float and calculate histogram bin index.
 * @tparam T Data type of the input pointer (ccfType or histogram_storage_t).
 * @param input_ptr Pointer to the input CCF data.
 * @param pixel_idx Output: calculated histogram bin index.
 * @param address Linear memory address of the CCF value.
 * @return The CCF value converted to float.
 */
template <typename T>
inline __device__ float convert_input(const T* __restrict__ input_ptr,
                                      int& pixel_idx,
                                      int  address) {

    // int((val - min) / step)
    // int((val -min) * inverse_step)
    // using these compile time constants do a fused multiply add
    float val;

    if constexpr ( std::is_same_v<T, __half> ) {
        val = __half2float(input_ptr[address]);
    }
    if constexpr ( std::is_same_v<T, __nv_bfloat16> ) {
        val = __bfloat162float(input_ptr[address]);
    }
    if constexpr ( std::is_same_v<T, histogram_storage_t> ) {
        val = input_ptr[address];
    }
    constexpr float neg_hist_min_div_hist_step = -TM::histogram_min * TM::histogram_step_inverse;

    pixel_idx = __float2int_rd(__fmaf_rn(val, TM::histogram_step_inverse, neg_hist_min_div_hist_step));

    return val;
}

/**
 * @brief Device function to update sum, sum of squares using Kahan summation, and track max CCF value.
 * Implements a trimming logic based on standard deviation for robust statistics.
 * @param val Current CCF value.
 * @param sum Accumulated sum (updated by reference).
 * @param sum_sq Accumulated sum of squares (updated by reference).
 * @param sum_counter_val Accumulation counter (updated by reference).
 * @param sum_err Kahan summation error term for sum (updated by reference).
 * @param sum_sq_err Kahan summation error term for sum_sq (updated by reference).
 * @param max_val Current maximum CCF value found for this pixel (updated by reference).
 * @param max_idx Index of the image in the batch corresponding to max_val (updated by reference).
 * @param idx Current image index in the batch.
 * @param min_counter_val Minimum count for robust statistics calculation.
 * @param threshold_val Sigma threshold for outlier rejection.
 */
inline __device__ void sum_squares_and_check_max(const float val,
                                                 float&      sum,
                                                 float&      sum_sq,
                                                 float&      sum_counter_val,
                                                 float&      sum_err,
                                                 float&      sum_sq_err,
                                                 float&      max_val,
                                                 int&        max_idx,
                                                 int         idx,
                                                 const float min_counter_val,
                                                 const float threshold_val) {

    if ( val > max_val ) {
        max_val = val;
        max_idx = idx;
    }

    // if ( sum_counter_val == 0.f || fabsf(val - sum / sum_counter_val) < sqrtf(((sum_sq / sum_counter_val) - powf(sum / sum_counter_val, 2))) * 3.0f ) {

    // for Welfords
    // For Kahan summation
    float mean_val = sum / sum_counter_val;

    if ( sum_counter_val < min_counter_val || fabsf((val - mean_val) * rsqrtf(sum_sq / sum_counter_val - mean_val * mean_val)) < threshold_val ) {
        sum_counter_val += 1.0f;

        // Kahan summation
        const float y = val - sum_err;
        const float t = sum + y;
        sum_err       = (t - sum) - y;
        sum           = t;

        const float y2 = __fmaf_ieee_rn(val, val, -sum_sq_err);
        const float t2 = sum_sq + y2;
        sum_sq_err     = (t2 - sum_sq) - y2;
        sum_sq         = t2;
    }
}

/**
 * @brief Device function to write accumulated statistics and MIP data to global memory.
 * @tparam ccfType Data type of CCF values.
 * @tparam mipType Data type of MIP values (packed).
 * @param sum_array Device pointer to sum array.
 * @param sum_sq_array Device pointer to sum of squares array.
 * @param sum_counter Device pointer to counter array.
 * @param mip_psi Device pointer to packed MIP value and psi angle array.
 * @param theta_phi Device pointer to packed theta and phi angle array.
 * @param sum Current sum for the pixel.
 * @param sum_sq Current sum of squares for the pixel.
 * @param sum_counter_val Current counter for the pixel.
 * @param psi Device pointer to psi angles for the current batch.
 * @param theta Device pointer to theta angles for the current batch.
 * @param phi Device pointer to phi angles for the current batch.
 * @param max_val Maximum CCF value found for this pixel in the current batch.
 * @param max_idx Index within the batch corresponding to max_val.
 * @param address Linear memory address for the current pixel.
 */
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
                                           const float max_val,
                                           const int   max_idx,
                                           const int   address) {

    // There may be rare cases where no stats have been evaluated, but then sum/sum_sq == 0. Rather than introduce extra branching logic, just do the extra io for those rare cases.
    sum_array[address]    = sum;
    sum_sq_array[address] = sum_sq;
    sum_counter[address]  = sum_counter_val;

    // TODO: I'm assuming we can avoid reading the mip value when <= histogram min based on short circuit logic, but
    // there may prefetching going on that might be prevented with a second nested if?
    if constexpr ( std::is_same_v<ccfType, __half> ) {
        // I though short circuit logic would be equivalent, but maybe the cuda driver is pre-fetching values? The nested conditional is ~3% faster on total run time
        // indicating we are skipping unnecessary reads.
        if ( max_val > TM::MIN_VALUE_TO_MIP ) {
            if ( max_val > __low2float(mip_psi[address]) ) {
                mip_psi[address]   = __halves2half2(__float2half_rn(max_val), psi[max_idx]);
                theta_phi[address] = __halves2half2(theta[max_idx], phi[max_idx]);
            }
        }
    }
    else if constexpr ( std::is_same_v<ccfType, __nv_bfloat16> ) {
        if ( max_val > TM::MIN_VALUE_TO_MIP ) {
            if ( max_val > __low2float(mip_psi[address]) ) {
                mip_psi[address]   = __halves2bfloat162(__float2bfloat16_rn(max_val), psi[max_idx]);
                theta_phi[address] = __halves2bfloat162(theta[max_idx], phi[max_idx]);
            }
        }
    }
    else if constexpr ( std::is_same_v<ccfType, histogram_storage_t> ) {
        if ( max_val > TM::MIN_VALUE_TO_MIP ) {
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

/**
 * @brief CUDA kernel to accumulate CCF statistics and update MIPs.
 *
 * This kernel processes a batch of CCF images. For each pixel:
 * 1. Iterates through all images in the batch.
 * 2. Converts CCF value, calculates histogram bin, and updates shared memory histogram using atomicAdd.
 * 3. Updates sum, sum of squares (with Kahan summation and outlier trimming), and tracks the maximum CCF value and corresponding angles.
 * 4. After processing all images in the batch for a pixel, writes the updated sum, sum_sq, counter, and MIP data (if current max is greater than stored MIP) to global memory.
 * 5. Finally, writes the block's partial histogram from shared memory to its designated spot in global memory.
 *
 * @note Each thread block processes a tile of the image.
 * @note Shared memory `smem` is used for efficient, coalesced updates to the histogram within a block.
 * @note Angle data (psi, theta, phi) for the current batch is read from global memory.
 */
template <typename ccfType, typename mipType>
__global__ void __launch_bounds__(TM::histogram_number_of_points)
        AccumulateDistributionKernel(const ccfType* __restrict__ input_ptr,
                                     histogram_storage_t* __restrict__ output_ptr,
                                     const __grid_constant__ int  plane_stride_pixels_img,
                                     const __grid_constant__ int  pitch_in_pixels_img,
                                     const __grid_constant__ int2 pre_padding,
                                     const __grid_constant__ int2 roi,
                                     const __grid_constant__ int  n_slices_to_process,
                                     float*                       sum_array,
                                     float*                       sum_sq_array,
                                     float*                       sum_counter,
                                     mipType* __restrict__ mip_psi,
                                     mipType* __restrict__ theta_phi,
                                     const ccfType* __restrict__ psi,
                                     const ccfType* __restrict__ theta,
                                     const ccfType* __restrict__ phi,
                                     const __grid_constant__ float min_counter_val,
                                     const __grid_constant__ float threshold_val) {

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

    // updates our block's partial histogram input_ptr shared memory

    // Currently, threads_per_block x is generally < image_dims.x, but we should be launching enough blocks in X to cover the image, making the grid stride loop overkill.
    // If we were to launch fewer threads_per_grid in x/y then we would still be okay here. The pre_padding values shift the physical x/y indices from the ROI to
    // the physical x/y indicies in the image. The linear address is then calculated in the usual fashion by convert_input using the image coordinates.
    for ( int j = pre_padding.y + physical_Y( ); j < pre_padding.y + roi.y; j += GridStride_2dGrid_Y( ) ) {
        for ( int i = pre_padding.x + physical_X( ); i < pre_padding.x + roi.x; i += GridStride_2dGrid_X( ) ) {
            // address = ((z * NY + y) * pitch_in_pixels) + x;
            const int address = j * pitch_in_pixels_img + i;
            float     max_val{TM::histogram_min};
            int       max_idx = 0;
            // even though we only use kahan summation over ~ 20 numbers, the increase in accuracy is worth it.
            float sum    = sum_array[address];
            float sum_sq = sum_sq_array[address];
            float sum_err{0.f}, sum_sq_err{0.f};
            float sum_counter_val = sum_counter[address];
            for ( int k = 0; k < n_slices_to_process; k++ ) {
                // pixel_idx = __half2int_rd((input_ptr[j * dims.w + i] -  TM::histogram_min) /  TM::histogram_step);
                int         pixel_idx;
                const float val = convert_input(input_ptr, pixel_idx, address + k * plane_stride_pixels_img);

                if ( pixel_idx >= 0 && pixel_idx < TM::histogram_number_of_points )
                    atomicAdd(&smem[pixel_idx], 1);
                sum_squares_and_check_max(val,
                                          sum,
                                          sum_sq,
                                          sum_counter_val,
                                          sum_err,
                                          sum_sq_err,
                                          max_val,
                                          max_idx,
                                          k,
                                          min_counter_val,
                                          threshold_val);
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

/**
 * @brief CUDA kernel to finalize histogram accumulation.
 *
 * This kernel sums up the partial histograms computed by each block in `AccumulateDistributionKernel`.
 * Each thread accumulates values for a specific bin across all block-local histograms.
 * The final aggregated histogram is written back to the first block's portion of the histogram memory.
 *
 * @param input_ptr Device pointer to the histogram data (contains block-wise partial histograms).
 * @param n_bins Total number of histogram bins.
 * @param n_blocks Total number of thread blocks that contributed to partial histograms.
 */
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
 * @param n_images_this_batch - number of slices to accumulate, must be <= n_imgs_to_process_at_once_
 *
 * @note Design:
 * - Ensures the number of images in the batch is valid.
 * - Asynchronously copies the current batch's angle data from host-pinned memory to device memory
 *   using `UpdateDeviceAngleArrays()`, which enqueues the copy on `calc_stream_[0]`.
 * - Launches `AccumulateDistributionKernel` on `calc_stream_[0]`. This kernel reads from
 *   `ccf_array_.at(active_idx_)` and `device_host_angle_arrays_.at(active_idx_)`.
 * - After launching the kernel, it calls `SetActive_idx()` to switch the `active_idx_`.
 *   This allows the host to start filling the *next* `ccf_array_` buffer and `host_angle_arrays_`
 *   while the current batch is being processed on the GPU, achieving H2D-D2D overlap.
 *
 * @note Thread Safety: This method is not thread-safe for concurrent calls from multiple host threads.
 * It relies on `active_idx_` for internal double buffering, managed by a single calling sequence.
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
            image_dims_.y * image_dims_.w,
            image_dims_.w,
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
    // This allows the CPU to prepare the next batch of CCF data and angles in the inactive buffers
    // while the GPU is processing the current batch using the (previously) active buffers.
    SetActive_idx( );
};

/**
 * @brief Performs the final accumulation of the histogram.
 *
 * Launches `FinalAccumulateKernel` to sum partial histograms from each block
 * into a single global histogram. This is typically called once after all batches
 * have been processed by `AccumulateDistribution`.
 *
 * @note All GPU operations are enqueued on `calc_stream_[0]`.
 */
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

/**
 * @brief Copies the final aggregated histogram from GPU to a host array and adds to it.
 *
 * This function performs a synchronous D2H copy of the histogram.
 *
 * @param array_to_add_to Host array to which the GPU histogram data will be added.
 *
 * @note This function involves a synchronous `cudaMemcpyDeviceToHost`.
 * It will block the calling host thread until the copy is complete.
 * This implies that all preceding work on `calc_stream_[0]` affecting the histogram
 * (i.e., `AccumulateDistributionKernel` calls and `FinalAccumulateKernel`)
 * should ideally be synchronized with `calc_stream_[0]` before this copy,
 * or `FinalAccumulate` should be called and synchronized before this.
 * The current implementation copies from `histogram_` which is the target of `FinalAccumulateKernel`.
 */
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

/**
 * @brief CUDA kernel to accumulate sums and sums of squares into global sum images, then zero per-batch accumulators.
 *
 * This kernel is called by `CopySumAndSumSqAndZero`. It adds the per-batch `sum_array` and `sum_sq_array`
 * (which hold sums across `n_imgs_to_process_at_once_` images for each pixel) to the global
 * `sum_img_array` and `sq_sum_img_array` (which are GpuImage objects holding the total sums across all batches).
 * After adding, it zeros out `sum_array`, `sum_sq_array`, and `sum_counter` to prepare for the next call
 * to `AccumulateDistributionKernel` (though typically `CopySumAndSumSqAndZero` is called at the very end).
 *
 * @param sum Device pointer to the per-batch sum array (read from, then zeroed).
 * @param sumsq Device pointer to the per-batch sum of squares array (read from, then zeroed).
 * @param sum_img_array Device pointer to the global sum image array (accumulated into).
 * @param sq_sum_img_array Device pointer to the global sum of squares image array (accumulated into).
 * @param sum_counter Device pointer to the per-batch counter array (zeroed).
 * @param numel Total number of elements (pixels) in the image.
 *
 * @note The use of `cudaStreamPerThread` in the calling function `CopySumAndSumSqAndZero` is unusual
 *       for a class that primarily uses a dedicated stream (`calc_stream_`). This might be an oversight
 *       or intended for a specific reason not immediately obvious. If `sum_array`, etc., are written to
 *       by kernels on `calc_stream_`, using `cudaStreamPerThread` here without proper synchronization
 *       could lead to race conditions. It should ideally use `calc_stream_[0]`.
 */
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

/**
 * @brief Copies accumulated sum and sum_sq arrays to provided GpuImage objects and zeros the internal accumulators.
 *
 * This function launches `AccumulateSumsKernel` to transfer the content of the internal
 * `sum_array` and `sum_sq_array` (which are accumulated per batch by `AccumulateDistributionKernel`)
 * into the provided `sum_img` and `sq_sum_img` GpuImage objects. After the copy/accumulation,
 * the internal `sum_array`, `sum_sq_array`, and `sum_counter` are zeroed.
 * This is typically called after all image batches have been processed to get the final
 * sum and sum of squares images.
 *
 * @param sum_img Output GpuImage to store the total sum array.
 * @param sq_sum_img Output GpuImage to store the total sum_sq array.
 *
 * @note Critical: The kernel launch `AccumulateSumsKernel<<<..., cudaStreamPerThread>>>` uses `cudaStreamPerThread` (default stream).
 * If `sum_array`, `sum_sq_array`, `sum_counter` were last written by kernels on `calc_stream_[0]` (e.g., `AccumulateDistributionKernel`),
 * there's a potential race condition unless `calc_stream_[0]` was synchronized before this call.
 * This should likely use `calc_stream_[0]` for consistency and safety within the class's stream model.
 * Consider changing `cudaStreamPerThread` to `calc_stream_[0]`.
 */
template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::CopySumAndSumSqAndZero(GpuImage& sum_img, GpuImage& sq_sum_img) {
    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((image_plane_mem_allocated_ + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // Potential Stream Issue: Uses cudaStreamPerThread. If sum_array etc. are populated
    // by kernels on calc_stream_[0], this needs synchronization or to use calc_stream_[0].
    AccumulateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(sum_array,
                                                                                sum_sq_array,
                                                                                sum_img.real_values,
                                                                                sq_sum_img.real_values,
                                                                                sum_counter,
                                                                                sq_sum_img.real_memory_allocated);
    postcheck;
}

/**
 * @brief CUDA kernel to convert packed MIP and angle data to separate float arrays.
 *
 * This kernel reads from `mip_psi` (packed MIP value and psi angle) and
 * `theta_phi` (packed theta and phi angles) and writes them out to separate
 * float arrays corresponding to GpuImage real_values.
 *
 * @tparam mipType Data type of the packed MIP/angle arrays (e.g., __half2, __nv_bfloat162).
 * @param mip_psi Device pointer to packed MIP value and psi angle.
 * @param theta_phi Device pointer to packed theta and phi angles.
 * @param numel Total number of elements (pixels).
 * @param mip Output device pointer for MIP values (float).
 * @param psi Output device pointer for psi angles (float).
 * @param theta Output device pointer for theta angles (float).
 * @param phi Output device pointer for phi angles (float).
 *
 * @note The kernel unpacks types like `__half2` into two float values.
 * @note The use of `cudaStreamPerThread` in the calling function `MipToImage` has similar
 *       concerns as in `CopySumAndSumSqAndZero` regarding synchronization with `calc_stream_`.
 */
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

/**
 * @brief Converts internal MIP and angle representations to output GpuImage objects.
 *
 * Launches `MipToImageKernel` to unpack the `mip_psi` and `theta_phi` arrays
 * (which store MIPs and angles in a packed format like `__half2` or `__nv_bfloat162`)
 * into the provided floating-point `GpuImage` objects.
 * This is typically called after all processing to retrieve the final MIPs and angle maps.
 *
 * @param d_max_intensity_projection Output GpuImage for the MIP.
 * @param d_best_psi Output GpuImage for the psi angle map.
 * @param d_best_theta Output GpuImage for the theta angle map.
 * @param d_best_phi Output GpuImage for the phi angle map.
 *
 * @note Critical: The kernel launch `MipToImageKernel<mipType><<<..., cudaStreamPerThread>>>` uses `cudaStreamPerThread`.
 * If `mip_psi` and `theta_phi` were last written by `AccumulateDistributionKernel` on `calc_stream_[0]`,
 * a race condition is possible without prior synchronization of `calc_stream_[0]`.
 * This should likely use `calc_stream_[0]` for safety.
 */
template <typename ccfType, typename mipType>
void TM_EmpiricalDistribution<ccfType, mipType>::MipToImage(GpuImage& d_max_intensity_projection,
                                                            GpuImage& d_best_psi,
                                                            GpuImage& d_best_theta,
                                                            GpuImage& d_best_phi) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((image_plane_mem_allocated_ + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // FIXME: Potential Stream Issue: Uses cudaStreamPerThread. If mip_psi and theta_phi are populated
    // by kernels on calc_stream_[0], this needs synchronization or to use calc_stream_[0].
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