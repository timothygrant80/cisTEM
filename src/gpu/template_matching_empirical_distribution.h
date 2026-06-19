#ifndef _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_
#define _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_

#include <memory>

/**
 * @file template_matching_empirical_distribution.h
 * @brief Defines the TM_EmpiricalDistribution class for GPU-accelerated empirical distribution calculations.
 *
 * This class is designed to accumulate statistics (sum, sum of squares, and optionally a histogram)
 * from a series of 2D cross-correlation function (CCF) images, typically generated during template matching.
 * It processes images in batches on the GPU for efficiency.
 *
 * @tparam ccfType The data type for CCF values (e.g., __half, bfloat16).
 * @tparam mipType The data type for Maximum Intensity Projection (MIP) values.
 *
 * @note This class is intended to be used by a single host thread controlling a GPU.
 * While it uses CUDA streams for asynchronous operations, its methods are not designed
 * to be called concurrently from multiple host threads. Synchronization is managed internally
 * for GPU operations but relies on the calling thread for overall coordination.
 */

/**
 * @brief Construct a new tm empiricaldistribution object
 * 
 * @tparam input_type - this is the the data storage type holding the values to be tracked.
 * Internally, this class uses cascading summation in single precision (fp32)
 *  
 * @tparam per_image - this is a boolean flag indicating whether the class should track the statistics per image 
 * like the cpu version of EmpiricalDistribution or per pixel across many images.
 */

/** @brief Data type used for storing histogram bins. */
using histogram_storage_t = float;

/**
 * @class TM_EmpiricalDistribution
 * @brief Manages GPU-accelerated accumulation of empirical distribution statistics from CCF images.
 *
 * This class handles the allocation of GPU memory, the setup of CUDA streams and events,
 * and the execution of CUDA kernels to compute sums, sums of squares, and histograms
 * from batches of CCF images. It also manages the transfer of data between host and device.
 *
 * Key operations include:
 * - Initialization with image dimensions and ROI.
 * - Allocation and zeroing of statistical arrays on the GPU.
 * - Accumulation of statistics from batches of CCF images.
 * - Finalization of accumulation.
 * - Copying results back to the host.
 * - Generation of Maximum Intensity Projections (MIPs) and corresponding angle maps.
 *
 * @tparam ccfType Data type for CCF values (e.g., __half, bfloat16).
 * @tparam mipType Data type for MIP values.
 *
 * @note Thread Safety: This class is not designed for concurrent access from multiple host threads.
 * All method calls should be serialized by the owning host thread. Internal GPU operations
 * are managed with CUDA streams and events for asynchronicity and synchronization with the GPU.
 * The `mip_dbl_buffer_idx_` member is used for double buffering of CCF and angle arrays to allow
 * data transfer to overlap with computation, but this is managed internally and does not
 * imply thread safety for external calls.
 */
template <typename ccfType, typename mipType>
class TM_EmpiricalDistribution {

  private:
    bool higher_order_moments_;
    bool object_initialized_{ };
    int  current_image_index_;

    int2 pre_padding_;
    int2 roi_;

    std::unique_ptr<RandomNumberGenerator> my_rng_;

    const int image_plane_mem_allocated_;

    /** @brief Number of images to process in a single batch on the GPU.
     *  Automatically sized based on image dimensions to balance memory usage. */
    const int n_imgs_to_process_at_once_;

    float*   sum_array;
    float*   sum_sq_array;
    float*   sum_counter;
    float*   sum_error_array;
    float*   sum_sq_error_array;
    mipType* mip_psi;
    mipType* theta_phi;
    ccfType* psi;
    ccfType* theta;
    ccfType* phi;

    // Can be 0 or 1
    int mip_dbl_buffer_idx_{ };
    // Can be 0 n_mips_to_process_at_once - 1
    std::array<int, 2> mip_active_slice_{ };

    std::array<ccfType*, 2> host_angle_arrays_;
    std::array<ccfType*, 2> device_angle_arrays_;

    std::array<ccfType*, 2> ccf_array_;

    __half* statistics_buffer_;

    dim3 threadsPerBlock_;
    dim3 gridDims_;

    int4 image_dims_;

    histogram_storage_t* histogram_;

    cudaStream_t calc_stream_[1];
    cudaEvent_t  mip_stack_is_ready_event_[1];
    cudaEvent_t  ccf_dbl_buffer_ready_event_[2]; ///< Signals CCF writes complete for each double buffer

  public:
    /**
     * @brief Construct a new TM_EmpiricalDistribution object.
     *
     * Initializes GPU resources, including memory for statistical arrays, CCF batches,
     * angle maps, and the histogram. Sets up CUDA stream and event for asynchronous operations.
 * 
     * @param reference_image Pointer to a GpuImage object used to determine image dimensions and properties.
     *                        This image's properties (width, height) are used to configure GPU launch parameters.
     * @param pre_padding Pre-padding applied to images.
     * @param roi Region of Interest for processing.
     *
     * @note Both histogram_min and histogram_step (if applicable, though not direct params here)
     *       must be > 0 for histogram creation. The number of histogram bins is typically fixed.
     * @note The `n_images_to_accumulate_concurrently` parameter mentioned in the original comment
     *       is now a compile-time constant `n_imgs_to_process_at_once_`.
 */
    TM_EmpiricalDistribution(GpuImage* reference_image,
                             int2      pre_padding,
                             int2      roi);

    /**
     * @brief Destructor for TM_EmpiricalDistribution.
     * Calls Delete() to release GPU resources.
     */
    ~TM_EmpiricalDistribution( );

    /**
     * @brief Releases all allocated GPU memory and destroys CUDA stream/event.
     * Ensures proper cleanup of resources.
     */
    void Delete( );

    // delete the copy and move constructors as these are not handled or needed.
    TM_EmpiricalDistribution(const TM_EmpiricalDistribution&)            = delete;
    TM_EmpiricalDistribution& operator=(const TM_EmpiricalDistribution&) = delete;
    TM_EmpiricalDistribution(TM_EmpiricalDistribution&&)                 = delete;
    TM_EmpiricalDistribution& operator=(TM_EmpiricalDistribution&&)      = delete;

    /**
     * @brief Toggles the active index for double buffering.
     * Switches between 0 and 1.
     * Because all work on these buffers is in calc_stream_ switching buffers internally has no race condition.
     */
    inline void ToggleActiveDoubleBufferIdx( ) {
        if ( mip_dbl_buffer_idx_ == 1 )
            mip_dbl_buffer_idx_ = 0;
        else
            mip_dbl_buffer_idx_ = 1;

        // Also reset the current mip index for the new active buffer
        mip_active_slice_.at(mip_dbl_buffer_idx_) = 0;
    }

    inline int GetCurrentMip_idx( ) {
        return mip_active_slice_.at(mip_dbl_buffer_idx_);
    }

    inline void IncrementCurrentMip_idx( ) {
        mip_active_slice_.at(mip_dbl_buffer_idx_)++;
    }

    /**
     * @brief Gets the number of images processed at once in a batch.
     * @return The batch size.
     */
    inline int n_imgs_to_process_at_once( ) const { return n_imgs_to_process_at_once_; }

    /** @brief Index offset for psi angle data within batched angle arrays. */
    inline int psi_idx( ) const { return 0; }

    /** @brief Index offset for theta angle data within batched angle arrays. */
    inline int theta_idx( ) const { return n_imgs_to_process_at_once_; }

    /** @brief Index offset for phi angle data within batched angle arrays. */
    inline int phi_idx( ) const { return 2 * n_imgs_to_process_at_once_; }

    /**
     * @brief Gets a device pointer to the CCF array for the current slice in the active buffer.
     * @param current_slice_to_process The index of the slice within the current batch.
     * @return Device pointer to the CCF data for the specified slice.
     */
    inline ccfType* GetCCFArray( ) {

        // Provides a pointer to the start of the CCF data for the 'current_slice_to_process'
        // within the currently active batch buffer ('mip_dbl_buffer_idx_').
        const int current_mip_to_process = GetCurrentMip_idx( );
        MyDebugAssertTrue(current_mip_to_process >= 0 && current_mip_to_process < n_imgs_to_process_at_once_, "current_mip_to_process (%d) should be >= 0 and < n_imgs_to_process_at_once_ (%d)", current_mip_to_process, n_imgs_to_process_at_once_);

        return &ccf_array_.at(mip_dbl_buffer_idx_)[image_plane_mem_allocated_ * current_mip_to_process];
    }

    /**
     * @brief Allocates and zeros the statistical arrays (sum, sum_sq, sum_counter) on the GPU.
     * Also allocates the histogram if configured.
     */
    void AllocateAndZeroStatisticalArrays( );

    /**
     * @brief Zeros the histogram array on the GPU.
     */
    void ZeroHistogram( );

    /**
     * @brief Accumulates statistics from a batch of CCF images.
     *
     * Launches CUDA kernels to update sum, sum_sq, sum_counter, and histogram arrays
     * based on the CCF data in the active device buffer.
     * This is the core GPU processing step for each batch.
     *
     */
    void AccumulateDistribution( );

    /**
     * @brief Performs final accumulation steps if needed (e.g., for higher-order moments, though not fully implemented).
     * Currently, this function might be a placeholder or for future extensions.
     */
    void FinalAccumulate( );

    /**
     * @brief Copies the histogram data from GPU to host and adds it to a host-side array.
     * @param array_to_add_to Pointer to the host array where histogram data will be added.
     */
    void CopyToHostAndAdd(long* array_to_add_to);

    /**
     * @brief Records an event in the calculation stream to signal that the MIP stack processing is complete.
     * This event can be used by other streams or the host to synchronize.
     * @note The commented-out lines show examples of how a stream or host could wait for this event.
     */
    inline void
    RecordTmEmpricalDist_Event( ) {
        // Records an event into calc_stream_[0] after all preceding work in that stream is complete.
        cudaErr(cudaEventRecord(mip_stack_is_ready_event_[0], calc_stream_[0]));
        // This would make a stream wait
        // cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, mip_stack_is_ready_event_[0], cudaEventWaitDefault));

        // This would make the host wait
        // cudaErr(cudaEventSynchronize(mip_stack_is_ready_event_[0]));
    }

    /**
     * @brief Makes the host CPU wait until the `mip_stack_is_ready_event_` is recorded.
     * This ensures that GPU operations related to MIP stack generation are finished before the host proceeds.
     */
    inline void
    MakeHostWaitOnTmEmpricalDist_Stream( ) {
        // Blocks the calling host thread until the mip_stack_is_ready_event_[0] has been recorded.
        cudaErr(cudaEventSynchronize(mip_stack_is_ready_event_[0]));
    }

    /**
     * @brief Records an event signaling that CCF writes to the current double buffer are complete.
     *
     * This should be called after the final FFT of a batch writes to the CCF buffer.
     * The event is recorded on the stream where CCF writes occur (typically cudaStreamPerThread).
     * AccumulateDistribution will wait on this event before the kernel reads the CCF data.
     *
     * @param ccf_write_stream The CUDA stream on which CCF data was written (e.g., cudaStreamPerThread).
     */
    inline void
    RecordCCFBufferReadyEvent(cudaStream_t ccf_write_stream) {
        cudaErr(cudaEventRecord(ccf_dbl_buffer_ready_event_[mip_dbl_buffer_idx_], ccf_write_stream));
    }

    /**
     * @brief Makes calc_stream_ wait for CCF buffer writes to complete before kernel reads.
     *
     * Called internally at the start of AccumulateDistribution to ensure the kernel
     * does not read CCF data before it has been fully written by the FFT operations.
     */
    inline void
    WaitOnCCFBufferReady( ) {
        cudaErr(cudaStreamWaitEvent(calc_stream_[0], ccf_dbl_buffer_ready_event_[mip_dbl_buffer_idx_], cudaEventWaitDefault));
    }

    /**
     * @brief Updates the host-side pinned memory for angle arrays with new angle values.
     * This data will be subsequently copied to the device.
     *
     * @param current_psi Current psi angle.
     * @param current_theta Current theta angle.
     * @param current_phi Current phi angle.
     */
    inline void UpdateHostAngleArrays(const float current_psi, const float current_theta, const float current_phi) {
        const int current_mip_to_process = GetCurrentMip_idx( );
        MyDebugAssertTrue(current_mip_to_process >= 0 && current_mip_to_process < n_imgs_to_process_at_once_, "current_mip_to_process (%d) should be >= 0 and < n_imgs_to_process_at_once_ (%d)", current_mip_to_process, n_imgs_to_process_at_once_);

        // Populates the host-pinned buffer for angle data for the current image in the batch.
        // This buffer is then copied asynchronously to the device.
        // The `mip_dbl_buffer_idx_` ensures writing to the correct buffer in the double-buffering scheme.
        if constexpr ( std::is_same_v<ccfType, __half> ) {
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + psi_idx( )]   = __float2half_rn(current_psi);
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + theta_idx( )] = __float2half_rn(current_theta);
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + phi_idx( )]   = __float2half_rn(current_phi);
        }
        else {
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + psi_idx( )]   = __float2bfloat16_rn(current_psi);
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + theta_idx( )] = __float2bfloat16_rn(current_theta);
            host_angle_arrays_.at(mip_dbl_buffer_idx_)[current_mip_to_process + phi_idx( )]   = __float2bfloat16_rn(current_phi);
        }
        IncrementCurrentMip_idx( );
    }

    /**
     * @brief Asynchronously copies the updated angle arrays from host pinned memory to device memory.
     * Uses the calculation stream `calc_stream_[0]`.
     * @note The FIXME comment suggests a potential optimization to use a single contiguous memory block
     *       for all angle arrays to reduce to one `cudaMemcpyAsync` call.
     */
    // This would probably be better if all the arrays were contiguous in memory so we only have one api call per round FIXME
    inline void UpdateDeviceAngleArrays( ) {
        // Asynchronously copies the entire batch of angle data (psi, theta, phi for all images in the batch)
        // from the host-pinned memory (`host_angle_arrays_`) to the corresponding device memory (`device_angle_arrays_`).
        // This operation is enqueued in `calc_stream_[0]`.
        cudaErr(cudaMemcpyAsync(device_angle_arrays_.at(mip_dbl_buffer_idx_), host_angle_arrays_.at(mip_dbl_buffer_idx_), n_imgs_to_process_at_once_ * sizeof(ccfType) * 3, cudaMemcpyHostToDevice, calc_stream_[0]));
    }

    /**
     * @brief Copies sum and sum_sq arrays from GPU to host GpuImage objects and zeros the device arrays.
     * @param sum GpuImage to store the sum array.
     * @param sq_sum GpuImage to store the sum_sq array.
     */
    void CopySumAndSumSqAndZero(GpuImage& sum, GpuImage& sq_sum);

    /**
     * @brief Generates Maximum Intensity Projection (MIP) images and corresponding angle maps.
     *
     * Processes the accumulated CCF data to find the maximum intensity projection and the
     * psi, theta, and phi angles associated with that maximum for each pixel.
     * The results are stored in the provided GpuImage objects.
     *
     * @param d_max_intensity_projection Output GpuImage for the MIP.
     * @param d_best_psi Output GpuImage for the psi angle map.
     * @param d_best_theta Output GpuImage for the theta angle map.
     * @param d_best_phi Output GpuImage for the phi angle map.
     */
    void MipToImage(GpuImage& d_max_intensity_projection,
                    GpuImage& d_best_psi,
                    GpuImage& d_best_theta,
                    GpuImage& d_best_phi);
};

#endif