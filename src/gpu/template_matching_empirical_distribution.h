#ifndef _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_
#define _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_

/**
 * @brief Construct a new tm empiricaldistribution object
 * 
 * @tparam input_type - this is the the data storage type holding the values to be tracked.
 * Internally, this class uses cascading summation in single precision (fp32)
 *  
 * @tparam per_image - this is a boolean flag indicating whether the class should track the statistics per image 
 * like the cpu version of EmpiricalDistribution or per pixel across many images.
 */

constexpr int n_imgs_to_process_at_once_ = 20;

constexpr int psi_idx          = 0;
constexpr int theta_idx        = 1 * n_imgs_to_process_at_once_;
constexpr int phi_idx          = 2 * n_imgs_to_process_at_once_;
constexpr int do_histogram_idx = 3 * n_imgs_to_process_at_once_;
constexpr int do_stats_idx     = 4 * n_imgs_to_process_at_once_;

using histogram_storage_t = float;

template <typename ccfType, typename mipType, bool per_image = false>
class TM_EmpiricalDistribution {

  private:
    bool        higher_order_moments_;
    int         current_image_index_;
    ccfType     histogram_min_;
    ccfType     histogram_step_;
    int         histogram_n_bins_;
    int2        pre_padding_;
    int2        roi_;
    const float histogram_sampling_rate_;
    const float stats_sampling_rate_;

    std::unique_ptr<RandomNumberGenerator> my_rng_;

    const int image_plane_mem_allocated_;

    float*   sum_array;
    float*   sum_sq_array;
    mipType* mip_psi;
    mipType* theta_phi;
    ccfType* psi;
    ccfType* theta;
    ccfType* phi;

    int active_idx_{ };

    std::array<ccfType*, 2> host_angle_and_bool_arrays_;
    std::array<ccfType*, 2> device_host_angle_and_bool_arrays_;

    std::array<int, 2> n_histogram_samples_this_batch_;
    std::array<int, 2> n_stats_samples_this_batch_;

    std::array<ccfType*, 2> ccf_array_;

    __half* statistics_buffer_;

    dim3 threadsPerBlock_;
    dim3 gridDims_;

    int4 image_dims_;

    histogram_storage_t* histogram_;

    cudaStream_t calc_stream_[1];
    cudaEvent_t  mip_stack_is_ready_event_[1];

  public:
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
    TM_EmpiricalDistribution(GpuImage&           reference_image,
                             histogram_storage_t histogram_min,
                             histogram_storage_t histogram_step,
                             int2                pre_padding,
                             int2                roi,
                             const float         histogram_sampling_rate,
                             const float         stats_sampling_rate);

    ~TM_EmpiricalDistribution( );

    // delete the copy and move  constructors as these are not handled or needed.
    TM_EmpiricalDistribution(const TM_EmpiricalDistribution&)            = delete;
    TM_EmpiricalDistribution& operator=(const TM_EmpiricalDistribution&) = delete;
    TM_EmpiricalDistribution(TM_EmpiricalDistribution&&)                 = delete;
    TM_EmpiricalDistribution& operator=(TM_EmpiricalDistribution&&)      = delete;

    inline int GetActiveIdx( ) { return active_idx_; }

    inline void SetActive_idx( ) {
        if ( active_idx_ == 1 )
            active_idx_ = 0;
        else
            active_idx_ = 1;
    }

    inline int n_imgs_to_process_at_once( ) { return n_imgs_to_process_at_once_; }

    inline ccfType* GetCCFArray(const int current_slice_to_process) {
        return &ccf_array_.at(active_idx_)[image_plane_mem_allocated_ * current_slice_to_process];
    }

    void AllocateAndZeroStatisticalArrays( );
    void ZeroHistogram( );

    void ZeroSamplingCounters( ) {
        for ( auto& counter : n_histogram_samples_this_batch_ )
            counter = 0;
        for ( auto& counter : n_stats_samples_this_batch_ )
            counter = 0;
    };

    void AccumulateDistribution(int n_images_this_batch, long& histogram_sampling_counter, long& stats_sampling_counter);
    void FinalAccumulate( );
    void CopyToHostAndAdd(long* array_to_add_to);

    inline void
    RecordMipStackIsReadyBlockingHost( ) {
        cudaErr(cudaEventRecord(mip_stack_is_ready_event_[0], calc_stream_[0]));
        // This would make a stream wait
        // cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, mip_stack_is_ready_event_[0], cudaEventWaitDefault));

        // This would make the host wait
        // cudaErr(cudaEventSynchronize(mip_stack_is_ready_event_[0]));
    }

    inline void
    MakeHostWaitOnMipStackIsReadyEvent( ) {
        cudaErr(cudaEventSynchronize(mip_stack_is_ready_event_[0]));
    }

    inline void UpdateHostAngleArrays(const int current_mip_to_process, const float current_psi, const float current_theta, const float current_phi) {
        MyDebugAssertTrue(current_mip_to_process >= 0 && current_mip_to_process <= n_imgs_to_process_at_once_, "current_mip_to_process is out of bounds");
        if constexpr ( std::is_same_v<ccfType, __half> ) {
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + psi_idx]   = __float2half_rn(current_psi);
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + theta_idx] = __float2half_rn(current_theta);
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + phi_idx]   = __float2half_rn(current_phi);
        }
        else {
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + psi_idx]   = __float2bfloat16_rn(current_psi);
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + theta_idx] = __float2bfloat16_rn(current_theta);
            host_angle_and_bool_arrays_.at(active_idx_)[current_mip_to_process + phi_idx]   = __float2bfloat16_rn(current_phi);
        }
    }

    // This would probably be better if all the arrays were contiguous in memory so we only have one api call per round FIXME
    inline void UpdateDeviceAngleArrays( ) {
        ZeroSamplingCounters( );
        for ( int i_bool = 0; i_bool < n_imgs_to_process_at_once_; i_bool++ ) {
            host_angle_and_bool_arrays_.at(active_idx_)[i_bool + do_histogram_idx] = (my_rng_->GetUniformRandomSTD(0.0f, 1.0f) <= histogram_sampling_rate_) ? ccfType{1.0} : ccfType{0.0f};
            if ( host_angle_and_bool_arrays_.at(active_idx_)[i_bool + do_histogram_idx] )
                n_histogram_samples_this_batch_.at(active_idx_)++;
            host_angle_and_bool_arrays_.at(active_idx_)[i_bool + do_stats_idx] = (my_rng_->GetUniformRandomSTD(0.0f, 1.0f) <= stats_sampling_rate_) ? ccfType{1.0} : ccfType{0.0f};
            if ( host_angle_and_bool_arrays_.at(active_idx_)[i_bool + do_stats_idx] )
                n_stats_samples_this_batch_.at(active_idx_)++;
        }
        cudaErr(cudaMemcpyAsync(device_host_angle_and_bool_arrays_.at(active_idx_), host_angle_and_bool_arrays_.at(active_idx_), n_imgs_to_process_at_once_ * sizeof(ccfType) * 5, cudaMemcpyHostToDevice, calc_stream_[0]));
    }

    void CopySumAndSumSqAndZero(GpuImage& sum, GpuImage& sq_sum);
    void MipToImage(GpuImage& d_max_intensity_projection,
                    GpuImage& d_best_psi,
                    GpuImage& d_best_theta,
                    GpuImage& d_best_phi);
};

#endif