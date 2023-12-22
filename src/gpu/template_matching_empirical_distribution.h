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

using histogram_storage_t = float;

template <typename ccfType, typename mipType, bool per_image = false>
class TM_EmpiricalDistribution {

  private:
    bool      higher_order_moments_;
    int       current_image_index_;
    ccfType   histogram_min_;
    ccfType   histogram_step_;
    int       histogram_n_bins_;
    int       n_border_pixels_to_ignore_for_histogram_;
    const int n_images_to_accumulate_concurrently_;

    float*   sum_array;
    float*   sum_sq_array;
    mipType* mip_psi;
    mipType* mip_theta;
    ccfType* psi;
    ccfType* theta;
    ccfType* phi;

    __half* statistics_buffer_;

    dim3 threadsPerBlock_;
    dim3 gridDims_;

    int4 image_dims_;

    histogram_storage_t* histogram_;
    cudaStream_t         calc_stream_; // Managed by some external resource

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
                             int                 n_border_pixels_to_ignore_for_histogram,
                             const int           n_images_to_accumulate_before_final_accumulation,
                             cudaStream_t        calc_stream = cudaStreamPerThread);

    ~TM_EmpiricalDistribution( );

    void AccumulateDistribution(ccfType* input_data, int n_images_this_batch);
    void FinalAccumulate( );
    void CopyToHostAndAdd(long* array_to_add_to);

    void SetCalcStream(cudaStream_t calc_stream) {
        MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");
        calc_stream_ = calc_stream;
    }
};

#endif