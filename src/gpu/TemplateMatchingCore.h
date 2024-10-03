#ifndef TemplateMatchingCore_H_
#define TemplateMatchingCore_H_

#include "GpuImage.h"
#include "DeviceManager.h"
#include "Histogram.h"
#include "template_matching_empirical_distribution.h"

class TemplateMatchingCore {

  private:
    bool object_initialized_;

  public:
    TemplateMatchingCore( ) : object_initialized_{false} { };

    // block copy and move explicitly
    TemplateMatchingCore(const TemplateMatchingCore&)            = delete;
    TemplateMatchingCore& operator=(const TemplateMatchingCore&) = delete;
    TemplateMatchingCore(TemplateMatchingCore&&)                 = delete;
    TemplateMatchingCore& operator=(TemplateMatchingCore&&)      = delete;

    void Init(int number_of_jobs);

    DeviceManager gpuDev;

    int nGPUs;
    int nThreads;
    int number_of_jobs_per_image_in_gui;

    // CPU images to be passed in -
    Image                     template_reconstruction;
    std::shared_ptr<GpuImage> template_gpu_shared;
    Image                     input_image; // These will be modified on the host from withing Template Matching Core so Allocate locally

    bool  use_lerp_for_resizing{ };
    float binning_factor = 1.f;

    std::vector<Image> current_projection;

    // These are assumed to be empty containers at the outset, so xfer host-->device is skipped
    GpuImage d_max_intensity_projection;
    GpuImage d_best_psi;
    GpuImage d_best_phi;
    GpuImage d_best_theta;
    GpuImage d_best_defocus;
    GpuImage d_best_pixel_size;

    GpuImage d_sum1, d_sum2;
    GpuImage d_sumSq1, d_sumSq2;
    bool     is_allocated_sum_buffer = false;
    int      is_non_zero_sum_buffer;

    // This will need to be copied in
    GpuImage              d_input_image;
    std::vector<GpuImage> d_current_projection;

    std::vector<GpuImage*> d_statistical_buffers_ptrs;

    GpuImage d_padded_reference;

    // Search range parameters
    float pixel_size_search_range;
    float pixel_size_step;
    float pixel_size;
    float defocus_search_range;
    float defocus_step;
    float defocus1;
    float defocus2;
    float psi_max;
    float psi_start;
    float psi_step;
    float minimum_threshold = 20.0f; //  Optionally override this to limit what is considered for refinement

    float c_defocus;
    float c_pixel;

    int  current_search_position;
    int  first_search_position;
    int  last_search_position;
    long total_number_of_cccs_calculated;
    long total_number_of_histogram_samples;
    long total_number_of_stats_samples;
    long total_correlation_positions;

    int n_global_search_images_to_save;

    bool      is_running_locally;
    bool      is_gpu_3d_swapped;
    bool      use_fast_fft;
    Histogram histogram;

    std::unique_ptr<TM_EmpiricalDistribution<__half, __half2>> my_dist;

    float histogram_min_scaled;
    float histogram_step_scaled;
    int2  pre_padding;
    int2  roi;

    // Search objects
    AnglesAndShifts angles;
    EulerSearch     global_euler_search;

    int dummy;

    ProgressBar* my_progress;

    MyApp* parent_pointer;

    __half2* sum_sumsq;
    __half2* mip_psi;
    __half2* theta_phi; // for passing euler angles to the callback
    __half*  secondary_peaks;

    void SumPixelWise(GpuImage& image);
    void MipPixelWise(__half psi, __half theta, __half phi);
    void MipPixelWiseStack(__half* mip_array, __half* psi, __half* theta, __half* phi, int n_mips_this_round);
    void MipToImage( );
    void AccumulateSums(__half2* sum_sumsq, GpuImage& sum, GpuImage& sq_sum);

    void UpdateSecondaryPeaks( );

    void SetMinimumThreshold(float wanted_threshold) { minimum_threshold = wanted_threshold; }

    void Init(MyApp*                    parent_pointer,
              std::shared_ptr<GpuImage> template_reconstruction,
              Image&                    input_image,
              Image&                    current_projection,
              float                     pixel_size_search_range,
              float                     pixel_size_step,
              float                     pixel_size,
              float                     defocus_search_range,
              float                     defocus_step,
              float                     defocus1,
              float                     defocus2,
              float                     psi_max,
              float                     psi_start,
              float                     psi_step,
              AnglesAndShifts&          angles,
              EulerSearch&              global_euler_search,
              float                     histogram_min_scaled,
              float                     histogram_step_scaled,
              int                       histogram_number_of_bins,
              const int2                pre_padding,
              const int2                roi,
              int                       first_search_position,
              int                       last_search_position,
              ProgressBar*              my_progress,
              long                      total_correlation_positions,
              bool                      is_running_locally,
              bool                      use_fast_fft,
              int                       number_of_global_search_images_to_save = 1);

    void RunInnerLoop(Image& projection_filter, float pixel_i, float defocus_i, int threadIDX, long& current_correlation_position);
};

#endif
