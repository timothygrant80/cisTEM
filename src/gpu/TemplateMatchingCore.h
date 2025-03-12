#ifndef TemplateMatchingCore_H_
#define TemplateMatchingCore_H_

#include <memory>

#include "GpuImage.h"
#include "DeviceManager.h"
#include "template_matching_empirical_distribution.h"

class TemplateMatchingCore {

  private:
    bool object_initialized_;
    bool use_gpu_prj;

  public:
    TemplateMatchingCore( ) : object_initialized_{false} { };

    // block copy and move explicitly
    TemplateMatchingCore(const TemplateMatchingCore&)            = delete;
    TemplateMatchingCore& operator=(const TemplateMatchingCore&) = delete;
    TemplateMatchingCore(TemplateMatchingCore&&)                 = delete;
    TemplateMatchingCore& operator=(TemplateMatchingCore&&)      = delete;

    void Init(int number_of_jobs);

    int number_of_jobs_per_image_in_gui;

    // CPU images to be passed in -
    std::shared_ptr<GpuImage> template_gpu_shared;
    std::shared_ptr<GpuImage> d_input_image;
    std::shared_ptr<GpuImage> d_input_image_sq;
    bool                      is_set_input_image_ptr{ };

    bool  use_lerp_for_resizing{ };
    float binning_factor = 1.f;

    std::vector<Image> current_projection;
    // Generally not used, except for --disable-gpu-prj
    Image* cpu_template = nullptr;

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

    std::vector<GpuImage> d_current_projection;

    std::vector<GpuImage*> d_statistical_buffers_ptrs;

    GpuImage d_padded_reference;

    // Search range parameters
    float psi_max;
    float psi_start;
    float psi_step;

    int  current_search_position;
    int  first_search_position;
    int  last_search_position;
    long total_number_of_cccs_calculated;
    long total_correlation_positions;

    int n_global_search_images_to_save;

    bool is_running_locally;
    bool use_fast_fft;

    std::unique_ptr<TM_EmpiricalDistribution<__half, __half2>> my_dist;

    int2 pre_padding;
    int2 roi;

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

    void SetCpuTemplate(Image* cpu_template) {
        this->cpu_template = cpu_template;
    }

    void Init(MyApp*                    parent_pointer,
              std::shared_ptr<GpuImage> template_reconstruction,
              std::shared_ptr<GpuImage> input_image,
              Image&                    current_projection,
              float                     psi_max,
              float                     psi_start,
              float                     psi_step,
              AnglesAndShifts&          angles,
              EulerSearch&              global_euler_search,
              const int2                pre_padding,
              const int2                roi,
              int                       first_search_position,
              int                       last_search_position,
              ProgressBar*              my_progress,
              long                      total_correlation_positions,
              bool                      is_running_locally,
              bool                      use_fast_fft,
              bool                      use_gpu_prj,
              int                       number_of_global_search_images_to_save = 1);

    bool                is_set_L2_cache_persisting{ };
    cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
    size_t              SetL2CachePersisting(const float L2_persistance_fraction);
    void                ClearL2CachePersisting( );
    void                SetL2AccessPolicy(size_t window_size);
    void                ClearL2AccessPolicy( );

    void RunInnerLoop(Image&      projection_filter,
                      int         threadIDX,
                      long&       current_correlation_position,
                      const float min_counter_val,
                      const float threshold_val);
};

#endif
