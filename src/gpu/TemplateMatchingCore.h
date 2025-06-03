#ifndef TemplateMatchingCore_H_
#define TemplateMatchingCore_H_

#include <memory>

#include "GpuImage.h"
#include "DeviceManager.h"
#include "template_matching_empirical_distribution.h"

/**
 * @class TemplateMatchingCore
 * @brief Manages GPU resources and executes the core template matching logic.
 *
 * This class encapsulates the GPU-specific operations for template matching,
 * including managing CUDA streams, memory, and kernel launches.
 * Each instance of this class is intended to be used by a single CPU thread
 * to manage its corresponding GPU operations, ensuring thread safety by
 * avoiding shared mutable state between threads at this level.
 *
 * Key responsibilities:
 * - Initialization of GPU resources (memory allocation, stream creation).
 * - Transferring data (templates, input images) between CPU and GPU.
 * - Executing the template matching search loop on the GPU.
 * - Managing intermediate and final results on the GPU.
 * - Handling CUDA stream synchronization and L2 cache persistence.
 *
 * Thread Safety:
 * - Instances of TemplateMatchingCore are NOT thread-safe if shared among multiple CPU threads.
 * - Designed to be instantiated per-thread (e.g., one instance per OpenMP thread)
 *   to manage a dedicated set of GPU resources and a specific portion of the workload.
 * - Internal CUDA operations are managed to be safe with respect to the GPU execution model,
 *   often relying on separate CUDA streams for concurrent operations managed by a single instance.
 * - Shared data (like input images or global results) passed to or from this class
 *   must be managed with appropriate synchronization mechanisms by the calling code (e.g., `MatchTemplateApp`).
 */
class TemplateMatchingCore {

  private:
    bool object_initialized_;
    bool use_gpu_prj;

  public:
    /**
     * @brief Default constructor. Initializes object_initialized_ to false.
     */
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

    /**
     * @brief Unique pointer to the empirical distribution object for template matching.
     * This object handles the statistical calculations for CCF values.
     * Being a unique_ptr ensures that each TemplateMatchingCore instance owns its
     * TM_EmpiricalDistribution, which is crucial if TM_EmpiricalDistribution itself
     * maintains state or uses GPU resources tied to a specific stream/context.
     */
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
    /**
     * @brief Performs pixel-wise Maximum Intensity Projection (MIP) update.
     * @param psi Current Psi angle.
     * @param theta Current Theta angle.
     * @param phi Current Phi angle.
     */
    void MipPixelWise(__half psi, __half theta, __half phi);
    /**
     * @brief Performs pixel-wise MIP update for a stack of projections.
     * @param mip_array Array to store MIP values.
     * @param psi Array of Psi angles.
     * @param theta Array of Theta angles.
     * @param phi Array of Phi angles.
     * @param n_mips_this_round Number of MIPs to process in this round.
     */
    void MipPixelWiseStack(__half* mip_array, __half* psi, __half* theta, __half* phi, int n_mips_this_round);
    /**
     * @brief Converts MIP data to an image format.
     */
    void MipToImage( );
    /**
     * @brief Accumulates sums into provided sum and sum-of-squares GPU images.
     * @param sum_sumsq GPU buffer containing sum and sum-of-squares to accumulate.
     * @param sum GpuImage to store accumulated sums.
     * @param sq_sum GpuImage to store accumulated sum of squares.
     */
    void AccumulateSums(__half2* sum_sumsq, GpuImage& sum, GpuImage& sq_sum);

    /**
     * @brief Updates secondary peak information.
     */
    void UpdateSecondaryPeaks( );

    /**
     * @brief Sets the CPU template image.
     * @param cpu_template Pointer to the CPU template image. Used when GPU projections are disabled.
     */
    void SetCpuTemplate(Image* cpu_template) {
        this->cpu_template = cpu_template;
    }

    /**
     * @brief Comprehensive initialization for the TemplateMatchingCore.
     *
     * This method sets up the TemplateMatchingCore instance with all necessary
     * parameters and data for performing template matching. It prepares GPU resources
     * and configures the search parameters.
     *
     * @param parent_pointer Pointer to the parent application (e.g., MatchTemplateApp).
     * @param template_reconstruction Shared pointer to the template reconstruction on the GPU.
     * @param input_image Shared pointer to the input image on the GPU.
     * @param current_projection A CPU Image object, potentially used as a template for GPU projection dimensions or if use_gpu_prj is false.
     * @param psi_max Maximum Psi angle for the search.
     * @param psi_start Starting Psi angle for the search.
     * @param psi_step Step size for Psi angle search.
     * @param angles AnglesAndShifts object defining the search space.
     * @param global_euler_search EulerSearch object defining global search parameters.
     * @param pre_padding Pre-padding values for images.
     * @param roi Region of Interest.
     * @param first_search_position Starting index for the search positions this instance will handle.
     * @param last_search_position Ending index for the search positions this instance will handle.
     * @param my_progress Pointer to a ProgressBar for updates.
     * @param total_correlation_positions Total number of correlation positions to be evaluated across all threads/instances.
     * @param is_running_locally Flag indicating if running in local mode.
     * @param use_fast_fft Flag to enable faster FFT.
     * @param use_gpu_prj Flag to enable GPU-based projections.
     * @param number_of_global_search_images_to_save Number of top global search result images to retain.
     */
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
