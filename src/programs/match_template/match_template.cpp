#include <cistem_config.h>
#include <filesystem>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#if defined(cisTEM_USING_FastFFT) && defined(ENABLEGPU)
#ifdef cisTEM_BUILDING_FastFFT
#include "../../../include/FastFFT/include/FastFFT.h"
#else
#include "/opt/FastFFT/include/FastFFT.h"
#endif
#endif

#include "template_matching_data_sizer.h"

// The profiling for development is under conrtol of --enable-profiling.
#ifdef CISTEM_PROFILING
using namespace cistem_timer;
#else
#define PRINT_VERBOSE
using namespace cistem_timer_noop;
#endif

#define USE_LERP_NOT_FOURIER_RESIZING
#define IMPLICIT_TEMPLATE_POWER_2

// TODO: This seems good, let's fix it in place rather than a define

// FIXME: Probably need to disable resizing, or make sure it is handled
#define TEST_LOCAL_NORMALIZATION

/**
 * @class AggregatedTemplateResult
 * @brief Stores and aggregates template matching results from multiple processing units (e.g., worker threads or nodes).
 *
 * This class is used by the master process to collect partial results for each input image being searched.
 * When all parts of a result for a given image are received, they are combined and finalized.
 * The `collated_data_array` holds all the raw floating point data, and other `collated_*` members
 * are pointers that provide structured views into this single block of memory for different types of data
 * (MIP, angles, defocus, pixel sums, etc.).
 */
class AggregatedTemplateResult {

  public:
    int   image_number;
    int   number_of_received_results;
    float total_number_of_angles_searched;
    bool  disable_flat_fielding;

    float* collated_data_array;
    float* collated_mip_data;
    float* collated_psi_data;
    float* collated_theta_data;
    float* collated_phi_data;
    float* collated_defocus_data;
    float* collated_pixel_size_data;
    float* collated_pixel_sums;
    float* collated_pixel_square_sums;
    long*  collated_histogram_data;

    AggregatedTemplateResult( );
    ~AggregatedTemplateResult( );
    /**
     * @brief Adds a partial result to the aggregated data.
     *
     * This method is called when a worker sends its computed results. It merges the incoming
     * `result_array` into the appropriate sections of `collated_data_array`. For MIP and
     * best angle/parameter maps, it updates values if the new result has a higher correlation.
     * For sums and histograms, it accumulates the values.
     *
     * @param result_array The partial result data from a worker.
     * @param array_size The total size of the `result_array`.
     * @param result_number The image number this result pertains to (used for verification).
     * @param number_of_expected_results The total number of partial results expected for this image.
     */
    void AddResult(float* result_array, long array_size, int result_number, int number_of_expected_results);
};

WX_DECLARE_OBJARRAY(AggregatedTemplateResult, ArrayOfAggregatedTemplateResults);
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAggregatedTemplateResults);

/**
 * @class MatchTemplateApp
 * @brief Main application class for the template matching program.
 *
 * This class orchestrates the template matching process, handling command-line arguments,
 * user input (if interactive), distributing work (if in a parallel environment),
 * and processing/saving the results. It can operate in both CPU and GPU accelerated modes.
 * Thread safety for parallel execution (primarily OpenMP for GPU path) is managed through
 * thread-local data where possible (e.g., `TemplateMatchingCore` instances) and critical sections
 * for updating shared aggregated results.
 */
class
        MatchTemplateApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    /**
     * @brief Handles program-defined results received by the master process from worker processes.
     *
     * In a distributed environment, worker processes send their partial results (packaged as a float array)
     * back to the master. This method is called on the master to receive, aggregate, and process these results.
     * When all results for a particular image are collected, it finalizes them (e.g., peak picking)
     * and may send them to a GUI or save them to disk.
     * @param result_array The float array containing the partial result data.
     * @param array_size The size of the `result_array`.
     * @param result_number The identifier of the image this result pertains to.
     * @param number_of_expected_results The total number of partial results expected for this image.
     */
    void MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results);
    void ProgramSpecificInit( );

    // for master collation

    ArrayOfAggregatedTemplateResults aggregated_results;

    float GetMaxJobWaitTimeInSeconds( ) { return 360.0f; }

  private:
    void AddCommandLineOptions( );
    /**
     * @brief Calculates the global mean and standard deviation of Cross-Correlation Coefficients (CCCs).
     *
     * This function iterates over the per-pixel sums and sums-of-squares of CCCs (accumulated
     * across all search angles) to determine the overall statistics of the CCC distribution.
     * These global statistics are then used for normalizing the MIP and rescaling the histogram.
     *
     * @tparam StatsType The data type of the sum and sum_of_sqs arrays (e.g., float, double).
     * @param[out] global_ccc_mean The calculated global mean of CCCs.
     * @param[out] global_ccc_std_dev The calculated global standard deviation of CCCs.
     * @param sum Pointer to the array of per-pixel sums of CCCs.
     * @param sum_of_sqs Pointer to the array of per-pixel sums of squared CCCs.
     * @param n_angles_in_search The total number of angles/orientations searched.
     * @param N The total number of pixels in the sum/sum_of_sqs arrays (image_real_memory_allocated).
     */
    template <typename StatsType>
    void CalcGlobalCCCScalingFactor(double&     global_ccc_mean,
                                    double&     global_ccc_std_dev,
                                    StatsType*  sum,
                                    StatsType*  sum_of_sqs,
                                    const float n_angles_in_search,
                                    const int   N);

    /**
     * @brief Resamples the histogram data based on global CCC mean and standard deviation.
     *
     * The raw histogram is collected based on initial estimates. After calculating the true
     * global mean and standard deviation of the CCC distribution, this function rescales
     * the histogram bins and interpolates counts from the original histogram (after
     * Savitzky-Golay smoothing) to populate the resampled histogram. This provides a
     * normalized view of the CCC distribution.
     *
     * @param histogram_ptr Pointer to the histogram data array (long type).
     * @param global_ccc_mean The global mean of CCCs.
     * @param global_ccc_std_dev The global standard deviation of CCCs.
     */
    void ResampleHistogramData(long*        histogram_ptr,
                               const double global_ccc_mean,
                               const double global_ccc_std_dev);

    /**
     * @brief Rescales the Maximum Intensity Projection (MIP) and statistical arrays using global CCC mean and standard deviation.
     *
     * This function normalizes the MIP values to represent Signal-to-Noise Ratio (SNR) like scores: (CCC - mean) / std_dev.
     * It also rescales the per-pixel sum and sum-of-squares arrays accordingly. The histogram data is also
     * adjusted using `ResampleHistogramData`. If `disable_flat_fielding` is false, the `scaled_mip`
     * is further normalized by local statistics (mean and std_dev derived from `correlation_pixel_sum`
     * and `correlation_pixel_sum_of_squares`).
     *
     * @tparam StatsType The data type of the correlation_pixel_sum and correlation_pixel_sum_of_squares arrays.
     * @param mip_image Pointer to the MIP image to be rescaled (modified in place).
     * @param scaled_mip Pointer to the output scaled MIP image.
     * @param correlation_pixel_sum Pointer to the array of per-pixel sums of CCCs (modified in place).
     * @param correlation_pixel_sum_of_squares Pointer to the array of per-pixel sums of squared CCCs (modified in place).
     * @param histogram Pointer to the histogram data array (modified in place via `ResampleHistogramData`).
     * @param n_angles_in_search The total number of angles/orientations searched.
     * @param disable_flat_fielding If true, local normalization for `scaled_mip` is skipped.
     */
    template <typename StatsType>
    void RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(Image*      mip_image,
                                                             Image*      scaled_mip,
                                                             StatsType*  correlation_pixel_sum,
                                                             StatsType*  correlation_pixel_sum_of_squares,
                                                             long*       histogram,
                                                             const float n_angles_in_search,
                                                             const bool  disable_flat_fielding);
};

IMPLEMENT_APP(MatchTemplateApp)

// TODO: why is this here?
void MatchTemplateApp::ProgramSpecificInit( ) {
}

// Optional command-line stuff
void MatchTemplateApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-gpu-prj", "Disable projection using the gpu. Default false");
    command_line_parser.AddLongSwitch("disable-flat-fielding", "Disable flat fielding. Default false");
    command_line_parser.AddOption("", "n-expected-false-positives", "average number of false positives per image, (defaults to 1)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongSwitch("ignore-defocus-for-threshold", "assume the defocus planes are not independent locs for threshold calc, (defaults false)");
    command_line_parser.AddLongSwitch("apply-result-rescaling", "Rescale the results their original size, (defaults false)");

#ifdef TEST_LOCAL_NORMALIZATION
    command_line_parser.AddOption("", "healpix-file", "Healpix file for the input images", wxCMD_LINE_VAL_STRING);
    command_line_parser.AddOption("", "min-stats-counter", "Minimum number of pixels to calculate the threshold (defaults to 10.f)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "threshold-val", "n_stddev to threshold value for the trimmed local variance (defaults to 3.0f)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "L2-peristance-fraction", "min L2 cache available for persisting as fraction of input image size in fp16 bytes (defaults to 0 [off])", wxCMD_LINE_VAL_DOUBLE);
#endif
}

// override the DoInteractiveUserInput

void MatchTemplateApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction;

    wxString mip_output_file;
    wxString best_psi_output_file;
    wxString best_theta_output_file;
    wxString best_phi_output_file;
    wxString best_defocus_output_file;
    wxString best_pixel_size_output_file;

    wxString output_histogram_file;
    wxString correlation_std_output_file;
    wxString correlation_avg_output_file;
    wxString scaled_mip_output_mrcfile;

    float input_pixel_size        = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;
    ;
    float    defocus_angle;
    float    phase_shift;
    float    low_resolution_limit      = 300.0;
    float    high_resolution_limit     = 8.0;
    float    angular_step              = 5.0;
    int      best_parameters_to_keep   = 20;
    float    defocus_search_range      = 500;
    float    defocus_step              = 50;
    float    pixel_size_search_range   = 0.1f;
    float    pixel_size_step           = 0.02f;
    float    padding                   = 1.0;
    bool     ctf_refinement            = false;
    float    particle_radius_angstroms = 0.0f;
    wxString my_symmetry               = "C1";
    float    in_plane_angular_step     = 0;
    bool     use_gpu_input             = false;
    int      max_threads               = 1; // Only used for the GPU code
    bool     use_fast_fft              = false;

    UserInput* my_input = new UserInput("MatchTemplate", 1.00);

    input_search_images         = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction        = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    mip_output_file             = my_input->GetFilenameFromUser("Output MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
    scaled_mip_output_mrcfile   = my_input->GetFilenameFromUser("Output Scaled MIP file", "The file for saving the maximum intensity projection image divided by correlation variance", "mip_scaled.mrc", false);
    best_psi_output_file        = my_input->GetFilenameFromUser("Output psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_output_file      = my_input->GetFilenameFromUser("Output theta file", "The file for saving the best psi image", "theta.mrc", false);
    best_phi_output_file        = my_input->GetFilenameFromUser("Output phi file", "The file for saving the best psi image", "phi.mrc", false);
    best_defocus_output_file    = my_input->GetFilenameFromUser("Output defocus file", "The file for saving the best defocus image", "defocus.mrc", false);
    best_pixel_size_output_file = my_input->GetFilenameFromUser("Output pixel size file", "The file for saving the best pixel size image", "pixel_size.mrc", false);
    correlation_avg_output_file = my_input->GetFilenameFromUser("Correlation average value", "The file for saving the average value of all correlation images", "corr_average.mrc", false);
    correlation_std_output_file = my_input->GetFilenameFromUser("Correlation variance output file", "The file for saving the variance of all correlation images", "corr_stddev.mrc", false);
    output_histogram_file       = my_input->GetFilenameFromUser("Output histogram of correlation values", "histogram of all correlation values", "histogram.txt", false);
    input_pixel_size            = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV                  = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm     = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast          = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                    = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                    = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle               = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift                 = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
    //    low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    angular_step          = my_input->GetFloatFromUser("Out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    //    best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
    defocus_search_range    = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
    defocus_step            = my_input->GetFloatFromUser("Defocus step (A) (0.0 = no search)", "Step size used in the defocus search", "50.0", 0.0);
    pixel_size_search_range = my_input->GetFloatFromUser("Pixel size search range (A)", "Search range (-value ... + value) around current pixel size", "0.1", 0.0);
    pixel_size_step         = my_input->GetFloatFromUser("Pixel size step (A) (0.0 = no search)", "Step size used in the pixel size search", "0.01", 0.0);
    padding                 = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0, 2.0);
    //    ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
    particle_radius_angstroms = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "0.0", 0.0);
    my_symmetry               = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
#ifdef ENABLEGPU
    use_gpu_input = my_input->GetYesNoFromUser("Use GPU", "Offload expensive calcs to GPU", "Yes");
#ifdef cisTEM_USING_FastFFT
    use_fast_fft = my_input->GetYesNoFromUser("Use Fast FFT", "Use the Fast FFT library", "Yes");
#endif
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#endif

    int   first_search_position           = -1;
    int   last_search_position            = -1;
    int   image_number_for_gui            = 0;
    int   number_of_jobs_per_image_in_gui = 0;
    float min_peak_radius                 = 10.0f;

    wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive
    wxString result_filename       = "/dev/null"; // shouldn't be used in interactive

    delete my_input;

    my_current_job.ManualSetArguments("ttffffffffffifffffbfftttttttttftiiiitttfbbi",
                                      input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      input_pixel_size,
                                      voltage_kV,
                                      spherical_aberration_mm,
                                      amplitude_contrast,
                                      defocus1,
                                      defocus2,
                                      defocus_angle,
                                      low_resolution_limit,
                                      high_resolution_limit,
                                      angular_step,
                                      best_parameters_to_keep,
                                      defocus_search_range,
                                      defocus_step,
                                      pixel_size_search_range,
                                      pixel_size_step,
                                      padding,
                                      ctf_refinement,
                                      particle_radius_angstroms,
                                      phase_shift,
                                      mip_output_file.ToUTF8( ).data( ),
                                      best_psi_output_file.ToUTF8( ).data( ),
                                      best_theta_output_file.ToUTF8( ).data( ),
                                      best_phi_output_file.ToUTF8( ).data( ),
                                      best_defocus_output_file.ToUTF8( ).data( ),
                                      best_pixel_size_output_file.ToUTF8( ).data( ),
                                      scaled_mip_output_mrcfile.ToUTF8( ).data( ),
                                      correlation_avg_output_file.ToUTF8( ).data( ),
                                      my_symmetry.ToUTF8( ).data( ),
                                      in_plane_angular_step,
                                      output_histogram_file.ToUTF8( ).data( ),
                                      first_search_position,
                                      last_search_position,
                                      image_number_for_gui,
                                      number_of_jobs_per_image_in_gui,
                                      correlation_std_output_file.ToUTF8( ).data( ),
                                      directory_for_results.ToUTF8( ).data( ),
                                      result_filename.ToUTF8( ).data( ),
                                      min_peak_radius,
                                      use_gpu_input,
                                      use_fast_fft,
                                      max_threads);
}

// override the do calculation method which will be what is actually run..

/**
 * @brief Performs the core template matching calculation.
 *
 * This is the main computational part of the program. It involves the following major steps:
 * 1.  Parameter Retrieval: Parses command-line arguments and job parameters.
 * 2.  Initialization:
 *     - Reads the input image stack and the 3D template reconstruction.
 *     - Initializes `TemplateMatchingDataSizer` to manage image/template resizing and preprocessing based on resolution limits and FFT requirements.
 *     - Allocates memory for output images (MIP, best angles, defocus, pixel size) and statistical accumulation arrays (sum of CCCs, sum of squared CCCs, histogram).
 * 3.  Search Setup:
 *     - Initializes `EulerSearch` to define the grid of orientations to be searched.
 *     - Sets up CTF parameters and generates the initial projection filter (CTF * whitening_filter).
 * 4.  Main Search Loop: Iterates over:
 *     - Pixel size variations (if enabled).
 *     - Defocus variations (if enabled).
 *     - For each pixel size/defocus combination:
 *         - Updates the 3D template (if pixel size changed) and the projection filter (if defocus changed).
 *         - **GPU Path (`use_gpu` is true and `ENABLEGPU` is defined):**
 *             - Initializes `TemplateMatchingCore` instances, one per OpenMP thread. Each `TemplateMatchingCore` manages its own CUDA stream and resources.
 *             - The input image and 3D template (if `use_gpu_prj` is true) are transferred to GPU memory (potentially shared via `std::shared_ptr<GpuImage>`).
 *             - An OpenMP parallel region distributes the Euler search angles among threads.
 *             - Each thread calls `TemplateMatchingCore::RunInnerLoop` to perform template projections (on GPU or CPU if `use_gpu_prj` is false),
 *               cross-correlation (on GPU), and updates its local MIP and best parameter maps.
 *             - After the parallel region, results from each thread (MIP, best angles, sums, histogram) are aggregated under an `#pragma omp critical` section.
 *         - **CPU Path (`use_gpu` is false or `ENABLEGPU` is not defined):**
 *             - Iterates through Euler search positions and in-plane rotations (psi).
 *             - For each orientation:
 *                 - Extracts a 2D projection from the 3D template.
 *                 - Applies the projection filter.
 *                 - Performs cross-correlation with the input image using FFTs.
 *                 - Updates the MIP, best angle/parameter maps, and accumulates statistics (sums, histogram).
 * 5.  Post-Processing and Output:
 *     - If image resizing/rotation was performed for the search, results are mapped back to the original image dimensions using `TemplateMatchingDataSizer::ResizeImage_postSearch`.
 *     - Global CCC statistics are calculated, and the MIP and histogram are rescaled/normalized using `RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev`.
 *     - If running locally (interactive or single job), output MRC files for MIP, scaled MIP, best angles, defocus, pixel size, and statistical maps are written. A histogram text file is also generated.
 *     - If running as a worker in a distributed job, the processed results are packaged into a float array and sent to the master process via `SendProgramDefinedResultToMaster`.
 *
 * Thread Safety Considerations:
 * - GPU Path: Each OpenMP thread gets its own `TemplateMatchingCore` instance, which encapsulates GPU resources and operations for that thread.
 *   Shared GPU data (input image, 3D template) is managed with `std::shared_ptr` and accessed read-only by projection/correlation kernels.
 *   Aggregation of results from different GPU threads into shared host arrays (e.g., global MIP, `correlation_pixel_sum`) is protected by an `#pragma omp critical` section.
 * - CPU Path: If `max_threads` > 1 is specified for CPU, it's currently a warning, and `max_threads` is reset to 1, as the CPU path is serial within `DoCalculation`.
 *   Parallelism in a CPU-only distributed job is achieved by running multiple `MatchTemplateApp` processes, each handling a subset of images or search angles, with results collated by a master process.
 *
 * @return True if the calculation completes successfully, false otherwise.
 */
bool MatchTemplateApp::DoCalculation( ) {

    bool is_rotated_by_90 = false;

    // In particular histogram_min, histogram_max, histogram_step, histogram_number_of_points, histogram_first_bin_midpoint
    using namespace cistem::match_template;
    StopWatch profile_timing;

    wxDateTime start_time = wxDateTime::Now( );

    double temp_double;
    long   temp_long;
    bool   use_gpu_prj                  = true;
    bool   disable_flat_fielding        = false;
    bool   ignore_defocus_for_threshold = false;
    bool   apply_result_rescaling{ };
    double n_expected_false_positives{1.0};

    if ( command_line_parser.FoundSwitch("apply-result-rescaling") ) {
        SendInfo("Applying result rescaling\n");
        apply_result_rescaling = true;
    }
    if ( command_line_parser.FoundSwitch("ignore-defocus-for-threshold") ) {
        SendInfo("Using one defocus plane for threshold calc\n");
        ignore_defocus_for_threshold = true;
    }
    if ( command_line_parser.FoundSwitch("disable-flat-fielding") ) {
        SendInfo("Disabling flat fielding\n");
        disable_flat_fielding = true;
    }

    if ( command_line_parser.FoundSwitch("disable-gpu-prj") ) {
        SendInfo("Disabling GPU projection\n");
        use_gpu_prj = false;
    }
    float L2_persistance_fraction{ };
    if ( command_line_parser.Found("L2-peristance-fraction", &temp_double) ) {
        L2_persistance_fraction = float(temp_double);
    }

    if ( command_line_parser.Found("n-expected-false-positives", &temp_double) ) {
        SendInfo("Using n expected false positives: " + wxString::Format("%f", temp_double) + "\n");
        n_expected_false_positives = temp_double;
    }
    // This allows an override for the TEST_LOCAL_NORMALIZATION
    bool allow_rotation_for_speed{true};
    // This allows us to not use local normalization while also compiling with this option
    bool  use_local_normalization{false};
    float min_counter_val{std::numeric_limits<float>::max( )}; // This way, if we aren't using it, we short-circute the calculation of the SD every pixel in the OR clause
    float threshold_val{3.0f};

#ifdef TEST_LOCAL_NORMALIZATION
    wxString healpix_file;
    if ( command_line_parser.Found("healpix-file", &healpix_file) ) {
        SendInfo("Using healpix file: " + healpix_file + "\n");
        healpix_file             = healpix_file;
        use_local_normalization  = true;
        allow_rotation_for_speed = false;
        min_counter_val          = 10.f; // If we are testing local normalization, set the default value here, and possible update it in the next lines.
    }
    if ( command_line_parser.Found("min-stats-counter", &temp_double) ) {
        min_counter_val = float(temp_double);
    }
    if ( command_line_parser.Found("threshold-val", &temp_double) ) {
        threshold_val = float(temp_double);
    }

    if ( use_local_normalization ) {
        wxPrintf("Using local normalization bool: %d\n", use_local_normalization);
        wxPrintf("Using min stats counter: %f\n", min_counter_val);
        wxPrintf("Using threshold value: %f\n", threshold_val);
    }
    // I guess this breaks the local normalization so provide an override for TM data sizer
#endif

    wxString input_search_images_filename    = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename   = my_current_job.arguments[1].ReturnStringArgument( );
    float    input_pixel_size                = my_current_job.arguments[2].ReturnFloatArgument( );
    float    voltage_kV                      = my_current_job.arguments[3].ReturnFloatArgument( );
    float    spherical_aberration_mm         = my_current_job.arguments[4].ReturnFloatArgument( );
    float    amplitude_contrast              = my_current_job.arguments[5].ReturnFloatArgument( );
    float    defocus1                        = my_current_job.arguments[6].ReturnFloatArgument( );
    float    defocus2                        = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus_angle                   = my_current_job.arguments[8].ReturnFloatArgument( );
    float    low_resolution_limit            = my_current_job.arguments[9].ReturnFloatArgument( );
    float    high_resolution_limit_search    = my_current_job.arguments[10].ReturnFloatArgument( );
    float    angular_step                    = my_current_job.arguments[11].ReturnFloatArgument( );
    int      best_parameters_to_keep         = my_current_job.arguments[12].ReturnIntegerArgument( );
    float    defocus_search_range            = my_current_job.arguments[13].ReturnFloatArgument( );
    float    defocus_step                    = my_current_job.arguments[14].ReturnFloatArgument( );
    float    pixel_size_search_range         = my_current_job.arguments[15].ReturnFloatArgument( );
    float    pixel_size_step                 = my_current_job.arguments[16].ReturnFloatArgument( );
    float    padding                         = my_current_job.arguments[17].ReturnFloatArgument( );
    bool     ctf_refinement                  = my_current_job.arguments[18].ReturnBoolArgument( );
    float    particle_radius_angstroms       = my_current_job.arguments[19].ReturnFloatArgument( );
    float    phase_shift                     = my_current_job.arguments[20].ReturnFloatArgument( );
    wxString mip_output_file                 = my_current_job.arguments[21].ReturnStringArgument( );
    wxString best_psi_output_file            = my_current_job.arguments[22].ReturnStringArgument( );
    wxString best_theta_output_file          = my_current_job.arguments[23].ReturnStringArgument( );
    wxString best_phi_output_file            = my_current_job.arguments[24].ReturnStringArgument( );
    wxString best_defocus_output_file        = my_current_job.arguments[25].ReturnStringArgument( );
    wxString best_pixel_size_output_file     = my_current_job.arguments[26].ReturnStringArgument( );
    wxString scaled_mip_output_file          = my_current_job.arguments[27].ReturnStringArgument( );
    wxString correlation_avg_output_file     = my_current_job.arguments[28].ReturnStringArgument( );
    wxString my_symmetry                     = my_current_job.arguments[29].ReturnStringArgument( );
    float    in_plane_angular_step           = my_current_job.arguments[30].ReturnFloatArgument( );
    wxString output_histogram_file           = my_current_job.arguments[31].ReturnStringArgument( );
    int      first_search_position           = my_current_job.arguments[32].ReturnIntegerArgument( );
    int      last_search_position            = my_current_job.arguments[33].ReturnIntegerArgument( );
    int      image_number_for_gui            = my_current_job.arguments[34].ReturnIntegerArgument( );
    int      number_of_jobs_per_image_in_gui = my_current_job.arguments[35].ReturnIntegerArgument( );
    wxString correlation_std_output_file     = my_current_job.arguments[36].ReturnStringArgument( );
    wxString directory_for_results           = my_current_job.arguments[37].ReturnStringArgument( );
    wxString result_output_filename          = my_current_job.arguments[38].ReturnStringArgument( );
    float    min_peak_radius                 = my_current_job.arguments[39].ReturnFloatArgument( );
    bool     use_gpu                         = my_current_job.arguments[40].ReturnBoolArgument( );
    bool     use_fast_fft                    = my_current_job.arguments[41].ReturnBoolArgument( );

    int max_threads = my_current_job.arguments[42].ReturnIntegerArgument( );

    if ( is_running_locally == false )
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...

    // This condition applies to GUI and CLI - it is just a recommendation to the user.
    if ( use_gpu && max_threads <= 1 ) {
        SendInfo("Warning, you are only using one thread on the GPU. Suggested minimum is 2. Check compute saturation using nvidia-smi -l 1\n");
    }
    if ( ! use_gpu && max_threads > 1 ) {
        SendInfo("Using more than one thread only works in the GPU implementation\nSet No. of threads per copy to 1 in your Run Profile\n.");
        max_threads = 1;
    }

#ifdef USE_LERP_NOT_FOURIER_RESIZING
    const bool use_lerp_not_fourier_resampling = true;
#else
    const bool use_lerp_not_fourier_resampling = false;
#endif

    if ( use_gpu && ! use_gpu_prj && use_lerp_not_fourier_resampling ) {
        SendError("LERP resampling is only supported on the GPU implementation\n");
    }

    profile_timing.start("Init");
    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );

    float outer_mask_radius;
    float current_psi;
    float psi_step;
    float psi_max;
    float psi_start;

    float expected_threshold;
    float actual_number_of_angles_searched{0.f};

    long* histogram_data;

    int current_bin;

    float  temp_float;
    float  variance;
    double temp_double_array[5];

    int   number_of_rotations;
    float fraction_of_search_positions_that_are_independent{1.f};
    long  current_correlation_position;
    long  number_of_search_positions{ };
    long  number_of_search_positions_per_thread{ };
    long  pixel_counter;

    int current_search_position;

    int i;

#ifdef TEST_LOCAL_NORMALIZATION
    NumericTextFile healpix_binning;
#endif

    EulerSearch     global_euler_search;
    AnglesAndShifts angles;

    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file;

    // Open input files for search images and template reconstruction
    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    Image input_image;
    Image padded_reference;
    Image input_reconstruction;
    Image template_reconstruction;
    Image current_projection;
    Image padded_projection;

    Image projection_filter;

    Image max_intensity_projection;

    Image best_psi;
    Image best_theta;
    Image best_phi;
    Image best_defocus;
    Image best_pixel_size;

    Image correlation_pixel_sum_image;
    Image correlation_pixel_sum_of_squares_image;

    Image temp_image;

    profile_timing.lap("Init");

    // Read input image and reconstruction
    profile_timing.start("Read input images");
    input_image.ReadSlice(&input_search_image_file, 1);

    float histogram_padding_trim_rescale; // scale the counts to

    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
    MyAssertTrue(input_reconstruction.IsCubic( ), "Input reconstruction should be cubic");
    profile_timing.lap("Read input images");

    // Initialize TemplateMatchingDataSizer for managing image sizes and preprocessing
    profile_timing.start("PreProcessInputImage");
    TemplateMatchingDataSizer data_sizer(this, input_image, input_reconstruction, input_pixel_size, padding);
    if ( use_local_normalization && data_sizer.IsResamplingNeeded( ) ) {
        SendError("Local normalization is not yet supported with resampling.");
    }

    data_sizer.PreProcessInputImage(input_image, false, true);
    profile_timing.lap("PreProcessInputImage");

    // Resize template and image for the search resolution
    profile_timing.start("Resize_preSearch");
    data_sizer.SetImageAndTemplateSizing(high_resolution_limit_search, use_fast_fft);
    data_sizer.ResizeTemplate_preSearch(input_reconstruction, use_lerp_not_fourier_resampling, true);

    // FIXME: we could use available threads to accelerate this. (Reduction of Allocations would also be good)
    data_sizer.ResizeImage_preSearch(input_image, allow_rotation_for_speed);
    profile_timing.lap("Resize_preSearch");

    if ( data_sizer.IsRotatedBy90( ) )
        defocus_angle += 90.0f;

    data_sizer.PrintImageSizes( );

    // Apply padding to the reconstruction if specified
    if ( padding != 1.0f ) {
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }

    // Allocate memory for output images and statistical arrays
    profile_timing.start("Allocate and zero arrays");
    padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    max_intensity_projection.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_psi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_theta.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_phi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_defocus.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_pixel_size.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    correlation_pixel_sum_image.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    correlation_pixel_sum_of_squares_image.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    double* correlation_pixel_sum            = new double[input_image.real_memory_allocated];
    double* correlation_pixel_sum_of_squares = new double[input_image.real_memory_allocated];

// FIXME: some of these arrays can be local variables.
#ifdef TEST_LOCAL_NORMALIZATION
    const int   BUFFER_SIZE       = 10;
    const float OUTLIER_THRESHOLD = 3.0f;
    // variables for Welford's algorithm
    double* mean_image; // replaces correlation_pixel_sum
    double* M2_image;
    int*    n_image;
    double* variance_image;
    double* stddev_image;
    double* local_stats;
    if ( use_local_normalization ) {
        n_image        = new int[input_image.real_memory_allocated];
        local_stats    = new double[4 * input_image.real_memory_allocated];
        mean_image     = (double*)&local_stats[0 * input_image.real_memory_allocated];
        M2_image       = (double*)&local_stats[1 * input_image.real_memory_allocated];
        variance_image = (double*)&local_stats[2 * input_image.real_memory_allocated];
        stddev_image   = (double*)&local_stats[3 * input_image.real_memory_allocated];
    }
#endif

    padded_reference.SetToConstant(0.f);
    max_intensity_projection.SetToConstant(0.f);
    best_psi.SetToConstant(0.f);
    best_theta.SetToConstant(0.f);
    best_phi.SetToConstant(0.f);
    best_defocus.SetToConstant(0.f);

    ZeroArray(correlation_pixel_sum, input_image.real_memory_allocated);
    ZeroArray(correlation_pixel_sum_of_squares, input_image.real_memory_allocated);

// FIXME: some of these arrays can be local variables.
#ifdef TEST_LOCAL_NORMALIZATION
    if ( use_local_normalization ) {
        ZeroArray(local_stats, 4 * input_image.real_memory_allocated);
        ZeroArray(n_image, input_image.real_memory_allocated);
    }
#endif

    histogram_data = new long[histogram_number_of_points];

    for ( int counter = 0; counter < histogram_number_of_points; counter++ ) {
        histogram_data[counter] = 0;
    }

    // assume cube

    // if we are using lerp for projection, we want the ctfs to be full size not search sized.
    int   wanted_pre_projection_template_size;
    float wanted_pre_projection_pixel_size;
    if ( use_lerp_not_fourier_resampling ) {

        wanted_pre_projection_pixel_size = data_sizer.GetPixelSize( );
    }
    else {
        wanted_pre_projection_pixel_size = data_sizer.GetSearchPixelSize( );
    }

    wanted_pre_projection_template_size = data_sizer.GetTemplateSizeX( );

    wxPrintf("values are %i %i %f %f %f\n", wanted_pre_projection_template_size, data_sizer.GetTemplateSearchSizeX( ), wanted_pre_projection_pixel_size, data_sizer.GetPixelSize( ), data_sizer.GetSearchPixelSize( ));

    CTF input_ctf;
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, wanted_pre_projection_pixel_size, deg_2_rad(phase_shift));
    projection_filter.Allocate(wanted_pre_projection_template_size, wanted_pre_projection_template_size, false);
    template_reconstruction.Allocate(wanted_pre_projection_template_size, wanted_pre_projection_template_size, wanted_pre_projection_template_size, true);

// We want the output projection to always be the search size
#ifdef IMPLICIT_TEMPLATE_POWER_2
    current_projection.Allocate(wanted_pre_projection_template_size, wanted_pre_projection_template_size, false);
#else
    current_projection.Allocate(data_sizer.GetTemplateSearchSizeX( ), data_sizer.GetTemplateSearchSizeX( ), false);
#endif

    // NOTE: note supported with resampling
    if ( padding != 1.0f )
        padded_projection.Allocate(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_x_dimension * padding, false);

    profile_timing.lap("Allocate and zero arrays");
    // angular step

    float mask_radius_search;
    if ( particle_radius_angstroms < 1.0f ) {
        mask_radius_search = 200.0f;
    } // This was the original default value.
    else
        mask_radius_search = particle_radius_angstroms;

    if ( angular_step <= 0 ) {
        angular_step = CalculateAngularStep(high_resolution_limit_search, mask_radius_search);
    }

    if ( in_plane_angular_step <= 0 ) {
        psi_step = rad_2_deg(data_sizer.GetSearchPixelSize( ) / mask_radius_search);
        psi_step = 360.0 / int(360.0 / psi_step + 0.5);
    }
    else {
        psi_step = in_plane_angular_step;
    }

    psi_start = 0.0f;
    psi_max   = 360.0f;
    if ( use_local_normalization ) {
#ifdef TEST_LOCAL_NORMALIZATION

        healpix_binning.Open(healpix_file, OPEN_TO_READ, 0);
        std::vector<float> orientations(healpix_binning.records_per_line);
        number_of_search_positions                     = healpix_binning.number_of_lines;
        global_euler_search.number_of_search_positions = number_of_search_positions;
        Allocate2DFloatArray(global_euler_search.list_of_search_parameters, number_of_search_positions, 2);
        for ( int counter = 0; counter < healpix_binning.number_of_lines; counter++ ) {
            healpix_binning.ReadLine(orientations.data( ));
            global_euler_search.list_of_search_parameters[counter][0] = orientations.at(0);
            global_euler_search.list_of_search_parameters[counter][1] = orientations.at(1);
        }
        healpix_binning.Close( );

#endif
    }
    else {
        // search grid
        // Note: resolution limit is only used in euler search in particle extraction and whitening. It does not affect template matching.
        global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, psi_max, psi_step, psi_start, data_sizer.GetSearchPixelSize( ) / high_resolution_limit_search, parameter_map, best_parameters_to_keep);

        // TODO 2x check me - w/o this O symm at least is broken
        if ( my_symmetry.StartsWith("C") ) {
            // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
            if ( global_euler_search.test_mirror ) {
                global_euler_search.theta_max = 180.0f;
            }
        }

        // Normally this is called in EulerSearch::InitGrid, but we need to re-call it here to get the search positions WITHOUT the default randomization to phi (azimuthal angle.)
        global_euler_search.CalculateGridSearchPositions(false);
    }

    // for now, I am assuming the MTF has been applied already.
    // work out the filter to just whiten the image..

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    profile_timing.start("PreProcessResizedInputImage");
    data_sizer.PreProcessResizedInputImage(input_image);
    profile_timing.lap("PreProcessResizedInputImage");
    // count total searches (lazy)

    current_correlation_position = 0;

    // if running locally, search over all of them

    if ( is_running_locally == true ) {
        first_search_position = 0;
        last_search_position  = global_euler_search.number_of_search_positions - 1;
    }

    // TODO unroll these loops and multiply the product.
    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        //loop over each rotation angle

        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            number_of_search_positions++;
        }
    }

    if ( defocus_step <= 0.0 ) {
        defocus_search_range = 0.0f;
        defocus_step         = 100.0f;
    }

    if ( pixel_size_step <= 0.0f ) {
        pixel_size_search_range = 0.0f;
        pixel_size_step         = 0.02f;
    }

    float n_defocus_steps = (2.f * myroundint(float(defocus_search_range) / float(defocus_step)) + 1.f);
    if ( ignore_defocus_for_threshold ) {
        fraction_of_search_positions_that_are_independent /= n_defocus_steps;
    }

    number_of_search_positions *= n_defocus_steps;
    number_of_search_positions_per_thread = number_of_search_positions;

    number_of_rotations = 0;

    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }

    ProgressBar* my_progress;

    //Loop over ever search position

    wxPrintf("\n\tFor image id %i\n", image_number_for_gui);
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
    wxPrintf("There are %li correlation positions total.\n\n", number_of_search_positions);

    wxPrintf("Performing Search...\n\n");

    wxDateTime overall_start;
    wxDateTime overall_finish;
    overall_start = wxDateTime::Now( );

    // These vars are only needed in the GPU code, but also need to be set out here to compile.
    std::vector<bool> first_gpu_loop(max_threads, true);

    int nThreads = 2;
    int nGPUs    = 1;
    int nJobs    = last_search_position - first_search_position + 1; // Number of primary Euler angles
    if ( use_gpu && max_threads > nJobs ) {
        SendInfo(wxString::Format("\n\tWarning, you request more threads (%d) than there are search positions (%d)\n", max_threads, nJobs));
        max_threads = nJobs; // Cap threads to number of jobs if over-requested
    }

    int minPos = first_search_position;
    int maxPos = last_search_position;
    int incPos = (nJobs) / (max_threads); // Increment for distributing jobs to threads

    //    wxPrintf("First last and inc %d, %d, %d\n", minPos, maxPos, incPos);

#ifdef ENABLEGPU
    profile_timing.start("Init GPU");
    TemplateMatchingCore* GPU;
    DeviceManager         gpuDev;
    profile_timing.lap("Init GPU");

    // We want to share the input image so that we can get the best L2 caching behavior on the GPU.
    std::shared_ptr<GpuImage> d_input_image = std::make_shared<GpuImage>( );
    if ( use_gpu ) {
        d_input_image->Init(input_image);
        d_input_image->CopyHostToDevice(input_image);
    }
#ifdef cisTEM_USING_FastFFT
    // With FastFFT, the zero-padding of the template is "post" whereas in the original method it was symmetric.
    // So we need to shift the logical origin of the input image to template size / 2
    // FastFFT also expects the data to be transposed in XY in Fourier space, and it is easiest to just to the FFT.
    if ( use_fast_fft ) {
        // FastFFT pads from the upper left corner, so we need to shift the image so the origins coinicide
        d_input_image->PhaseShift(-(d_input_image->physical_address_of_box_center.x - current_projection.physical_address_of_box_center_x),
                                  -(d_input_image->physical_address_of_box_center.y - current_projection.physical_address_of_box_center_y),
                                  0);

        d_input_image->BackwardFFT( );

        // We will have done an "extra" round trip iFFT/FFT her since the input image was originally normalized to STD 1.0, so re-normalize by 1/n
        // In TemplateMatchingCore.cu, we normalize the whitened reference in real space, so we also "take off" sqrt(N_template) here, so that on
        // FwdFFT the variance is still 1 under the footprint of the tempalte. A final accounting for the invFFT is applied (1/sqrt(N_image)) during the invFFT.
        const float round_trip_scale = 1.f / d_input_image->number_of_real_space_pixels / sqrtf(float(current_projection.number_of_real_space_pixels));

        FastFFT::FourierTransformer<float, float, float2, 2> FT;
        FT.SetForwardFFTPlan(input_image.logical_x_dimension,
                             input_image.logical_y_dimension,
                             d_input_image->logical_z_dimension,
                             d_input_image->dims.x,
                             d_input_image->dims.y,
                             d_input_image->dims.z,
                             true);
        FT.SetInverseFFTPlan(d_input_image->dims.x,
                             d_input_image->dims.y,
                             d_input_image->dims.z,
                             d_input_image->dims.x,
                             d_input_image->dims.y,
                             d_input_image->dims.z,
                             true);

        FT.FwdFFT(d_input_image->real_values);

        // We've done a round trip iFFT/FFT since the input image was normalized to STD 1.0, so re-normalize by 1/n
        d_input_image->is_in_real_space = false;
        d_input_image->MultiplyByConstant(round_trip_scale);
    }
#endif

    // Convert input image to FP16 on GPU if using GPU
    // Finally, we want to move this into the FP16 buffer. TODO: we can probably safely deallocate the single precision buffer here
    if ( use_gpu )
        d_input_image->CopyFP32toFP16buffer(false);
#endif

    // Initialize TemplateMatchingCore instances for GPU processing
    if ( use_gpu ) {
        number_of_search_positions_per_thread = number_of_search_positions / max_threads;

#ifdef ENABLEGPU
        profile_timing.start("Init GPU");
        GPU = new TemplateMatchingCore[max_threads];
        gpuDev.Init(nGPUs, this);
        profile_timing.lap("Init GPU");
        //    wxPrintf("Host: %s is running\nnThreads: %d\nnGPUs: %d\n:nSearchPos %d \n",hostNameBuffer,nThreads, nGPUs, maxPos);

        //    TemplateMatchingCore GPU(number_of_jobs_per_image_in_gui);
#endif
    }

    // Initialize progress bar if running locally
    if ( is_running_locally == true ) {
        my_progress = new ProgressBar(number_of_search_positions_per_thread);
    }

    // Main search loop: iterates over pixel size and defocus variations
    for ( int size_i = -myroundint(float(pixel_size_search_range) / float(pixel_size_step)); size_i <= myroundint(float(pixel_size_search_range) / float(pixel_size_step)); size_i++ ) {
        profile_timing.start("ChangePixelSize");
        // Adjust template reconstruction for current pixel size
        // Manually set this so that if we do loop over the pixel size, it doesn't create any problems with the gpu branch
        template_reconstruction.is_fft_centered_in_box = false;

        input_reconstruction.ChangePixelSize(&template_reconstruction, (data_sizer.GetSearchPixelSize( ) + float(size_i) * pixel_size_step) / data_sizer.GetSearchPixelSize( ), 0.001f, true);
        //    template_reconstruction.ForwardFFT();
        profile_timing.lap("ChangePixelSize");
        profile_timing.start("SwapRealSpaceQuadrants");
        template_reconstruction.ZeroCentralPixel( );
        template_reconstruction.SwapRealSpaceQuadrants( );
        profile_timing.lap("SwapRealSpaceQuadrants");

        // Prepare template for GPU if enabled
        if ( use_gpu ) {
#ifdef ENABLEGPU
            // Shared pointer for the 3D template on GPU, accessible by all threads.
            std::shared_ptr<GpuImage> template_reconstruction_gpu;
            if ( use_gpu_prj ) {

                // FIXME: move this (and the above CPU steps) into a method to prepare the 3d reference.
                profile_timing.start("Swap Fourier Quadrants");
                template_reconstruction.BackwardFFT( );
                // FIXME: this should be a GPU method (or also a ...) and it should optionally be combined with a copy to texture.
                template_reconstruction.SwapFourierSpaceQuadrants(false);
                profile_timing.lap("Swap Fourier Quadrants");
                // We only want to have one copy of the 3d template in texture memory that each thread can then reference.
                // First allocate a shared pointer and construct the GpuImage based on the CPU template
                // TODO: Initially, i had this set to use
                // GpuImage::InitializeBasedOnCpuImage(tmp_vol, false, true); where the memory is instructed not to be pinned.
                // This should be fine now, but .
                profile_timing.start("CopyHostToDeviceTextureComplex");

                // Create GpuImage from the CPU template and copy to GPU texture memory.
                template_reconstruction_gpu = std::make_shared<GpuImage>(template_reconstruction);
                template_reconstruction_gpu->CopyHostToDeviceTextureComplex<3>(template_reconstruction);

                profile_timing.lap("CopyHostToDeviceTextureComplex");
            }

            data_sizer.whitening_filter_ptr->MakeThreadSafeForNThreads(max_threads);
            size_t L2_window_size;
            // note that we need the firstprivate so the shared ptr is intialized the first time it is encountered
#pragma omp parallel num_threads(max_threads) default(none) shared(L2_window_size, first_gpu_loop, GPU, first_search_position, incPos, maxPos, max_threads,                                      \
                                                                   d_input_image, angles, my_progress, template_reconstruction, use_fast_fft, projection_filter,                                 \
                                                                   min_counter_val, profile_timing, current_projection, psi_start, psi_step, psi_max,                                            \
                                                                   global_euler_search, number_of_search_positions, number_of_search_positions_per_thread, use_gpu_prj,                          \
                                                                   data_sizer, best_psi, best_theta, best_phi, best_defocus, best_pixel_size,                                                    \
                                                                   correlation_pixel_sum, correlation_pixel_sum_image, correlation_pixel_sum_of_squares, correlation_pixel_sum_of_squares_image, \
                                                                   actual_number_of_angles_searched, defocus_step) firstprivate(template_reconstruction_gpu, L2_persistance_fraction, current_correlation_position)
            {
                int tIDX = ReturnThreadNumberOfCurrentThread( );
                // gpuDev.SetGpu( );

                // Initialize TemplateMatchingCore for this thread on its first loop iteration
                if ( first_gpu_loop.at(tIDX) ) {

#ifdef USE_LERP_NOT_FOURIER_RESIZING
                    GPU[tIDX].use_lerp_for_resizing = true;
                    GPU[tIDX].binning_factor        = data_sizer.GetFullBinningFactor( );
#endif

                    // Determine the range of Euler search positions for this thread
                    int t_first_search_position = first_search_position + (tIDX * incPos);
                    int t_last_search_position  = first_search_position + (incPos - 1) + (tIDX * incPos);
                    if ( tIDX == (max_threads - 1) ) // Last thread takes any remaining positions
                        t_last_search_position = maxPos;
                    profile_timing.start("Init GPU");
                    // Initialize the TemplateMatchingCore instance for this thread
                    GPU[tIDX].Init(this,
                                   template_reconstruction_gpu, // Shared pointer to GPU template
                                   d_input_image, // Shared pointer to GPU input image
                                   current_projection, // CPU projection image (used if use_gpu_prj is false)
                                   psi_max,
                                   psi_start,
                                   psi_step,
                                   angles,
                                   global_euler_search,
                                   data_sizer.GetPrePadding( ),
                                   data_sizer.GetRoi( ),
                                   t_first_search_position,
                                   t_last_search_position,
                                   my_progress,
                                   number_of_search_positions_per_thread,
                                   is_running_locally,
                                   use_fast_fft,
                                   use_gpu_prj);

                    profile_timing.lap("Init GPU");
                    if ( ! use_gpu_prj ) { // If projections are on CPU, set CPU template for the core
                        GPU[tIDX].SetCpuTemplate(&template_reconstruction);
                    }
                    // Critical section for printing, ensuring orderly output from threads
#pragma omp critical
                    {
                        wxPrintf("Staring TemplateMatchingCore object %d to work on position range %d-%d\n", tIDX, t_first_search_position, t_last_search_position);
                    }
                    first_gpu_loop.at(tIDX) = false;

                    if ( tIDX == 0 )
                        L2_window_size = GPU[tIDX].SetL2CachePersisting(L2_persistance_fraction);

#pragma omp barrier // all threads need to wait on tIDX 0 to set the device props

                    // Now that the L2 cache is set persisiting and all threads have reached this point, set up the individual policies in their respective streams.
                    if ( L2_window_size > 0 )
                        GPU[tIDX].SetL2AccessPolicy(L2_window_size);
                }
                else {
                    GPU[tIDX].template_gpu_shared = template_reconstruction_gpu;
                }
            } // end of omp block
#endif
        }

        for ( int defocus_i = -myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i <= myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i++ ) {

            profile_timing.start("Ctf and whitening filter");
            // Create projection filter (CTF * whitening_filter) for current defocus
            input_ctf.SetDefocus((defocus1 + float(defocus_i) * defocus_step) / wanted_pre_projection_pixel_size,
                                 (defocus2 + float(defocus_i) * defocus_step) / wanted_pre_projection_pixel_size,
                                 deg_2_rad(defocus_angle));
            // Reset this bool since we will overwrite all values in the CTF image.
            projection_filter.is_fft_centered_in_box = false;
            projection_filter.CalculateCTFImage(input_ctf);
            projection_filter.ApplyCurveFilter(data_sizer.whitening_filter_ptr.get( ));

            profile_timing.lap("Ctf and whitening filter");

            // GPU execution path
            if ( use_gpu ) {
#ifdef ENABLEGPU
                // Prepare projection filter for GPU (swap quadrants if projecting on GPU)
                // Rather than multiplying the projection by the ctf_image, we will interpolate from it
                // This allows intra projection down sampling and also keeps the ctf in fast read only cache.
                // As with the 3d volume, we have to swap the fourier space quadrants and shift x by 1 to have spatially local interp
                if ( use_gpu_prj )
                    projection_filter.SwapFourierSpaceQuadrants(false, true);

#pragma omp parallel num_threads(max_threads) default(none) shared(min_counter_val, threshold_val, data_sizer, best_psi, best_theta, best_phi, best_defocus, best_pixel_size, max_intensity_projection,                            \
                                                                   correlation_pixel_sum, correlation_pixel_sum_image, correlation_pixel_sum_of_squares, correlation_pixel_sum_of_squares_image, actual_number_of_angles_searched, \
                                                                   profile_timing, GPU, projection_filter, current_projection, angles, global_euler_search, number_of_search_positions_per_thread, use_gpu_prj,                    \
                                                                   defocus_i, defocus_step, size_i, pixel_size_step, histogram_data) private(current_correlation_position)

                {
                    int tIDX = ReturnThreadNumberOfCurrentThread( );
                    // gpuDev.SetGpu( ); // Potentially set GPU for this thread

                    // Each thread calls RunInnerLoop on its TemplateMatchingCore instance
                    profile_timing.start("RunInnerLoop");
                    GPU[tIDX].RunInnerLoop(projection_filter, // Current projection filter
                                           tIDX,
                                           current_correlation_position, // Used for progress, might need adjustment for per-thread
                                           min_counter_val,
                                           threshold_val);
                    profile_timing.lap("RunInnerLoop");

                    // Critical section to aggregate results from each thread's GPU buffers to shared host arrays.
                    // This prevents race conditions when updating global MIP, angle maps, sums, and histogram.
#pragma omp critical
                    {
                        profile_timing.start("Transfer data back to host");
                        // Copy results from this thread's GPU buffers to temporary host Images
                        Image mip_buffer   = GPU[tIDX].d_max_intensity_projection.CopyDeviceToNewHost(true, false);
                        Image psi_buffer   = GPU[tIDX].d_best_psi.CopyDeviceToNewHost(true, false);
                        Image phi_buffer   = GPU[tIDX].d_best_phi.CopyDeviceToNewHost(true, false);
                        Image theta_buffer = GPU[tIDX].d_best_theta.CopyDeviceToNewHost(true, false);

                        Image sum   = GPU[tIDX].d_sum2.CopyDeviceToNewHost(true, false);
                        Image sumSq = GPU[tIDX].d_sumSq2.CopyDeviceToNewHost(true, false);

                        // Aggregate results into global host arrays
                        // Note: even if we have ignored some invalid boundary values, copy over everything here
                        for ( int current_y = data_sizer.GetPrePaddingY( ); current_y < data_sizer.GetPrePaddingY( ) + data_sizer.GetRoiY( ); current_y++ ) {
                            for ( int current_x = data_sizer.GetPrePaddingX( ); current_x < data_sizer.GetPrePaddingX( ) + data_sizer.GetRoiX( ); current_x++ ) {
                                // first mip
                                long address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, 0);

                                // Update global MIP and best angle/parameter maps if current thread found a better CCC
                                if ( mip_buffer.real_values[address] > max_intensity_projection.real_values[address] ) {
                                    max_intensity_projection.real_values[address] = mip_buffer.real_values[address];
                                    best_psi.real_values[address]                 = psi_buffer.real_values[address];
                                    best_theta.real_values[address]               = theta_buffer.real_values[address];
                                    best_phi.real_values[address]                 = phi_buffer.real_values[address];
                                    best_defocus.real_values[address]             = float(defocus_i) * defocus_step;
                                    best_pixel_size.real_values[address]          = float(size_i) * pixel_size_step;
                                }

                                // Accumulate sums and sums of squares
                                correlation_pixel_sum[address] += (double)sum.real_values[address];
                                correlation_pixel_sum_of_squares[address] += (double)sumSq.real_values[address];
                            }
                        }

                        // Add this thread's histogram data to the global histogram
                        // GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);
                        GPU[tIDX].my_dist->CopyToHostAndAdd(histogram_data);

                        // Update total number of angles searched
                        //                    current_correlation_position += GPU[tIDX].total_number_of_cccs_calculated;
                        actual_number_of_angles_searched += float(GPU[tIDX].total_number_of_cccs_calculated);
                        profile_timing.lap("Transfer data back to host");
                    } // end of omp critical block
                } // end of parallel block

#endif
            }
            else { // CPU execution path (serial within this function call)
                profile_timing.start("RunInnerLoop cpu");
                // Loop over Euler search positions
                for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
                    //loop over each in-plane rotation angle (psi)

                    //current_rotation = 0;
                    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

                        angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
                        //                    angles.Init(130.0, 30.0, 199.5, 0.0, 0.0);

                        // Extract 2D projection from 3D template
                        if ( padding != 1.0f ) {
                            template_reconstruction.ExtractSlice(padded_projection, angles, 1.0f, false);
                            padded_projection.SwapRealSpaceQuadrants( );
                            padded_projection.BackwardFFT( );
                            padded_projection.ClipInto(&current_projection);
                            current_projection.ForwardFFT( );
                        }
                        else {
                            template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
                            current_projection.SwapRealSpaceQuadrants( );
                        }

                        // Apply projection filter (CTF * whitening)
                        current_projection.MultiplyPixelWise(projection_filter);

                        // Inverse FFT to get cross-correlation map (CCM)
                        current_projection.BackwardFFT( );
                        //current_projection.ReplaceOutliersWithMean(6.0f);

                        current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));

                        // Normalize projection
                        // We want a variance of 1 in the padded FFT. Scale the small SumOfSquares (which is already divided by n) and then re-divide by N.
                        variance = current_projection.ReturnSumOfSquares( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - powf(current_projection.ReturnAverageOfRealValues( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels, 2);
                        current_projection.DivideByConstant(sqrtf(variance));
                        current_projection.ClipIntoLargerRealSpace2D(&padded_reference); // Padded_reference now holds the normalized template projection

                        // Forward FFT the padded reference (template projection)
                        // Note: The real space variance is set to 1.0 (for the padded size image) and that results in a variance of N in the FFT do to the scaling of the FFT,
                        // but the FFT values are divided by 1/N so the variance becomes N / (N^2) = is 1/N
                        padded_reference.ForwardFFT( );
                        // Zeroing the central pixel is probably not doing anything useful...
                        padded_reference.ZeroCentralPixel( );

#ifdef MKL
                        // Use the MKL
                        vmcMulByConj(padded_reference.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(input_image.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                        for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter++ ) {
                            padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * input_image.complex_values[pixel_counter];
                        }
#endif

                        // Note: the cross correlation will have variance 1/N (the product of variance of the two FFTs assuming the means are both zero and the distributions independent.)
                        // Taking the inverse FFT scales this variance by N resulting in a MIP with variance 1
                        padded_reference.BackwardFFT( );

                        // update mip, and histogram..

                        for ( int current_y = data_sizer.GetPrePaddingY( ); current_y < data_sizer.GetPrePaddingY( ) + data_sizer.GetRoiY( ); current_y++ ) {
                            for ( int current_x = data_sizer.GetPrePaddingX( ); current_x < data_sizer.GetPrePaddingX( ) + data_sizer.GetRoiX( ); current_x++ ) {
                                // first mip
                                long address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, 0);
                                if ( padded_reference.real_values[address] > max_intensity_projection.real_values[address] ) {
                                    max_intensity_projection.real_values[address] = padded_reference.real_values[address];
                                    best_psi.real_values[address]                 = current_psi;
                                    best_theta.real_values[address]               = global_euler_search.list_of_search_parameters[current_search_position][1];
                                    best_phi.real_values[address]                 = global_euler_search.list_of_search_parameters[current_search_position][0];
                                    best_defocus.real_values[address]             = float(defocus_i) * defocus_step;
                                    best_pixel_size.real_values[address]          = float(size_i) * pixel_size_step;
                                    //                                if (size_i != 0) wxPrintf("size_i = %i\n", size_i);
                                    //                                correlation_pixel_sum[pixel_counter] = variance;
                                }

                                // histogram
                                float mip_value = padded_reference.real_values[address];
                                current_bin     = int((mip_value - histogram_min) / histogram_step);
                                //current_bin = int(double((padded_reference.real_values[address]) - histogram_min) / histogram_step);

                                if ( current_bin >= 0 && current_bin <= histogram_number_of_points ) {
                                    histogram_data[current_bin] += 1;
                                }

                                // Note: this one is outside the ifdefs so we can leave the "normal" stats images in places.
                                if ( use_local_normalization ) {
                                    // Local normalization
#ifdef TEST_LOCAL_NORMALIZATION
                                    float value = padded_reference.real_values[address]; //* (float)sqrt_input_pixels;
                                    // Welford's algorithm for trimming
                                    // For the GPU implementation we'll have at least 10 (though currently 20) mip values the first time we go through a stack, so
                                    // rather than just skipping the first 10 and assuming no outliers, we can probably be more clever.
                                    if ( n_image[address] < BUFFER_SIZE ) {
                                        // Buffering phase
                                        n_image[address]++;
                                        float delta = value - mean_image[address];
                                        mean_image[address] += delta / n_image[address];
                                        float delta2 = value - mean_image[address];
                                        M2_image[address] += delta * delta2;
                                    }
                                    else {
                                        // Outlier trimming
                                        variance_image[address] = M2_image[address] / (n_image[address] - 1);
                                        stddev_image[address]   = std::sqrt(variance_image[address]);
                                        if ( std::abs(value - mean_image[address]) > OUTLIER_THRESHOLD * stddev_image[address] ) {
                                            // Skip outlier

                                            continue;
                                        }

                                        // Update running statistics for non-outliers
                                        n_image[address]++;
                                        float delta = value - mean_image[address];
                                        mean_image[address] += delta / n_image[address];
                                        float delta2 = value - mean_image[address];
                                        M2_image[address] += delta * delta2;
                                    }

#endif
                                }
                                else {
                                    correlation_pixel_sum[address] += mip_value;
                                    correlation_pixel_sum_of_squares[address] += mip_value * mip_value;
                                }
                            }
                        }

                        current_projection.is_in_real_space = false;
                        padded_reference.is_in_real_space   = true;

                        current_correlation_position++;
                        if ( is_running_locally == true )
                            my_progress->Update(current_correlation_position);

                        actual_number_of_angles_searched++;

                        if ( is_running_locally == false ) {

                            // Currently there is no subsampling in the CPU implementation
                            temp_float             = current_correlation_position;
                            JobResult* temp_result = new JobResult;
                            temp_result->SetResult(1, &temp_float);
                            AddJobToResultQueue(temp_result);
                        }
                    }
                }
                profile_timing.lap("RunInnerLoop cpu");
            } // if/else on use_gpu for inner loop
        } // defocus loop
    } // pixel size loop

    // Most of the time we can get away without synchronizing here, however,

    wxPrintf("\n\n\tTimings: Overall: %s\n", (wxDateTime::Now( ) - overall_start).Format( ));

    // Post-search processing: resize results back to original dimensions if necessary
    profile_timing.start("Resize_postSearch");
    // We may have rotated or re-sized the image for performance. To map the results back, it will be
    // easiest to convert the statistical arrays back to images.
    if ( use_local_normalization ) {
#ifdef TEST_LOCAL_NORMALIZATION
        if ( ! use_gpu )
            wxPrintf("\n\n\nLocal normalization: Done on cpu!\n");
        else {
            // FIXME: redundant
            for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
                correlation_pixel_sum_image.real_values[pixel_counter]            = (float)correlation_pixel_sum[pixel_counter];
                correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares[pixel_counter];
            }
        }
#endif
    }
    else {
        for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
            correlation_pixel_sum_image.real_values[pixel_counter]            = (float)correlation_pixel_sum[pixel_counter];
            correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares[pixel_counter];
        }
    }

    // Remove any unwanted values in the padding area from FFTs
    correlation_pixel_sum_image.ZeroFFTWPadding( );
    correlation_pixel_sum_of_squares_image.ZeroFFTWPadding( );

    // Resize output images (MIP, angles, etc.) and statistical maps back to original dimensions if search was on resized images.
    // Even if we apply_result_rescaling, we still want to remove any padding.
    data_sizer.ResizeImage_postSearch(max_intensity_projection,
                                      best_psi,
                                      best_phi,
                                      best_theta,
                                      best_defocus,
                                      best_pixel_size,
                                      correlation_pixel_sum_image,
                                      correlation_pixel_sum_of_squares_image,
                                      apply_result_rescaling,
                                      max_threads);

    float output_pixel_size = apply_result_rescaling ? data_sizer.GetPixelSize( ) : data_sizer.GetSearchPixelSize( );

    profile_timing.lap("Resize_postSearch");
    profile_timing.print_times( );

    // If running locally, finalize results (rescale MIP, save files)
    if ( is_running_locally ) {
        delete my_progress;

// FIXME: This needs to go into the other functions
#ifdef TEST_LOCAL_NORMALIZATION
        // The gpu implementation is returning the sum and sum of squares images
        if ( use_local_normalization && ! use_gpu ) {
            for ( long pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
                correlation_pixel_sum[pixel_counter]            = mean_image[pixel_counter];
                correlation_pixel_sum_of_squares[pixel_counter] = stddev_image[pixel_counter];
            }
        }
#endif

        // Rescale MIP and statistical arrays based on global CCC mean and stddev
        // Adjust the MIP by the measured mean and stddev of the full search CCC which is an estimate for the moments of the noise distribution of CCCs.
        Image scaled_mip = max_intensity_projection;
        RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(&max_intensity_projection,
                                                            &scaled_mip,
                                                            correlation_pixel_sum_image.real_values,
                                                            correlation_pixel_sum_of_squares_image.real_values,
                                                            histogram_data,
                                                            actual_number_of_angles_searched,
                                                            disable_flat_fielding);
        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0f;
        double      temp_threshold;
        double      erf_input = (n_expected_false_positives * 2.0) / (1.0 * double(data_sizer.GetNumberOfValidSearchPixels( )) * double(actual_number_of_angles_searched) * double(fraction_of_search_positions_that_are_independent));
#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        cisTEM_erfcinv(erf_input); // Note: This seems to be a custom function, ensure it's correctly defined and linked.
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        // Write output MRC files for MIP, scaled MIP, best angles, defocus, pixel size, and statistical maps
        temp_image.CopyFrom(&max_intensity_projection);
        MRCFile mip_out(mip_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        mip_out.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&mip_out, 1);
        mip_out.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile scaled_mip_output_mrcfile(scaled_mip_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        scaled_mip_output_mrcfile.SetOutputToFP16( );
#endif
        scaled_mip.WriteSlice(&scaled_mip_output_mrcfile, 1);
        scaled_mip_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile correlation_pixel_sum_output_mrcfile(correlation_avg_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        correlation_pixel_sum_output_mrcfile.SetOutputToFP16( );
#endif
        correlation_pixel_sum_image.WriteSlice(&correlation_pixel_sum_output_mrcfile, 1);
        correlation_pixel_sum_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile correlation_pixel_sum_of_squares_output_mrcfile(correlation_std_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        correlation_pixel_sum_of_squares_output_mrcfile.SetOutputToFP16( );
#endif
        correlation_pixel_sum_of_squares_image.WriteSlice(&correlation_pixel_sum_of_squares_output_mrcfile, 1);
        correlation_pixel_sum_of_squares_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile best_psi_output_mrcfile(best_psi_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        best_psi_output_mrcfile.SetOutputToFP16( );
#endif
        best_psi.WriteSlice(&best_psi_output_mrcfile, 1);
        best_psi_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile best_theta_output_mrcfile(best_theta_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        best_theta_output_mrcfile.SetOutputToFP16( );
#endif
        best_theta.WriteSlice(&best_theta_output_mrcfile, 1);
        best_theta_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile best_phi_output_mrcfile(best_phi_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        best_phi_output_mrcfile.SetOutputToFP16( );
#endif
        best_phi.WriteSlice(&best_phi_output_mrcfile, 1);
        best_phi_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile best_defocus_output_mrcfile(best_defocus_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        best_defocus_output_mrcfile.SetOutputToFP16( );
#endif
        best_defocus.WriteSlice(&best_defocus_output_mrcfile, 1);
        best_defocus_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        MRCFile best_pixel_size_output_mrcfile(best_pixel_size_output_file.ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        best_pixel_size_output_mrcfile.SetOutputToFP16( );
#endif
        best_pixel_size.WriteSlice(&best_pixel_size_output_mrcfile, 1);
        best_pixel_size_output_mrcfile.SetPixelSizeAndWriteHeader(output_pixel_size);

        // Write out histogram text file
        NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 4);

        double* expected_survival_histogram = new double[histogram_number_of_points];
        double* survival_histogram          = new double[histogram_number_of_points];
        ZeroDoubleArray(survival_histogram, histogram_number_of_points);

        for ( int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++ ) {
            expected_survival_histogram[line_counter] = (erfc((histogram_first_bin_midpoint + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * float(actual_number_of_angles_searched) * float(fraction_of_search_positions_that_are_independent) / n_expected_false_positives;
        }

        survival_histogram[histogram_number_of_points - 1] = histogram_data[histogram_number_of_points - 1];

        for ( int line_counter = histogram_number_of_points - 2; line_counter >= 0; line_counter-- ) {
            survival_histogram[line_counter] = survival_histogram[line_counter + 1] + histogram_data[line_counter];
        }

        histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
        histogram_file.WriteCommentLine("SNR, histogram, survival histogram, random survival histogram");

        for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
            temp_double_array[0] = histogram_first_bin_midpoint + histogram_step * float(line_counter);
            temp_double_array[1] = histogram_data[line_counter];
            temp_double_array[2] = survival_histogram[line_counter];
            temp_double_array[3] = expected_survival_histogram[line_counter];
            histogram_file.WriteLine(temp_double_array);
        }

        histogram_file.Close( );

        // memory cleanup
        delete[] survival_histogram;
        delete[] expected_survival_histogram;
    }
    else {
        // send back the final images to master (who should merge them, and send to the gui)

        long   result_array_counter;
        long   number_of_result_floats = cistem::match_template::COUNT; // first float is x size, 2nd is y size of images, 3rd is number allocated, 4th  float is number of doubles in the histogram
        long   pixel_counter;
        float* pointer_to_histogram_data;

        pointer_to_histogram_data = (float*)histogram_data;

        // Make sure there is enough space allocated for all results
        number_of_result_floats += max_intensity_projection.real_memory_allocated * cistem::match_template::number_of_output_images;
        number_of_result_floats += histogram_number_of_points * sizeof(long) / sizeof(float); // histogram are longs

        float* result = new float[number_of_result_floats];
        // Not zero floating this array since all additions are assignments. This can help to expose any indexing errors.

        // Brackets are just there to limit the scope of the using declaration.
        {
            using cm_t = cistem::match_template::Enum;

            result[cm_t::image_size_x]                                      = float(max_intensity_projection.logical_x_dimension);
            result[cm_t::image_size_y]                                      = float(max_intensity_projection.logical_y_dimension);
            result[cm_t::image_real_memory_allocated]                       = float(max_intensity_projection.real_memory_allocated);
            result[cm_t::number_of_histogram_bins]                          = float(histogram_number_of_points);
            result[cm_t::number_of_angles_searched]                         = actual_number_of_angles_searched;
            result[cm_t::fraction_of_search_positions_that_are_independent] = fraction_of_search_positions_that_are_independent;
            result[cm_t::ccc_scalar]                                        = 1.0f; // (float)sqrt_input_pixels is redundant, but we need all the results to calculate the scaling from the global CCC moments
            result[cm_t::search_pixel_size]                                 = output_pixel_size;
            result[cm_t::input_binning_factor]                              = apply_result_rescaling ? 1.0f : data_sizer.GetFullBinningFactor( );
            result[cm_t::number_of_valid_search_pixels]                     = float(data_sizer.GetNumberOfValidSearchPixels( )); // if apply_result_rescaling is false, this will = image_size_x * image_size_y as they are cropped to the ROI
            result[cm_t::disable_flat_fielding]                             = float(disable_flat_fielding);
            result[cm_t::number_of_expected_false_positives]                = float(n_expected_false_positives);

            if ( ! apply_result_rescaling ) {
                MyDebugAssertTrue(data_sizer.GetNumberOfValidSearchPixels( ) == (max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension),
                                  "If apply_result_rescaling is true, the number of valid search pixels (%ld) must equal the image size (%d, %d)\n", data_sizer.GetNumberOfValidSearchPixels( ), max_intensity_projection.logical_x_dimension, max_intensity_projection.logical_y_dimension);
            }
        }

        result_array_counter = cistem::match_template::COUNT;

        // Copy the MIP, best angles, defocus, pixel size, and statistical data to the result array
        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = max_intensity_projection.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = best_psi.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = best_theta.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = best_phi.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = best_defocus.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = best_pixel_size.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = correlation_pixel_sum_image.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++ ) {
            result[result_array_counter] = correlation_pixel_sum_of_squares_image.real_values[pixel_counter];
            result_array_counter++;
        }

        for ( pixel_counter = 0; pixel_counter < histogram_number_of_points * 2; pixel_counter++ ) {
            result[result_array_counter] = pointer_to_histogram_data[pixel_counter];
            result_array_counter++;
        }

        // Send the packed result to the master process
        SendProgramDefinedResultToMaster(result, number_of_result_floats, image_number_for_gui, number_of_jobs_per_image_in_gui);
        // The result should not be deleted here, as the worker thread will free it up once it has been send to the master
        // delete [] result;
    }

    delete[] histogram_data;
    delete[] correlation_pixel_sum;
    delete[] correlation_pixel_sum_of_squares;
#ifdef ENABLEGPU
    if ( use_gpu ) {
        delete[] GPU;
    }

//  gpuDev.ResetGpu();
#endif

    if ( is_running_locally == true ) {
        wxPrintf("\nMatch Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}

/**
 * @brief Handles program-defined results received by the master process from worker processes.
 *
 * This method is called on the master process when it receives a partial result from a worker.
 * It finds or creates an `AggregatedTemplateResult` object for the specific image (`result_number`).
 * The incoming `result_array` is then added to this `AggregatedTemplateResult`.
 *
 * If all expected partial results for an image have been received:
 * 1.  The aggregated data (MIP, angles, sums, histogram) is retrieved.
 * 2.  The MIP and statistical arrays are rescaled using global CCC statistics (`RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev`).
 * 3.  Output MRC files are written for the finalized MIP, angle maps, defocus, pixel size, and statistical maps.
 * 4.  A histogram text file is generated.
 * 5.  Peak picking is performed on the scaled MIP to identify potential particle locations.
 * 6.  The identified peaks and associated parameters are used to generate a result image (projections of the template at peak locations).
 * 7.  This result image is saved, and the peak information is sent to the GUI socket if connected.
 * 8.  The `AggregatedTemplateResult` object for the completed image is removed.
 *
 * @param result_array The float array containing the partial result data from a worker.
 * @param array_size The size of the `result_array`.
 * @param result_number The identifier of the image this result pertains to (job ID, essentially).
 * @param number_of_expected_results The total number of partial results expected for this image.
 */
void MatchTemplateApp::MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results) {

    // do we have this image number already?

    bool need_a_new_result = true;
    int  array_location    = -1;
    long pixel_counter;

    // constexpr values for histogram values.
    using namespace cistem::match_template;

    wxPrintf("Master Handling result for image %i..", result_number);

    // Check if an AggregatedTemplateResult already exists for this image
    for ( int result_counter = 0; result_counter < aggregated_results.GetCount( ); result_counter++ ) {
        if ( aggregated_results[result_counter].image_number == result_number ) {
            aggregated_results[result_counter].AddResult(result_array, array_size, result_number, number_of_expected_results);
            need_a_new_result = false;
            array_location    = result_counter;
            wxPrintf("Found array location for image %i, at %i\n", result_number, array_location);
            break;
        }
    }

    // If no existing result, create a new one
    if ( need_a_new_result == true ) {
        AggregatedTemplateResult result_to_add;
        // So I guess this Add, then index into size - 1 is like push_back kinda?
        aggregated_results.Add(result_to_add);
        aggregated_results[aggregated_results.GetCount( ) - 1].image_number = result_number;
        aggregated_results[aggregated_results.GetCount( ) - 1].AddResult(result_array, array_size, result_number, number_of_expected_results);
        array_location = aggregated_results.GetCount( ) - 1;
        wxPrintf("Adding new result to array for image %i, at %i\n", result_number, array_location);
    }

    // Check if all expected results for this image have been received
    if ( aggregated_results[array_location].number_of_received_results == number_of_expected_results ) {
        // All parts of the result for this image are now collected. Proceed to finalize.
        // TODO send the result back to the GUI, for now hack mode to save the files to the directory..

        wxString directory_for_writing_results = current_job_package.jobs[0].arguments[37].ReturnStringArgument( );

        // Image objects for storing and processing results
        Image temp_image;

        Image scaled_mip;
        Image psi_image;
        Image phi_image;
        Image theta_image;
        Image defocus_image;
        Image pixel_size_image;

        Image result_image;
        Image input_reconstruction;
        Image current_projection;

        int   number_of_peaks_found = 0;
        float sq_dist_x;
        float sq_dist_y;
        float current_phi;
        float current_psi;
        float current_theta;
        int   i;
        int   j;
        long  address;

        ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;
        TemplateMatchFoundPeakInfo         temp_peak_info;

        Peak            current_peak;
        AnglesAndShifts angles;

        bool use_gpu = current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[40].ReturnBoolArgument( );

        int    image_size_x                                      = int(aggregated_results[array_location].collated_data_array[cistem::match_template::image_size_x]);
        int    image_size_y                                      = int(aggregated_results[array_location].collated_data_array[cistem::match_template::image_size_y]);
        int    image_real_memory_allocated                       = int(aggregated_results[array_location].collated_data_array[cistem::match_template::image_real_memory_allocated]);
        float  search_pixel_size                                 = aggregated_results[array_location].collated_data_array[cistem::match_template::search_pixel_size];
        float  input_binning_factor                              = aggregated_results[array_location].collated_data_array[cistem::match_template::input_binning_factor];
        long   number_of_valid_search_pixels                     = long(aggregated_results[array_location].collated_data_array[cistem::match_template::number_of_valid_search_pixels]);
        double n_expected_false_positives                        = double(aggregated_results[array_location].collated_data_array[cistem::match_template::number_of_expected_false_positives]);
        float  fraction_of_search_positions_that_are_independent = aggregated_results[array_location].collated_data_array[cistem::match_template::fraction_of_search_positions_that_are_independent];

        double number_of_search_positions = double(aggregated_results[array_location].total_number_of_angles_searched * fraction_of_search_positions_that_are_independent);

        bool using_binned_ref = input_binning_factor > 1.0f ? true : false;

        ImageFile input_reconstruction_file;
        input_reconstruction_file.OpenFile(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[1].ReturnStringArgument( ), false);

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);

        // Fill the temp_image with data form the collatged mip before passing it on to be rescaled.
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_mip_data[pixel_counter];
        }

        scaled_mip.CopyFrom(&temp_image);
        RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(&temp_image,
                                                            &scaled_mip,
                                                            aggregated_results[array_location].collated_pixel_sums,
                                                            aggregated_results[array_location].collated_pixel_square_sums,
                                                            aggregated_results[array_location].collated_histogram_data,
                                                            aggregated_results[array_location].total_number_of_angles_searched,
                                                            aggregated_results[array_location].disable_flat_fielding);

        // Update the collated mip data which is used downstream for the scaled mip and other calcs
        // Fill the temp_image with data form the collatged mip before passing it on to be rescaled.
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            aggregated_results[array_location].collated_mip_data[pixel_counter] = temp_image.real_values[pixel_counter];
        }

        MRCFile mip_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[21].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        mip_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&mip_output_file, 1);
        mip_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        wxPrintf("Writing result %i\n", aggregated_results[array_location].image_number - 1);

        // psi
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_psi_data[pixel_counter];
        }
        MRCFile psi_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[22].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        psi_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&psi_output_file, 1);
        psi_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        psi_image.CopyFrom(&temp_image);
        //theta

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_theta_data[pixel_counter];
        }
        MRCFile theta_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[23].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        theta_output_file.SetOutputToFP16( );
#endif

        temp_image.WriteSlice(&theta_output_file, 1);
        theta_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        theta_image.CopyFrom(&temp_image);

        // phi

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_phi_data[pixel_counter];
        }
        MRCFile phi_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[24].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        phi_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&phi_output_file, 1);
        phi_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        phi_image.CopyFrom(&temp_image);

        // defocus

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_defocus_data[pixel_counter];
        }
        MRCFile defocus_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[25].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        defocus_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&defocus_output_file, 1);
        defocus_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        defocus_image.CopyFrom(&temp_image);

        // pixel size

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_size_data[pixel_counter];
        }
        MRCFile pixel_size_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[26].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        pixel_size_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&pixel_size_output_file, 1);
        pixel_size_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        pixel_size_image.CopyFrom(&temp_image);

        // Write scaled MIP (locally normalized if flat fielding enabled)
        MRCFile scaled_mip_output_mrcfile(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[27].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        scaled_mip_output_mrcfile.SetOutputToFP16( );
#endif
        scaled_mip.WriteSlice(&scaled_mip_output_mrcfile, 1);
        scaled_mip_output_mrcfile.SetPixelSizeAndWriteHeader(search_pixel_size);

        // Write statistical sum and sum_of_squares maps
        // sums

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_sums[pixel_counter];
        }
        MRCFile sum_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[36].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        sum_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&sum_output_file, 1);
        sum_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);
        // square sums

        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
        }
        MRCFile square_sum_output_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[28].ReturnStringArgument( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        square_sum_output_file.SetOutputToFP16( );
#endif
        temp_image.WriteSlice(&square_sum_output_file, 1);
        square_sum_output_file.SetPixelSizeAndWriteHeader(search_pixel_size);

        // Write histogram text file
        //NumericTextFile histogram_file(wxString::Format("%s/histogram_%i.txt", directory_for_writing_results, aggregated_results[array_location].image_number), OPEN_TO_WRITE, 4);
        NumericTextFile histogram_file(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[31].ReturnStringArgument( ), OPEN_TO_WRITE, 4);

        double* expected_survival_histogram = new double[histogram_number_of_points];
        double* survival_histogram          = new double[histogram_number_of_points];

        float expected_threshold;

        double temp_double_array[5];

        ZeroDoubleArray(survival_histogram, histogram_number_of_points);
        survival_histogram[histogram_number_of_points - 1] = aggregated_results[array_location].collated_histogram_data[histogram_number_of_points - 1];

        for ( int line_counter = histogram_number_of_points - 2; line_counter >= 0; line_counter-- ) {
            survival_histogram[line_counter] = survival_histogram[line_counter + 1] + aggregated_results[array_location].collated_histogram_data[line_counter];
        }

        for ( int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++ ) {
            expected_survival_histogram[line_counter] = (erfc((histogram_first_bin_midpoint + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * (number_of_valid_search_pixels * number_of_search_positions / n_expected_false_positives);
        }

        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0;
        double      temp_threshold   = 0.0;
        double      erf_input        = (n_expected_false_positives * 2.0) / (1.0 * (double(number_of_valid_search_pixels) * double(number_of_search_positions)));
        //        wxPrintf("ox oy total %3.3e %3.3e %3.3e\n", (double)result_array[5] , (double)result_array[6] , (double)aggregated_results[array_location].total_number_of_angles_searched, erf_input);

#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        temp_threshold       = cisTEM_erfcinv(erf_input);
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
        histogram_file.WriteCommentLine("histogram, expected histogram, survival histogram, expected survival histogram");

        if ( use_gpu ) {
            // In the GPU code, I am not histogramming the padding regions which are not valid. Adjust the counts here. Maybe not the best approach. FIXME also the cpu counts.
            // FIXME: since I'm using number_of_valid_search_pixels this should not be needed.
#ifdef ENABLEGPU
            double sum_expected = 0.0;
            double sum_counted  = 0.0;

            for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
                sum_counted += survival_histogram[line_counter];
                sum_expected += expected_survival_histogram[line_counter];
            }
            for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
                if ( sum_counted > 0.0 )
                    survival_histogram[line_counter] *= (float)(sum_expected / sum_counted);
            }
#endif
        }

        for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
            temp_double_array[0] = histogram_first_bin_midpoint + histogram_step * float(line_counter);
            temp_double_array[1] = aggregated_results[array_location].collated_histogram_data[line_counter];
            temp_double_array[2] = survival_histogram[line_counter];
            temp_double_array[3] = expected_survival_histogram[line_counter];
            histogram_file.WriteLine(temp_double_array);
        }

        histogram_file.Close( );

        // Calculate the result image, and keep the peak info to send back...

        int   min_peak_radius         = current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[39].ReturnFloatArgument( );
        float min_peak_radius_squared = powf(float(min_peak_radius), 2);

        if ( using_binned_ref )
            min_peak_radius_squared /= (input_binning_factor * input_binning_factor);

        result_image.Allocate(scaled_mip.logical_x_dimension, scaled_mip.logical_y_dimension, 1);
        result_image.SetToConstant(0.0f);

        input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
        if ( using_binned_ref ) {
            // Not exact but just for visualization
            int new_size = int(input_reconstruction.logical_x_dimension / input_binning_factor + 0.5f);
            if ( IsOdd(new_size) )
                new_size++;
            input_reconstruction.ForwardFFT( );
            input_reconstruction.Resize(new_size, new_size, new_size);
            input_reconstruction.BackwardFFT( );
        }

        float max_density = input_reconstruction.ReturnAverageOfMaxN( );
        input_reconstruction.DivideByConstant(max_density);

        input_reconstruction.ForwardFFT( );
        input_reconstruction.MultiplyByConstant(sqrtf(input_reconstruction.logical_x_dimension * input_reconstruction.logical_y_dimension * sqrtf(input_reconstruction.logical_z_dimension)));
        input_reconstruction.ZeroCentralPixel( );
        input_reconstruction.SwapRealSpaceQuadrants( );

        // assume cube

        current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);

        // loop until the found peak is below the threshold

#ifdef CISTEM_TEST_FILTERED_MIP
        int exclusion_radius = input_pixel_size / objective_aperture_resolution;
#else
        int exclusion_radius = input_reconstruction.logical_x_dimension / cistem::fraction_of_box_size_to_exclude_for_border + 1;
#endif

        // if we used a resampled search and have elected to skip resampling the results images, this border region is already removed.
        // this should be true for any binning > 1
        if ( input_binning_factor > 1.0f ) {
            exclusion_radius = 0;
        }

        long nTrys = 0;
        while ( 1 == 1 ) {
            // look for a peak..
            nTrys++;
            //            wxPrintf("Trying the %ld'th peak\n",nTrys);
            // FIXME min-distance from edges would be better to set dynamically.
            current_peak = scaled_mip.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, exclusion_radius);
            if ( current_peak.value < expected_threshold )
                break;

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            address = 0;

            current_peak.x = current_peak.x + scaled_mip.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + scaled_mip.physical_address_of_box_center_y;

            // arguments[2] = search_pixel_size
            temp_peak_info.x_pos = current_peak.x * search_pixel_size; // RETURNING IN ANGSTROMS (also takes care of binning if present)
            temp_peak_info.y_pos = current_peak.y * search_pixel_size; // RETURNING IN ANGSTROMS

            //            wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

            for ( j = std::max(myroundint(current_peak.y) - min_peak_radius, 0); j < std::min(myroundint(current_peak.y) + min_peak_radius, scaled_mip.logical_y_dimension); j++ ) {
                sq_dist_y = float(j) - current_peak.y;
                sq_dist_y *= sq_dist_y;

                for ( i = std::max(myroundint(current_peak.x) - min_peak_radius, 0); i < std::min(myroundint(current_peak.x) + min_peak_radius, scaled_mip.logical_x_dimension); i++ ) {
                    sq_dist_x = float(i) - current_peak.x;
                    sq_dist_x *= sq_dist_x;
                    address = phi_image.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);

                    // The square centered at the pixel
                    if ( sq_dist_x == 0 && sq_dist_y == 0 ) {
                        current_phi   = phi_image.real_values[address];
                        current_theta = theta_image.real_values[address];
                        current_psi   = psi_image.real_values[address];

                        temp_peak_info.phi   = phi_image.real_values[address];
                        temp_peak_info.theta = theta_image.real_values[address];
                        temp_peak_info.psi   = psi_image.real_values[address];

                        temp_peak_info.defocus     = defocus_image.real_values[address]; // RETURNING MINUS
                        temp_peak_info.pixel_size  = pixel_size_image.real_values[address];
                        temp_peak_info.peak_height = scaled_mip.real_values[address];
                    }

                    if ( sq_dist_x + sq_dist_y <= min_peak_radius_squared ) {
                        scaled_mip.real_values[address] = -FLT_MAX;
                    }

                    //                    address++;
                }
                //                address += scaled_mip.padding_jump_value;
            }

            //        wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x, current_peak.y, current_psi, current_theta, current_phi, current_defocus, current_pixel_size, current_peak.value);
            //        coordinates[0] = current_peak.x * search_pixel_size;
            //        coordinates[1] = current_peak.y * search_pixel_size;
            ////        coordinates[2] = binned_pixel_size * (slab.physical_address_of_box_center_z - binned_reconstruction.physical_address_of_box_center_z) - current_defocus;
            //        coordinates[2] = binned_pixel_size * slab.physical_address_of_box_center_z - current_defocus;
            //        coordinate_file.WriteLine(coordinates);

            // ok get a projection

            //////////////////////////////////////////////
            // CURRENTLY HARD CODED TO ONLY DO 1000 MAX //
            //////////////////////////////////////////////

            if ( number_of_peaks_found <= cistem::maximum_number_of_detections ) {

                angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

                input_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
                current_projection.SwapRealSpaceQuadrants( );

                current_projection.MultiplyByConstant(sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension));
                current_projection.BackwardFFT( );
                current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));

                // insert it into the output image

                result_image.InsertOtherImageAtSpecifiedPosition(&current_projection, current_peak.x - result_image.physical_address_of_box_center_x, current_peak.y - result_image.physical_address_of_box_center_y, 0, 0.0f);
                all_peak_infos.Add(temp_peak_info);
            }
            else {
                SendInfo("WARNING: More than 1000 peaks above threshold were found. Limiting results to 1000 peaks.\n");
                break;
            }
        }

        // save the output image

        result_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[38].ReturnStringArgument( ), 1, true, search_pixel_size);

        // tell the gui that this result is available...

        ArrayOfTemplateMatchFoundPeakInfos blank_changes;
        SendTemplateMatchingResultToSocket(controller_socket, aggregated_results[array_location].image_number, expected_threshold, all_peak_infos, blank_changes);

        // Clean up: remove the completed AggregatedTemplateResult and associated memory
        // this should be done now.. so delete it

        aggregated_results.RemoveAt(array_location);
        delete[] expected_survival_histogram;
        delete[] survival_histogram;
    }
}

/**
 * @brief Constructor for AggregatedTemplateResult.
 * Initializes members to default values, pointers to NULL.
 */
AggregatedTemplateResult::AggregatedTemplateResult( ) {
    image_number                    = -1;
    number_of_received_results      = 0;
    total_number_of_angles_searched = 0.0f;
    disable_flat_fielding           = false;

    collated_data_array        = NULL;
    collated_mip_data          = NULL;
    collated_psi_data          = NULL;
    collated_theta_data        = NULL;
    collated_phi_data          = NULL;
    collated_defocus_data      = NULL;
    collated_pixel_size_data   = NULL;
    collated_pixel_sums        = NULL;
    collated_pixel_square_sums = NULL;
    collated_histogram_data    = NULL;
}

/**
 * @brief Destructor for AggregatedTemplateResult.
 * Frees the `collated_data_array` if it was allocated.
 */
AggregatedTemplateResult::~AggregatedTemplateResult( ) {
    if ( collated_data_array != NULL )
        delete[] collated_data_array;
}

/**
 * @brief Adds a partial result to this AggregatedTemplateResult object.
 *
 * This method is responsible for merging a new `result_array` (typically from a worker process)
 * into the existing aggregated data.
 *
 * If this is the first result being added (`collated_data_array == NULL`):
 * - Allocates `collated_data_array` to the size of the incoming `result_array`.
 * - Initializes `collated_*` pointers to point to the correct offsets within `collated_data_array`.
 * - Copies header information (image size, histogram bins, etc.) from `result_array` to `collated_data_array`.
 *
 * For all results (first or subsequent):
 * - Updates `total_number_of_angles_searched`.
 * - Iterates through the pixel data:
 *   - If the MIP value in `result_array` is greater than the current `collated_mip_data` for a pixel,
 *     updates `collated_mip_data` and the corresponding best angle/defocus/pixel_size values for that pixel.
 * - Accumulates `collated_pixel_sums` and `collated_pixel_square_sums`.
 * - Accumulates `collated_histogram_data`.
 * - Increments `number_of_received_results`.
 *
 * @param result_array The float array containing the partial result data.
 * @param array_size The total size of the `result_array`.
 * @param result_number The image number this result pertains to (used for verification, though not explicitly here).
 * @param number_of_expected_results The total number of partial results expected for this image (used for logging).
 */
void AggregatedTemplateResult::AddResult(float* result_array, long array_size, int result_number, int number_of_expected_results) {

    int offset = cistem::match_template::COUNT; // Offset to the start of actual image data after header info

    const int histogram_number_of_points = cistem::match_template::histogram_number_of_points;

    // Extract metadata from the incoming result_array
    int   image_size_x                = result_array[cistem::match_template::image_size_x];
    int   image_size_y                = result_array[cistem::match_template::image_size_y];
    int   image_real_memory_allocated = result_array[cistem::match_template::image_real_memory_allocated];
    float search_pixel_size           = result_array[cistem::match_template::search_pixel_size];

    // If this is the first result, allocate memory and set up pointers
    // FIXME: change to nullptr
    if ( collated_data_array == NULL ) {
        collated_data_array = new float[array_size]; // Allocate the main buffer
        ZeroFloatArray(collated_data_array, array_size);
        number_of_received_results      = 0;
        total_number_of_angles_searched = 0.0f;
        disable_flat_fielding           = result_array[cistem::match_template::disable_flat_fielding]; // FIXME: shouldn't we check that these are consistent across all results?

        // Set up pointers to different data sections within collated_data_array
        // nasty..

        collated_mip_data          = &collated_data_array[offset + image_real_memory_allocated * 0];
        collated_psi_data          = &collated_data_array[offset + image_real_memory_allocated * 1];
        collated_theta_data        = &collated_data_array[offset + image_real_memory_allocated * 2];
        collated_phi_data          = &collated_data_array[offset + image_real_memory_allocated * 3];
        collated_defocus_data      = &collated_data_array[offset + image_real_memory_allocated * 4];
        collated_pixel_size_data   = &collated_data_array[offset + image_real_memory_allocated * 5];
        collated_pixel_sums        = &collated_data_array[offset + image_real_memory_allocated * 6];
        collated_pixel_square_sums = &collated_data_array[offset + image_real_memory_allocated * 7];

        collated_histogram_data = (long*)&collated_data_array[offset + image_real_memory_allocated * 8];

        for ( int i_header_info = 0; i_header_info < offset; i_header_info++ ) {
            collated_data_array[i_header_info] = result_array[i_header_info];
        }
    }

    // Update the total number of angles searched across all results
    total_number_of_angles_searched += result_array[cistem::match_template::number_of_angles_searched];

    // Pointers for the different data sections in result_array
    float* result_mip_data          = &result_array[offset + image_real_memory_allocated * 0];
    float* result_psi_data          = &result_array[offset + image_real_memory_allocated * 1];
    float* result_theta_data        = &result_array[offset + image_real_memory_allocated * 2];
    float* result_phi_data          = &result_array[offset + image_real_memory_allocated * 3];
    float* result_defocus_data      = &result_array[offset + image_real_memory_allocated * 4];
    float* result_pixel_size_data   = &result_array[offset + image_real_memory_allocated * 5];
    float* result_pixel_sums        = &result_array[offset + image_real_memory_allocated * 6];
    float* result_pixel_square_sums = &result_array[offset + image_real_memory_allocated * 7];

    long* input_histogram_data = (long*)&result_array[offset + image_real_memory_allocated * 8];

    long pixel_counter;
    long result_array_counter;

    // handle the images..

    for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
        if ( result_mip_data[pixel_counter] > collated_mip_data[pixel_counter] ) {
            collated_mip_data[pixel_counter]        = result_mip_data[pixel_counter];
            collated_psi_data[pixel_counter]        = result_psi_data[pixel_counter];
            collated_theta_data[pixel_counter]      = result_theta_data[pixel_counter];
            collated_phi_data[pixel_counter]        = result_phi_data[pixel_counter];
            collated_defocus_data[pixel_counter]    = result_defocus_data[pixel_counter];
            collated_pixel_size_data[pixel_counter] = result_pixel_size_data[pixel_counter];
        }
    }

    // sums and sum of squares

    for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
        collated_pixel_sums[pixel_counter] += result_pixel_sums[pixel_counter];
    }

    for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
        collated_pixel_square_sums[pixel_counter] += result_pixel_square_sums[pixel_counter];
    }

    // handle the histogram..

    for ( pixel_counter = 0; pixel_counter < histogram_number_of_points; pixel_counter++ ) {
        collated_histogram_data[pixel_counter] += input_histogram_data[pixel_counter];
    }

    number_of_received_results++;
    wxPrintf("Received %i of %i results\n", number_of_received_results, number_of_expected_results);
}

/**
 * @brief Get the measured moments of the CCC distribution. Note that this method relies on the FFT padding having been appropriately managed at earlier steps.
 * 
 * @param global_ccc_mean 
 * @param global_ccc_std_dev 
 * @param sum_of_sqs 
 * @param sum 
 * @param n_angles_in_search 
 * @param N Total number of elements in sum and sum_of_sqs arrays (image_real_memory_allocated).
 */
template <typename StatsType>
void MatchTemplateApp::CalcGlobalCCCScalingFactor(double&     global_ccc_mean,
                                                  double&     global_ccc_std_dev,
                                                  StatsType*  sum,
                                                  StatsType*  sum_of_sqs,
                                                  const float n_angles_in_search,
                                                  const int   N) {

    MyDebugAssertTrue(N > 0, "N must be greater than 0");

    double global_sum            = 0.0;
    double global_sum_of_squares = 0.0;

    long counted_values = 0;
    long address        = 0;

    for ( int address = 0; address < N; address++ ) {
        if ( sum_of_sqs[address] > cistem::float_epsilon ) {
            global_sum += double(sum[address]);
            global_sum_of_squares += double(sum_of_sqs[address]);
            counted_values++;
        }
    }

    const double total_number_of_ccs = double(n_angles_in_search) * double(counted_values);
    std::cerr << "Counted Values: " << counted_values << " out of " << N << " fractions: " << float(counted_values) / float(N) << std::endl;

    global_ccc_mean    = global_sum / total_number_of_ccs;
    global_ccc_std_dev = sqrt(global_sum_of_squares / total_number_of_ccs - double(global_ccc_mean * global_ccc_mean));

    return;
}

/**
 * @brief Resamples the histogram data based on global CCC mean and standard deviation.
 *
 * The raw histogram is collected based on initial estimates. After calculating the true
 * global mean and standard deviation of the CCC distribution, this function rescales
 * the histogram bins and interpolates counts from the original histogram (after
 * Savitzky-Golay smoothing) to populate the resampled histogram. This provides a
 * normalized view of the CCC distribution.
 *
 * @param histogram_ptr Pointer to the histogram data array (long type).
 * @param global_ccc_mean The global mean of CCCs.
 * @param global_ccc_std_dev The global standard deviation of CCCs.
 */
void MatchTemplateApp::ResampleHistogramData(long*        histogram_ptr,
                                             const double global_ccc_mean,
                                             const double global_ccc_std_dev) {

    // constexpr values for histogram values.
    using namespace cistem::match_template;
    // Sample the existing histogram onto a curve object that we can rescale smoothly.
    Curve histogram_curve;
    for ( int i_hist = 0; i_hist < histogram_number_of_points; ++i_hist ) {
        // Original x-value is scaled to (x - mean) / std_dev before adding to curve
        histogram_curve.AddPoint(float((double(histogram_first_bin_midpoint + histogram_step * float(i_hist)) - global_ccc_mean) / global_ccc_std_dev), float(histogram_ptr[i_hist]));
    }

    // We expect the curve to be fairly smooth already, so we'll use a small window size for the fitting.
    // I'm not sure if the polynomial order should be 1 or 3.
    // Smooth the histogram data using Savitzky-Golay filter
    histogram_curve.FitSavitzkyGolayToData(5, 3);

    // We accumulated the histogram based on our best guess at the values from the ccg, but we now need to rescale each x-value and
    // interoplate that from the existing histogram.
    // Populate the original histogram_ptr with new counts interpolated from the smoothed, rescaled curve.
    double scaled_histogram_midpoint;
    for ( int i_hist = 0; i_hist < histogram_number_of_points; ++i_hist ) {
        // Calculate the midpoint of the current bin in the *target* rescaled space
        scaled_histogram_midpoint = (double(histogram_first_bin_midpoint + histogram_step * float(i_hist)) - global_ccc_mean) / global_ccc_std_dev;
        // mip values are scaled by (measured_value - global_ccc_mean) / global_ccc_std_dev.
        // To find the corresponding unscaled value in the measured histogram then
        // we need to divide by mip_rescaling_factor.
        // histogram_ptr[i_hist] = long(histogram_curve.ReturnSavitzkyGolayInterpolationFromX(global_ccc_mean + (scaled_histogram_midpoint * global_ccc_std_dev)));
        // Interpolate the count for this scaled midpoint from the smoothed curve (which is already in the scaled domain)
        histogram_ptr[i_hist] = long(histogram_curve.ReturnSavitzkyGolayInterpolationFromX(scaled_histogram_midpoint));
    }
}

/**
 * @brief Rescales the Maximum Intensity Projection (MIP) and statistical arrays (sum, sum_of_squares)
 *        using the calculated global mean and standard deviation of the Cross-Correlation Coefficients (CCCs).
 *        Also resamples the histogram based on these global statistics.
 *
 * The primary goal is to transform raw CCC values into more meaningful scores, often resembling SNR.
 * - MIP values are rescaled as: `(mip_value - global_ccc_mean) / global_ccc_std_dev`.
 * - Per-pixel sums and sums-of-squares are also rescaled to reflect this transformation.
 * - The histogram is resampled to align with the new distribution of scaled CCCs using `ResampleHistogramData`.
 * - If `disable_flat_fielding` is false, the `scaled_mip` is further normalized by dividing the
 *   globally rescaled MIP by the local standard deviation (derived from the rescaled per-pixel sums and sums-of-squares),
 *   after subtracting the local mean. This aims to correct for local variations in background or noise.
 *
 * @tparam StatsType The data type of `correlation_pixel_sum` and `correlation_pixel_sum_of_squares`.
 * @param mip_image Pointer to the MIP image. Its real_values are modified to be the globally rescaled CCCs.
 * @param scaled_mip Pointer to the output scaled MIP image. This will contain the `mip_image` values,
 *                   potentially further normalized by local statistics if flat fielding is enabled.
 * @param correlation_pixel_sum Pointer to the array of per-pixel sums of CCCs. Modified to reflect global rescaling.
 * @param correlation_pixel_sum_of_squares Pointer to the array of per-pixel sums of squared CCCs. Modified to reflect global rescaling.
 * @param histogram Pointer to the histogram data. Modified by `ResampleHistogramData`.
 * @param n_angles_in_search The total number of angles/orientations searched, used for calculating moments.
 * @param disable_flat_fielding If true, the local normalization step for `scaled_mip` is skipped.
 */
template <typename StatsType>
void MatchTemplateApp::RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(Image*      mip_image,
                                                                           Image*      scaled_mip,
                                                                           StatsType*  correlation_pixel_sum,
                                                                           StatsType*  correlation_pixel_sum_of_squares,
                                                                           long*       histogram,
                                                                           const float n_angles_in_search,
                                                                           const bool  disable_flat_fielding) {

    double global_ccc_mean    = 0.0;
    double global_ccc_std_dev = 0.0;
    CalcGlobalCCCScalingFactor(global_ccc_mean, global_ccc_std_dev, correlation_pixel_sum, correlation_pixel_sum_of_squares, n_angles_in_search, mip_image->real_memory_allocated);

    std::cerr << "Over n_cccs " << n_angles_in_search << " the Global mean and std_dev are " << global_ccc_mean << " and " << global_ccc_std_dev << std::endl;
    // Use the global statistics to resample the histogram from a smoothed curve fit to the measured data.
    ResampleHistogramData(histogram, global_ccc_mean, global_ccc_std_dev);

    // Assuming we want to measure SNR = (CCC - mean) / std_dev, but really we measure std_dev * SNR + mean.
    // Scaling the pixel-wise sum over the search space (A) requires (A - N * mean) / stddev
    // Scaling the pixel-wse sum_of_sqs over the search space (B) requires (B - 2 * mean * A + N * mean^2) / stddev^2
    double N_x_mean    = n_angles_in_search * global_ccc_mean;
    double N_x_mean_sq = N_x_mean * global_ccc_mean;
    for ( long pixel_counter = 0; pixel_counter < mip_image->real_memory_allocated; pixel_counter++ ) {
        // We need the estimated value of the sum (A) so we have to calculate sum_sq first
        mip_image->real_values[pixel_counter] = (mip_image->real_values[pixel_counter] - global_ccc_mean) / global_ccc_std_dev;
        if ( correlation_pixel_sum_of_squares[pixel_counter] > cistem::float_epsilon ) {
            correlation_pixel_sum_of_squares[pixel_counter] = (correlation_pixel_sum_of_squares[pixel_counter] - (2.0 * global_ccc_mean * correlation_pixel_sum[pixel_counter]) + N_x_mean_sq) / (global_ccc_std_dev * global_ccc_std_dev);
            correlation_pixel_sum[pixel_counter]            = (correlation_pixel_sum[pixel_counter] - N_x_mean) / global_ccc_std_dev;

            // TODO: this could be done in one step, but for now I'm brining it in so that local/gui rescaling happens in the same place and leaving it written as it was there.
            correlation_pixel_sum[pixel_counter] /= n_angles_in_search;
            correlation_pixel_sum_of_squares[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares[pixel_counter] /
                                                                            n_angles_in_search -
                                                                    powf(correlation_pixel_sum[pixel_counter], 2));

            if ( disable_flat_fielding )
                scaled_mip->real_values[pixel_counter] = mip_image->real_values[pixel_counter];
            else
                scaled_mip->real_values[pixel_counter] = (mip_image->real_values[pixel_counter] - correlation_pixel_sum[pixel_counter]) / correlation_pixel_sum_of_squares[pixel_counter];
        }
        else {
            scaled_mip->real_values[pixel_counter] = mip_image->real_values[pixel_counter];
        }
    }

// TODO: This would normally have followed the above where currently we are just calculating the rescaled mip.
#ifdef CISTEM_TEST_FILTERED_MIP
    MyAssertTrue(false, "This block is broken by the new resizing routines.");
#endif

    //     {
    //         {}
    // #ifndef CISTEM_TEST_FILTERED_MIP
    //         // ifdef, we want to modify the avg and stdDev image first
    //         if ( aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] > 0.0f ) {
    //             // Save the variance, not the stdDev
    //             aggregated_results[array_location].collated_mip_data[pixel_counter] = (aggregated_results[array_location].collated_mip_data[pixel_counter] - aggregated_results[array_location].collated_pixel_sums[pixel_counter]) /
    //                                                                                   aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
    //         }
    //         else {
    //             aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = 0.0f;
    //             aggregated_results[array_location].collated_mip_data[pixel_counter]          = 0.0f;
    //         }
    // #endif
    //     } // leave OOB values at -FLTMAX
    // }

    // #ifdef CISTEM_TEST_FILTERED_MIP
    // MyAssertTrue(false, "This block is broken by the new resizing routines.");
    // // We assume the user has set the min pixel radius in pixels to match the expected radius of the particle, which is only true if
    // // a) they are aware of this hack
    // // b) the sample is a single particle (layered sample will have a different radius)
    // float estimated_radius_in_pixels = current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[39].ReturnFloatArgument( );

    // // The factor of 4 (two particle diameters) is in no way optimized.
    // float objective_aperture_resolution = search_pixel_size * estimated_radius_in_pixels * 4.0f;
    // float mask_falloff                  = 7.f;

    // // std::cerr << "Inside test filtered mip" << std::endl;
    // // std::cerr << "Objective aperture resolution: " << objective_aperture_resolution << std::endl;
    // // std::cerr << "Mask falloff: " << mask_falloff << std::endl;
    // // std::cerr << "Pixel size: " << search_pixel_size << std::endl;
    // // std::cerr << "Estimated radius in pixels: " << estimated_radius_in_pixels << std::endl;

    // Image temp_filtered_img;
    // temp_filtered_img.Allocate(temp_image.logical_x_dimension, temp_image.logical_y_dimension, true);
    // temp_filtered_img.ReturnCosineMaskBandpassResolution(search_pixel_size, objective_aperture_resolution, mask_falloff);

    // // Direct at the avg image first
    // for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
    //     temp_filtered_img.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_sums[pixel_counter];
    // }

    // temp_filtered_img.ForwardFFT( );
    // temp_filtered_img.CosineRingMask(-1.0f, objective_aperture_resolution, mask_falloff);
    // temp_filtered_img.BackwardFFT( );

    // // Now filter, subtracting the means
    // // Direct at the avg image first
    // for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
    //     aggregated_results[array_location].collated_mip_data[pixel_counter] -= temp_filtered_img.real_values[pixel_counter];
    //     aggregated_results[array_location].collated_pixel_sums[pixel_counter] = temp_filtered_img.real_values[pixel_counter];
    // }

    // // Direct to the stdDev image
    // for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
    //     temp_filtered_img.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
    // }

    // temp_filtered_img.ForwardFFT( );
    // temp_filtered_img.CosineRingMask(-1.0f, objective_aperture_resolution, mask_falloff);
    // temp_filtered_img.BackwardFFT( );

    // // Now filter the stdDev
    // for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
    //     aggregated_results[array_location].collated_mip_data[pixel_counter]          = (temp_filtered_img.real_values[pixel_counter] > 0.00001) ? aggregated_results[array_location].collated_mip_data[pixel_counter] / temp_filtered_img.real_values[pixel_counter] : 0.0f;
    //     aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = temp_filtered_img.real_values[pixel_counter];
    // }
}