#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#if defined(cisTEM_USING_FastFFT) && defined(ENABLEGPU)
#include "../../include/FastFFT/include/FastFFT.h"
#endif

#include "template_matching_data_sizer.h"

// #define USE_LERP_NOT_FOURIER_RESIZING

class AggregatedTemplateResult {

  public:
    int   image_number;
    int   number_of_received_results;
    float total_number_of_angles_searched;

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
    void AddResult(float* result_array, long array_size, int result_number, int number_of_expected_results);
};

WX_DECLARE_OBJARRAY(AggregatedTemplateResult, ArrayOfAggregatedTemplateResults);
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAggregatedTemplateResults);

class
        MatchTemplateApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results);
    void ProgramSpecificInit( );

    // for master collation

    ArrayOfAggregatedTemplateResults aggregated_results;

    float GetMaxJobWaitTimeInSeconds( ) { return 360.0f; }

  private:
    template <typename StatsType>
    void CalcGlobalCCCScalingFactor(double&     global_ccc_mean,
                                    double&     global_ccc_std_dev,
                                    StatsType*  sum,
                                    StatsType*  sum_of_sqs,
                                    const float n_angles_in_search,
                                    const long  n_values);

    void ResampleHistogramData(long*        histogram_ptr,
                               const double global_ccc_mean,
                               const double global_ccc_std_dev);

    template <typename StatsType>
    void RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(Image*      mip_image,
                                                             Image*      scaled_mip,
                                                             StatsType*  correlation_pixel_sum,
                                                             StatsType*  correlation_pixel_sum_of_squares,
                                                             long*       histogram,
                                                             const float n_angles_in_search);
};

IMPLEMENT_APP(MatchTemplateApp)

// TODO: why is this here?
void MatchTemplateApp::ProgramSpecificInit( ) {
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
    wxString scaled_mip_output_file;

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
    scaled_mip_output_file      = my_input->GetFilenameFromUser("Output Scaled MIP file", "The file for saving the maximum intensity projection image divided by correlation variance", "mip_scaled.mrc", false);
    best_psi_output_file        = my_input->GetFilenameFromUser("Output psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_output_file      = my_input->GetFilenameFromUser("Output theta file", "The file for saving the best psi image", "theta.mrc", false);
    best_phi_output_file        = my_input->GetFilenameFromUser("Output phi file", "The file for saving the best psi image", "phi.mrc", false);
    best_defocus_output_file    = my_input->GetFilenameFromUser("Output defocus file", "The file for saving the best defocus image", "defocus.mrc", false);
    best_pixel_size_output_file = my_input->GetFilenameFromUser("Output pixel size file", "The file for saving the best pixel size image", "pixel_size.mrc", false);
    correlation_avg_output_file = my_input->GetFilenameFromUser("Correlation average value", "The file for saving the average value of all correlation images", "corr_average.mrc", false);
    correlation_std_output_file = my_input->GetFilenameFromUser("Correlation variance output file", "The file for saving the variance of all correlation images", "corr_variance.mrc", false);
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
                                      scaled_mip_output_file.ToUTF8( ).data( ),
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

bool MatchTemplateApp::DoCalculation( ) {

    bool is_rotated_by_90 = false;

    const int   histogram_number_of_points = cistem::match_template::histogram_number_of_points;
    const float histogram_min              = cistem::match_template::histogram_min;
    const float histogram_max              = cistem::match_template::histogram_max;

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_search_images_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename = my_current_job.arguments[1].ReturnStringArgument( );
    float    input_pixel_size              = my_current_job.arguments[2].ReturnFloatArgument( );
    float    voltage_kV                    = my_current_job.arguments[3].ReturnFloatArgument( );
    float    spherical_aberration_mm       = my_current_job.arguments[4].ReturnFloatArgument( );
    float    amplitude_contrast            = my_current_job.arguments[5].ReturnFloatArgument( );
    float    defocus1                      = my_current_job.arguments[6].ReturnFloatArgument( );
    float    defocus2                      = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus_angle                 = my_current_job.arguments[8].ReturnFloatArgument( );
    ;
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

    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );

    float outer_mask_radius;
    float current_psi;
    float psi_step;
    float psi_max;
    float psi_start;
    float histogram_step;

    float expected_threshold;
    float actual_number_of_angles_searched;

    double histogram_min_scaled; // scaled for the x*y scaling which is only applied at the end.
    double histogram_step_scaled; // scaled for the x*y scaling which is only applied at the end.

    long* histogram_data;

    int current_bin;

    float  temp_float;
    float  variance;
    double temp_double;
    double temp_double_array[5];

    int  number_of_rotations;
    long total_correlation_positions;
    long current_correlation_position;
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int current_search_position;
    int current_x;
    int current_y;

    int defocus_i;
    int size_i;

    int i;

    int remove_npix_from_edge = 0;

    EulerSearch     global_euler_search;
    AnglesAndShifts angles;

    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    //
    remove_npix_from_edge = myroundint(particle_radius_angstroms / input_pixel_size);
    //    wxPrintf("Removing %d pixels around the edge.\n", remove_npix_from_edge);

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

    input_image.ReadSlice(&input_search_image_file, 1);

    int   max_padding = 0; // To restrict histogram calculation
    float histogram_padding_trim_rescale; // scale the counts to

    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
    MyAssertTrue(input_reconstruction.IsCubic( ), "Input reconstruction should be cubic");

    TemplateMatchingDataSizer data_sizer(this, input_image, input_reconstruction, input_pixel_size, padding);


#ifdef USE_LERP_NOT_FOURIER_RESIZING
    const bool use_lerp_not_fourier_resampling = true;
#else
    const bool use_lerp_not_fourier_resampling = false;
#endif
    data_sizer.PreProcessInputImage(input_image);
    data_sizer.SetImageAndTemplateSizing(high_resolution_limit_search, use_fast_fft);
    data_sizer.ResizeTemplate_preSearch(input_reconstruction, use_lerp_not_fourier_resampling);
    data_sizer.ResizeImage_preSearch(input_image);


    float wanted_binning_factor = data_sizer.GetSearchPixelSize( ) / data_sizer.GetPixelSize( );
    std::cerr << "Binning factor is " << wanted_binning_factor << std::endl;

    if ( data_sizer.IsRotatedBy90( ) )
        defocus_angle += 90.0f;

    data_sizer.PrintImageSizes( );

    // FIXME this should be changed out for the upper and lower bounds in x and y
    int lx, ly, hx, hy;
    data_sizer.GetValidXYPhysicalIdicies(lx, ly, hx, hy);
    max_padding = std::max(lx, ly);
    wxPrintf("Max padding is %d\n", max_padding);
#ifdef DEBUG
    if ( use_fast_fft )
        SendInfo("Using FastFFT is\n\n");
    else
        SendInfo("Using FastFFT is NO\n\n");
#endif

    if ( padding != 1.0f ) {
        MyDebugAssertFalse(data_sizer.IsResamplingNeeded( ), "Currently, padding of 1.0 is required when resampling.");
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }

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

    padded_reference.SetToConstant(0.0f);
    max_intensity_projection.SetToConstant(-FLT_MAX);
    best_psi.SetToConstant(0.0f);
    best_theta.SetToConstant(0.0f);
    best_phi.SetToConstant(0.0f);
    best_defocus.SetToConstant(0.0f);

    ZeroDoubleArray(correlation_pixel_sum, input_image.real_memory_allocated);
    ZeroDoubleArray(correlation_pixel_sum_of_squares, input_image.real_memory_allocated);

    // setup curve
    // FIXME: the scaled versions are now redundant as the ccgs are scaled to be independent of image size with std ~ 1
    // and the measure stddev is used to rescale the histogram (and mips etc) at the end.
    histogram_step        = (histogram_max - histogram_min) / float(histogram_number_of_points);
    histogram_min_scaled  = histogram_min;
    histogram_step_scaled = histogram_step;

    histogram_data = new long[histogram_number_of_points];

    for ( int counter = 0; counter < histogram_number_of_points; counter++ ) {
        histogram_data[counter] = 0;
    }

    CTF input_ctf;
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, data_sizer.GetSearchPixelSize( ), deg_2_rad(phase_shift));

    // assume cube

    current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);
    projection_filter.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);
    template_reconstruction.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_y_dimension, input_reconstruction.logical_z_dimension, true);
    if ( padding != 1.0f )
        padded_projection.Allocate(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_x_dimension * padding, false);

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

    //psi_step = 5;

    // search grid

    // TODO: when checking the impact of limiting the resolution, it may be worthwile to NOT limit the number of search positions
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

    // for now, I am assuming the MTF has been applied already.
    // work out the filter to just whiten the image..

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    data_sizer.PreProcessResizedInputImage(input_image);

    // count total searches (lazy)

    total_correlation_positions  = 0;
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
            total_correlation_positions++;
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

    total_correlation_positions *= (2 * myroundint(float(defocus_search_range) / float(defocus_step)) + 1);
    total_correlation_positions_per_thread = total_correlation_positions;

    number_of_rotations = 0;

    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }

    ProgressBar* my_progress;

    //Loop over ever search position

    wxPrintf("\n\tFor image id %i\n", image_number_for_gui);
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);

    wxPrintf("Performing Search...\n\n");

    //    wxPrintf("Searching %i - %i of %i total positions\n", first_search_position, last_search_position, global_euler_search.number_of_search_positions);
    //    wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);

    actual_number_of_angles_searched = 0.0;

    wxDateTime overall_start;
    wxDateTime overall_finish;
    overall_start = wxDateTime::Now( );

    // These vars are only needed in the GPU code, but also need to be set out here to compile.
    bool first_gpu_loop = true;
    int  nThreads       = 2;
    int  nGPUs          = 1;
    int  nJobs          = last_search_position - first_search_position + 1;
    if ( use_gpu && max_threads > nJobs ) {
        wxPrintf("\n\tWarning, you request more threads (%d) than there are search positions (%d)\n", max_threads, nJobs);
        max_threads = nJobs;
    }

    int minPos = first_search_position;
    int maxPos = last_search_position;
    int incPos = (nJobs) / (max_threads);

//    wxPrintf("First last and inc %d, %d, %d\n", minPos, maxPos, incPos);
#ifdef ENABLEGPU
    TemplateMatchingCore* GPU;
    DeviceManager         gpuDev;
#endif

    if ( use_gpu ) {
        total_correlation_positions_per_thread = total_correlation_positions / max_threads;

#ifdef ENABLEGPU
        //    checkCudaErrors(cudaGetDeviceCount(&nGPUs));
        GPU = new TemplateMatchingCore[max_threads];
        gpuDev.Init(nGPUs);

//    wxPrintf("Host: %s is running\nnThreads: %d\nnGPUs: %d\n:nSearchPos %d \n",hostNameBuffer,nThreads, nGPUs, maxPos);

//    TemplateMatchingCore GPU(number_of_jobs_per_image_in_gui);
#endif
    }

    if ( is_running_locally == true ) {
        my_progress = new ProgressBar(total_correlation_positions_per_thread);
    }

    //    wxPrintf("Starting job\n");
    for ( size_i = -myroundint(float(pixel_size_search_range) / float(pixel_size_step)); size_i <= myroundint(float(pixel_size_search_range) / float(pixel_size_step)); size_i++ ) {

        //        template_reconstruction.CopyFrom(&input_reconstruction);
        input_reconstruction.ChangePixelSize(&template_reconstruction, (data_sizer.GetSearchPixelSize( ) + float(size_i) * pixel_size_step) / data_sizer.GetSearchPixelSize( ), 0.001f, true);
        //    template_reconstruction.ForwardFFT();
        template_reconstruction.ZeroCentralPixel( );
        template_reconstruction.SwapRealSpaceQuadrants( );

        if ( use_gpu ) {
#ifdef ENABLEGPU
            // FIXME: move this (and the above CPU steps) into a method to prepare the 3d reference.
            // Swapping the fourier space quadrants is a one way operation, so we need a copy in case the user has a loop over pixel size
            // TODO: we could check this and avoid the copy
            Image tmp_vol = template_reconstruction;
            if ( ! tmp_vol.is_fft_centered_in_box ) {
                // FIXME: The extra RealSpace swap could be avoided
                tmp_vol.SwapRealSpaceQuadrants( );
                tmp_vol.BackwardFFT( );
                tmp_vol.SwapFourierSpaceQuadrants(true);
            }
            // We only want to have one copy of the 3d template in texture memory that each thread can then reference.
            // First allocate a shared pointer and construct the GpuImage based on the CPU template
            // TODO: Initially, i had this set to use
            // GpuImage::InitializeBasedOnCpuImage(tmp_vol, false, true); where the memory is instructed not to be pinned.
            // This should be fine now, but .
            std::shared_ptr<GpuImage> template_reconstruction_gpu = std::make_shared<GpuImage>(tmp_vol);
            template_reconstruction_gpu->CopyHostToDeviceTextureComplex3d(tmp_vol);
#pragma omp parallel num_threads(max_threads)
            {
                int tIDX = ReturnThreadNumberOfCurrentThread( );
                gpuDev.SetGpu(tIDX);

                if ( first_gpu_loop ) {

                    int t_first_search_position = first_search_position + (tIDX * incPos);
                    int t_last_search_position  = first_search_position + (incPos - 1) + (tIDX * incPos);

                    if ( tIDX == (max_threads - 1) )
                        t_last_search_position = maxPos;

                    GPU[tIDX].Init(this, template_reconstruction_gpu, input_image, current_projection,
                                   pixel_size_search_range, pixel_size_step, data_sizer.GetSearchPixelSize( ),
                                   defocus_search_range, defocus_step, defocus1, defocus2,
                                   psi_max, psi_start, psi_step,
                                   angles, global_euler_search,
                                   histogram_min_scaled, histogram_step_scaled, histogram_number_of_points,
                                   max_padding, t_first_search_position, t_last_search_position,
                                   my_progress, total_correlation_positions_per_thread, is_running_locally, use_fast_fft);

#ifdef USE_LERP_NOT_FOURIER_RESIZING
                    std::cerr << "\n\nUsing LERP\n\n";
                    GPU[tIDX].use_lerp_for_resizing = true;
                    GPU[tIDX].binning_factor        = wanted_binning_factor;
#endif
                    wxPrintf("%d\n", tIDX);
                    wxPrintf("%d\n", t_first_search_position);
                    wxPrintf("%d\n", t_last_search_position);
                    wxPrintf("Staring TemplateMatchingCore object %d to work on position range %d-%d\n", tIDX, t_first_search_position, t_last_search_position);

                    first_gpu_loop = false;
                }
                else {
                    GPU[tIDX].template_gpu_shared = template_reconstruction_gpu;
                }
            } // end of omp block
#endif
        }
        for ( defocus_i = -myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i <= myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i++ ) {

            // make the projection filter, which will be CTF * whitening filter
            input_ctf.SetDefocus((defocus1 + float(defocus_i) * defocus_step) / data_sizer.GetSearchPixelSize( ), (defocus2 + float(defocus_i) * defocus_step) / data_sizer.GetSearchPixelSize( ), deg_2_rad(defocus_angle));
            //            input_ctf.SetDefocus((defocus1 + 200) / data_sizer.GetSearchPixelSize(), (defocus2 + 200) / data_sizer.GetSearchPixelSize(), deg_2_rad(defocus_angle));
            projection_filter.CalculateCTFImage(input_ctf);
            projection_filter.ApplyCurveFilter(data_sizer.whitening_filter_ptr.get());

            //            projection_filter.QuickAndDirtyWriteSlices("/tmp/projection_filter.mrc",1,projection_filter.logical_z_dimension,true,1.5);
            if ( use_gpu ) {
#ifdef ENABLEGPU
                //            wxPrintf("\n\n\t\tsizeI defI %d %d\n\n\n", size_i, defocus_i);

#pragma omp parallel num_threads(max_threads)
                {
                    int tIDX = ReturnThreadNumberOfCurrentThread( );
                    gpuDev.SetGpu(tIDX);

                    GPU[tIDX].RunInnerLoop(projection_filter, size_i, defocus_i, tIDX, current_correlation_position);

#pragma omp critical
                    {

                        Image mip_buffer   = GPU[tIDX].d_max_intensity_projection.CopyDeviceToNewHost(true, false);
                        Image psi_buffer   = GPU[tIDX].d_best_psi.CopyDeviceToNewHost(true, false);
                        Image phi_buffer   = GPU[tIDX].d_best_phi.CopyDeviceToNewHost(true, false);
                        Image theta_buffer = GPU[tIDX].d_best_theta.CopyDeviceToNewHost(true, false);

                        Image sum   = GPU[tIDX].d_sum3.CopyDeviceToNewHost(true, false);
                        Image sumSq = GPU[tIDX].d_sumSq3.CopyDeviceToNewHost(true, false);

                        // TODO swap max_padding for explicit padding in x/y and limit calcs to that region.
                        pixel_counter = 0;
                        for ( current_y = 0; current_y < max_intensity_projection.logical_y_dimension; current_y++ ) {
                            for ( current_x = 0; current_x < max_intensity_projection.logical_x_dimension; current_x++ ) {
                                // first mip

                                if ( mip_buffer.real_values[pixel_counter] > max_intensity_projection.real_values[pixel_counter] ) {
                                    max_intensity_projection.real_values[pixel_counter] = mip_buffer.real_values[pixel_counter];
                                    best_psi.real_values[pixel_counter]                 = psi_buffer.real_values[pixel_counter];
                                    best_theta.real_values[pixel_counter]               = theta_buffer.real_values[pixel_counter];
                                    best_phi.real_values[pixel_counter]                 = phi_buffer.real_values[pixel_counter];
                                    best_defocus.real_values[pixel_counter]             = float(defocus_i) * defocus_step;
                                    best_pixel_size.real_values[pixel_counter]          = float(size_i) * pixel_size_step;
                                }

                                correlation_pixel_sum[pixel_counter] += (double)sum.real_values[pixel_counter];
                                correlation_pixel_sum_of_squares[pixel_counter] += (double)sumSq.real_values[pixel_counter];

                                pixel_counter++;
                            }

                            pixel_counter += max_intensity_projection.padding_jump_value;
                        }

                        // GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);
                        GPU[tIDX].my_dist.at(0).CopyToHostAndAdd(histogram_data);

                        //                    current_correlation_position += GPU[tIDX].total_number_of_cccs_calculated;
                        actual_number_of_angles_searched += GPU[tIDX].total_number_of_cccs_calculated;

                    } // end of omp critical block
                } // end of parallel block

                continue;

#endif
            }

            for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
                //loop over each rotation angle

                //current_rotation = 0;
                for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

                    angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
                    //                    angles.Init(130.0, 30.0, 199.5, 0.0, 0.0);

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

                    current_projection.MultiplyPixelWise(projection_filter);

                    current_projection.BackwardFFT( );
                    //current_projection.ReplaceOutliersWithMean(6.0f);

                    current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));

                    // We want a variance of 1 in the padded FFT. Scale the small SumOfSquares (which is already divided by n) and then re-divide by N.
                    variance = current_projection.ReturnSumOfSquares( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - powf(current_projection.ReturnAverageOfRealValues( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels, 2);
                    current_projection.DivideByConstant(sqrtf(variance));
                    current_projection.ClipIntoLargerRealSpace2D(&padded_reference);

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
                    pixel_counter = 0;

                    for ( current_y = 0; current_y < max_intensity_projection.logical_y_dimension; current_y++ ) {
                        for ( current_x = 0; current_x < max_intensity_projection.logical_x_dimension; current_x++ ) {
                            // first mip

                            if ( padded_reference.real_values[pixel_counter] > max_intensity_projection.real_values[pixel_counter] ) {
                                max_intensity_projection.real_values[pixel_counter] = padded_reference.real_values[pixel_counter];
                                best_psi.real_values[pixel_counter]                 = current_psi;
                                best_theta.real_values[pixel_counter]               = global_euler_search.list_of_search_parameters[current_search_position][1];
                                best_phi.real_values[pixel_counter]                 = global_euler_search.list_of_search_parameters[current_search_position][0];
                                best_defocus.real_values[pixel_counter]             = float(defocus_i) * defocus_step;
                                best_pixel_size.real_values[pixel_counter]          = float(size_i) * pixel_size_step;
                                //                                if (size_i != 0) wxPrintf("size_i = %i\n", size_i);
                                //                                correlation_pixel_sum[pixel_counter] = variance;
                            }

                            // histogram

                            current_bin = int(double((padded_reference.real_values[pixel_counter]) - histogram_min_scaled) / histogram_step_scaled);
                            //current_bin = int(double((padded_reference.real_values[pixel_counter]) - histogram_min) / histogram_step);

                            if ( current_bin >= 0 && current_bin <= histogram_number_of_points ) {
                                histogram_data[current_bin] += 1;
                            }

                            pixel_counter++;
                        }

                        pixel_counter += padded_reference.padding_jump_value;
                    }

                    //                    correlation_pixel_sum.AddImage(&padded_reference);
                    for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated; pixel_counter++ ) {
                        correlation_pixel_sum[pixel_counter] += padded_reference.real_values[pixel_counter];
                    }
                    padded_reference.SquareRealValues( );
                    //                    correlation_pixel_sum_of_squares.AddImage(&padded_reference);
                    for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated; pixel_counter++ ) {
                        correlation_pixel_sum_of_squares[pixel_counter] += padded_reference.real_values[pixel_counter];
                    }

                    //max_intensity_projection.QuickAndDirtyWriteSlice("/tmp/mip.mrc", 1);

                    current_projection.is_in_real_space = false;
                    padded_reference.is_in_real_space   = true;

                    current_correlation_position++;
                    if ( is_running_locally == true )
                        my_progress->Update(current_correlation_position);

                    if ( is_running_locally == false ) {
                        actual_number_of_angles_searched++;
                        temp_float             = current_correlation_position;
                        JobResult* temp_result = new JobResult;
                        temp_result->SetResult(1, &temp_float);
                        AddJobToResultQueue(temp_result);
                    }
                }
            }
        }
    }

    wxPrintf("\n\n\tTimings: Overall: %s\n", (wxDateTime::Now( ) - overall_start).Format( ));

    // We may have rotated or re-sized the image for performance. To map the results back, it will be
    // easiest to convert the statistical arrays back to images.
    for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
        correlation_pixel_sum_image.real_values[pixel_counter]            = (float)correlation_pixel_sum[pixel_counter];
        correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares[pixel_counter];
    }
    // Remove any unwanted values in the padding area
    correlation_pixel_sum_image.ZeroFFTWPadding( );
    correlation_pixel_sum_of_squares_image.ZeroFFTWPadding( );

    data_sizer.ResizeImage_postSearch(input_image,
                                      max_intensity_projection,
                                      best_psi,
                                      best_phi,
                                      best_theta,
                                      best_defocus,
                                      best_pixel_size,
                                      correlation_pixel_sum_image,
                                      correlation_pixel_sum_of_squares_image);

    // FIXME: we shouldn't need these, just here for convenience on transition
    int final_resize_x = data_sizer.GetImageSizeX( );
    int final_resize_y = data_sizer.GetImageSizeY( );

    if ( is_running_locally ) {
        delete my_progress;

        // Adjust the MIP by the measured mean and stddev of the full search CCC which is an estimate for the moments of the noise distribution of CCCs.
        // FIXME: the histogram will include many non-valid values corresponding to the padding regions.
        Image scaled_mip = max_intensity_projection;
        RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(&max_intensity_projection, &scaled_mip, correlation_pixel_sum_image.real_values, correlation_pixel_sum_of_squares_image.real_values, histogram_data, total_correlation_positions);
        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0;
        double      temp_threshold;
        double      erf_input = 2.0 / (1.0 * (double)final_resize_x * (double)final_resize_y * (double)total_correlation_positions);
#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        cisTEM_erfcinv(erf_input);
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        temp_image.CopyFrom(&max_intensity_projection);
        // temp_image.Resize(final_resize_x, final_resize_y, 1, temp_image.ReturnAverageOfRealValuesOnEdges( ));
        temp_image.QuickAndDirtyWriteSlice(mip_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));

        // max_intensity_projection.Resize(final_resize_x, final_resize_y, 1, max_intensity_projection.ReturnAverageOfRealValuesOnEdges( ));
        scaled_mip.QuickAndDirtyWriteSlice(scaled_mip_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));

        // correlation_pixel_sum_image.Resize(final_resize_x, final_resize_y, 1, correlation_pixel_sum_image.ReturnAverageOfRealValuesOnEdges( ));
        correlation_pixel_sum_image.QuickAndDirtyWriteSlice(correlation_avg_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // correlation_pixel_sum_of_squares_image.Resize(final_resize_x, final_resize_y, 1, correlation_pixel_sum_of_squares_image.ReturnAverageOfRealValuesOnEdges( ));
        correlation_pixel_sum_of_squares_image.QuickAndDirtyWriteSlice(correlation_std_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // best_psi.Resize(final_resize_x, final_resize_y, 1, 0.0f);
        best_psi.QuickAndDirtyWriteSlice(best_psi_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // best_theta.Resize(final_resize_x, final_resize_y, 1, 0.0f);
        best_theta.QuickAndDirtyWriteSlice(best_theta_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // best_phi.Resize(final_resize_x, final_resize_y, 1, 0.0f);
        best_phi.QuickAndDirtyWriteSlice(best_phi_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // best_defocus.Resize(final_resize_x, final_resize_y, 1, 0.0f);
        best_defocus.QuickAndDirtyWriteSlice(best_defocus_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));
        // best_pixel_size.Resize(final_resize_x, final_resize_y, 1, 0.0f);
        best_pixel_size.QuickAndDirtyWriteSlice(best_pixel_size_output_file.ToStdString( ), 1, data_sizer.GetPixelSize( ));

        // write out histogram..

        float           histogram_first_bin_midpoint = histogram_min + (histogram_step / 2.0f); // start position
        NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 4);

        double* expected_survival_histogram = new double[histogram_number_of_points];
        double* survival_histogram          = new double[histogram_number_of_points];
        ZeroDoubleArray(survival_histogram, histogram_number_of_points);

        for ( int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++ ) {
            expected_survival_histogram[line_counter] = (erfc((histogram_first_bin_midpoint + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * float(total_correlation_positions);
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
        long   number_of_result_floats = cistem::match_template::number_of_meta_data_values; // first float is x size, 2nd is y size of images, 3rd is number allocated, 4th  float is number of doubles in the histogram
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

            result[cm_t::image_size_x]                  = max_intensity_projection.logical_x_dimension;
            result[cm_t::image_size_y]                  = max_intensity_projection.logical_y_dimension;
            result[cm_t::image_real_memory_allocated]   = max_intensity_projection.real_memory_allocated;
            result[cm_t::number_of_histogram_bins]      = histogram_number_of_points;
            result[cm_t::number_of_angles_searched]     = actual_number_of_angles_searched;
            result[cm_t::ccc_scalar]                    = 1.0f; // (float)sqrt_input_pixels is redundant, but we need all the results to calculate the scaling from the global CCC moments
            result[cm_t::input_pixel_size]              = data_sizer.GetPixelSize( );
            result[cm_t::number_of_valid_search_pixels] = data_sizer.GetNumberOfValidSearchPixels( );
        }

        result_array_counter = cistem::match_template::number_of_meta_data_values;

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

void MatchTemplateApp::MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results) {

    // do we have this image number already?

    bool need_a_new_result = true;
    int  array_location    = -1;
    long pixel_counter;

    // constexpr values for histogram values.
    using namespace cistem::match_template;

    wxPrintf("Master Handling result for image %i..", result_number);

    for ( int result_counter = 0; result_counter < aggregated_results.GetCount( ); result_counter++ ) {
        if ( aggregated_results[result_counter].image_number == result_number ) {
            aggregated_results[result_counter].AddResult(result_array, array_size, result_number, number_of_expected_results);
            need_a_new_result = false;
            array_location    = result_counter;
            wxPrintf("Found array location for image %i, at %i\n", result_number, array_location);
            break;
        }
    }

    if ( need_a_new_result == true ) // we aren't collecting data for this result yet.. start
    {
        AggregatedTemplateResult result_to_add;
        aggregated_results.Add(result_to_add);
        aggregated_results[aggregated_results.GetCount( ) - 1].image_number = result_number;
        aggregated_results[aggregated_results.GetCount( ) - 1].AddResult(result_array, array_size, result_number, number_of_expected_results);
        array_location = aggregated_results.GetCount( ) - 1;
        wxPrintf("Adding new result to array for image %i, at %i\n", result_number, array_location);
    }

    // did this complete a result?

    if ( aggregated_results[array_location].number_of_received_results == number_of_expected_results ) {
        // TODO send the result back to the GUI, for now hack mode to save the files to the directory..

        wxString directory_for_writing_results = current_job_package.jobs[0].arguments[37].ReturnStringArgument( );

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

        ImageFile input_reconstruction_file;
        input_reconstruction_file.OpenFile(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[1].ReturnStringArgument( ), false);
        // FIXME: this is not correct BROKEN needs to use valid area (add to constants and reference here based on data_sizer)
        int   image_size_x                  = aggregated_results[array_location].collated_data_array[cistem::match_template::image_size_x];
        int   image_size_y                  = aggregated_results[array_location].collated_data_array[cistem::match_template::image_size_y];
        int   image_real_memory_allocated   = aggregated_results[array_location].collated_data_array[cistem::match_template::image_real_memory_allocated];
        float input_pixel_size              = aggregated_results[array_location].collated_data_array[cistem::match_template::input_pixel_size];
        long  number_of_valid_search_pixels = aggregated_results[array_location].collated_data_array[cistem::match_template::number_of_valid_search_pixels];

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
                                                            aggregated_results[array_location].total_number_of_angles_searched);

        // Update the collated mip data which is used downstream for the scaled mip and other calcs
        // Fill the temp_image with data form the collatged mip before passing it on to be rescaled.
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            aggregated_results[array_location].collated_mip_data[pixel_counter] = temp_image.real_values[pixel_counter];
        }

        wxPrintf("Writing result %i\n", aggregated_results[array_location].image_number - 1);
        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[21].ReturnStringArgument( ), 1, false, input_pixel_size);
        temp_image.Deallocate( );

        // psi

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_psi_data[pixel_counter];
        }

        //temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/psi.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);
        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[22].ReturnStringArgument( ), 1, false, input_pixel_size);
        psi_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        //theta

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_theta_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[23].ReturnStringArgument( ), 1, false, input_pixel_size);
        theta_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // phi

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_phi_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[24].ReturnStringArgument( ), 1, false, input_pixel_size);
        phi_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // defocus

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_defocus_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[25].ReturnStringArgument( ), 1, false, input_pixel_size);
        defocus_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // pixel size

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_size_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[26].ReturnStringArgument( ), 1, false, input_pixel_size);
        pixel_size_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        scaled_mip.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[27].ReturnStringArgument( ), 1, false, input_pixel_size);

        // sums

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_sums[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[36].ReturnStringArgument( ), 1, false, input_pixel_size);
        temp_image.Deallocate( );

        // square sums

        temp_image.Allocate(int(image_size_x), int(image_size_y), true);
        for ( pixel_counter = 0; pixel_counter < image_real_memory_allocated; pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[28].ReturnStringArgument( ), 1, false, input_pixel_size);
        temp_image.Deallocate( );

        // histogram

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
            expected_survival_histogram[line_counter] = (erfc((histogram_first_bin_midpoint + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * (number_of_valid_search_pixels * aggregated_results[array_location].total_number_of_angles_searched);
        }

        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0;
        double      temp_threshold   = 0.0;
        double      erf_input        = 2.0 / (1.0 * (double(number_of_valid_search_pixels) * double(aggregated_results[array_location].total_number_of_angles_searched)));
        //        wxPrintf("ox oy total %3.3e %3.3e %3.3e\n", (double)result_array[5] , (double)result_array[6] , (double)aggregated_results[array_location].total_number_of_angles_searched, erf_input);

#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        temp_threshold       = cisTEM_erfcinv(erf_input);
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        //        expected_threshold = sqrtf(2.0f)*cisTEM_erfcinv((2.0f*(1))/(((final_resize_x * final_resize_y * aggregated_results[array_location].total_number_of_angles_searched))));

        histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
        histogram_file.WriteCommentLine("histogram, expected histogram, survival histogram, expected survival histogram");

        if ( use_gpu ) {
            // In the GPU code, I am not histogramming the padding regions which are not valid. Adjust the counts here. Maybe not the best approach. FIXME also the cpu counts.
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

        result_image.Allocate(scaled_mip.logical_x_dimension, scaled_mip.logical_y_dimension, 1);
        result_image.SetToConstant(0.0f);

        input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
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

            // arguments[2] = input_pixel_size
            temp_peak_info.x_pos = current_peak.x * current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[2].ReturnFloatArgument( ); // RETURNING IN ANGSTROMS
            temp_peak_info.y_pos = current_peak.y * current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[2].ReturnFloatArgument( ); // RETURNING IN ANGSTROMS

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
            //        coordinates[0] = current_peak.x * input_pixel_size;
            //        coordinates[1] = current_peak.y * input_pixel_size;
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

                //current_projection.QuickAndDirtyWriteSlice("/tmp/projs.mrc", all_peak_infos.GetCount());
            }
            else {
                SendInfo("WARNING: More than 1000 peaks above threshold were found. Limiting results to 1000 peaks.\n");
                break;
            }
        }

        // save the output image

        result_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[38].ReturnStringArgument( ), 1, true, input_pixel_size);

        // tell the gui that this result is available...

        ArrayOfTemplateMatchFoundPeakInfos blank_changes;
        SendTemplateMatchingResultToSocket(controller_socket, aggregated_results[array_location].image_number, expected_threshold, all_peak_infos, blank_changes);

        // this should be done now.. so delete it

        aggregated_results.RemoveAt(array_location);
        delete[] expected_survival_histogram;
        delete[] survival_histogram;
    }
}

AggregatedTemplateResult::AggregatedTemplateResult( ) {
    image_number                    = -1;
    number_of_received_results      = 0;
    total_number_of_angles_searched = 0.0f;

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

AggregatedTemplateResult::~AggregatedTemplateResult( ) {
    if ( collated_data_array != NULL )
        delete[] collated_data_array;
}

void AggregatedTemplateResult::AddResult(float* result_array, long array_size, int result_number, int number_of_expected_results) {

    int offset = cistem::match_template::number_of_meta_data_values;

    const int histogram_number_of_points = cistem::match_template::histogram_number_of_points;

    int   image_size_x                = result_array[cistem::match_template::image_size_x];
    int   image_size_y                = result_array[cistem::match_template::image_size_y];
    int   image_real_memory_allocated = result_array[cistem::match_template::image_real_memory_allocated];
    float input_pixel_size            = result_array[cistem::match_template::input_pixel_size];

    // FIXME: change to nullptr
    if ( collated_data_array == NULL ) {
        collated_data_array = new float[array_size];
        ZeroFloatArray(collated_data_array, array_size);
        number_of_received_results      = 0;
        total_number_of_angles_searched = 0.0f;

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

    total_number_of_angles_searched += result_array[cistem::match_template::number_of_angles_searched];

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
 * @param n_values 
 */
template <typename StatsType>
void MatchTemplateApp::CalcGlobalCCCScalingFactor(double&     global_ccc_mean,
                                                  double&     global_ccc_std_dev,
                                                  StatsType*  sum,
                                                  StatsType*  sum_of_sqs,
                                                  const float n_angles_in_search,
                                                  const long  n_values) {

    double global_sum            = 0.0;
    double global_sum_of_squares = 0.0;
    // const double total_number_of_angles_searched = double(n_angles_in_search * float(n_values));

    long counted_values = 0;
    for ( long pixel_counter = 0; pixel_counter < n_values; pixel_counter++ ) {
        // Using -float_max to indicate that the value is not valid. When doing NN interpolation.
        if ( sum[pixel_counter] > -std::numeric_limits<float>::max( ) ) {
            global_sum += double(sum[pixel_counter]);
            global_sum_of_squares += double(sum_of_sqs[pixel_counter]);
            counted_values++;
        }
    }
    const double total_number_of_ccs = double(n_angles_in_search * float(counted_values));
    std::cerr << "Counted Values: " << counted_values << " out of " << n_values << " fractions: " << float(n_values) / float(counted_values) << std::endl;

    global_ccc_mean    = global_sum / total_number_of_ccs;
    global_ccc_std_dev = sqrt(global_sum_of_squares / total_number_of_ccs - double(global_ccc_mean * global_ccc_mean));

    return;
}

void MatchTemplateApp::ResampleHistogramData(long*        histogram_ptr,
                                             const double global_ccc_mean,
                                             const double global_ccc_std_dev) {

    // constexpr values for histogram values.
    using namespace cistem::match_template;
    // Sample the existing histogram onto a curve object that we can rescale smoothly.
    Curve histogram_curve;
    for ( int i_hist = 0; i_hist < histogram_number_of_points; ++i_hist ) {
        histogram_curve.AddPoint(histogram_first_bin_midpoint + histogram_step * float(i_hist),
                                 float(histogram_ptr[i_hist]));
    }

    // We expect the curve to be fairly smooth already, so we'll use a small window size for the fitting.
    // I'm not sure if the polynomial order should be 1 or 3.
    histogram_curve.FitSavitzkyGolayToData(5, 3);

    // We accumulated the histogram based on our best guess at the values from the ccg, but we now need to rescale each x-value and
    // interoplate that from the existing histogram.
    double scaled_histogram_midpoint;
    for ( int i_hist = 0; i_hist < histogram_number_of_points; ++i_hist ) {
        scaled_histogram_midpoint = histogram_first_bin_midpoint + histogram_step * double(i_hist);
        // mip values are scaled by (measured_value - global_ccc_mean) / global_ccc_std_dev.
        // To find the corresponding unscaled value in the measured histogram then
        // we need to divide by mip_rescaling_factor.
        histogram_ptr[i_hist] = long(histogram_curve.ReturnSavitzkyGolayInterpolationFromX(global_ccc_mean + (scaled_histogram_midpoint * global_ccc_std_dev)));
    }
}

template <typename StatsType>
void MatchTemplateApp::RescaleMipAndStatisticalArraysByGlobalMeanAndStdDev(Image*      mip_image,
                                                                           Image*      scaled_mip,
                                                                           StatsType*  correlation_pixel_sum,
                                                                           StatsType*  correlation_pixel_sum_of_squares,
                                                                           long*       histogram,
                                                                           const float n_angles_in_search) {

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
        if ( correlation_pixel_sum[pixel_counter] > -std::numeric_limits<float>::max( ) ) {
            mip_image->real_values[pixel_counter]           = (mip_image->real_values[pixel_counter] - global_ccc_mean) / global_ccc_std_dev;
            correlation_pixel_sum_of_squares[pixel_counter] = (correlation_pixel_sum_of_squares[pixel_counter] - (2.0 * global_ccc_mean * correlation_pixel_sum[pixel_counter]) + N_x_mean_sq) / (global_ccc_std_dev * global_ccc_std_dev);
            correlation_pixel_sum[pixel_counter]            = (correlation_pixel_sum[pixel_counter] - N_x_mean) / global_ccc_std_dev;

            // TODO: this could be done in one step, but for now I'm brining it in so that local/gui rescaling happens in the same place and leaving it written as it was there.
            correlation_pixel_sum[pixel_counter] /= n_angles_in_search;
            correlation_pixel_sum_of_squares[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares[pixel_counter] /
                                                                            n_angles_in_search -
                                                                    powf(correlation_pixel_sum[pixel_counter], 2));

            scaled_mip->real_values[pixel_counter] = (mip_image->real_values[pixel_counter] - correlation_pixel_sum[pixel_counter]) / correlation_pixel_sum_of_squares[pixel_counter];
            if ( ! std::isfinite(scaled_mip->real_values[pixel_counter]) ) {
                scaled_mip->real_values[pixel_counter] = -std::numeric_limits<float>::max( );
            }
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
    // float objective_aperture_resolution = input_pixel_size * estimated_radius_in_pixels * 4.0f;
    // float mask_falloff                  = 7.f;

    // // std::cerr << "Inside test filtered mip" << std::endl;
    // // std::cerr << "Objective aperture resolution: " << objective_aperture_resolution << std::endl;
    // // std::cerr << "Mask falloff: " << mask_falloff << std::endl;
    // // std::cerr << "Pixel size: " << input_pixel_size << std::endl;
    // // std::cerr << "Estimated radius in pixels: " << estimated_radius_in_pixels << std::endl;

    // Image temp_filtered_img;
    // temp_filtered_img.Allocate(temp_image.logical_x_dimension, temp_image.logical_y_dimension, true);
    // temp_filtered_img.ReturnCosineMaskBandpassResolution(input_pixel_size, objective_aperture_resolution, mask_falloff);

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