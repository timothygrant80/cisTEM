#include "../../core/core_headers.h"

#include "../../constants/constants.h"

// Values for data that are passed around in the results.
const int number_of_output_images     = 8; //mip, psi, theta, phi, pixel, defocus, sums, sqsums (scaled mip is not sent out)
const int number_of_meta_data_values  = 6; // img_x, img_y, number cccs, histogram values.
const int MAX_ALLOWED_NUMBER_OF_PEAKS = 1000; // An error will be thrown and job aborted if this number of peaks is exceeded in the make template results block

class AggregatedTemplateResult {
  public:
    int   image_number;
    int   number_of_received_results;
    float total_number_of_ccs;

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

// nasty globals to track histogram size

int   histogram_number_of_points = 512;
float histogram_min              = -12.5f;
float histogram_max              = 22.5f;

class
        MatchTemplateApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results);
    void ProgramSpecificInit( );

    // for master collation

    ArrayOfAggregatedTemplateResults aggregated_results;
    bool                             is_rotated_by_90 = false;

    float GetMaxJobWaitTimeInSeconds( ) { return 360.0f; }

  private:
};

class ImageProjectionComparison {
  public:
    Particle*            particle;
    ReconstructedVolume* reference_volume;
    Image*               projection_image;
    //    Image                        *temp_image;
};

// This is the function which will be minimized
float FrealignObjectiveFunction(void* scoring_parameters, float* array_of_values) {
    ImageProjectionComparison* comparison_object = reinterpret_cast<ImageProjectionComparison*>(scoring_parameters);
    comparison_object->particle->temp_parameters = comparison_object->particle->current_parameters;
    comparison_object->particle->UnmapParameters(array_of_values);

    //    comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image, *comparison_object->particle->ctf_image,
    //            comparison_object->particle->alignment_parameters, 0.0, 0.0,
    //            comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true);

    if ( comparison_object->particle->no_ctf_weighting )
        comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
                                                                 *comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
                                                                 comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, false, false, false, false);
    // Case for normal parameter refinement with weighting applied to particle images and 3D reference
    else if ( comparison_object->particle->includes_reference_ssnr_weighting )
        comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
                                                                 *comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
                                                                 comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true, true, false, false);
    // Case for normal parameter refinement with weighting applied only to particle images
    else
        comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
                                                                 *comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
                                                                 comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true, false, true, true);

    //    if (comparison_object->particle->origin_micrograph < 0) comparison_object->particle->origin_micrograph = 0;
    //    comparison_object->particle->origin_micrograph++;
    //    for (int i = 0; i < comparison_object->projection_image->real_memory_allocated; i++) {comparison_object->projection_image->real_values[i] *= fabs(comparison_object->projection_image->real_values[i]);}
    //    comparison_object->projection_image->ForwardFFT();
    //    comparison_object->projection_image->CalculateCrossCorrelationImageWith(comparison_object->particle->particle_image);
    //    comparison_object->projection_image->SwapRealSpaceQuadrants();
    //    comparison_object->projection_image->BackwardFFT();
    //    comparison_object->projection_image->QuickAndDirtyWriteSlice("proj.mrc", comparison_object->particle->origin_micrograph);
    //    comparison_object->projection_image->SwapRealSpaceQuadrants();
    //    comparison_object->particle->particle_image->SwapRealSpaceQuadrants();
    //    comparison_object->particle->particle_image->BackwardFFT();
    //    comparison_object->particle->particle_image->QuickAndDirtyWriteSlice("part.mrc", comparison_object->particle->origin_micrograph);
    //    comparison_object->particle->particle_image->SwapRealSpaceQuadrants();
    //    exit(0);

    //    float score =      - comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
    //              comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit)
    //            - comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float);
    //    wxPrintf("psi, theta, phi, x, y, = %g, %g, %g, %g, %g, score = %g\n",
    //            comparison_object->particle->alignment_parameters.ReturnPsiAngle(),
    //            comparison_object->particle->alignment_parameters.ReturnThetaAngle(),
    //            comparison_object->particle->alignment_parameters.ReturnPhiAngle(),
    //            comparison_object->particle->alignment_parameters.ReturnShiftX(),
    //            comparison_object->particle->alignment_parameters.ReturnShiftY(), score);
    //    return score;
    //    wxPrintf("sigma_noise, mask_volume, penalty = %g %g %g\n", comparison_object->particle->sigma_noise, comparison_object->particle->mask_volume,
    //            comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float));
    return -comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
                                                                                         comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit) -
           comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_parameters);
    // This penalty term assumes a Gaussian x,y distribution that is probably not correct in most cases. It might be better to leave it out.
}

IMPLEMENT_APP(MatchTemplateApp)

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

    float pixel_size              = 1.0f;
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
    pixel_size                  = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
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
    use_gpu_input = my_input->GetYesNoFromUser("Use GPU", "Offload expensive calcs to GPU", "No");
    max_threads   = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#endif

    int   first_search_position           = -1;
    int   last_search_position            = -1;
    int   image_number_for_gui            = 0;
    int   number_of_jobs_per_image_in_gui = 0;
    float min_peak_radius                 = 10.0f;

    wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive
    wxString result_filename       = "/dev/null"; // shouldn't be used in interactive

    delete my_input;

    my_current_job.ManualSetArguments("ttffffffffffifffffbfftttttttttftiiiitttfbi", input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      pixel_size,
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
                                      max_threads);
}

// override the do calculation method which will be what is actually run..

bool MatchTemplateApp::DoCalculation( ) {
    /*
    wxString input_particle_images                 = my_current_job.arguments[0].ReturnStringArgument(); // global
    wxString input_parameter_file                 = my_current_job.arguments[1].ReturnStringArgument(); // not sure
    wxString input_reconstruction                = my_current_job.arguments[2].ReturnStringArgument(); // global
    wxString input_reconstruction_statistics     = my_current_job.arguments[3].ReturnStringArgument(); // global
    bool     use_statistics                        = my_current_job.arguments[4].ReturnBoolArgument();   // global
    wxString ouput_matching_projections         = my_current_job.arguments[5].ReturnStringArgument(); // ignore (always false)
    wxString ouput_parameter_file                = my_current_job.arguments[6].ReturnStringArgument(); // not sure par file
    wxString ouput_shift_file                    = my_current_job.arguments[7].ReturnStringArgument(); // not sure output
    wxString my_symmetry                        = my_current_job.arguments[8].ReturnStringArgument(); // global
    int         first_particle                        = my_current_job.arguments[9].ReturnIntegerArgument(); // local (effectively ignore)
    int         last_particle                        = my_current_job.arguments[10].ReturnIntegerArgument(); // local (effectively ignore)
    float     percent_used                        = my_current_job.arguments[11].ReturnFloatArgument();
    float      pixel_size                            = my_current_job.arguments[12].ReturnFloatArgument(); // local
    float    voltage_kV                            = my_current_job.arguments[13].ReturnFloatArgument(); // local
    float      spherical_aberration_mm            = my_current_job.arguments[14].ReturnFloatArgument(); // local
    float    amplitude_contrast                    = my_current_job.arguments[15].ReturnFloatArgument(); // local
    float     molecular_mass_kDa                    = my_current_job.arguments[16].ReturnFloatArgument(); // global
    float    inner_mask_radius                    = my_current_job.arguments[17].ReturnFloatArgument(); // global
    float    outer_mask_radius                    = my_current_job.arguments[18].ReturnFloatArgument(); // global
    float    low_resolution_limit                = my_current_job.arguments[19].ReturnFloatArgument(); // global
    float    high_resolution_limit                = my_current_job.arguments[20].ReturnFloatArgument(); // global
    float     signed_CC_limit                    = my_current_job.arguments[21].ReturnFloatArgument(); // global
    float     classification_resolution_limit    = my_current_job.arguments[22].ReturnFloatArgument(); // global
    float    particle_radius_angstroms                    = my_current_job.arguments[23].ReturnFloatArgument(); // global
    float     high_resolution_limit_search        = my_current_job.arguments[24].ReturnFloatArgument(); // global
    float     angular_step                        = my_current_job.arguments[25].ReturnFloatArgument(); // global
    int         best_parameters_to_keep            = my_current_job.arguments[26].ReturnIntegerArgument(); // global
    float     max_search_x                        = my_current_job.arguments[27].ReturnFloatArgument(); // global
    float     max_search_y                        = my_current_job.arguments[28].ReturnFloatArgument(); // global
    refine_particle.mask_center_2d_x            = my_current_job.arguments[29].ReturnFloatArgument(); // global
    refine_particle.mask_center_2d_y            = my_current_job.arguments[30].ReturnFloatArgument(); // global
    refine_particle.mask_center_2d_z            = my_current_job.arguments[31].ReturnFloatArgument(); // global
    refine_particle.mask_radius_2d                = my_current_job.arguments[32].ReturnFloatArgument(); // global
    float     defocus_search_range                = my_current_job.arguments[33].ReturnFloatArgument(); // global
    float     defocus_step                        = my_current_job.arguments[34].ReturnFloatArgument(); // global
    float     padding                            = my_current_job.arguments[35].ReturnFloatArgument(); // global
//    float     filter_constant                    = my_current_job.arguments[35].ReturnFloatArgument();
    bool     global_search                        = my_current_job.arguments[36].ReturnBoolArgument(); // global
    bool     local_refinement                    = my_current_job.arguments[37].ReturnBoolArgument(); // global
// Psi, Theta, Phi, ShiftX, ShiftY
    refine_particle.parameter_map[3]            = my_current_job.arguments[38].ReturnBoolArgument(); //global
    refine_particle.parameter_map[2]            = my_current_job.arguments[39].ReturnBoolArgument(); //global
    refine_particle.parameter_map[1]            = my_current_job.arguments[40].ReturnBoolArgument(); // global
    refine_particle.parameter_map[4]            = my_current_job.arguments[41].ReturnBoolArgument(); // global
    refine_particle.parameter_map[5]            = my_current_job.arguments[42].ReturnBoolArgument(); // global
    bool      calculate_matching_projections        = my_current_job.arguments[43].ReturnBoolArgument(); // global - but ignore
    refine_particle.apply_2D_masking            = my_current_job.arguments[44].ReturnBoolArgument(); // global
    bool     ctf_refinement                        = my_current_job.arguments[45].ReturnBoolArgument(); // global
    bool     normalize_particles                = my_current_job.arguments[46].ReturnBoolArgument();
    bool     invert_contrast                    = my_current_job.arguments[47].ReturnBoolArgument(); // global - but ignore.
    bool     exclude_blank_edges                = my_current_job.arguments[48].ReturnBoolArgument();
    bool     normalize_input_3d                    = my_current_job.arguments[49].ReturnBoolArgument();
    bool     threshold_input_3d                    = my_current_job.arguments[50].ReturnBoolArgument();
    bool     local_global_refine                = my_current_job.arguments[51].ReturnBoolArgument();
    int         current_class                        = my_current_job.arguments[52].ReturnIntegerArgument(); // global - but ignore.
    bool     ignore_input_angles                = my_current_job.arguments[53].ReturnBoolArgument(); // during global search, ignore the starting parameters (this helps reduce bias)
*/

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_search_images_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename = my_current_job.arguments[1].ReturnStringArgument( );
    float    pixel_size                    = my_current_job.arguments[2].ReturnFloatArgument( );
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
    int      max_threads                     = my_current_job.arguments[41].ReturnIntegerArgument( );

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
    float actual_number_of_ccs_calculated;

    double histogram_min_scaled; // scaled for the x*y scaling which is only applied at the end.
    double histogram_step_scaled; // scaled for the x*y scaling which is only applied at the end.

    long* histogram_data;

    int current_bin;

    float  temp_float;
    float  variance;
    double temp_double;
    double temp_double_array[5];
    float  factor_score;

    int  number_of_rotations;
    long total_correlation_positions;
    long current_correlation_position;
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int current_search_position;
    int current_x;
    int current_y;

    int factorizable_x;
    int factorizable_y;
    int factor_result_pos;
    int factor_result_neg;

    int defocus_i;
    int size_i;

    int i;

    long   original_input_image_x;
    long   original_input_image_y;
    int    remove_npix_from_edge = 0;
    double sqrt_input_pixels;

    EulerSearch     global_euler_search;
    AnglesAndShifts angles;

    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file;

    Curve whitening_filter;
    Curve number_of_terms;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    //
    remove_npix_from_edge = myroundint(particle_radius_angstroms / pixel_size);
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

    // Resize input image to be factorizable by small numbers
    original_input_image_x = input_image.logical_x_dimension;
    original_input_image_y = input_image.logical_y_dimension;
    factorizable_x         = input_image.logical_x_dimension;
    factorizable_y         = input_image.logical_y_dimension;

    bool      DO_FACTORIZATION                       = true;
    bool      MUST_BE_POWER_OF_TWO                   = false; // Required for half-preicision xforms
    bool      MUST_BE_FACTOR_OF_FOUR                 = true; // May be faster
    const int max_number_primes                      = 6;
    int       primes[max_number_primes]              = {2, 3, 5, 7, 9, 13};
    float     max_reduction_by_fraction_of_reference = 0.000001f; // FIXME the cpu version is crashing when the image is reduced, but not the GPU
    float     max_increas_by_fraction_of_image       = 0.1f;
    int       max_padding                            = 0; // To restrict histogram calculation
    float     histogram_padding_trim_rescale; // scale the counts to

    // for 5760 this will return
    // 5832 2     2     2     3     3     3     3     3     3 - this is ~ 10% faster than the previous solution BUT
    if ( DO_FACTORIZATION ) {
        for ( i = 0; i < max_number_primes; i++ ) {

            factor_result_neg = ReturnClosestFactorizedLower(original_input_image_x, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
            factor_result_pos = ReturnClosestFactorizedUpper(original_input_image_x, primes[i], true, MUST_BE_FACTOR_OF_FOUR);

            if ( (float)(original_input_image_x - factor_result_neg) < (float)input_reconstruction_file.ReturnXSize( ) * max_reduction_by_fraction_of_reference ) {
                factorizable_x = factor_result_neg;
                break;
            }
            if ( (float)(-original_input_image_x + factor_result_pos) < (float)input_image.logical_x_dimension * max_increas_by_fraction_of_image ) {
                factorizable_x = factor_result_pos;
                break;
            }
        }
        factor_score = FLT_MAX;
        for ( i = 0; i < max_number_primes; i++ ) {

            factor_result_neg = ReturnClosestFactorizedLower(original_input_image_y, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
            factor_result_pos = ReturnClosestFactorizedUpper(original_input_image_y, primes[i], true, MUST_BE_FACTOR_OF_FOUR);

            if ( (float)(original_input_image_y - factor_result_neg) < (float)input_reconstruction_file.ReturnYSize( ) * max_reduction_by_fraction_of_reference ) {
                factorizable_y = factor_result_neg;
                break;
            }
            if ( (float)(-original_input_image_y + factor_result_pos) < (float)input_image.logical_y_dimension * max_increas_by_fraction_of_image ) {
                factorizable_y = factor_result_pos;
                break;
            }
        }
        if ( factorizable_x - original_input_image_x > max_padding )
            max_padding = factorizable_x - original_input_image_x;
        if ( factorizable_y - original_input_image_y > max_padding )
            max_padding = factorizable_y - original_input_image_y;

        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("old x, y = %i %i\n  new x, y = %i %i\n", input_image.logical_x_dimension, input_image.logical_y_dimension, factorizable_x, factorizable_y);
        }

        input_image.Resize(factorizable_x, factorizable_y, 1, input_image.ReturnAverageOfRealValuesOnEdges( ));

        input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
        if ( padding != 1.0f ) {
            input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
        }
#undef ROTATEFORSPEED
#ifdef ROTATEFORSPEED
        if ( ! is_power_of_two(factorizable_x) && is_power_of_two(factorizable_y) ) {
            // The speedup in the FFT for better factorization is also dependent on the dimension. The full transform (in cufft anyway) is faster if the best dimension is on X.
            // TODO figure out how to check the case where there is no factor of two, but one dimension is still faster. Probably getting around to writing an explicit planning tool would be useful.
            if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
                wxPrintf("Rotating the search image for speed\n");
            }
            is_rotated_by_90 = true;
            input_image.RotateInPlaceAboutZBy90Degrees(true);
            // bool preserve_origin = true;
            // input_reconstruction.RotateInPlaceAboutZBy90Degrees(true, preserve_origin);
            // The amplitude spectrum is also rotated
            defocus_angle += 90.0f;
        }
        else {
            if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
                wxPrintf("Not rotating the search image for speed even though it is enabled\n");
            }
            is_rotated_by_90 = false;
        }
#endif
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

    sqrt_input_pixels = sqrt((double)(input_image.logical_x_dimension * input_image.logical_y_dimension));

    // setup curve
    histogram_step        = (histogram_max - histogram_min) / float(histogram_number_of_points);
    histogram_min_scaled  = histogram_min / sqrt_input_pixels;
    histogram_step_scaled = histogram_step / sqrt_input_pixels;

    histogram_data = new long[histogram_number_of_points];

    for ( int counter = 0; counter < histogram_number_of_points; counter++ ) {
        histogram_data[counter] = 0;
    }

    CTF input_ctf;
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));

    // assume cube

    current_projection.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);
    projection_filter.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);
    template_reconstruction.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_y_dimension, input_reconstruction.logical_z_dimension, true);
    if ( padding != 1.0f )
        padded_projection.Allocate(input_reconstruction_file.ReturnXSize( ) * padding, input_reconstruction_file.ReturnXSize( ) * padding, false);

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
        psi_step = rad_2_deg(pixel_size / mask_radius_search);
        psi_step = 360.0 / int(360.0 / psi_step + 0.5);
    }
    else {
        psi_step = in_plane_angular_step;
    }

    //psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
    psi_start = 0.0f;
    psi_max   = 360.0f;

    //psi_step = 5;

    //wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);

    // search grid

    global_euler_search.InitGrid(my_symmetry, angular_step, 20.0f, 80.0f, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }
    global_euler_search.theta_max = 100.0f;
    global_euler_search.phi_max   = 15.0f;
    global_euler_search.for_mt    = true;
    global_euler_search.CalculateGridSearchPositions(false);

    // for now, I am assuming the MTF has been applied already.
    // work out the filter to just whiten the image..

    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    // remove outliers
    // This won't work for movie frames (13.0 is used in unblur) TODO use poisson stats
    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );
    input_image.SwapRealSpaceQuadrants( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    //whitening_filter.WriteToFile("/tmp/filter.txt");
    input_image.ApplyCurveFilter(&whitening_filter);
    input_image.ZeroCentralPixel( );
    input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));
    //input_image.QuickAndDirtyWriteSlice("/tmp/white.mrc", 1);
    //exit(-1);

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

    actual_number_of_ccs_calculated = 0.0;

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
        input_reconstruction.ChangePixelSize(&template_reconstruction, (pixel_size + float(size_i) * pixel_size_step) / pixel_size, 0.001f, true);
        //    template_reconstruction.ForwardFFT();
        template_reconstruction.ZeroCentralPixel( );
        template_reconstruction.SwapRealSpaceQuadrants( );

        //        wxPrintf("First search last search position %d/ %d\n",first_search_position, last_search_position);

        if ( use_gpu ) {
#ifdef ENABLEGPU

#pragma omp parallel num_threads(max_threads)
            {
                int tIDX = ReturnThreadNumberOfCurrentThread( );
                gpuDev.SetGpu(tIDX);

                if ( first_gpu_loop ) {

                    int t_first_search_position = first_search_position + (tIDX * incPos);
                    int t_last_search_position  = first_search_position + (incPos - 1) + (tIDX * incPos);

                    if ( tIDX == (max_threads - 1) )
                        t_last_search_position = maxPos;

                    GPU[tIDX].Init(this, template_reconstruction, input_image, current_projection,
                                   pixel_size_search_range, pixel_size_step, pixel_size,
                                   defocus_search_range, defocus_step, defocus1, defocus2,
                                   psi_max, psi_start, psi_step,
                                   angles, global_euler_search,
                                   histogram_min_scaled, histogram_step_scaled, histogram_number_of_points,
                                   max_padding, t_first_search_position, t_last_search_position,
                                   my_progress, total_correlation_positions_per_thread, is_running_locally);

                    wxPrintf("%d\n", tIDX);
                    wxPrintf("%d\n", t_first_search_position);
                    wxPrintf("%d\n", t_last_search_position);
                    wxPrintf("Staring TemplateMatchingCore object %d to work on position range %d-%d\n", tIDX, t_first_search_position, t_last_search_position);

                    first_gpu_loop = false;
                }
                else {
                    GPU[tIDX].template_reconstruction.CopyFrom(&template_reconstruction);
                }
            } // end of omp block
#endif
        }
        for ( defocus_i = -myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i <= myroundint(float(defocus_search_range) / float(defocus_step)); defocus_i++ ) {

            // make the projection filter, which will be CTF * whitening filter
            input_ctf.SetDefocus((defocus1 + float(defocus_i) * defocus_step) / pixel_size, (defocus2 + float(defocus_i) * defocus_step) / pixel_size, deg_2_rad(defocus_angle));
            //            input_ctf.SetDefocus((defocus1 + 200) / pixel_size, (defocus2 + 200) / pixel_size, deg_2_rad(defocus_angle));
            projection_filter.CalculateCTFImage(input_ctf);
            projection_filter.ApplyCurveFilter(&whitening_filter);

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

                        Image mip_buffer;
                        mip_buffer.CopyFrom(&max_intensity_projection);
                        Image psi_buffer;
                        psi_buffer.CopyFrom(&max_intensity_projection);
                        Image phi_buffer;
                        phi_buffer.CopyFrom(&max_intensity_projection);
                        Image theta_buffer;
                        theta_buffer.CopyFrom(&max_intensity_projection);

                        GPU[tIDX].d_max_intensity_projection.CopyDeviceToHost(mip_buffer, true, false);
                        GPU[tIDX].d_best_psi.CopyDeviceToHost(psi_buffer, true, false);
                        GPU[tIDX].d_best_phi.CopyDeviceToHost(phi_buffer, true, false);
                        GPU[tIDX].d_best_theta.CopyDeviceToHost(theta_buffer, true, false);

                        //                    mip_buffer.QuickAndDirtyWriteSlice("/tmp/tmpMipBuffer.mrc",1,1);
                        // TODO should prob aggregate these across all workers
                        // TODO add a copySum method that allocates a pinned buffer, copies there then sumes into the wanted image.
                        Image sum;
                        Image sumSq;

                        sum.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
                        sumSq.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);

                        sum.SetToConstant(0.0f);
                        sumSq.SetToConstant(0.0f);

                        GPU[tIDX].d_sum3.CopyDeviceToHost(sum, true, false);
                        GPU[tIDX].d_sumSq3.CopyDeviceToHost(sumSq, true, false);

                        GPU[tIDX].d_max_intensity_projection.Wait( );

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

                        GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);

                        //                    current_correlation_position += GPU[tIDX].total_number_of_cccs_calculated;
                        actual_number_of_ccs_calculated += GPU[tIDX].total_number_of_cccs_calculated;

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
                    //                    current_projection.QuickAndDirtyWriteSlice("proj.mrc", 1);
                    //if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_nofilter.mrc", 1);

                    current_projection.MultiplyPixelWise(projection_filter);

                    //if (first_search_position == 0) projection_filter.QuickAndDirtyWriteSlice("/tmp/projection_filter.mrc", 1);
                    //if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_afterfilter.mrc", 1);

                    //current_projection.ZeroCentralPixel();
                    //current_projection.DivideByConstant(sqrt(current_projection.ReturnSumOfSquares()));
                    current_projection.BackwardFFT( );
                    //current_projection.ReplaceOutliersWithMean(6.0f);

                    // find the pixel with the largest absolute density, and shift it to the centre

                    /*    pixel_counter = 0;
                    int best_x;
                    int best_y;
                    float max_value = -FLT_MAX;

                    for ( int y = 0; y < current_projection.logical_y_dimension; y ++ )
                    {
                        for ( int x = 0; x < current_projection.logical_x_dimension; x ++ )
                        {
                            if (fabsf(current_projection.real_values[pixel_counter]) > max_value)
                            {
                                max_value = fabsf(current_projection.real_values[pixel_counter]);
                                best_x = x - current_projection.physical_address_of_box_center_x;
                                best_y = y - current_projection.physical_address_of_box_center_y;;
                            }
                            pixel_counter++;
                        }
                        pixel_counter += current_projection.padding_jump_value;
                    }

                    current_projection.RealSpaceIntegerShift(best_x, best_y, 0);
    */
                    ///

                    current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));

                    //                    variance = current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels;
                    //                    current_projection.DivideByConstant(sqrtf(variance));
                    //                    variance = current_projection.ReturnSumOfSquares();
                    variance = current_projection.ReturnSumOfSquares( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - powf(current_projection.ReturnAverageOfRealValues( ) * current_projection.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels, 2);
                    current_projection.DivideByConstant(sqrtf(variance));
                    current_projection.ClipIntoLargerRealSpace2D(&padded_reference);

                    padded_reference.ForwardFFT( );
                    // Zeroing the central pixel is probably not doing anything useful...
                    padded_reference.ZeroCentralPixel( );
                    //                    padded_reference.DivideByConstant(sqrtf(variance));

                    //if (first_search_position == 0)  padded_reference.QuickAndDirtyWriteSlice("/tmp/proj.mrc", 1);

#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(padded_reference.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(input_image.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter++ ) {
                        padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * input_image.complex_values[pixel_counter];
                    }
#endif

                    padded_reference.BackwardFFT( );
                    //                    padded_reference.QuickAndDirtyWriteSlice("cc.mrc", 1);
                    //                    exit(0);

                    //                    for (pixel_counter = 0; pixel_counter <  padded_reference.real_memory_allocated; pixel_counter++)
                    //                    {
                    //                        temp_float = padded_reference.real_values[pixel_counter] / variance;
                    //                        padded_reference.real_values[pixel_counter] = temp_float * padded_reference.real_values[pixel_counter] - powf(temp_float, 2) * variance;
                    ////                        if (pixel_counter == 1000) wxPrintf("l, value = %g %g\n", temp_float, padded_reference.real_values[pixel_counter]);
                    ////                        padded_reference.real_values[pixel_counter] *= powf(float(padded_reference.number_of_real_space_pixels), 2);
                    ////                        padded_reference.real_values[pixel_counter] = temp_float;
                    //                    }

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
                        actual_number_of_ccs_calculated++;
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

    for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
        correlation_pixel_sum_image.real_values[pixel_counter]            = (float)correlation_pixel_sum[pixel_counter];
        correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares[pixel_counter];
    }

    if ( is_rotated_by_90 ) {
        // swap back all the images prior to re-sizing
        input_image.BackwardFFT( );
        input_image.RotateInPlaceAboutZBy90Degrees(false);
        max_intensity_projection.RotateInPlaceAboutZBy90Degrees(false);

        best_psi.RotateInPlaceAboutZBy90Degrees(false);
        // If the template is also rotated, then this additional accounting is not needed.
        // To account for the pre-rotation, psi needs to have 90 added to it.
        best_psi.AddConstant(90.0f);
        // We also want the angles to remain in (0,360] so loop over and clamp
        for ( int idx = 0; idx < best_psi.real_memory_allocated; idx++ ) {
            best_psi.real_values[idx] = clamp_angular_range_0_to_2pi(best_psi.real_values[idx], true);
        }
        best_theta.RotateInPlaceAboutZBy90Degrees(false);
        best_phi.RotateInPlaceAboutZBy90Degrees(false);
        best_defocus.RotateInPlaceAboutZBy90Degrees(false);
        best_pixel_size.RotateInPlaceAboutZBy90Degrees(false);

        correlation_pixel_sum_image.RotateInPlaceAboutZBy90Degrees(false);
        correlation_pixel_sum_of_squares_image.RotateInPlaceAboutZBy90Degrees(false);

        // This is ineffecient, but a quick way to ensure consistent results.
        delete[] correlation_pixel_sum;
        delete[] correlation_pixel_sum_of_squares;
        // Now we have the rotated values which may also be a different total amount of memory
        correlation_pixel_sum            = new double[input_image.real_memory_allocated];
        correlation_pixel_sum_of_squares = new double[input_image.real_memory_allocated];
        ZeroDoubleArray(correlation_pixel_sum, input_image.real_memory_allocated);
        ZeroDoubleArray(correlation_pixel_sum_of_squares, input_image.real_memory_allocated);
        for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
            correlation_pixel_sum[pixel_counter]            = (double)correlation_pixel_sum_image.real_values[pixel_counter];
            correlation_pixel_sum_of_squares[pixel_counter] = (double)correlation_pixel_sum_of_squares_image.real_values[pixel_counter];
        }
    }

    if ( is_running_locally == true ) {
        delete my_progress;

        // scale images..

        for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {

            //            correlation_pixel_sum.real_values[pixel_counter] /= float(total_correlation_positions);
            //            correlation_pixel_sum_of_squares.real_values[pixel_counter] = correlation_pixel_sum_of_squares.real_values[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum.real_values[pixel_counter], 2);
            //            if (correlation_pixel_sum_of_squares.real_values[pixel_counter] > 0.0f)
            //            {
            //                correlation_pixel_sum_of_squares.real_values[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares.real_values[pixel_counter]) * sqrtf(correlation_pixel_sum.logical_x_dimension * correlation_pixel_sum.logical_y_dimension);
            //            }
            //            else correlation_pixel_sum_of_squares.real_values[pixel_counter] = 0.0f;
            correlation_pixel_sum[pixel_counter] /= float(total_correlation_positions);
            correlation_pixel_sum_of_squares[pixel_counter] = correlation_pixel_sum_of_squares[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum[pixel_counter], 2);
            if ( correlation_pixel_sum_of_squares[pixel_counter] > 0.0f ) {
                correlation_pixel_sum_of_squares[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares[pixel_counter]) * (float)sqrt_input_pixels;
            }
            else
                correlation_pixel_sum_of_squares[pixel_counter] = 0.0f;
            correlation_pixel_sum[pixel_counter] *= (float)sqrt_input_pixels;
        }

        max_intensity_projection.MultiplyByConstant((float)sqrt_input_pixels);
        //        correlation_pixel_sum.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
        //        correlation_pixel_sum_of_squares.MultiplyByConstant(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension);

        // we need to quadrant swap the images, also shift them, with an extra pixel shift.  This is because I take the conjugate of the input image, not the reference..

        //        max_intensity_projection.InvertPixelOrder();
        //        max_intensity_projection.SwapRealSpaceQuadrants();

        //        best_psi.InvertPixelOrder();
        //        best_psi.SwapRealSpaceQuadrants();

        //        best_theta.InvertPixelOrder();
        //        best_theta.SwapRealSpaceQuadrants();

        //        best_phi.InvertPixelOrder();
        //        best_phi.SwapRealSpaceQuadrants();

        //        best_defocus.InvertPixelOrder();
        //        best_defocus.SwapRealSpaceQuadrants();

        //        correlation_pixel_sum.InvertPixelOrder();
        //        correlation_pixel_sum.SwapRealSpaceQuadrants();

        //        correlation_pixel_sum_of_squares.InvertPixelOrder();
        //        correlation_pixel_sum_of_squares.SwapRealSpaceQuadrants();

        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0;
        double      temp_threshold;
        double      erf_input = 2.0 / (1.0 * (double)original_input_image_x * (double)original_input_image_y * (double)total_correlation_positions);
#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        cisTEM_erfcinv(erf_input);
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        //        expected_threshold = sqrtf(2.0f)*cisTEM_erfcinv((2.0f*(1))/((original_input_image_x * original_input_image_y * double(total_correlation_positions))));

        // write out images..

        //        wxPrintf("\nPeak at %g, %g : %g\n", max_intensity_projection.FindPeakWithIntegerCoordinates().x, max_intensity_projection.FindPeakWithIntegerCoordinates().y, max_intensity_projection.FindPeakWithIntegerCoordinates().value);
        //        wxPrintf("Sigma = %g, ratio = %g\n", sqrtf(max_intensity_projection.ReturnVarianceOfRealValues()), max_intensity_projection.FindPeakWithIntegerCoordinates().value / sqrtf(max_intensity_projection.ReturnVarianceOfRealValues()));

        temp_image.CopyFrom(&max_intensity_projection);
        temp_image.Resize(original_input_image_x, original_input_image_y, 1, temp_image.ReturnAverageOfRealValuesOnEdges( ));
        temp_image.QuickAndDirtyWriteSlice(mip_output_file.ToStdString( ), 1, pixel_size);
        //        max_intensity_projection.SubtractImage(&correlation_pixel_sum);
        for ( pixel_counter = 0; pixel_counter < input_image.real_memory_allocated; pixel_counter++ ) {
            max_intensity_projection.real_values[pixel_counter] -= correlation_pixel_sum[pixel_counter];
            if ( correlation_pixel_sum_of_squares[pixel_counter] > 0.0f ) {
                max_intensity_projection.real_values[pixel_counter] /= correlation_pixel_sum_of_squares[pixel_counter];
            }
            else
                max_intensity_projection.real_values[pixel_counter] = 0.0f;
            correlation_pixel_sum_image.real_values[pixel_counter]            = correlation_pixel_sum[pixel_counter];
            correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = correlation_pixel_sum_of_squares[pixel_counter];
        }
        //        max_intensity_projection.DividePixelWise(correlation_pixel_sum_of_squares);
        max_intensity_projection.Resize(original_input_image_x, original_input_image_y, 1, max_intensity_projection.ReturnAverageOfRealValuesOnEdges( ));
        max_intensity_projection.QuickAndDirtyWriteSlice(scaled_mip_output_file.ToStdString( ), 1, pixel_size);

        correlation_pixel_sum_image.Resize(original_input_image_x, original_input_image_y, 1, correlation_pixel_sum_image.ReturnAverageOfRealValuesOnEdges( ));
        correlation_pixel_sum_image.QuickAndDirtyWriteSlice(correlation_avg_output_file.ToStdString( ), 1, pixel_size);
        correlation_pixel_sum_of_squares_image.Resize(original_input_image_x, original_input_image_y, 1, correlation_pixel_sum_of_squares_image.ReturnAverageOfRealValuesOnEdges( ));
        correlation_pixel_sum_of_squares_image.QuickAndDirtyWriteSlice(correlation_std_output_file.ToStdString( ), 1, pixel_size);
        best_psi.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        best_psi.QuickAndDirtyWriteSlice(best_psi_output_file.ToStdString( ), 1, pixel_size);
        best_theta.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        best_theta.QuickAndDirtyWriteSlice(best_theta_output_file.ToStdString( ), 1, pixel_size);
        best_phi.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        best_phi.QuickAndDirtyWriteSlice(best_phi_output_file.ToStdString( ), 1, pixel_size);
        best_defocus.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        best_defocus.QuickAndDirtyWriteSlice(best_defocus_output_file.ToStdString( ), 1, pixel_size);
        best_pixel_size.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        best_pixel_size.QuickAndDirtyWriteSlice(best_pixel_size_output_file.ToStdString( ), 1, pixel_size);

        // write out histogram..

        temp_float = histogram_min + (histogram_step / 2.0f); // start position
        NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 4);

        double* expected_survival_histogram = new double[histogram_number_of_points];
        double* survival_histogram          = new double[histogram_number_of_points];
        ZeroDoubleArray(survival_histogram, histogram_number_of_points);

        for ( int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++ ) {
            expected_survival_histogram[line_counter] = (erfc((temp_float + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * ((float)(sqrt_input_pixels * sqrt_input_pixels) * float(total_correlation_positions));
        }

        survival_histogram[histogram_number_of_points - 1] = histogram_data[histogram_number_of_points - 1];

        for ( int line_counter = histogram_number_of_points - 2; line_counter >= 0; line_counter-- ) {
            survival_histogram[line_counter] = survival_histogram[line_counter + 1] + histogram_data[line_counter];
        }

        histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
        histogram_file.WriteCommentLine("SNR, histogram, survival histogram, random survival histogram");

        for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
            temp_double_array[0] = temp_float + histogram_step * float(line_counter);
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
        long   number_of_result_floats = number_of_meta_data_values; // first float is x size, 2nd is y size of images, 3rd is number allocated, 4th  float is number of doubles in the histogram
        long   pixel_counter;
        float* pointer_to_histogram_data;

        pointer_to_histogram_data = (float*)histogram_data;

        //        max_intensity_projection.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        correlation_pixel_sum_image.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        correlation_pixel_sum_of_squares_image.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        best_psi.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        best_theta.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        best_phi.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        best_defocus.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);
        //        best_pixel_size.Resize(original_input_image_x, original_input_image_y, 1, 0.0f);

        // If the padded image is large, we want to resize, then trim to valid area, otherwise we want to trim to valid area and then resize.
        // Default to the case where the padding increases the image size. A call to resize a same size image only cost the function call time.
        int trim_x = original_input_image_x - remove_npix_from_edge;
        int trim_y = original_input_image_y - remove_npix_from_edge;

        float central_average;
        float central_region = 0.35 * (max_intensity_projection.logical_x_dimension + max_intensity_projection.logical_y_dimension - 2 * remove_npix_from_edge);

        if ( original_input_image_x > max_intensity_projection.logical_x_dimension ) {
            trim_x = max_intensity_projection.logical_x_dimension - remove_npix_from_edge;
        }
        if ( original_input_image_y > max_intensity_projection.logical_y_dimension ) {
            trim_y = max_intensity_projection.logical_y_dimension - remove_npix_from_edge;
        }

        // mip
        central_average = max_intensity_projection.ReturnAverageOfRealValues(central_region, false);
        max_intensity_projection.Resize(trim_x, trim_y, 1, central_average);

        max_intensity_projection.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        //sum
        central_average = correlation_pixel_sum_image.ReturnAverageOfRealValues(central_region, false);
        correlation_pixel_sum_image.Resize(trim_x, trim_y, 1, central_average);
        correlation_pixel_sum_image.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // sq sum
        central_average = correlation_pixel_sum_of_squares_image.ReturnAverageOfRealValues(central_region, false);
        correlation_pixel_sum_of_squares_image.Resize(trim_x, trim_y, 1, central_average);
        correlation_pixel_sum_of_squares_image.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // psi
        central_average = best_psi.ReturnAverageOfRealValues(central_region, false);
        best_psi.Resize(trim_x, trim_y, 1, central_average);
        best_psi.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // theta
        central_average = best_theta.ReturnAverageOfRealValues(central_region, false);
        best_theta.Resize(trim_x, trim_y, 1, central_average);
        best_theta.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // phi
        central_average = best_phi.ReturnAverageOfRealValues(central_region, false);
        best_phi.Resize(trim_x, trim_y, 1, central_average);
        best_phi.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // pixel
        central_average = best_pixel_size.ReturnAverageOfRealValues(central_region, false);
        best_pixel_size.Resize(trim_x, trim_y, 1, central_average);
        best_pixel_size.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // defocus
        central_average = best_defocus.ReturnAverageOfRealValues(central_region, false);
        best_defocus.Resize(trim_x, trim_y, 1, central_average);
        best_defocus.Resize(original_input_image_x, original_input_image_y, 1, central_average);

        // Make sure there is enough space allocated for all results
        number_of_result_floats += max_intensity_projection.real_memory_allocated * number_of_output_images;
        number_of_result_floats += histogram_number_of_points * sizeof(long) / sizeof(float); // histogram are longs

        float* result = new float[number_of_result_floats];
        // Not zero floating this array since all additions are assignments. This can help to expose any indexing errors.

        result[0] = max_intensity_projection.logical_x_dimension;
        result[1] = max_intensity_projection.logical_y_dimension;
        result[2] = max_intensity_projection.real_memory_allocated;
        result[3] = histogram_number_of_points;
        result[4] = actual_number_of_ccs_calculated;
        result[5] = (float)sqrt_input_pixels;
        //        result[5] = original_input_image_x;
        //        result[6] = original_input_image_y;

        result_array_counter = number_of_meta_data_values;

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

    if ( aggregated_results[array_location].number_of_received_results == number_of_expected_results ) // we should be done for this image
    {
        // TODO send the result back to the GUI, for now hack mode to save the files to the directory..

        wxString directory_for_writing_results = current_job_package.jobs[0].arguments[37].ReturnStringArgument( );

        //        wxPrintf("temp x, y, n, resize x, y = %i %i %i %i %i \n", int(aggregated_results[array_location].collated_data_array[0]), \
//            int(aggregated_results[array_location].collated_data_array[1]), int(result_array[2]), int(result_array[5]), int(result_array[6]));

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

        double sqrt_input_pixels = aggregated_results[array_location].collated_data_array[5];
        bool   use_gpu           = current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[40].ReturnBoolArgument( );

        ImageFile input_reconstruction_file;
        input_reconstruction_file.OpenFile(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[1].ReturnStringArgument( ), false);

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);

        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_mip_data[pixel_counter] * sqrt_input_pixels;
        }

        wxPrintf("Writing result %i\n", aggregated_results[array_location].image_number - 1);
        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[21].ReturnStringArgument( ), 1);
        temp_image.Deallocate( );

        // psi

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_psi_data[pixel_counter];
        }

        //temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/psi.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);
        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[22].ReturnStringArgument( ), 1);
        psi_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        //theta

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_theta_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[23].ReturnStringArgument( ), 1);
        theta_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // phi

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_phi_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[24].ReturnStringArgument( ), 1);
        phi_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // defocus

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_defocus_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[25].ReturnStringArgument( ), 1);
        defocus_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // pixel size

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_size_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[26].ReturnStringArgument( ), 1);
        pixel_size_image.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // do the scaling...

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            aggregated_results[array_location].collated_pixel_sums[pixel_counter] /= aggregated_results[array_location].total_number_of_ccs;
            aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = sqrtf(aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] /
                                                                                                         aggregated_results[array_location].total_number_of_ccs -
                                                                                                 powf(aggregated_results[array_location].collated_pixel_sums[pixel_counter], 2));
            if ( aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] > 0.0f ) {

                // Save the variance, not the stdDev
                //                aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = sqrtf(aggregated_results[array_location].collated_pixel_square_sums[pixel_counter]);
                aggregated_results[array_location].collated_mip_data[pixel_counter] = (aggregated_results[array_location].collated_mip_data[pixel_counter] - aggregated_results[array_location].collated_pixel_sums[pixel_counter]) /
                                                                                      aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
            }
            else {
                aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = 0.0f;
                aggregated_results[array_location].collated_mip_data[pixel_counter]          = 0.0f;
            }
        }

        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_mip_data[pixel_counter];
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[27].ReturnStringArgument( ), 1);
        scaled_mip.CopyFrom(&temp_image);
        temp_image.Deallocate( );

        // sums

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_sums[pixel_counter] * sqrt_input_pixels;
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[28].ReturnStringArgument( ), 1);
        temp_image.Deallocate( );

        // square sums

        temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);
        for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
            temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] * sqrt_input_pixels;
        }

        temp_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[36].ReturnStringArgument( ), 1);
        temp_image.Deallocate( );

        // histogram

        float histogram_step = (histogram_max - histogram_min) / float(histogram_number_of_points);
        float temp_float     = histogram_min + (histogram_step / 2.0f); // start position
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
            expected_survival_histogram[line_counter] = (erfc((temp_float + histogram_step * float(line_counter)) / sqrtf(2.0f)) / 2.0f) * (aggregated_results[array_location].collated_data_array[0] * aggregated_results[array_location].collated_data_array[1] * aggregated_results[array_location].total_number_of_ccs);
        }

        // calculate the expected threshold (from peter's paper)
        const float CCG_NOISE_STDDEV = 1.0;
        double      temp_threshold   = 0.0;
        double      erf_input        = 2.0 / (1.0 * ((double)aggregated_results[array_location].collated_data_array[0] * (double)aggregated_results[array_location].collated_data_array[1] * (double)aggregated_results[array_location].total_number_of_ccs));
        //        wxPrintf("ox oy total %3.3e %3.3e %3.3e\n", (double)result_array[5] , (double)result_array[6] , (double)aggregated_results[array_location].total_number_of_ccs, erf_input);

#ifdef MKL
        vdErfcInv(1, &erf_input, &temp_threshold);
#else
        temp_threshold = cisTEM_erfcinv(erf_input);
#endif
        expected_threshold = sqrtf(2.0f) * (float)temp_threshold * CCG_NOISE_STDDEV;

        //        expected_threshold = sqrtf(2.0f)*cisTEM_erfcinv((2.0f*(1))/(((original_input_image_x * original_input_image_y * aggregated_results[array_location].total_number_of_ccs))));

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
                survival_histogram[line_counter] *= (float)(sum_expected / sum_counted);
            }
#endif
        }

        for ( int line_counter = 0; line_counter < histogram_number_of_points; line_counter++ ) {
            temp_double_array[0] = temp_float + histogram_step * float(line_counter);
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

        long nTrys = 0;
        while ( 1 == 1 ) {
            // look for a peak..
            nTrys++;
            //            wxPrintf("Trying the %ld'th peak\n",nTrys);
            // FIXME min-distance from edges would be better to set dynamically.
            current_peak = scaled_mip.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction.logical_x_dimension / cistem::fraction_of_box_size_to_exclude_for_border + 1);
            if ( current_peak.value < expected_threshold )
                break;

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            address = 0;

            current_peak.x = current_peak.x + scaled_mip.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + scaled_mip.physical_address_of_box_center_y;

            // arguments[2] = pixel_size
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
            //        coordinates[0] = current_peak.x * pixel_size;
            //        coordinates[1] = current_peak.y * pixel_size;
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

        result_image.QuickAndDirtyWriteSlice(current_job_package.jobs[(aggregated_results[array_location].image_number - 1) * number_of_expected_results].arguments[38].ReturnStringArgument( ), 1, true);

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
    image_number               = -1;
    number_of_received_results = 0;
    total_number_of_ccs        = 0.0f;

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

    int offset = number_of_meta_data_values;

    if ( collated_data_array == NULL ) {
        collated_data_array = new float[array_size];
        ZeroFloatArray(collated_data_array, array_size);
        number_of_received_results = 0;
        total_number_of_ccs        = 0.0f;

        // nasty..

        collated_mip_data          = &collated_data_array[offset + int(result_array[2]) * 0];
        collated_psi_data          = &collated_data_array[offset + int(result_array[2]) * 1];
        collated_theta_data        = &collated_data_array[offset + int(result_array[2]) * 2];
        collated_phi_data          = &collated_data_array[offset + int(result_array[2]) * 3];
        collated_defocus_data      = &collated_data_array[offset + int(result_array[2]) * 4];
        collated_pixel_size_data   = &collated_data_array[offset + int(result_array[2]) * 5];
        collated_pixel_sums        = &collated_data_array[offset + int(result_array[2]) * 6];
        collated_pixel_square_sums = &collated_data_array[offset + int(result_array[2]) * 7];

        collated_histogram_data = (long*)&collated_data_array[offset + int(result_array[2]) * 8];

        collated_data_array[0] = result_array[0];
        collated_data_array[1] = result_array[1];
        collated_data_array[2] = result_array[2];
        collated_data_array[3] = result_array[3];

        collated_data_array[5] = result_array[5];
    }

    total_number_of_ccs += result_array[4];

    float* result_mip_data          = &result_array[offset + int(result_array[2]) * 0];
    float* result_psi_data          = &result_array[offset + int(result_array[2]) * 1];
    float* result_theta_data        = &result_array[offset + int(result_array[2]) * 2];
    float* result_phi_data          = &result_array[offset + int(result_array[2]) * 3];
    float* result_defocus_data      = &result_array[offset + int(result_array[2]) * 4];
    float* result_pixel_size_data   = &result_array[offset + int(result_array[2]) * 5];
    float* result_pixel_sums        = &result_array[offset + int(result_array[2]) * 6];
    float* result_pixel_square_sums = &result_array[offset + int(result_array[2]) * 7];

    long* input_histogram_data = (long*)&result_array[offset + int(result_array[2]) * 8];

    long pixel_counter;
    long result_array_counter;

    // handle the images..

    for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
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

    for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
        collated_pixel_sums[pixel_counter] += result_pixel_sums[pixel_counter];
    }

    for ( pixel_counter = 0; pixel_counter < int(result_array[2]); pixel_counter++ ) {
        collated_pixel_square_sums[pixel_counter] += result_pixel_square_sums[pixel_counter];
    }

    // handle the histogram..

    for ( pixel_counter = 0; pixel_counter < histogram_number_of_points; pixel_counter++ ) {
        collated_histogram_data[pixel_counter] += input_histogram_data[pixel_counter];
    }

    number_of_received_results++;
    wxPrintf("Received %i of %i results\n", number_of_received_results, number_of_expected_results);
}
