#include "../../core/core_headers.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

// check scaling

class
        AzimuthalAverageNew : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

typedef struct ctf_parameters {
    float acceleration_voltage; // keV
    float spherical_aberration; // mm
    float amplitude_contrast;
    float defocus_1; // A
    float defocus_2; // A
    float astigmatism_angle; // degrees
    float lowest_frequency_for_fitting; // 1/A
    float highest_frequency_for_fitting; // 1/A
    float astigmatism_tolerance; // A
    float pixel_size; // A
    float additional_phase_shift; // rad
} ctf_parameters;

std::vector<float>  sum_image_columns(Image* current_image);
float               sum_image_columns_float(Image* current_image);
void                save_all_columns_sum_to_file(const std::vector<std::vector<float>>& all_columns_sum, const std::string& filename);
std::pair<int, int> findOuterTubeEdges(const std::vector<float>& cols, float min_tube_diameter, float max_tube_diameter);

// new way
void create_white_sphere_mask(Image* mask_file, int x_sphere_center, int y_sphere_center, int z_spehere_center, float radius);
void create_black_sphere_mask(Image* mask_file, int x_sphere_center, int y_sphere_center, int z_spehere_center, float radius);

//Functions for the average images bins
void  InitializeCTFSumOfSquares(int numBins, Image& current_image, std::vector<std::vector<float>>* ctf_sum_of_squares);
void  ApplyCTFAndReturnCTFSumOfSquares(Image& image, CTF ctf_to_apply, bool absolute, bool apply_beam_tilt, bool apply_envelope, std::vector<float>& ctf_sum_of_squares);
void  divide_by_ctf_sum_of_squares(Image& current_image, std::vector<float>& ctf_sum_of_squares);
void  sum_image_direction(Image* current_image, int dim);
void  apply_ctf(Image* current_image, CTF ctf_to_apply, float* ctf_sum_of_squares, bool absolute, bool do_fill_sum_of_squares);
float angle_within360(float angle);
float ReturnAverageOfRealValuesOnVerticalEdges(Image* current_image);

IMPLEMENT_APP(AzimuthalAverageNew)

// override the DoInteractiveUserInput

void AzimuthalAverageNew::DoInteractiveUserInput( ) {
    // intial parameters
    float pixel_size;
    // ctf parameters
    wxString input_star_filename;
    float    acceleration_voltage;
    float    spherical_aberration;
    float    amplitude_contrast;
    float    defocus_1              = 0.0;
    float    defocus_2              = 0.0;
    float    astigmatism_angle      = 0.0;
    float    additional_phase_shift = 0.0;
    bool     input_ctf_values_from_star_file;
    bool     phase_flip_only;
    wxString output_average_per_bin_filename;
    wxString output_azimuthal_average_volume_filename;

    // tube searching parameters
    float min_tube_diameter = 0.0;
    float max_tube_diameter = 0.0;
    int   bins_count        = 1; // The number of bins to use when clssifying images based on tube diameter, the min number is 1 class
    int   outer_mask_radius = 0;
    bool  low_pass;
    float low_pass_resolution = 50.0;

    // finding helix axis method
    bool use_auto_corr;
    bool use_ft;

    // RASTR mask parameters
    bool     RASTR      = false;
    bool     input_mask = false;
    wxString input_mask_filename;
    int      x_mask_center      = 1;
    int      y_mask_center      = 1;
    int      z_mask_center      = 1;
    int      sphere_mask_radius = 1;
    int      number_of_models   = 1;
    bool     mask_upweighted    = false; // don't mask for now
    bool     align_upweighted   = false; // we want the tubes to be aligned vertically
    bool     center_upweighted  = false; // center the upweighted region in the middle so full image is aligned and centered
    wxString RASTR_output_filename;
    wxString RASTR_output_star_filename;

    // SPOT RASTR
    bool     SPOT_RASTR = false;
    wxString SPOT_RASTR_output_filename;
    wxString SPOT_RASTR_output_star_filename;

    // expert options
    // If user chose No in set expert options the below values will be used by default
    bool set_expert_options;
    // rotation (degrees)
    float psi_min              = 0.0;
    float psi_max              = 180.0;
    float psi_step             = 5.0;
    float fine_tuning_psi_step = 0.25;
    float padding_factor       = 2; //(sqrt of 2)
    // bool     find_positive_peaks       = true;
    // bool     find_negative_peaks       = true;
    wxString output_peaks_filename     = "peaks_output.txt";
    wxString output_diameters_filename = "diameters_output.txt";
    float    cosine_edge               = 10.0;
    float    outside_weight            = 0.0;
    float    filter_radius             = 0.0;
    float    outside_value             = 0.0;
    bool     use_outside_value         = false;
    bool     use_memory                = false;
    int      max_threads;

    UserInput* my_input = new UserInput("AzimuthalAverageNew", 1.0);

    wxString input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of input stack", "input_stack.mrc", true);
    // get CTF from user
    pixel_size           = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (keV)", "Acceleration voltage, in keV", "300.0", 0.0, 500.0);
    spherical_aberration = my_input->GetFloatFromUser("Spherical aberration (mm)", "Objective lens spherical aberration", "2.7", 0.0);
    amplitude_contrast   = my_input->GetFloatFromUser("Amplitude contrast", "Fraction of total contrast attributed to amplitude contrast", "0.07", 0.0);

    input_ctf_values_from_star_file = my_input->GetYesNoFromUser("Use a star file to input defocus values?", "If yes, defocus values will be extracted from star file", "NO");

    if ( input_ctf_values_from_star_file == true ) {
        input_star_filename = my_input->GetFilenameFromUser("Input star file", "The input star file", "my_parameters.star", true);
    }
    else {
        defocus_1              = my_input->GetFloatFromUser("Underfocus 1 (A)", "In Angstroms, the objective lens underfocus along the first axis", "1.2");
        defocus_2              = my_input->GetFloatFromUser("Underfocus 2 (A)", "In Angstroms, the objective lens underfocus along the second axis", "1.2");
        astigmatism_angle      = my_input->GetFloatFromUser("Astigmatism angle", "Angle between the first axis and the x axis of the image", "0.0");
        additional_phase_shift = my_input->GetFloatFromUser("Additional phase shift (rad)", "Additional phase shift relative to undiffracted beam, as introduced for example by a phase plate", "0.0");
    }

    phase_flip_only = my_input->GetYesNoFromUser("Phase Flip Only", "If Yes, only phase flipping is performed", "NO");

    // tube diameter arguments
    min_tube_diameter = my_input->GetFloatFromUser("Minimum tube diameter", "The minimum tube diameter for searching and bining tubes", "30.0", 0.0);
    max_tube_diameter = my_input->GetFloatFromUser("Maximum tube diameter", "The maximum tube diameter for searching and bining tubes", "60.0", 0.0);
    bins_count        = my_input->GetIntFromUser("Number of classes for classifying the tube diameters", "The number of classes (bins) to classify the tubes based on specified min. and max. diameter", "1", 1);
    outer_mask_radius = my_input->GetIntFromUser("Outer mask radius for tube search (pixels)", "Outer mask radius to use when searching for tubes in pixels. If 0, outer mask radius will be x dimension * 0.45", "0", 0);
    // tubes_centered    = my_input->GetYesNoFromUser("Are tubes centered?", "If yes, the outer radius of peak search from cross-correlation will be 1/5 x-dimension", "NO");
    low_pass = my_input->GetYesNoFromUser("Apply low-pass gaussian filter when finding peaks or aligning?", "If yes, will apply a gaussian low pass filter to the original images before finding peaks or aligining the azimuthal average projection", "NO");
    if ( low_pass ) {
        low_pass_resolution = my_input->GetFloatFromUser("Resolution limit for low pass filtering", "Resolution limit for low pass filter. Only this resolution or worse information will be retained", "50.0", 0.0);
    }

    // use autocorrelation or Fourier Transform to find the axis of the helix
    use_auto_corr = my_input->GetYesNoFromUser("Use auto correlation to find helix axis psi rotation?", "If yes, will use auto correlation to find helix axis psi rotation", "NO");
    if ( ! use_auto_corr ) {
        use_ft = my_input->GetYesNoFromUser("Use Fourier Transform to find helix axis psi rotation?", "If yes, will use Fourier Transform to find helix axis psi rotation", "Yes");
    }

    // output arguments for azimuthal average volume and class projections
    output_average_per_bin_filename          = my_input->GetFilenameFromUser("Output name for the average projection images per class stack", "The output name for the average images generated per class MRC file", "output_average_per_class.mrc", false);
    output_azimuthal_average_volume_filename = my_input->GetFilenameFromUser("Output name for the azimuthal average volume per class stack", "The output name for the azimuthal average volume generated per class MRC file", "output_azimuthal_average_volume.mrc", false);

    //RASTR mask parameters
    RASTR = my_input->GetYesNoFromUser("Perform RASTR masking to 3D azimuthal average?", "Mask out region of interest", "NO");
    // get mask properties from user
    if ( RASTR == true ) {
        input_mask = my_input->GetYesNoFromUser("Use input mask file?", "Do you want to provide an input mask file?", "No");
        if ( input_mask == true ) {
            input_mask_filename = my_input->GetFilenameFromUser("Input mask file name", "The mask to be applied to the 3D azimuthal average model", "my_mask.mrc", true);
        }
        x_mask_center      = my_input->GetIntFromUser("X center of the mask (pixels)", "The X center of the provided mask or the created mask by the program, default is 0.75 * box size", "1", 1);
        y_mask_center      = my_input->GetIntFromUser("Y center pf the mask (pixels)", "The Y center of the provided mask or the created mask by the program, default is 0.5 * box size", "1", 1);
        z_mask_center      = my_input->GetIntFromUser("Z center of the mask (pixels)", "The Z center of the provided mask or the created mask by the program, default is 0.5 * box size", "1", 1);
        sphere_mask_radius = my_input->GetIntFromUser("Sphere mask radius (pixels)", "The radius of the provided mask or the created mask by the program, default is 0.25 * box size", "1", 1);
        number_of_models   = my_input->GetIntFromUser("Number of models to generate", "3D azimuthal average model is rotated in increments of (360/n) degrees", "4", 1);
        mask_upweighted    = my_input->GetYesNoFromUser("Mask the upweighted regions?", "Do you want to mask the upweighted regions? if no it will save the centered aligned full subtracted image", "No");
        if ( mask_upweighted ) {
            align_upweighted  = my_input->GetYesNoFromUser("Align the upweighted regions vertically?", "Do you want to align the upweighted regions vertically?", "No");
            center_upweighted = my_input->GetYesNoFromUser("Center the upweighted regions?", "Do you want to center the upweighted regions?", "No");
        }
        RASTR_output_filename      = my_input->GetFilenameFromUser("Output masked upweighted regions filename", "The output MRC file containing average subtracted masked upweighted regions", "masked_upweighted_regions_output_filename.mrc", false);
        RASTR_output_star_filename = my_input->GetFilenameFromUser("Output star file with the psi and shift changes", "The output star file, containing the new psi and shift parameters", "my_RASTR_output_parameters.star", false);
    }

    SPOT_RASTR = my_input->GetYesNoFromUser("Perform SPOT-RASTR tube subtraction?", "If yes, azimuthal average model projections will be subtracted from original images", "NO");
    if ( SPOT_RASTR == true ) {
        SPOT_RASTR_output_filename      = my_input->GetFilenameFromUser("Output average subtracted stack filename", "The output average subtracted MRC file", "average_subtracted_output_filename.mrc", false);
        SPOT_RASTR_output_star_filename = my_input->GetFilenameFromUser("Output star file with the psi and shift changes", "The output star file, containing the new psi and shift parameters", "my_SPOT_RASTR_output_parameters.star", false);
    }

    if ( (RASTR == true) || (SPOT_RASTR == true) ) {
    }
    // expert options parameters
    set_expert_options = my_input->GetYesNoFromUser("Set Expert Options?", "Set these for more control, hopefully not needed", "NO");

    // set alignment options from user
    if ( set_expert_options == true ) {
        psi_min              = my_input->GetFloatFromUser("Minimum rotation for initial search (degrees)", "The minimum angle rotation of initial search will be limited to this value.", "0.0", -180.0);
        psi_max              = my_input->GetFloatFromUser("Maximum rotation for initial search (degrees)", "The maximum angle rotation of initial search will be limited to this value.", "180.0", 180.0);
        psi_step             = my_input->GetFloatFromUser("Rotation step size (degrees)", "The step size of each rotation will be limited to this value.", "5.0", 0.0);
        fine_tuning_psi_step = my_input->GetFloatFromUser("Local refinement angle rotation step size (degrees)", "The local refinement search step size will be this value", "0.25", 0.0);
        padding_factor       = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the average image is padded to improve subtraction, defau;t is sqrt(2)", "1.4", 1.0);
        // find_positive_peaks       = my_input->GetYesNoFromUser("Find the positive peaks?", "If yes, will find the highest positive peaks.", "Yes");
        // find_negative_peaks       = my_input->GetYesNoFromUser("Find the negative peaks?", "If yes, will find the highest negative peaks.", "Yes");
        output_peaks_filename     = my_input->GetFilenameFromUser("Output peaks file name", "Filename of the peaks ", "peaks_output.txt", false);
        output_diameters_filename = my_input->GetFilenameFromUser("Output diameters file name", "Filename of the saved diameters ", "diameters_output.txt", false);
        cosine_edge               = my_input->GetFloatFromUser("Width of cosine edge (A)", "Width of the smooth edge to add to the mask in Angstroms", "10.0", 0.0);
        outside_weight            = my_input->GetFloatFromUser("Weight of density outside mask", "Factor to multiply density outside of the mask", "0.0", 0.0, 1.0);
        filter_radius             = my_input->GetFloatFromUser("Low-pass filter outside mask (A)", "Low-pass filter to be applied to the density outside the mask", "0.0", 0.0);
        outside_value             = my_input->GetFloatFromUser("Outside mask value", "Value used to set density outside the mask", "0.0", 0.0);
        use_outside_value         = my_input->GetYesNoFromUser("Use outside mask value", "Should the density outside the mask be set to the user-provided value", "No");
    }

    use_memory = my_input->GetYesNoFromUser("Allocate images to memory?", "Choice between memory allocation or using functions; no is recommended for systems with limited memory.", "NO");

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    my_current_job.Reset(54);
    my_current_job.ManualSetArguments("tffffbtffffbffiibfbbttbbtiiiiibbbttbttbfffffttffffbbi", input_filename.ToUTF8( ).data( ),
                                      pixel_size,
                                      acceleration_voltage,
                                      spherical_aberration,
                                      amplitude_contrast,
                                      input_ctf_values_from_star_file,
                                      input_star_filename.ToUTF8( ).data( ),
                                      defocus_1,
                                      defocus_2,
                                      astigmatism_angle,
                                      additional_phase_shift,
                                      phase_flip_only,
                                      min_tube_diameter,
                                      max_tube_diameter,
                                      bins_count,
                                      outer_mask_radius,
                                      low_pass,
                                      low_pass_resolution,
                                      use_auto_corr,
                                      use_ft,
                                      output_average_per_bin_filename.ToUTF8( ).data( ),
                                      output_azimuthal_average_volume_filename.ToUTF8( ).data( ),
                                      RASTR,
                                      input_mask,
                                      input_mask_filename.ToUTF8( ).data( ),
                                      x_mask_center,
                                      y_mask_center,
                                      z_mask_center,
                                      sphere_mask_radius,
                                      number_of_models,
                                      mask_upweighted,
                                      align_upweighted,
                                      center_upweighted,
                                      RASTR_output_filename.ToUTF8( ).data( ),
                                      RASTR_output_star_filename.ToUTF8( ).data( ),
                                      SPOT_RASTR,
                                      SPOT_RASTR_output_filename.ToUTF8( ).data( ),
                                      SPOT_RASTR_output_star_filename.ToUTF8( ).data( ),
                                      set_expert_options,
                                      psi_min,
                                      psi_max,
                                      psi_step,
                                      fine_tuning_psi_step,
                                      padding_factor,
                                      //   find_positive_peaks,
                                      //   find_negative_peaks,
                                      output_peaks_filename.ToUTF8( ).data( ),
                                      output_diameters_filename.ToUTF8( ).data( ),
                                      cosine_edge,
                                      outside_weight,
                                      filter_radius,
                                      outside_value,
                                      use_outside_value,
                                      use_memory,
                                      max_threads);
}

// override the do calculation method which will be what is actually run..

bool AzimuthalAverageNew::DoCalculation( ) {
    // get the arguments for this job..
    wxString input_filename                           = my_current_job.arguments[0].ReturnStringArgument( );
    float    pixel_size                               = my_current_job.arguments[1].ReturnFloatArgument( );
    float    acceleration_voltage                     = my_current_job.arguments[2].ReturnFloatArgument( );
    float    spherical_aberration                     = my_current_job.arguments[3].ReturnFloatArgument( );
    float    amplitude_contrast                       = my_current_job.arguments[4].ReturnFloatArgument( );
    bool     input_ctf_values_from_star_file          = my_current_job.arguments[5].ReturnBoolArgument( );
    wxString input_star_filename                      = my_current_job.arguments[6].ReturnStringArgument( );
    float    defocus_1                                = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus_2                                = my_current_job.arguments[8].ReturnFloatArgument( );
    float    astigmatism_angle                        = my_current_job.arguments[9].ReturnFloatArgument( );
    float    additional_phase_shift                   = my_current_job.arguments[10].ReturnFloatArgument( );
    bool     phase_flip_only                          = my_current_job.arguments[11].ReturnBoolArgument( );
    float    min_tube_diameter                        = my_current_job.arguments[12].ReturnFloatArgument( );
    float    max_tube_diameter                        = my_current_job.arguments[13].ReturnFloatArgument( );
    int      bins_count                               = my_current_job.arguments[14].ReturnIntegerArgument( );
    int      outer_mask_radius                        = my_current_job.arguments[15].ReturnIntegerArgument( );
    bool     low_pass                                 = my_current_job.arguments[16].ReturnBoolArgument( );
    float    low_pass_resolution                      = my_current_job.arguments[17].ReturnFloatArgument( );
    bool     use_auto_corr                            = my_current_job.arguments[18].ReturnBoolArgument( );
    bool     use_ft                                   = my_current_job.arguments[19].ReturnBoolArgument( );
    wxString output_average_per_bin_filename          = my_current_job.arguments[20].ReturnStringArgument( );
    wxString output_azimuthal_average_volume_filename = my_current_job.arguments[21].ReturnStringArgument( );
    bool     RASTR                                    = my_current_job.arguments[22].ReturnBoolArgument( );
    bool     input_mask                               = my_current_job.arguments[23].ReturnBoolArgument( );
    wxString input_mask_filename                      = my_current_job.arguments[24].ReturnStringArgument( );
    int      x_mask_center                            = my_current_job.arguments[25].ReturnIntegerArgument( );
    int      y_mask_center                            = my_current_job.arguments[26].ReturnIntegerArgument( );
    int      z_mask_center                            = my_current_job.arguments[27].ReturnIntegerArgument( );
    int      sphere_mask_radius                       = my_current_job.arguments[28].ReturnIntegerArgument( );
    int      number_of_models                         = my_current_job.arguments[29].ReturnIntegerArgument( );
    bool     mask_upweighted                          = my_current_job.arguments[30].ReturnBoolArgument( );
    bool     align_upweighted                         = my_current_job.arguments[31].ReturnBoolArgument( );
    bool     center_upweighted                        = my_current_job.arguments[32].ReturnBoolArgument( );
    wxString RASTR_output_filename                    = my_current_job.arguments[33].ReturnStringArgument( );
    wxString RASTR_output_star_filename               = my_current_job.arguments[34].ReturnStringArgument( );
    bool     SPOT_RASTR                               = my_current_job.arguments[35].ReturnBoolArgument( );
    wxString SPOT_RASTR_output_filename               = my_current_job.arguments[36].ReturnStringArgument( );
    wxString SPOT_RASTR_output_star_filename          = my_current_job.arguments[37].ReturnStringArgument( );
    bool     set_expert_options                       = my_current_job.arguments[38].ReturnBoolArgument( );
    float    psi_min                                  = my_current_job.arguments[39].ReturnFloatArgument( );
    float    psi_max                                  = my_current_job.arguments[40].ReturnFloatArgument( );
    float    psi_step                                 = my_current_job.arguments[41].ReturnFloatArgument( );
    float    fine_tuning_psi_step                     = my_current_job.arguments[42].ReturnFloatArgument( );
    float    padding_factor                           = my_current_job.arguments[43].ReturnFloatArgument( );
    wxString output_peaks_filename                    = my_current_job.arguments[44].ReturnStringArgument( );
    wxString output_diameters_filename                = my_current_job.arguments[45].ReturnStringArgument( );
    float    cosine_edge                              = my_current_job.arguments[46].ReturnFloatArgument( );
    float    outside_weight                           = my_current_job.arguments[47].ReturnFloatArgument( );
    float    filter_radius                            = my_current_job.arguments[48].ReturnFloatArgument( );
    float    outside_value                            = my_current_job.arguments[49].ReturnFloatArgument( );
    bool     use_outside_value                        = my_current_job.arguments[50].ReturnBoolArgument( );
    bool     use_memory                               = my_current_job.arguments[51].ReturnBoolArgument( );
    int      max_threads                              = my_current_job.arguments[52].ReturnIntegerArgument( );

    // initiate I/O variables
    MRCFile  my_input_file(input_filename.ToStdString( ), false); // check all the functions and things done with the MRCFile and also check the wxPrintF statement
    MRCFile  my_output_sum_image_filename(output_average_per_bin_filename.ToStdString( ), true);
    MRCFile* my_output_SPOT_RASTR_filename;

    long number_of_input_images = my_input_file.ReturnNumberOfSlices( );
    int  x_dim                  = my_input_file.ReturnXSize( );
    int  y_dim                  = my_input_file.ReturnYSize( );

    // If use_memory to make things fast
    Image* image_stack;
    if ( use_memory )
        image_stack = new Image[number_of_input_images];
    else
        image_stack = nullptr;

    // input stack low pass filtered and masked
    Image* image_stack_filtered_masked;
    if ( use_memory )
        image_stack_filtered_masked = new Image[number_of_input_images];
    else
        image_stack_filtered_masked = nullptr;

    if ( use_memory ) {
        wxPrintf("\nLoading images to memory...\n\n");
        ProgressBar* loading_progress = new ProgressBar(number_of_input_images);
#pragma omp parallel for num_threads(max_threads) schedule(static) shared(loading_progress, my_input_file, image_stack, image_stack_filtered_masked, x_dim, y_dim, outer_mask_radius, low_pass, low_pass_resolution, pixel_size)

        for ( long image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
// Read from disk
#pragma omp critical
            image_stack[image_counter].ReadSlice(&my_input_file, image_counter + 1);
            image_stack_filtered_masked[image_counter].CopyFrom(&image_stack[image_counter]);
            // Normalize the image using cisTEM Normalize
            image_stack_filtered_masked[image_counter].Normalize( );
            // Here the masking is important as we want to only find the rotation of the tubes around the center or near the center
            // Any repeating signal near the signal will be seen in both the auto-correlation and FT even if only partial tube is present
            // This will ensure no aligning is done to tubes on edges

            if ( outer_mask_radius != 0 ) {
                image_stack_filtered_masked[image_counter].CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                image_stack_filtered_masked[image_counter].CircleMask(x_dim * 0.45);
            }
            // FT the image
            image_stack_filtered_masked[image_counter].ForwardFFT( );
            // convert the central pixel to zero (Is that done in real or Fourier space??)
            image_stack_filtered_masked[image_counter].ZeroCentralPixel( );

            if ( low_pass ) {
                // will applying a low pass filter here improve finding the correct rotation in FT
                image_stack_filtered_masked[image_counter].GaussianLowPassFilter((pixel_size * 2) / low_pass_resolution);
            }

            if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                loading_progress->Update(image_counter + 1);
        }
        delete loading_progress;
    }
    // initiate default parameters for the ApplyCTFAndReturnCTFSumOfSquares function
    // (May be change that later to be expert options inputs???)
    bool absolute        = false;
    bool apply_beam_tilt = false;
    bool apply_envelope  = false;

    // CTF object
    CTF current_ctf;

    // check if input defocus file is given for the CTF calculations
    // before running any code check if the file containing defocus is given to the program
    ctf_parameters* ctf_parameters_stack = new ctf_parameters[number_of_input_images];

    if ( input_ctf_values_from_star_file == true ) {
        //cisTEM star
        cisTEMParameters input_star_file;
        //Relion Star
        //BasicStarFileReader input_star_file;
        //wxString            star_error_text;
        if ( (is_running_locally && ! DoesFileExist(input_star_filename.ToStdString( ))) ) {
            SendErrorAndCrash(wxString::Format("Error: Input star file %s not found\n", input_star_filename));
        }
        //CisTEM star
        input_star_file.ReadFromcisTEMStarFile(input_star_filename.ToStdString( ));
        //RELION star
        //input_star_file.ReadFile(input_star_filename.ToStdString( ));
        for ( long image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            ctf_parameters_stack[image_counter].acceleration_voltage          = acceleration_voltage;
            ctf_parameters_stack[image_counter].spherical_aberration          = spherical_aberration;
            ctf_parameters_stack[image_counter].amplitude_contrast            = amplitude_contrast;
            ctf_parameters_stack[image_counter].defocus_1                     = input_star_file.ReturnDefocus1(image_counter);
            ctf_parameters_stack[image_counter].defocus_2                     = input_star_file.ReturnDefocus2(image_counter);
            ctf_parameters_stack[image_counter].astigmatism_angle             = input_star_file.ReturnDefocusAngle(image_counter);
            ctf_parameters_stack[image_counter].lowest_frequency_for_fitting  = 0.0;
            ctf_parameters_stack[image_counter].highest_frequency_for_fitting = 0.5;
            ctf_parameters_stack[image_counter].astigmatism_tolerance         = 0.0;
            ctf_parameters_stack[image_counter].pixel_size                    = pixel_size;
            ctf_parameters_stack[image_counter].additional_phase_shift        = input_star_file.ReturnPhaseShift(image_counter);
            //wxPrintf("The current image is %li and its defocus is %f\n", image_counter+1, input_star_file.ReturnDefocus1(image_counter) );
        }
    }

    current_ctf.Init(acceleration_voltage, spherical_aberration, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.5, 0.0, pixel_size, additional_phase_shift);

    // Initialize sum of squares vector of vectors
    // Create a dynamic memory for vector of vectors to save the CTF sum of squares for all images sumed in each bin
    Image current_image;

    // calculating the ctf_sum_of_squares
    current_image.ReadSlice(&my_input_file, 1);
    // CTFSumOfSquares for the initial sum image
    // CTFSumOfSquaresNew for the final sum image
    std::vector<std::vector<float>>* CTFSumOfSquares = new std::vector<std::vector<float>>( );
    CTFSumOfSquares->resize(bins_count);

    std::vector<std::vector<float>>* CTFSumOfSquaresNew = new std::vector<std::vector<float>>( );
    CTFSumOfSquaresNew->resize(bins_count);

    InitializeCTFSumOfSquares(bins_count, current_image, CTFSumOfSquares);
    InitializeCTFSumOfSquares(bins_count, current_image, CTFSumOfSquaresNew);

    // Initialize the sum images based on the number specified by the user
    Image sum_images[bins_count];

    for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
        sum_images[bin_index].Allocate(x_dim, y_dim, true); //allocate in real space
        sum_images[bin_index].SetToConstant(0.0);
    }

    std::vector<std::vector<float>> all_columns_sum(number_of_input_images, std::vector<float>(x_dim, 0.0)); // saving those values to ensure debugging if the user want to

    // calculate the bin range
    float bin_range = (max_tube_diameter - min_tube_diameter) / bins_count;

    std::vector<float> tube_rotation(number_of_input_images, 0.0f);
    std::vector<float> best_sum_column(number_of_input_images, 0.0f);
    std::vector<float> x_shift_column(number_of_input_images, 0.0f);
    std::vector<float> y_shift_row(number_of_input_images, 0.0f);
    std::vector<float> all_diameters(number_of_input_images, 0.0f);
    int                diameter_bins[number_of_input_images];
    float              center_peak_index = y_dim / 2;

    // saving all the diameters and peaks in a text file
    // Open the diameter file in write mode
    std::ofstream file(output_diameters_filename.ToStdString( ));
    if ( ! file.is_open( ) ) {
        std::cerr << "Error: Could not open diameters_output.txt\n";
    }

    file << std::fixed << std::setprecision(2); // Optional: set float precision
    file << "image_index, diameter\n";

    std::ofstream peak_file(output_peaks_filename.ToStdString( )); // Open file once

    if ( ! peak_file.is_open( ) ) {
        std::cerr << "Error: Could not open peaks_output.txt\n";
        //return;
    }

    peak_file << std::fixed << std::setprecision(2); // Optional: set float precision
    peak_file << "image_index, peak_one_value, peak_two_value\n";

    std::vector<std::pair<int, std::pair<int, int>>> results(number_of_input_images);

    //Initialize mask file
    MRCFile* my_mask_file;
    Image*   mask;
    //RASTR == true & input_mask == true
    if ( RASTR && input_mask ) {
        if ( ! DoesFileExist(input_mask_filename.ToStdString( )) ) {
            SendError(wxString::Format("Error: Mask %s not found\n", input_mask_filename.ToStdString( )));
            exit(-1);
        }

        my_mask_file = new MRCFile(input_mask_filename.ToStdString( ), false);

        //Curve hist;
        mask = new Image;
        mask->Allocate(my_mask_file->ReturnXSize( ), my_mask_file->ReturnYSize( ), my_mask_file->ReturnZSize( ));
        mask->ReadSlices(my_mask_file, 1, my_mask_file->ReturnNumberOfSlices( ));
        //mask.ComputeHistogramOfRealValuesCurve(&hist);
        //hist.PrintToStandardOut( );
        if ( mask->ReturnMaximumValue( ) != 1 & mask->ReturnMinimumValue( ) != 0 ) {
            SendError(wxString::Format("Error: The minimum value in the mask is %f not 0 and the maximum value in the mask is %f not 1", mask->ReturnMinimumValue( ), mask->ReturnMaximumValue( )));
            exit(-1);
        }
        delete mask;
    }

    // if input mask file is not provided make initialize the x,y,z centers of the mask to the default values
    if ( input_mask != true ) {
        if ( x_mask_center == 1 ) {
            x_mask_center = 0.75 * x_dim;
        }
        if ( y_mask_center == 1 ) {
            y_mask_center = 0.5 * x_dim;
        }
        if ( z_mask_center == 1 ) {
            z_mask_center = 0.5 * x_dim;
        }
        if ( sphere_mask_radius == 1 ) {
            sphere_mask_radius = 0.25 * x_dim; //previously was 0.1875
        }
    }

    // initiate other needed variables
    long  image_counter;
    Image final_image;
    Image final_image_copy; // IS THIS NEEDED ???

    wxPrintf("\nCalculating initial tube rotation...\n\n");
    ProgressBar* my_progress = new ProgressBar(number_of_input_images);

#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads) default(none) shared(my_input_file, best_sum_column, tube_rotation, all_columns_sum, number_of_input_images, max_threads, use_auto_corr, use_ft, low_pass_resolution, x_dim, y_dim, image_stack_filtered_masked, \
                                                                                            x_shift_column, all_diameters, psi_step, min_tube_diameter, max_tube_diameter, pixel_size, psi_min, psi_max, low_pass, use_memory,                                                          \
                                                                                            defocus_1, defocus_2, astigmatism_angle, additional_phase_shift, current_ctf, my_progress, cosine_edge, outside_weight, filter_radius, outside_value, use_outside_value,                    \
                                                                                            CTFSumOfSquares, bins_count, bin_range, sum_images, diameter_bins, outer_mask_radius, center_peak_index,                                                                                    \
                                                                                            absolute, apply_beam_tilt, apply_envelope, input_ctf_values_from_star_file, ctf_parameters_stack, I) private(current_image, final_image, image_counter)

    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        // read the current image in the stack
        if ( use_memory ) {
            current_image.CopyFrom(&image_stack_filtered_masked[image_counter]);
        }
        else {
#pragma omp critical
            current_image.ReadSlice(&my_input_file, image_counter + 1);
            // Normalize the image using cisTEM Normalize
            current_image.Normalize( );
            // Here the masking is important as we want to only find the rotation of the tubes around the center or near the center
            // Any repeating signal near the signal will be seen in both the auto-correlation and FT even if only partial tube is present
            // This will ensure no aligning is done to tubes on edges
            if ( outer_mask_radius != 0 ) {
                current_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                current_image.CircleMask(x_dim * 0.45);
            }
            // FT the image
            current_image.ForwardFFT( );
            // convert the central pixel to zero (Is that done in real or Fouriier space??)
            current_image.ZeroCentralPixel( );

            if ( low_pass ) {
                // will applying a low pass filter here improve finding the correct rotation in FT
                current_image.GaussianLowPassFilter((pixel_size * 2) / low_pass_resolution);
            }
        }

        if ( use_auto_corr ) {
            for ( long pixel_counter = 0; pixel_counter < current_image.real_memory_allocated / 2; pixel_counter++ ) {

                // calculating the amplitude is not needed by let's see and print its value
                float amplitude = abs(current_image.complex_values[pixel_counter]);

                //As the phase will be zero, the real is just the amplitude and the imaginary is 0
                // so we will set the complex number to be equal to apmlitude + 0
                current_image.complex_values[pixel_counter] = amplitude * amplitude + I * 0.0f;
            }
            // return the image to real space again and save them to see the correlation
            current_image.BackwardFFT( );
            current_image.SwapRealSpaceQuadrants( );
            // set the image is centered inside the box as true
            current_image.object_is_centred_in_box = true;
        }
        Image temp_image;
        Image power_image;
        // Now let's sum the images horizontally (across the columns and see how they will look) and rotate them then sum again after each rotation
        // finding the initial rotation angle when searching within 180 degrees and a step size 4 degrees

        if ( use_ft ) {
            temp_image.Allocate(x_dim, y_dim, false);
            temp_image.SetToConstant(0.0);
        }
        else {
            temp_image.Allocate(x_dim, y_dim, true);
            temp_image.SetToConstant(0.0);
        }

        float              local_best_sum = -FLT_MAX;
        float              local_best_psi = 0.0f;
        std::vector<float> local_columns_sum_vector;

        for ( float psi = psi_min; psi <= psi_max; psi += psi_step ) {
            // rotate by the temporary image by the rotation angle
            if ( use_auto_corr ) {
                temp_image.CopyFrom(&current_image);
                temp_image.Rotate2DInPlace(psi, FLT_MAX);
            }
            else if ( use_ft ) {
                AnglesAndShifts rotation_angle;
                rotation_angle.Init(0.0, 0.0, psi, 0.0, 0.0);
                temp_image.CopyFrom(&current_image);
                temp_image.SwapRealSpaceQuadrants( );
                Image rotated_image;
                rotated_image.Allocate(x_dim, y_dim, false); // the rotated images real values will be the rotated FT
                rotated_image.SetToConstant(0.0);
                temp_image.RotateFourier2D(rotated_image, rotation_angle);
                // allocate memory for power image
                power_image.Allocate(x_dim, y_dim, true);
                power_image.SetToConstant(0.0);
                rotated_image.ComputeAmplitudeSpectrumFull2D(&power_image);
                rotated_image.Deallocate( );
            }

            float column_sum;

            if ( use_auto_corr ) {
                // Filter the autocorrelation image so that any pixel values above average are kept to ensure correct rotation calculation
                // should we make that in the advanced options something to be optimized by user
                Image binary_mask;
                float image_average;
                image_average = temp_image.ReturnAverageOfRealValues( );
                binary_mask.Allocate(x_dim, y_dim, true);
                binary_mask.SetToConstant(0.0);
                binary_mask.CopyFrom(&temp_image);
                binary_mask.Binarise(image_average);

                float filter_edge = 40.0;
                temp_image.ApplyMask(binary_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
                binary_mask.Deallocate( );
                // Use a threshold to find the line corresponding to tube axis
                column_sum = sum_image_columns_float(&temp_image);
            }
            else if ( use_ft ) {
                // Use a threshold to find the line corresponding to tube axis angle of rotation
                float image_average;
                float image_sd;
                float image_threshold;
                Image binary_mask;
                image_average = power_image.ReturnAverageOfRealValues( );
                image_sd      = sqrt(power_image.ReturnVarianceOfRealValues( ));
                // Threshold value is 2 sd away from mean to eliminate any outliers
                image_threshold = image_average + (2 * image_sd);
                binary_mask.CopyFrom(&power_image);
                binary_mask.Binarise(image_threshold);

                float filter_edge = 40.0;
                power_image.ApplyMask(binary_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
                //power_image.QuickAndDirtyWriteSlice("binarized_power_image.mrc", 1);

                column_sum = sum_image_columns_float(&power_image);
            }

            std::vector<float> column_sum_vector;
            if ( use_auto_corr ) {
                column_sum_vector = sum_image_columns(&temp_image);
            }
            else if ( use_ft ) {
                column_sum_vector = sum_image_columns(&power_image);
            }

            if ( column_sum > local_best_sum ) {
                // at this point I will not add 90 to the saved psi if using FFT
                local_best_psi           = psi;
                local_best_sum           = column_sum;
                local_columns_sum_vector = column_sum_vector;
                //wxPrintf("The best psi angle with the highest sum is %f with sum %f \n\n", local_best_psi, local_best_sum);
            }
            power_image.Deallocate( ); // should this be here inside the psi loop or after it?? and reset the power_image to 0 constant ???
        }

        temp_image.Deallocate( );
        // tuning the rotation angle to search with 2 degrees before and after the initial angle
        // The step size will be rotation angle (2) /20

        float tuning_rotation_range = psi_step / 2;
        float tuning_step_size      = psi_step / 20;

        // column wise calculations
        float current_best_psi       = local_best_psi; // as the tube_rotation[image_counter] is still zero when having local variables!
        float tuning_psi_lower_range = current_best_psi - tuning_rotation_range;
        float tuning_psi_upper_range = current_best_psi + tuning_rotation_range;

        Image tuning_temp_image;
        Image tuning_power_image;

        if ( use_ft ) {
            tuning_temp_image.Allocate(x_dim, y_dim, false);
            tuning_temp_image.SetToConstant(0.0);
        }
        else {
            tuning_temp_image.Allocate(x_dim, y_dim, true);
            tuning_temp_image.SetToConstant(0.0);
        }

        for ( float tuning_psi = tuning_psi_lower_range; tuning_psi <= tuning_psi_upper_range; tuning_psi += tuning_step_size ) {
            // rotate by the temporary image by the rotation angle
            if ( use_auto_corr ) {
                tuning_temp_image.CopyFrom(&current_image);
                tuning_temp_image.Rotate2DInPlace(tuning_psi, FLT_MAX);
            }
            if ( use_ft ) {
                AnglesAndShifts tuning_rotation_angle;
                tuning_rotation_angle.Init(0.0, 0.0, tuning_psi, 0.0, 0.0);

                Image tuning_rotated_image;
                tuning_rotated_image.Allocate(x_dim, y_dim, false);
                tuning_rotated_image.SetToConstant(0.0);

                tuning_temp_image.CopyFrom(&current_image);
                tuning_temp_image.SwapRealSpaceQuadrants( );
                tuning_temp_image.RotateFourier2D(tuning_rotated_image, tuning_rotation_angle);

                // allocate memory for power image
                tuning_power_image.Allocate(x_dim, y_dim, true);
                tuning_power_image.SetToConstant(0.0);

                tuning_rotated_image.ComputeAmplitudeSpectrumFull2D(&tuning_power_image);
                tuning_rotated_image.Deallocate( );
            }

            float tuning_column_sum;
            float tuning_image_average;
            float tuning_image_sd;
            float tuning_image_threshold;
            Image tuning_binary_mask;

            if ( use_auto_corr ) {
                tuning_image_average = tuning_temp_image.ReturnAverageOfRealValues( );
                tuning_binary_mask.Allocate(x_dim, y_dim, true);
                tuning_binary_mask.SetToConstant(0.0);
                tuning_binary_mask.CopyFrom(&tuning_temp_image);
                tuning_binary_mask.Binarise(tuning_image_average);

                float filter_edge = 40.0;
                tuning_temp_image.ApplyMask(tuning_binary_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
                tuning_binary_mask.Deallocate( );
                tuning_column_sum = sum_image_columns_float(&tuning_temp_image);
            }
            else if ( use_ft ) {
                // Use a threshold to find the line corresponding to tube axis angle of rotation
                tuning_image_average = tuning_power_image.ReturnAverageOfRealValues( );
                tuning_image_sd      = sqrt(tuning_power_image.ReturnVarianceOfRealValues( ));
                // Threshold value is 2 sd away from mean to eliminate any outliers
                tuning_image_threshold = tuning_image_average + (2 * tuning_image_sd);
                tuning_binary_mask.CopyFrom(&tuning_power_image);
                tuning_binary_mask.Binarise(tuning_image_threshold);
                float filter_edge = 40.0;
                tuning_power_image.ApplyMask(tuning_binary_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
                tuning_column_sum = sum_image_columns_float(&tuning_power_image);
            }

            std::vector<float> tuning_column_sum_vector;

            if ( use_auto_corr ) {
                tuning_column_sum_vector = sum_image_columns(&tuning_temp_image);
            }
            else if ( use_ft ) {
                tuning_column_sum_vector = sum_image_columns(&tuning_power_image);
            }

            if ( tuning_column_sum > local_best_sum ) {
                // at this point I will not add 90 to the saved psi if using FFT
                local_best_psi           = tuning_psi;
                local_best_sum           = tuning_column_sum;
                local_columns_sum_vector = tuning_column_sum_vector;
                //wxPrintf("Tuning best psi angle with the highest sum is %f with sum %f \n\n", local_best_psi, local_best_sum);
            }
            tuning_power_image.Deallocate( ); // should this be here or outside the tuning psi loop?? and reset the image to zero inside the loop?
        }

        tuning_temp_image.Deallocate( );

        tube_rotation[image_counter]   = local_best_psi;
        best_sum_column[image_counter] = local_best_sum;
        all_columns_sum[image_counter] = local_columns_sum_vector;

        // After finding the correct initial psi, we need to find initial tube diameter and x shift to center images and sort images
        final_image.Allocate(x_dim, y_dim, true);
        final_image.SetToConstant(0.0);
        // find  the x,y shift
        // ReadSlice requires omp critical to avoid parallel reads, which may lead to the wrong slice being read
        if ( use_memory ) {
            final_image.CopyFrom(&image_stack_filtered_masked[image_counter]);
            final_image.BackwardFFT( );
        }
        else {
#pragma omp critical
            final_image.ReadSlice(&my_input_file, image_counter + 1);

            final_image.Normalize( );
            if ( outer_mask_radius != 0 ) {
                final_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                final_image.CircleMask(x_dim * 0.45);
            }
            final_image.ForwardFFT( );
            final_image.ZeroCentralPixel( );
            // to apply Gaussian filter you need to be in Fourier space
            // Nyquist frequency value is the pixel size * 2
            // if we want to apply a gaussian pass filter that will make the image at 150 angestrom to be well smoothened and get better peaks
            // It should be pixel_size/resolution limit ?? or pixel_size * 2
            final_image.GaussianLowPassFilter((pixel_size * 2) / low_pass_resolution); // use a sigma between 0-1 for best results as this will remove the high frequency information
            final_image.BackwardFFT( );
        }

        if ( use_auto_corr ) {
            final_image.Rotate2DInPlace(local_best_psi, FLT_MAX); // if not 0.0 it will not crop the images into circle after rotation as no mask will be applied
        }
        if ( use_ft ) { // here we need to add the 90 to correctly align the image for diameter determination
            final_image.Rotate2DInPlace(local_best_psi + 90.0, FLT_MAX);
        }

        all_columns_sum[image_counter] = sum_image_columns(&final_image);

        ////////////////////////////////////////////////////////////////////////////////////////////////////s/////////////////////////////////////////////////////
        ////////////// THIS PART MAY CAUSE ERRORS IN DIAMETERS CALCULATIONS IF THE MASK IS TOOO TIGHT AND REMOVED SOME OF THE TUBE EDGES/////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // calculate the required x shift to center the tubes
        auto [peak_one_column_sum, peak_two_column_sum] = findOuterTubeEdges(all_columns_sum[image_counter], min_tube_diameter, max_tube_diameter);

        // The next line not needed
        float tube_center_column_sum          = std::abs(peak_one_column_sum - peak_two_column_sum) / 2;
        float distance_from_center_column_sum = -((peak_one_column_sum + peak_two_column_sum) / 2 - center_peak_index);

        x_shift_column[image_counter] = distance_from_center_column_sum; // the x-shift needed
        // find the diameter and save it
        float tube_diameter          = std::abs((peak_one_column_sum - peak_two_column_sum));
        all_diameters[image_counter] = tube_diameter;

        // save an index for the class assignment of each image based on its diameter
        int which_bin_index_int = (tube_diameter - min_tube_diameter) / bin_range; // it will always round down so 0.9999 > 0
        // save the bin assignment so that I don't need to recalculate the diameter again outside that loop
        diameter_bins[image_counter] = which_bin_index_int;

        if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
            my_progress->Update(image_counter + 1);
    }

    delete my_progress;

    Image added_image; // This is the sum image based on the initial rotation angle calculated from auto-correlation
    wxPrintf("\nCreating Initial Sum Images...\n\n");
    ProgressBar* sum_progress = new ProgressBar(number_of_input_images);

    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        if ( use_memory ) {
            added_image.CopyFrom(&image_stack[image_counter]); // no need for counter + 1 anymore
        }
        else {
            added_image.ReadSlice(&my_input_file, image_counter + 1);
        }
        added_image.Normalize( );
        //added_image.QuickAndDirtyWriteSlice("added_image_after_normalization.mrc", image_counter +1);
        if ( input_ctf_values_from_star_file ) {
            current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
        }

        added_image.ForwardFFT( );
        added_image.ZeroCentralPixel( );

        // Any tube diameters within range will be considered
        // any diameter outside the specified range, its CTF will be disregarded to avoid wrong averaging calculations
        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            if ( diameter_bins[image_counter] == bin_index ) { // This should catch any diameter within the range
                ApplyCTFAndReturnCTFSumOfSquares(added_image, current_ctf, absolute, apply_beam_tilt, apply_envelope, (*CTFSumOfSquares)[bin_index]);
            }
        }

        added_image.BackwardFFT( );
        // Here will adjust the angles based on the method used to find tube rotation
        if ( use_auto_corr ) {
            added_image.Rotate2DInPlace(tube_rotation[image_counter], FLT_MAX);
        }
        if ( use_ft ) {
            added_image.Rotate2DInPlace(tube_rotation[image_counter] + 90.0, FLT_MAX);
        }

        added_image.PhaseShift(x_shift_column[image_counter], 0.0, 0.0);

        // using the dynamic memory allocation to add the current image to the sum images based on the tube diameter
        // Any tube diameters within range will be added
        // any diameter outside the specified range will be disregarded when generating the added image to avoid wrong averaging
        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            if ( diameter_bins[image_counter] == bin_index ) { // This should catch any diameter within the range
                sum_images[bin_index].AddImage(&added_image);
            }
        }

        if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
            sum_progress->Update(image_counter + 1);
    }

    delete sum_progress;

    // divide the sum image by CTF sum of squares and centering it
    for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
        // Vertically summing the image to ensure cross-correlation doesn't correlate by mistake to a wrong area if input images are pre-aligned
        sum_image_direction(&sum_images[bin_index], 2);
        sum_images[bin_index].ForwardFFT( );
        divide_by_ctf_sum_of_squares(sum_images[bin_index], (*CTFSumOfSquares)[bin_index]);
        sum_images[bin_index].BackwardFFT( );
        // shift sum image to the center after padding based on tube peaks
        // This shift is necessary at this point to ensure the cross-correlation shift is centered correctly later
        std::vector<float> column_sum                   = sum_image_columns(&sum_images[bin_index]);
        auto [peak_one_column_sum, peak_two_column_sum] = findOuterTubeEdges(column_sum, min_tube_diameter, max_tube_diameter);

        float tube_center_column_sum          = std::abs(peak_one_column_sum - peak_two_column_sum) / 2;
        float distance_from_center_column_sum = -((peak_one_column_sum + peak_two_column_sum) / 2 - center_peak_index);
        //to center the sum image
        sum_images[bin_index].PhaseShift(-distance_from_center_column_sum, 0.0, 0.0); // the x-shift needed to center the sum image
    }

    delete CTFSumOfSquares;

    // create average_images that are CTF corrected and then average_image copies directly from here
    Image* average_images;
    average_images = new Image[number_of_input_images];
    Image temporary_average_image;
    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            if ( diameter_bins[image_counter] == bin_index ) { // This should catch any diameter within the range
                temporary_average_image.CopyFrom(&sum_images[bin_index]);
            }
            else if ( diameter_bins[image_counter] < 0 ) {
                temporary_average_image.CopyFrom(&sum_images[0]);
            }
            else if ( diameter_bins[image_counter] >= bins_count ) {
                temporary_average_image.CopyFrom(&sum_images[bins_count - 1]);
            }
        }
        temporary_average_image.Normalize( );
        temporary_average_image.ForwardFFT( );
        if ( input_ctf_values_from_star_file ) {
            current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
        }
        temporary_average_image.ApplyCTF(current_ctf);
        temporary_average_image.BackwardFFT( );
        average_images[image_counter].CopyFrom(&temporary_average_image);
        //temporary_average_image.QuickAndDirtyWriteSlice("average_image_with_ctf.mrc", image_counter + 1);
    }

    /// Getting the correct rotation and shift using cross-correlation
    float inner_radius_for_peak_search;
    float outer_radius_for_peak_search;
    inner_radius_for_peak_search = 0.0; // inner radius should be set to 0
    outer_radius_for_peak_search = x_dim / 2;

    std::vector<float> best_correlation_score(number_of_input_images, -FLT_MAX);
    std::vector<float> best_psi_value(number_of_input_images, 0.0f);
    std::vector<float> best_x_shift_value(number_of_input_images, 0.0f);
    std::vector<float> best_y_shift_value(number_of_input_images, 0.0f);

    Image my_image;
    Image my_image_copy;
    Image my_image_tuned;
    Image average_image;
    Image tuning_average_image;
    float tuned_rotation_range = psi_step; ///2
    float tuned_step_size      = fine_tuning_psi_step;
    Image fine_tuning_average_image;

    wxPrintf("\nAligning Images...\n\n");
    ProgressBar* my_aln_progress = new ProgressBar(number_of_input_images);

#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads) default(none) shared(number_of_input_images, my_input_file, inner_radius_for_peak_search, outer_radius_for_peak_search, low_pass_resolution, x_dim, y_dim, use_memory, average_images,                                  \
                                                                                            best_correlation_score, best_psi_value, best_x_shift_value, best_y_shift_value, psi_step, tube_rotation, outer_mask_radius, use_auto_corr, use_ft, image_stack_filtered_masked, results,           \
                                                                                            max_threads, diameter_bins, sum_images, bins_count, tuned_rotation_range, tuned_step_size, my_aln_progress, x_shift_column, y_shift_row, all_diameters, bin_range, current_image, all_columns_sum, \
                                                                                            input_ctf_values_from_star_file, current_ctf, ctf_parameters_stack, min_tube_diameter, max_tube_diameter, center_peak_index, pixel_size, low_pass) private(image_counter, my_image, average_image, tuning_average_image, final_image, fine_tuning_average_image, my_image_copy, my_image_tuned)

    for ( long aln_image_counter = 0; aln_image_counter < number_of_input_images; aln_image_counter++ ) {

        if ( use_memory ) {
            my_image.CopyFrom(&image_stack_filtered_masked[aln_image_counter]);
        }
        else {
#pragma omp critical
            my_image.ReadSlice(&my_input_file, aln_image_counter + 1);

            my_image.Normalize( );

            if ( outer_mask_radius != 0 ) {
                my_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                my_image.CircleMask(x_dim * 0.45);
            }
            my_image.ForwardFFT( );
            my_image.ZeroCentralPixel( );
            //testing adding a low pass filter on the original image before getting the correct shift from the correlation and how that can affect the centering of the mask at the end
            if ( low_pass ) {
                my_image.GaussianLowPassFilter((pixel_size * 2) / low_pass_resolution); //150
            }
        }
        my_image.BackwardFFT( );
        //my_image.QuickAndDirtyWriteSlice("low_pass_filtered_image_for_comparison.mrc", aln_image_counter + 1);

        // initial angle search will start from the rotation angle we got from the auto-correlation
        // then change the auto-correlation angle to be within 180
        // do another search within +/- 90 degrees of the auto-correlation psi angle or the FT psi angle (+90)
        // only at this point we need to adjust the angle before cross-correlation but later it will be already correct and no further adjustments
        float local_best_corr_score = -FLT_MAX;
        float local_best_psi        = 0.0f;
        float local_best_x_shift    = 0.0f;
        float local_best_y_shift    = 0.0f;

        float current_best_psi;
        if ( use_auto_corr ) {
            current_best_psi = tube_rotation[aln_image_counter];
        }
        if ( use_ft ) {
            current_best_psi = tube_rotation[aln_image_counter] + 90.0;
        }

        float angle_range   = 180.0;
        float psi_min_angle = current_best_psi - 0.5 * angle_range;
        float psi_max_angle = current_best_psi + 0.5 * angle_range;

        for ( float psi = psi_min_angle; psi < psi_max_angle; psi += psi_step ) {
            // create a new peak to save the cross-correlation peak values
            Peak current_peak;
            average_image.CopyFrom(&average_images[aln_image_counter]);

            //will rotate the original image to be aligned with the sum image to facilitate the shift calculations
            my_image_copy.Allocate(x_dim, y_dim, true);
            my_image_copy.CopyFrom(&my_image);
            my_image_copy.Rotate2DInPlace(psi, FLT_MAX);
            // // make directional sum to eliminate any vertical signal
            // sum_image_direction(&my_image_copy, 2);
            // // calculate the cross correlation of the reference image with the rotated image
            average_image.CalculateCrossCorrelationImageWith(&my_image_copy);

            if ( outer_mask_radius != 0 ) {
                average_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                average_image.CircleMask(x_dim * 0.45);
            }

            // find the peak from the cross corrlation to get the values
            current_peak = average_image.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

            if ( current_peak.value > local_best_corr_score ) {
                local_best_corr_score = current_peak.value;
                local_best_psi        = psi;
                local_best_x_shift    = current_peak.x;
                local_best_y_shift    = current_peak.y;
            }
        }
        // end of initial search for the best psi and shift angles
        // Start the tuning loop
        // save the best_psi as the current_best_psi for that image
        // calculate the tuning psi range which is = to the rotation angle and step size = rotation angle / 20
        current_best_psi = local_best_psi;
        //current_best_psi            = best_psi_value[aln_image_counter];
        float tuned_psi_lower_range = current_best_psi - tuned_rotation_range;
        float tuned_psi_upper_range = current_best_psi + tuned_rotation_range;

        // loop over the range of +/- half the rotation angle
        // increment by 1/10 of the tuned rotation angle (rotation angle/2)/10 degrees for tuning
        for ( float tuned_psi = tuned_psi_lower_range; tuned_psi <= tuned_psi_upper_range; tuned_psi += tuned_step_size ) {

            // create a new peak to save the tuned values
            Peak current_tuned_peak;
            tuning_average_image.CopyFrom(&average_images[aln_image_counter]);
            my_image_tuned.Allocate(x_dim, y_dim, true);
            my_image_tuned.CopyFrom(&my_image);
            my_image_tuned.Rotate2DInPlace(tuned_psi, FLT_MAX);
            // // make directional sum to eliminate any vertical signal
            // sum_image_direction(&my_image_tuned, 2);

            // calculate the cross correlation of the reference image with the rotated image
            tuning_average_image.CalculateCrossCorrelationImageWith(&my_image_tuned); //rotated_image
            // Added this as usually tubes are centered so to avoid any extra shift after aligining
            // especially if using a gaussian filter

            if ( outer_mask_radius != 0 ) {
                tuning_average_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                tuning_average_image.CircleMask(x_dim * 0.45);
            }
            // find the peak from the cross corrlation to get the values from the tuning_average_image
            current_tuned_peak = tuning_average_image.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

            if ( current_tuned_peak.value > local_best_corr_score ) {
                local_best_corr_score = current_tuned_peak.value;
                local_best_psi        = tuned_psi;
                local_best_x_shift    = current_tuned_peak.x;
                local_best_y_shift    = current_tuned_peak.y;
            }
        }

        best_correlation_score[aln_image_counter] = local_best_corr_score;
        best_psi_value[aln_image_counter]         = local_best_psi;
        best_x_shift_value[aln_image_counter]     = local_best_x_shift;
        best_y_shift_value[aln_image_counter]     = local_best_y_shift;

        // Updating the tube diameters
        final_image.Allocate(x_dim, y_dim, true);
        final_image.SetToConstant(0.0);

        if ( use_memory ) {
            final_image.CopyFrom(&image_stack_filtered_masked[aln_image_counter]);
            final_image.BackwardFFT( );
        }
        else {
#pragma omp critical
            final_image.ReadSlice(&my_input_file, aln_image_counter + 1);
            final_image.Normalize( );

            if ( outer_mask_radius != 0 ) {
                final_image.CircleMask(outer_mask_radius);
            }
            else if ( outer_mask_radius == 0 ) {
                final_image.CircleMask(x_dim * 0.45);
            }
            final_image.ForwardFFT( );
            final_image.ZeroCentralPixel( );
            // to apply Gaussian filter you need to be in Fourier space
            // Nyquist frequency value is the pixel size * 2
            // if we want to apply a gaussian pass filter that will make the image at 150 angestrom to be well smoothened and get better peaks
            // It should be pixel_size/resolution limit ?? or pixel_size * 2
            final_image.GaussianLowPassFilter((pixel_size * 2) / low_pass_resolution); // use a sigma between 0-1 for best results as this will remove the high frequency information
            final_image.BackwardFFT( );
        }
        // removed the -psi from here as I want to rotate the image to be aligned with Y-axis as the average image to get the correct x-shift
        final_image.Rotate2DInPlace(best_psi_value[aln_image_counter], FLT_MAX);
        final_image.PhaseShift(best_x_shift_value[aln_image_counter], 0.0);
        // find the outer edges peaks
        all_columns_sum[aln_image_counter]              = sum_image_columns(&final_image);
        auto [peak_one_column_sum, peak_two_column_sum] = findOuterTubeEdges(all_columns_sum[aln_image_counter], min_tube_diameter, max_tube_diameter);
        // save the peaks to an output file later
        results[aln_image_counter] = {aln_image_counter, {peak_one_column_sum, peak_two_column_sum}};

        final_image.Deallocate( );
        my_image_copy.Deallocate( );
        my_image_tuned.Deallocate( );
        // find the diameter and save it
        float tube_diameter              = std::abs((peak_one_column_sum - peak_two_column_sum)); // * pixel_size
        all_diameters[aln_image_counter] = tube_diameter;

        // Update the index for the class assignment of each image based on its diameter
        int which_bin_index_int = (tube_diameter - min_tube_diameter) / bin_range; // it will always round down so 0.9999 > 0
        // save the bin assignment so that I don't need to recalculate the diameter again outside that loop
        diameter_bins[aln_image_counter] = which_bin_index_int;

        if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
            my_aln_progress->Update(aln_image_counter + 1);
    }

    delete my_aln_progress;
    delete[] average_images;
    final_image.Deallocate( );

    // Check if the all diameters file is open
    if ( file.is_open( ) ) {
        for ( size_t i = 0; i < all_diameters.size( ); ++i ) {
            file << all_diameters[i] << '\n';
        }
        file.close( );
    }

    //save the peaks to the output file
    if ( peak_file.is_open( ) ) {
        for ( auto& r : results ) {
            peak_file << r.first << ", " << r.second.first << ", " << r.second.second << "\n";
        }
        peak_file.close( );
    }

    save_all_columns_sum_to_file(all_columns_sum, "column_sums_output.txt");

    Image added_image_after_aln;
    Image sum_images_after_aln[bins_count];

    wxPrintf("\nCreating Final Sum Images...\n\n");
    ProgressBar* update_sum_progress = new ProgressBar(number_of_input_images);

    for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
        sum_images_after_aln[bin_index].Allocate(x_dim, y_dim, true); //allocate in real space
        sum_images_after_aln[bin_index].SetToConstant(0.0);
    }

    // creating a sum image after getting the best rotation from the alignment
    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        added_image_after_aln.ReadSlice(&my_input_file, image_counter + 1);
        added_image_after_aln.Normalize( );
        //added_image.QuickAndDirtyWriteSlice("added_image_after_normalization.mrc", image_counter +1);
        if ( input_ctf_values_from_star_file ) {
            current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
        }
        added_image_after_aln.ForwardFFT( );
        added_image_after_aln.ZeroCentralPixel( );

        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            if ( diameter_bins[image_counter] == bin_index ) { // This should catch any diameter within the range
                ApplyCTFAndReturnCTFSumOfSquares(added_image_after_aln, current_ctf, absolute, apply_beam_tilt, apply_envelope, (*CTFSumOfSquaresNew)[bin_index]);
            }
        }

        added_image_after_aln.BackwardFFT( );
        //changed this to +psi as I am now rotating the image to the azimuthal average reference
        added_image_after_aln.Rotate2DInPlace(best_psi_value[image_counter], FLT_MAX);
        // update the x_shift column to get the shift that is needed to center the tube after rotation
        // I removed the Y-shift as now the image was rotated to the reference and only X-shift is important
        added_image_after_aln.PhaseShift(best_x_shift_value[image_counter], 0.0, 0.0);
        // using the dynamic memory allocation to add the current image to the sum images based on the tube diameter
        // only images within the specified range will be added and their CTF will be saved to be used later
        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            if ( diameter_bins[image_counter] == bin_index ) { // This should catch any diameter within the range
                sum_images_after_aln[bin_index].AddImage(&added_image_after_aln);
            }
        }
        if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
            update_sum_progress->Update(image_counter + 1);
    }

    delete update_sum_progress;

    for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
        sum_images_after_aln[bin_index].ForwardFFT( );
        divide_by_ctf_sum_of_squares(sum_images_after_aln[bin_index], (*CTFSumOfSquaresNew)[bin_index]);
        sum_images_after_aln[bin_index].BackwardFFT( );
        //sum_images_after_aln[bin_index].QuickAndDirtyWriteSlice("final_sum_image_before_averaging.mrc", bin_index + 1);
    }
    delete CTFSumOfSquaresNew;
    sum_images_after_aln->Deallocate( );

    // rotationally averaged 3D reconstruction
    Image               model_volume[bins_count];
    ReconstructedVolume input_3d[bins_count];
    Image               projection_volume_3d[bins_count];
    Image               projection_volume_image;
    Image               padded_projection_volume_image;
    AnglesAndShifts     my_parameters;

    // generate masked 3D AA models
    Image               my_masked_volume[bins_count];
    Image               my_mask;
    ReconstructedVolume masked_3d[bins_count];
    Image               masked_projection_volume_3d[bins_count];
    Image               masked_projection_volume_image;
    Image               masked_padded_projection_volume_image;

    for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
        projection_volume_3d[bin_index].Allocate(x_dim * padding_factor, y_dim * padding_factor, x_dim * padding_factor, true); //allocate in real space
        projection_volume_3d[bin_index].SetToConstant(0.0);
        if ( RASTR == true ) {
            masked_projection_volume_3d[bin_index].Allocate(x_dim * padding_factor, y_dim * padding_factor, x_dim * padding_factor, true); //allocate in real space
            masked_projection_volume_3d[bin_index].SetToConstant(0.0);
        }
    }
    // mask needed for masking upweighted regions
    // create the mask
    Image my_white_mask;

    ReconstructedVolume mask_volume;
    Image*              mask_projection;

    if ( RASTR == true ) {
        //First prepare a black mask that will be multiplied by the azimuthal average to generate the masked volume
        // i.e masked volume (will be 0 inside the mask area)
        // Allocate memory for the mask file to be read
        my_mask.Allocate(x_dim, y_dim, x_dim, true);

        if ( input_mask == true ) {
            //#pragma omp critical
            my_mask.ReadSlices(my_mask_file, 1, my_mask_file->ReturnNumberOfSlices( ));
            my_mask.BinariseInverse(0.0f); // any pixel of 0.0 or less will be 1.0
        }
        else {
            my_mask.SetToConstant(0.0);
            create_black_sphere_mask(&my_mask, x_mask_center, y_mask_center, z_mask_center, sphere_mask_radius);
            //my_mask[bin_index].QuickAndDirtyWriteSlices("my_created_black_mask.mrc", 1, x_dim);
            //smoothen the mask
            my_mask.ForwardFFT( );
            my_mask.GaussianLowPassFilter((pixel_size * 2) / 150);
            my_mask.BackwardFFT( );
        }
        // create a padded mask as the model is padded now
        my_mask.Resize(padding_factor * x_dim, padding_factor * y_dim, padding_factor * x_dim, 1.0);
        //my_mask[bin_index].QuickAndDirtyWriteSlices("my_created_black_mask_after_resizing.mrc", 1, padding_factor * x_dim);

        // create the mask for masking the upweighted regions later after subtraction
        // it should be created from the previous mask or using the same values
        // Allocate memory for the mask file to be read
        my_white_mask.Allocate(x_dim, y_dim, x_dim, true);

        if ( input_mask == true ) {
            my_white_mask.ReadSlices(my_mask_file, 1, my_mask_file->ReturnNumberOfSlices( ));
            my_white_mask.Resize(padding_factor * my_mask_file->ReturnXSize( ), padding_factor * my_mask_file->ReturnYSize( ), padding_factor * my_mask_file->ReturnZSize( ));
            my_white_mask.BinariseInverse(0.0f); //any pixel of 0.0 or less will be 1.0
            my_white_mask.BinariseInverse(0.0f); // now we flip them again so the mask itself is 1.0 and background is 0.0
            delete my_mask_file;
        }
        else {
            my_white_mask.SetToConstant(0.0);
            create_white_sphere_mask(&my_white_mask, x_mask_center, y_mask_center, z_mask_center, sphere_mask_radius);
            // smoothen the mask
            my_white_mask.ForwardFFT( );
            my_white_mask.GaussianLowPassFilter((pixel_size * 2) / 150);
            my_white_mask.BackwardFFT( );
            my_white_mask.Resize(padding_factor * x_dim, padding_factor * y_dim, padding_factor * x_dim);
            //my_white_mask.QuickAndDirtyWriteSlices("my_white_padded_mask.mrc", 1, padding_factor * x_dim);
        }

        mask_volume.InitWithDimensions(my_white_mask.logical_x_dimension, my_white_mask.logical_y_dimension, my_white_mask.logical_z_dimension, pixel_size);
        mask_volume.density_map->CopyFrom(&my_white_mask);
        float mask_radius       = FLT_MAX; //100 - FLT_MAX
        mask_volume.mask_radius = mask_radius;
        mask_volume.PrepareForProjections(0.0, 2.0 * pixel_size);
        mask_projection = new Image;
        mask_projection->CopyFrom(mask_volume.density_map);
        my_white_mask.Deallocate( );
    }

    // Find the position of the dot if there is an extension in the filename provided by the user
    size_t extension_pos = output_azimuthal_average_volume_filename.find('.');
    if ( extension_pos != std::string::npos ) { // If dot is found
        // Extract substring before the dot
        output_azimuthal_average_volume_filename = output_azimuthal_average_volume_filename.substr(0, extension_pos);
    }

    std::string model_file_name = "";
    // // create a copy of the sum images to be used to create the azimuthal average model
    // Image        sum_images_copy[bins_count];
    long     model_dimension = x_dim;
    MRCFile* output_model_filename;
    if ( ! my_output_sum_image_filename.IsOpen( ) ) {
        my_output_sum_image_filename.OpenFile(output_average_per_bin_filename.ToStdString( ), true);
        if ( ! my_output_sum_image_filename.IsOpen( ) ) {
            wxPrintf("ERROR: Could not open '%s' for writing\n", output_average_per_bin_filename.ToStdString( ).c_str( ));
            DEBUG_ABORT;
        }
    }
    if ( SPOT_RASTR == true ) {
        number_of_models = 1;
    }

    my_output_sum_image_filename.my_header.SetNumberOfImages(bins_count * number_of_models);
    my_output_sum_image_filename.my_header.SetDimensionsImage(x_dim, y_dim);
    my_output_sum_image_filename.SetPixelSize(pixel_size);
    my_output_sum_image_filename.WriteHeader( );
    my_output_sum_image_filename.rewrite_header_on_close = true;

    wxPrintf("\nPreparing Azimuthal Average model for projection...\n\n");
    ProgressBar* prepare_projections_progress = new ProgressBar(bins_count);
    // save the azimuthal average if the RASTR and SPOT RASTR are not true in the correct contrast
    if ( RASTR == false && SPOT_RASTR == false ) {
        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
            model_volume[bin_index].Allocate(padding_factor * sum_images[bin_index].logical_x_dimension, padding_factor * sum_images[bin_index].logical_y_dimension, padding_factor * sum_images[bin_index].logical_x_dimension, true);
            model_volume[bin_index].SetToConstant(0.0);

            float edge_value = sum_images[bin_index].ReturnAverageOfRealValuesOnEdges( );
            sum_images[bin_index].Resize(model_volume[bin_index].logical_x_dimension, model_volume[bin_index].logical_y_dimension, 1, edge_value);

            // fill the padded version with the sum image
            // then do average rotationally before filling the volume
            sum_image_direction(&sum_images[bin_index], 2);
            sum_images[bin_index].ApplyRampFilter( );
            sum_images[bin_index].AverageRotationally( );
            sum_images[bin_index].InvertRealValues( );

            // fill in the model volume with the azimuthal average slice
            long pixel_coord_xy  = 0;
            long pixel_coord_xyz = 0;
            long volume_counter  = 0;
            for ( int z = 0; z < model_volume[bin_index].logical_z_dimension; z++ ) {
                for ( int y = 0; y < model_volume[bin_index].logical_y_dimension; y++ ) {
                    for ( int x = 0; x < model_volume[bin_index].logical_x_dimension; x++ ) {
                        pixel_coord_xy                                      = sum_images[bin_index].ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                        model_volume[bin_index].real_values[volume_counter] = sum_images[bin_index].real_values[pixel_coord_xy];
                        volume_counter++;
                    }
                    volume_counter += sum_images[bin_index].padding_jump_value;
                }
            }
            //resize the sum_images back to be unpadded
            sum_images[bin_index].Resize(x_dim, y_dim, 1, edge_value);

            // I need to move this to another location to ensure saving the azimuthal averages volume with correct contrast (I need to invert real values)
            // This can't happen here it needs to be later when sum_images will not be used anymore
            // save the azimuthal average model to an MRC file
            model_file_name       = output_azimuthal_average_volume_filename + "_" + std::to_string(bin_index + 1) + ".mrc";
            output_model_filename = new MRCFile(model_file_name, true, true);
            for ( long model_counter = 0; model_counter < model_dimension; model_counter++ ) {
                sum_images[bin_index].WriteSlice(output_model_filename, model_counter + 1);
            }
            // output_model_filename->my_header.SetDimensionsVolume(model_dimension, model_dimension, model_dimension);
            // output_model_filename->my_header.SetPixelSize(pixel_size);
            output_model_filename->WriteHeader( );
            delete output_model_filename;
            // here we are copying from the padded volume
            input_3d[bin_index].InitWithDimensions(model_volume[bin_index].logical_x_dimension, model_volume[bin_index].logical_y_dimension, model_volume[bin_index].logical_z_dimension, pixel_size);
            input_3d[bin_index].density_map->CopyFrom(&model_volume[bin_index]);
            float mask_radius               = FLT_MAX; //100 - FLT_MAX
            input_3d[bin_index].mask_radius = mask_radius;
            input_3d[bin_index].PrepareForProjections(0.0, 2.0 * pixel_size); // 0.0, 2.0 * pixel_size float low resolution limit and high resolution limit, bool approximate bining = F and apply_bining = T

            projection_volume_3d[bin_index].CopyFrom(input_3d[bin_index].density_map);
            // deallocate the reconstruction volume
            input_3d[bin_index].Deallocate( );

            projection_volume_image.Allocate(x_dim, y_dim, true);
            padded_projection_volume_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding

            my_parameters.Init(90.0, 90.0, 90.0, 0.0, 0.0);
            projection_volume_3d[bin_index].ExtractSlice(padded_projection_volume_image, my_parameters); //
            padded_projection_volume_image.SwapRealSpaceQuadrants( ); // must do this step as image is not centered in the box
            padded_projection_volume_image.BackwardFFT( );
            padded_projection_volume_image.object_is_centred_in_box = true;
            padded_projection_volume_image.ClipInto(&projection_volume_image);
            if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                my_output_sum_image_filename.OpenFile(output_average_per_bin_filename.ToStdString( ), true);
                if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                    wxPrintf("ERROR: Could not open '%s' for writing\n", output_average_per_bin_filename.ToStdString( ).c_str( ));
                    DEBUG_ABORT;
                }
            }

#pragma omp critical
            projection_volume_image.WriteSlice(&my_output_sum_image_filename, bin_index + 1);
            projection_volume_image.Deallocate( );
            padded_projection_volume_image.Deallocate( );
            prepare_projections_progress->Update(bin_index + 1);
        }
        delete prepare_projections_progress;

        float adjusted_x_shifts[number_of_input_images];

        cisTEMParameters image_output_params;

        image_output_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);

        image_output_params.PreallocateMemoryAndBlank(number_of_input_images);
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            // calculate the adjusted x shift of images based on their 3d projection into 2d
            // // generate the full rotation matrix
            RotationMatrix temp_matrix;
            float          rotated_x, rotated_y, rotated_z;
            temp_matrix.SetToEulerRotation(0.0, 90.0, (90.0 - best_psi_value[image_counter]));
            // assuming no Y shift will be applied to ensure everything is centered
            temp_matrix.RotateCoords((best_x_shift_value[image_counter]), 0.0, 0.0, rotated_x, rotated_y, rotated_z);
            // correct the shift happenning because of extractslice - Note x and y shift needs to be flipped
            // since projection is centered so negative rotation and shift is needed here
            adjusted_x_shifts[image_counter] = -rotated_y;

            image_output_params.all_parameters[image_counter].position_in_stack                  = image_counter + 1;
            image_output_params.all_parameters[image_counter].psi                                = 90.0 - best_psi_value[image_counter]; // -(best_psi_value[image_counter] - 90.0)  i.e. -rotation + 90 This is the angle that when the tube is rotated by it will align it to 90.0 degrees Psi
            image_output_params.all_parameters[image_counter].theta                              = 90.0f;
            image_output_params.all_parameters[image_counter].phi                                = 0;
            image_output_params.all_parameters[image_counter].x_shift                            = adjusted_x_shifts[image_counter]; // This the shift that will center the tube after rotating it to have 90 degree rotation
            image_output_params.all_parameters[image_counter].y_shift                            = 0.0;
            image_output_params.all_parameters[image_counter].defocus_1                          = ctf_parameters_stack[image_counter].defocus_1;
            image_output_params.all_parameters[image_counter].defocus_2                          = ctf_parameters_stack[image_counter].defocus_2;
            image_output_params.all_parameters[image_counter].defocus_angle                      = ctf_parameters_stack[image_counter].astigmatism_angle;
            image_output_params.all_parameters[image_counter].phase_shift                        = ctf_parameters_stack[image_counter].additional_phase_shift;
            image_output_params.all_parameters[image_counter].image_is_active                    = 1;
            image_output_params.all_parameters[image_counter].occupancy                          = 100.0f;
            image_output_params.all_parameters[image_counter].logp                               = -1000.0f;
            image_output_params.all_parameters[image_counter].sigma                              = 10.0f;
            image_output_params.all_parameters[image_counter].pixel_size                         = pixel_size;
            image_output_params.all_parameters[image_counter].microscope_voltage_kv              = ctf_parameters_stack[image_counter].acceleration_voltage;
            image_output_params.all_parameters[image_counter].microscope_spherical_aberration_mm = ctf_parameters_stack[image_counter].spherical_aberration;
            image_output_params.all_parameters[image_counter].amplitude_contrast                 = ctf_parameters_stack[image_counter].amplitude_contrast;
            image_output_params.all_parameters[image_counter].beam_tilt_x                        = 0.0f;
            image_output_params.all_parameters[image_counter].beam_tilt_y                        = 0.0f;
            image_output_params.all_parameters[image_counter].image_shift_x                      = 0.0f;
            image_output_params.all_parameters[image_counter].image_shift_y                      = 0.0f;
        }
        image_output_params.WriteTocisTEMStarFile("updated_rotation_parameters.star");
    }
    else {
        // RUN THIS ON SINGLE THREAD IS BETTER
//Image azimuthal_average_slice;
#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min(bins_count, max_threads)) default(none) shared(SPOT_RASTR, RASTR, prepare_projections_progress, current_image, bins_count, sum_images, model_volume, my_masked_volume, my_mask, input_3d, masked_3d, my_output_sum_image_filename,                   \
                                                                                                                  pixel_size, padding_factor, x_mask_center, y_mask_center, z_mask_center, sphere_mask_radius, filter_radius, use_outside_value, x_dim, y_dim,                                                          \
                                                                                                                  outside_value, outside_weight, cosine_edge, number_of_models, input_mask, model_file_name, output_average_per_bin_filename,                                                                           \
                                                                                                                  output_azimuthal_average_volume_filename, model_dimension, masked_projection_volume_3d, projection_volume_3d) private(projection_volume_image, padded_projection_volume_image, output_model_filename, \
                                                                                                                                                                                                                                        masked_projection_volume_image, masked_padded_projection_volume_image, my_white_mask, mask_volume, mask_projection, my_parameters)

        for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {

            model_volume[bin_index].Allocate(padding_factor * sum_images[bin_index].logical_x_dimension, padding_factor * sum_images[bin_index].logical_y_dimension, padding_factor * sum_images[bin_index].logical_x_dimension, true);
            model_volume[bin_index].SetToConstant(0.0);

            float edge_value = sum_images[bin_index].ReturnAverageOfRealValuesOnEdges( );
            sum_images[bin_index].Resize(model_volume[bin_index].logical_x_dimension, model_volume[bin_index].logical_y_dimension, 1, edge_value);

            // fill the padded version with the sum image
            // then do average rotationally before filling the volume
            sum_image_direction(&sum_images[bin_index], 2);
            sum_images[bin_index].ApplyRampFilter( );
            sum_images[bin_index].AverageRotationally( );

            // fill in the model volume with the azimuthal average slice
            long pixel_coord_xy  = 0;
            long pixel_coord_xyz = 0;
            long volume_counter  = 0;
            for ( int z = 0; z < model_volume[bin_index].logical_z_dimension; z++ ) {
                for ( int y = 0; y < model_volume[bin_index].logical_y_dimension; y++ ) {
                    for ( int x = 0; x < model_volume[bin_index].logical_x_dimension; x++ ) {
                        pixel_coord_xy                                      = sum_images[bin_index].ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                        model_volume[bin_index].real_values[volume_counter] = sum_images[bin_index].real_values[pixel_coord_xy];
                        volume_counter++;
                    }
                    volume_counter += sum_images[bin_index].padding_jump_value;
                }
            }

            //Invert contrast and resize the sum_images back to be unpadded
            sum_images[bin_index].InvertRealValues( );
            sum_images[bin_index].Resize(x_dim, y_dim, 1, edge_value);
            // save the azimuthal average model to an MRC file
            model_file_name       = output_azimuthal_average_volume_filename + "_" + std::to_string(bin_index + 1) + ".mrc";
            output_model_filename = new MRCFile(model_file_name, true, true);
            for ( long model_counter = 0; model_counter < model_dimension; model_counter++ ) {
                sum_images[bin_index].WriteSlice(output_model_filename, model_counter + 1);
            }
            // output_model_filename->my_header.SetDimensionsVolume(model_dimension, model_dimension, model_dimension);
            // output_model_filename->my_header.SetPixelSize(pixel_size);
            output_model_filename->WriteHeader( );
            delete output_model_filename;
            //}

            // if we will not apply mask and will subtract the projection of the azimuthal average as is
            if ( SPOT_RASTR == true ) {

                input_3d[bin_index].InitWithDimensions(model_volume[bin_index].logical_x_dimension, model_volume[bin_index].logical_y_dimension, model_volume[bin_index].logical_z_dimension, pixel_size);
                input_3d[bin_index].density_map->CopyFrom(&model_volume[bin_index]);
                float mask_radius               = FLT_MAX; //100 - FLT_MAX
                input_3d[bin_index].mask_radius = mask_radius;
                input_3d[bin_index].PrepareForProjections(0.0, 2.0 * pixel_size); // 0.0, 2.0 * pixel_size float low resolution limit and high resolution limit, bool approximate bining = F and apply_bining = T

                projection_volume_3d[bin_index].CopyFrom(input_3d[bin_index].density_map);
                // deallocate the reconstruction volume
                input_3d[bin_index].Deallocate( );

                projection_volume_image.Allocate(x_dim, y_dim, true);
                padded_projection_volume_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding

                my_parameters.Init(0.0, 90.0, 90.0, 0.0, 0.0);
                projection_volume_3d[bin_index].ExtractSlice(padded_projection_volume_image, my_parameters); //
                padded_projection_volume_image.SwapRealSpaceQuadrants( ); // must do this step as image is not centered in the box
                padded_projection_volume_image.BackwardFFT( );
                padded_projection_volume_image.object_is_centred_in_box = true;
                padded_projection_volume_image.ClipInto(&projection_volume_image);
                if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                    my_output_sum_image_filename.OpenFile(output_average_per_bin_filename.ToStdString( ), true);
                    if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                        wxPrintf("ERROR: Could not open '%s' for writing\n", output_average_per_bin_filename.ToStdString( ).c_str( ));
                        DEBUG_ABORT;
                    }
                }

                //invert contrast before writing to restore correct contrast as CTF correction inverted the contrast of the images
                projection_volume_image.InvertRealValues( );

#pragma omp critical
                projection_volume_image.WriteSlice(&my_output_sum_image_filename, bin_index + 1);
                projection_volume_image.Deallocate( );
                padded_projection_volume_image.Deallocate( );
            }

            if ( RASTR == true ) { // if a mask is applied and masked model is what will be projected and subtracted

                // prepare an unmasked azimuthal average volume to be used for getting the correct scaling factor
                // I will not save any projections from this generated volume, only the masked one to ensure they are masked correctly
                input_3d[bin_index].InitWithDimensions(model_volume[bin_index].logical_x_dimension, model_volume[bin_index].logical_y_dimension, model_volume[bin_index].logical_z_dimension, pixel_size);
                input_3d[bin_index].density_map->CopyFrom(&model_volume[bin_index]);
                float mask_radius               = FLT_MAX; //100 - FLT_MAX
                input_3d[bin_index].mask_radius = mask_radius;

                input_3d[bin_index].PrepareForProjections(0.0, 2.0 * pixel_size);
                projection_volume_3d[bin_index].CopyFrom(input_3d[bin_index].density_map);
                //deallocate the reconstruction volume
                input_3d[bin_index].Deallocate( );
                // prepare the masked azimuthal average volumes at different Phi angles
                my_masked_volume[bin_index].CopyFrom(&model_volume[bin_index]);

                float filter_edge = 40.0;
                float mask_volume_in_voxels;

                //wxPrintf("\nMasking Volume...\n");

                if ( ! model_volume[bin_index].HasSameDimensionsAs(&my_mask) ) {
                    wxPrintf("\nVolume and mask file have different dimensions\n");
                    DEBUG_ABORT;
                }
                if ( filter_radius == 0.0 )
                    filter_radius = pixel_size;
                mask_volume_in_voxels = my_masked_volume[bin_index].ApplyMask(my_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);

                //save the masked azimuthal average
                //my_masked_volume[bin_index].QuickAndDirtyWriteSlices("masked_azimuthal_average.mrc", 1, my_masked_volume[bin_index].logical_z_dimension);

                // create the density map to initiate the projections
                // pad 3D masked volume
                masked_3d[bin_index].InitWithDimensions(my_masked_volume[bin_index].logical_x_dimension, my_masked_volume[bin_index].logical_y_dimension, my_masked_volume[bin_index].logical_z_dimension, pixel_size);
                masked_3d[bin_index].density_map->CopyFrom(&my_masked_volume[bin_index]);
                mask_radius                      = FLT_MAX; //100 - FLT_MAX
                masked_3d[bin_index].mask_radius = mask_radius;
                masked_3d[bin_index].PrepareForProjections(0.0, 2.0 * pixel_size);
                masked_projection_volume_3d[bin_index].CopyFrom(masked_3d[bin_index].density_map);

                // deallocate the reconstruction volume
                masked_3d[bin_index].Deallocate( );

                float phi;
                for ( long model_counter = 0; model_counter < number_of_models; model_counter++ ) {
                    //Allocate memory for the masked and padded masked projections
                    masked_projection_volume_image.Allocate(x_dim, y_dim, true);
                    masked_padded_projection_volume_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding
                    //calculate the phi angle
                    phi = model_counter * 360.0 / number_of_models;
                    my_parameters.Init(phi, 90.0, 90.0, 0.0, 0.0);
                    masked_projection_volume_3d[bin_index].ExtractSlice(masked_padded_projection_volume_image, my_parameters); //
                    masked_padded_projection_volume_image.SwapRealSpaceQuadrants( ); // must do this step as image is not centered in the box
                    masked_padded_projection_volume_image.BackwardFFT( );
                    masked_padded_projection_volume_image.object_is_centred_in_box = true;
                    masked_padded_projection_volume_image.ClipInto(&masked_projection_volume_image);
                    if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                        my_output_sum_image_filename.OpenFile(output_average_per_bin_filename.ToStdString( ), true);
                        if ( ! my_output_sum_image_filename.IsOpen( ) ) {
                            wxPrintf("ERROR: Could not open '%s' for writing\n", output_average_per_bin_filename.ToStdString( ).c_str( ));
                            DEBUG_ABORT;
                        }
                    }

                    //invert contrast before writing to restore correct contrast as CTF correction inverted the contrast of the images
                    masked_projection_volume_image.InvertRealValues( );

#pragma omp critical
                    masked_projection_volume_image.WriteSlice(&my_output_sum_image_filename, bin_index * number_of_models + model_counter + 1);
                    masked_padded_projection_volume_image.Deallocate( );
                    masked_projection_volume_image.Deallocate( );
                }
            }
            model_volume[bin_index].Deallocate( );
            prepare_projections_progress->Update(bin_index + 1);
        }
        // deallocate all variables that are not needed from memory

        my_mask.Deallocate( );
        sum_images->Deallocate( );
        my_masked_volume->Deallocate( );
        masked_3d->Deallocate( );
        input_3d->Deallocate( );
        // check if the projection images file is still open then close it
        if ( my_output_sum_image_filename.IsOpen( ) ) {
            my_output_sum_image_filename.CloseFile( );
        }

        delete prepare_projections_progress;
    }

    Image           subtracted_image;
    Image           projection_3d;
    Image           projection_image;
    Image           padded_projection_image;
    AnglesAndShifts my_parameters_for_subtraction;
    float*          adjusted_x_shifts = nullptr;
    float*          adjusted_y_shifts = nullptr;

    if ( SPOT_RASTR == true ) {
        wxPrintf("\nSubtracting Azimuthal Average Projections...\n\n");

        ProgressBar* subtract_progress = new ProgressBar(number_of_input_images);
        MRCFile      my_output_SPOT_RASTR_filename(SPOT_RASTR_output_filename.ToStdString( ), true);
        my_output_SPOT_RASTR_filename.my_header.SetNumberOfImages(number_of_input_images);
        my_output_SPOT_RASTR_filename.my_header.SetDimensionsImage(x_dim, y_dim);
        my_output_SPOT_RASTR_filename.SetPixelSize(pixel_size);
        my_output_SPOT_RASTR_filename.WriteHeader( );
        my_output_SPOT_RASTR_filename.rewrite_header_on_close = true;

        //Added new as OMP was causing problems when writing images to a file that is not opened and have set dimensions and header information
        MRCFile SPOT_RASTR_projections_output("SPOT_RASTR_projection_image_to_be_subtracted.mrc", true);
        if ( ! SPOT_RASTR_projections_output.IsOpen( ) ) {
            SPOT_RASTR_projections_output.OpenFile("SPOT_RASTR_projection_image_to_be_subtracted.mrc", true);
            if ( ! SPOT_RASTR_projections_output.IsOpen( ) ) {
                wxPrintf("ERROR: Could not open '%s' for writing\n", "SPOT_RASTR_projection_image_to_be_subtracted.mrc");
                DEBUG_ABORT;
            }
        }
        SPOT_RASTR_projections_output.my_header.SetNumberOfImages(number_of_input_images);
        SPOT_RASTR_projections_output.my_header.SetDimensionsImage(x_dim, y_dim);
        SPOT_RASTR_projections_output.SetPixelSize(pixel_size);
        SPOT_RASTR_projections_output.WriteHeader( );
        SPOT_RASTR_projections_output.rewrite_header_on_close = true;
        // This is needed to adjust for extra shift happening in ExtractSlice
        adjusted_x_shifts = new float[number_of_input_images];
        adjusted_y_shifts = new float[number_of_input_images];

#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads) default(none) shared(number_of_input_images, my_input_file, best_psi_value, best_x_shift_value, best_y_shift_value, current_image, subtract_progress, SPOT_RASTR_projections_output,                                 \
                                                                                            ctf_parameters_stack, max_threads, diameter_bins, bins_count, my_output_SPOT_RASTR_filename, x_dim, y_dim, use_memory, image_stack, projection_volume_3d, adjusted_x_shifts, adjusted_y_shifts, \
                                                                                            input_ctf_values_from_star_file, current_ctf, pixel_size, padding_factor, input_3d, x_mask_center, y_mask_center, z_mask_center) private(subtracted_image, projection_3d, projection_image, padded_projection_image, my_parameters_for_subtraction)

        for ( long subtraction_image_counter = 0; subtraction_image_counter < number_of_input_images; subtraction_image_counter++ ) {
            // read the current image in the stack
            subtracted_image.Allocate(x_dim, y_dim, true);
            projection_image.Allocate(x_dim, y_dim, true);
            padded_projection_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding

            subtracted_image.SetToConstant(0.0);
            projection_image.SetToConstant(0.0);
            padded_projection_image.SetToConstant(0.0);
            if ( use_memory ) {
                subtracted_image.CopyFrom(&image_stack[subtraction_image_counter]);
            }
            else {
#pragma omp critical
                subtracted_image.ReadSlice(&my_input_file, subtraction_image_counter + 1);
            }
            if ( input_ctf_values_from_star_file ) {
                current_ctf.Init(ctf_parameters_stack[subtraction_image_counter].acceleration_voltage, ctf_parameters_stack[subtraction_image_counter].spherical_aberration, ctf_parameters_stack[subtraction_image_counter].amplitude_contrast, ctf_parameters_stack[subtraction_image_counter].defocus_1, ctf_parameters_stack[subtraction_image_counter].defocus_2, ctf_parameters_stack[subtraction_image_counter].astigmatism_angle, ctf_parameters_stack[subtraction_image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[subtraction_image_counter].highest_frequency_for_fitting, ctf_parameters_stack[subtraction_image_counter].astigmatism_tolerance, ctf_parameters_stack[subtraction_image_counter].pixel_size, ctf_parameters_stack[subtraction_image_counter].additional_phase_shift);
            }

            my_parameters_for_subtraction.Init(0.0, 90.0, 90.0 - best_psi_value[subtraction_image_counter], 0.0, 0.0);

            for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
                if ( diameter_bins[subtraction_image_counter] == bin_index ) { // This should catch any diameter within the range
                    projection_volume_3d[bin_index].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                    // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                    padded_projection_image.ZeroCentralPixel( );
                    padded_projection_image.complex_values[0] = projection_volume_3d[bin_index].complex_values[0];
                }
                else if ( diameter_bins[subtraction_image_counter] < 0 ) { // if the which_bin_index is negative this mean the tube diameter is less than the min. so will be considered with the first bin
                    projection_volume_3d[0].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                    // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                    padded_projection_image.ZeroCentralPixel( );
                    padded_projection_image.complex_values[0] = projection_volume_3d[0].complex_values[0];
                }
                else if ( diameter_bins[subtraction_image_counter] >= bins_count ) { // if the which_bin_index is larger than or equal the bins_count then the diameter is more than the max. so will be added to the last bin
                    projection_volume_3d[bins_count - 1].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                    // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                    padded_projection_image.ZeroCentralPixel( );
                    padded_projection_image.complex_values[0] = projection_volume_3d[bins_count - 1].complex_values[0];
                }
            }

            padded_projection_image.ApplyCTF(current_ctf, false, true);

            // // generate the full rotation matrix
            RotationMatrix temp_matrix;
            float          rotated_x, rotated_y, rotated_z;
            temp_matrix.SetToEulerRotation(0.0, 90.0, (90.0 - best_psi_value[subtraction_image_counter]));
            // assuming no Y shift will be applied to ensure everything is centered
            temp_matrix.RotateCoords((best_x_shift_value[subtraction_image_counter]), 0.0, 0.0, rotated_x, rotated_y, rotated_z);
            // correct the shift happenning because of extractslice - Note x and y shift needs to be flipped
            // since projection is centered so negative rotation and shift is needed here
            adjusted_x_shifts[subtraction_image_counter] = -rotated_y;
            adjusted_y_shifts[subtraction_image_counter] = -rotated_x;

            padded_projection_image.PhaseShift(adjusted_x_shifts[subtraction_image_counter], 0); //-best_x_shift_value[subtraction_image_counter]

            padded_projection_image.SwapRealSpaceQuadrants( );
            padded_projection_image.BackwardFFT( );

            // de-pad projection
            padded_projection_image.ClipInto(&projection_image);
            // scaling is needed as without it wither over-subtraction will happen or small negligible change can happen
            float average = ReturnAverageOfRealValuesOnVerticalEdges(&projection_image);
            projection_image.AddConstant(-average);

            int   pixel_counter            = 0;
            float sum_of_pixelwise_product = 0.0;
            float sum_of_squares           = 0.0;
            float scale_factor             = 0.0;

            for ( long j = 0; j < projection_image.logical_y_dimension; j++ ) {
                for ( long i = 0; i < projection_image.logical_x_dimension; i++ ) {

                    sum_of_pixelwise_product += projection_image.real_values[pixel_counter] * subtracted_image.real_values[pixel_counter];
                    sum_of_squares += projection_image.real_values[pixel_counter] * projection_image.real_values[pixel_counter];
                    pixel_counter++;
                }
                pixel_counter += projection_image.padding_jump_value;
            }

            scale_factor = sum_of_pixelwise_product / sum_of_squares;
            //wxPrintf("The scale factor of image %li is %f \n", subtraction_image_counter+1, scale_factor);

            // multiply by the scaling factor calculated
            projection_image.MultiplyByConstant((scale_factor));
#pragma omp critical
            projection_image.WriteSlice(&SPOT_RASTR_projections_output, subtraction_image_counter + 1);

            //subtract the averaged sum image
            subtracted_image.SubtractImage(&projection_image);

            padded_projection_image.Deallocate( );
            projection_image.Deallocate( );
            // write the subtracted images
#pragma omp critical
            subtracted_image.WriteSlice(&my_output_SPOT_RASTR_filename, subtraction_image_counter + 1);
            subtracted_image.Deallocate( );

            if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )

                subtract_progress->Update(subtraction_image_counter + 1);
        }

        // Add a header file for the saved subtracted images
        my_output_SPOT_RASTR_filename.my_header.SetDimensionsVolume(x_dim, y_dim, number_of_input_images);
        my_output_SPOT_RASTR_filename.my_header.SetPixelSize(my_input_file.ReturnPixelSize( ));
        my_output_SPOT_RASTR_filename.WriteHeader( );

        delete subtract_progress;
    }

    cisTEMParameters SPOT_RASTR_output_params;

    SPOT_RASTR_output_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);

    if ( SPOT_RASTR == true ) {
        SPOT_RASTR_output_params.PreallocateMemoryAndBlank(number_of_input_images);
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            SPOT_RASTR_output_params.all_parameters[image_counter].position_in_stack                  = image_counter + 1;
            SPOT_RASTR_output_params.all_parameters[image_counter].psi                                = 90.0 - best_psi_value[image_counter]; // -(best_psi_value[image_counter] - 90.0)  i.e. -rotation + 90 This is the angle that when the tube is rotated by it will align it to 90.0 degrees Psi
            SPOT_RASTR_output_params.all_parameters[image_counter].theta                              = 90.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].phi                                = 0;
            SPOT_RASTR_output_params.all_parameters[image_counter].x_shift                            = adjusted_x_shifts[image_counter]; // This the shift that will center the tube after rotating it to have 90 degree rotation
            SPOT_RASTR_output_params.all_parameters[image_counter].y_shift                            = 0.0;
            SPOT_RASTR_output_params.all_parameters[image_counter].defocus_1                          = ctf_parameters_stack[image_counter].defocus_1;
            SPOT_RASTR_output_params.all_parameters[image_counter].defocus_2                          = ctf_parameters_stack[image_counter].defocus_2;
            SPOT_RASTR_output_params.all_parameters[image_counter].defocus_angle                      = ctf_parameters_stack[image_counter].astigmatism_angle;
            SPOT_RASTR_output_params.all_parameters[image_counter].phase_shift                        = ctf_parameters_stack[image_counter].additional_phase_shift;
            SPOT_RASTR_output_params.all_parameters[image_counter].image_is_active                    = 1;
            SPOT_RASTR_output_params.all_parameters[image_counter].occupancy                          = 100.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].logp                               = -1000.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].sigma                              = 10.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].pixel_size                         = pixel_size;
            SPOT_RASTR_output_params.all_parameters[image_counter].microscope_voltage_kv              = ctf_parameters_stack[image_counter].acceleration_voltage;
            SPOT_RASTR_output_params.all_parameters[image_counter].microscope_spherical_aberration_mm = ctf_parameters_stack[image_counter].spherical_aberration;
            SPOT_RASTR_output_params.all_parameters[image_counter].amplitude_contrast                 = ctf_parameters_stack[image_counter].amplitude_contrast;
            SPOT_RASTR_output_params.all_parameters[image_counter].beam_tilt_x                        = 0.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].beam_tilt_y                        = 0.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].image_shift_x                      = 0.0f;
            SPOT_RASTR_output_params.all_parameters[image_counter].image_shift_y                      = 0.0f;
        }

        SPOT_RASTR_output_params.WriteTocisTEMStarFile(SPOT_RASTR_output_star_filename.ToStdString( ));
    }

    //////////// Continue with RASTR processing
    Image unmasked_projection_3d;
    Image unmasked_projection_image;
    Image unmasked_padded_projection_image;
    Image subtracted_RASTR_image;
    Image centered_upweighted_image;

    float* RASTR_adjusted_x_shifts;
    float* RASTR_adjusted_y_shifts;

    // Variables needed for RASTR mask file
    Image           mask_projection_image;
    Image           padded_mask_projection_image;
    AnglesAndShifts mask_parameters;
    float           phi;

    if ( RASTR == true ) { // if RASTR is true then we will use the masked model for subtraction
        wxPrintf("\nSubtracting Masked Azimuthal Average Projections...\n\n");

        ProgressBar* mask_subtract_progress = new ProgressBar(number_of_input_images * number_of_models);
        MRCFile      my_output_RASTR_filename(RASTR_output_filename.ToStdString( ), true);
        //Added new as OMP was causing problems when writing images to a file that is not opened and have set dimensions and header information
        my_output_RASTR_filename.my_header.SetNumberOfImages(number_of_input_images * number_of_models);
        my_output_RASTR_filename.my_header.SetDimensionsImage(x_dim, y_dim);
        my_output_RASTR_filename.SetPixelSize(pixel_size);
        my_output_RASTR_filename.WriteHeader( );
        my_output_RASTR_filename.rewrite_header_on_close = true;
        //Added new as OMP was causing problems when writing images to a file that is not opened and have set dimensions and header information
        MRCFile RASTR_projections_output("RASTR_projection_image_to_be_subtracted.mrc", true);
        if ( ! RASTR_projections_output.IsOpen( ) ) {
            RASTR_projections_output.OpenFile("RASTR_projection_image_to_be_subtracted.mrc", true);
            if ( ! RASTR_projections_output.IsOpen( ) ) {
                wxPrintf("ERROR: Could not open '%s' for writing\n", "RASTR_projection_image_to_be_subtracted.mrc");
                DEBUG_ABORT;
            }
        }
        RASTR_projections_output.my_header.SetNumberOfImages(number_of_input_images * number_of_models);
        RASTR_projections_output.my_header.SetDimensionsImage(x_dim, y_dim);
        RASTR_projections_output.SetPixelSize(pixel_size);
        RASTR_projections_output.WriteHeader( );
        RASTR_projections_output.rewrite_header_on_close = true;

        // MRCFile unmasked_projections_output("unmasked_projection_image_to_be_subtracted.mrc", true);
        // if ( ! unmasked_projections_output.IsOpen( ) ) {
        //     unmasked_projections_output.OpenFile("unmasked_projection_image_to_be_subtracted.mrc", true);
        //     if ( ! unmasked_projections_output.IsOpen( ) ) {
        //         wxPrintf("ERROR: Could not open '%s' for writing\n", "unmasked_projection_image_to_be_subtracted.mrc");
        //         DEBUG_ABORT;
        //     }
        // }
        // unmasked_projections_output.my_header.SetNumberOfImages(number_of_input_images * number_of_models);
        // unmasked_projections_output.my_header.SetDimensionsImage(x_dim, y_dim);
        // unmasked_projections_output.SetPixelSize(pixel_size);
        // unmasked_projections_output.WriteHeader( );
        // unmasked_projections_output.rewrite_header_on_close = true;

        // This is needed to adjust for extra shift happening in ExtractSlice
        adjusted_x_shifts = new float[number_of_input_images];
        adjusted_y_shifts = new float[number_of_input_images];

        RASTR_adjusted_x_shifts = new float[number_of_input_images * number_of_models];
        RASTR_adjusted_y_shifts = new float[number_of_input_images * number_of_models];
        ///padded_projected_volumes, padded_projected_masked_volumes,

        //Will make the outer loop on one thread but inner loop multi-threaded to ensure the sequential processing of the models!
        for ( long model_counter = 0; model_counter < number_of_models; model_counter++ ) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads) default(none) shared(number_of_input_images, my_input_file, best_psi_value, best_x_shift_value, best_y_shift_value, number_of_models, input_3d, mask_upweighted, image_stack, model_counter,                                                                                                    \
                                                                                            ctf_parameters_stack, max_threads, diameter_bins, bins_count, current_image, mask_subtract_progress, align_upweighted, RASTR_projections_output, use_memory, adjusted_x_shifts, adjusted_y_shifts,                                                                         \
                                                                                            input_ctf_values_from_star_file, current_ctf, pixel_size, padding_factor, masked_3d, x_mask_center, y_mask_center, x_dim, y_dim, z_mask_center, RASTR_adjusted_x_shifts, RASTR_adjusted_y_shifts, mask_projection, input_mask, filter_radius, outside_weight, cosine_edge, \
                                                                                            center_upweighted, sphere_mask_radius, RASTR_output_filename, my_output_RASTR_filename, projection_volume_3d, masked_projection_volume_3d) private(phi, projection_3d, projection_image, padded_projection_image, my_parameters_for_subtraction, mask_projection_image, padded_mask_projection_image, mask_parameters, subtracted_RASTR_image, centered_upweighted_image, unmasked_projection_image, unmasked_padded_projection_image, unmasked_projection_3d)

            for ( long subtraction_image_counter = 0; subtraction_image_counter < number_of_input_images; subtraction_image_counter++ ) {

                long current_counter = number_of_input_images * model_counter + subtraction_image_counter;

                subtracted_RASTR_image.Allocate(x_dim, y_dim, true);
                projection_image.Allocate(x_dim, y_dim, true);
                padded_projection_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding

                unmasked_projection_image.Allocate(x_dim, y_dim, true);
                unmasked_padded_projection_image.Allocate(x_dim * padding_factor, y_dim * padding_factor, false); // as my volume now is already padded so no need to add extra padding

                subtracted_RASTR_image.SetToConstant(0.0);
                projection_image.SetToConstant(0.0);
                padded_projection_image.SetToConstant(0.0);
                unmasked_projection_image.SetToConstant(0.0);
                unmasked_padded_projection_image.SetToConstant(0.0);

                if ( use_memory ) {
                    subtracted_RASTR_image.CopyFrom(&image_stack[subtraction_image_counter]);
                }
                else {
#pragma omp critical
                    subtracted_RASTR_image.ReadSlice(&my_input_file, subtraction_image_counter + 1);
                }

                if ( input_ctf_values_from_star_file ) {
                    current_ctf.Init(ctf_parameters_stack[subtraction_image_counter].acceleration_voltage, ctf_parameters_stack[subtraction_image_counter].spherical_aberration, ctf_parameters_stack[subtraction_image_counter].amplitude_contrast, ctf_parameters_stack[subtraction_image_counter].defocus_1, ctf_parameters_stack[subtraction_image_counter].defocus_2, ctf_parameters_stack[subtraction_image_counter].astigmatism_angle, ctf_parameters_stack[subtraction_image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[subtraction_image_counter].highest_frequency_for_fitting, ctf_parameters_stack[subtraction_image_counter].astigmatism_tolerance, ctf_parameters_stack[subtraction_image_counter].pixel_size, ctf_parameters_stack[subtraction_image_counter].additional_phase_shift);
                }

                phi = model_counter * 360.0 / number_of_models;

                // Extracting a slice from a 3D volume
                // angles and shifts are negative to align projection with original input stack
                // use the phi based on the number of models

                my_parameters_for_subtraction.Init(phi, 90.0, 90.0 - best_psi_value[subtraction_image_counter], 0.0, 0.0);

                for ( int bin_index = 0; bin_index < bins_count; bin_index++ ) {
                    if ( diameter_bins[subtraction_image_counter] == bin_index ) { // This should catch any diameter within the range

                        projection_volume_3d[bin_index].ExtractSlice(unmasked_padded_projection_image, my_parameters_for_subtraction);
                        masked_projection_volume_3d[bin_index].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                        unmasked_padded_projection_image.ZeroCentralPixel( );
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        unmasked_padded_projection_image.complex_values[0] = projection_volume_3d[bin_index].complex_values[0];
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        padded_projection_image.ZeroCentralPixel( );
                        padded_projection_image.complex_values[0] = masked_projection_volume_3d[bin_index].complex_values[0];
                    }
                    else if ( diameter_bins[subtraction_image_counter] < 0 ) { // if the which_bin_index is negative this mean the tube diameter is less than the min. so will be considered with the first bin

                        projection_volume_3d[0].ExtractSlice(unmasked_padded_projection_image, my_parameters_for_subtraction);
                        masked_projection_volume_3d[0].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        unmasked_padded_projection_image.complex_values[0] = projection_volume_3d[0].complex_values[0];
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        padded_projection_image.ZeroCentralPixel( );
                        padded_projection_image.complex_values[0] = masked_projection_volume_3d[0].complex_values[0];
                    }
                    else if ( diameter_bins[subtraction_image_counter] >= bins_count ) { // if the which_bin_index is larger than or equal the bins_count then the diameter is more than the max. so will be added to the last bin
                        projection_volume_3d[bins_count - 1].ExtractSlice(unmasked_padded_projection_image, my_parameters_for_subtraction);
                        masked_projection_volume_3d[bins_count - 1].ExtractSlice(padded_projection_image, my_parameters_for_subtraction);
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        unmasked_padded_projection_image.complex_values[0] = projection_volume_3d[bins_count - 1].complex_values[0];
                        // in project 3D the following line is added before apply ctf and apply ctf is done where absolute is false and apply beam tilt is true
                        padded_projection_image.ZeroCentralPixel( );
                        padded_projection_image.complex_values[0] = masked_projection_volume_3d[bins_count - 1].complex_values[0];
                    }
                }
                // save the adjusted shift as the extract slice function applied a rotation matrix to the model which shifted x, y shifts more
                // so saving those extra shifts is needed to center the images later and to save the correct shift in the output star file
                // we will need two adjustments one to get the correct shift for the unmasked/masked volume projections
                // and the other to get the correct shift needed to center the RASTR particle in the middle of the image

                // // generate the full rotation matrix
                RotationMatrix temp_matrix;
                float          rotated_x, rotated_y, rotated_z;
                temp_matrix.SetToEulerRotation(0.0, 90.0, (90.0 - best_psi_value[subtraction_image_counter]));
                // assuming no Y shift will be applied to ensure everything is centered
                temp_matrix.RotateCoords((best_x_shift_value[subtraction_image_counter]), 0.0, 0.0, rotated_x, rotated_y, rotated_z);
                // correct the shift happenning because of extractslice - Note x and y shift needs to be flipped
                // since projection is centered so negative rotation and shift is needed here
                adjusted_x_shifts[subtraction_image_counter] = -rotated_y;
                adjusted_y_shifts[subtraction_image_counter] = -rotated_x;

                padded_projection_image.ApplyCTF(current_ctf, false, true);
                padded_projection_image.PhaseShift(adjusted_x_shifts[subtraction_image_counter], 0); //-best_y_shift_value[subtraction_image_counter]
                padded_projection_image.SwapRealSpaceQuadrants( );
                padded_projection_image.BackwardFFT( );
                // de-pad projection
                padded_projection_image.ClipInto(&projection_image);

                float edge_average = ReturnAverageOfRealValuesOnVerticalEdges(&projection_image);
                projection_image.AddConstant(-edge_average);

                /////////////////////////////////////////////////////////////////////////////////////////
                // extract the unmasked projection image to calculate the correct scaling factor
                ////////////////////////////////////////////////////////////////////////////////////////

                unmasked_padded_projection_image.ApplyCTF(current_ctf, false, true);
                // since projection is centered so negative rotation and shift is needed here
                unmasked_padded_projection_image.PhaseShift(adjusted_x_shifts[subtraction_image_counter], 0.0); //RASTR_adjusted_x_shifts[current_counter] -best_x_shift_value[subtraction_image_counter]
                unmasked_padded_projection_image.SwapRealSpaceQuadrants( );
                unmasked_padded_projection_image.BackwardFFT( );

                // de-pad projection
                unmasked_padded_projection_image.ClipInto(&unmasked_projection_image);

                edge_average = ReturnAverageOfRealValuesOnVerticalEdges(&unmasked_projection_image);
                unmasked_projection_image.AddConstant(-edge_average);

                int   pixel_counter            = 0;
                float sum_of_pixelwise_product = 0.0;
                float sum_of_squares           = 0.0;
                float scale_factor             = 0.0;

                for ( long j = 0; j < unmasked_projection_image.logical_y_dimension; j++ ) {
                    for ( long i = 0; i < unmasked_projection_image.logical_x_dimension; i++ ) {

                        sum_of_pixelwise_product += unmasked_projection_image.real_values[pixel_counter] * subtracted_RASTR_image.real_values[pixel_counter];
                        sum_of_squares += unmasked_projection_image.real_values[pixel_counter] * unmasked_projection_image.real_values[pixel_counter];
                        pixel_counter++;
                    }
                    pixel_counter += unmasked_projection_image.padding_jump_value;
                }

                unmasked_projection_image.Deallocate( );
                unmasked_padded_projection_image.Deallocate( );
                // calculate the scaling factor based on comparing the unmasked projection of azimuthal average with original image
                // the masked projection mess up the scaling factor calculations
                scale_factor = sum_of_pixelwise_product / sum_of_squares;
                //wxPrintf("The scale factor of image %li is %f \n", subtraction_image_counter+1, scale_factor);

                // multiply by the scaling factor calculated
                projection_image.MultiplyByConstant((scale_factor));

#pragma omp critical
                projection_image.WriteSlice(&RASTR_projections_output, current_counter + 1);

                //subtract the averaged sum image after rotation and translation from the original image
                subtracted_RASTR_image.SubtractImage(&projection_image);

                padded_projection_image.Deallocate( );
                projection_image.Deallocate( );

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // if we want to mask the final upweighted regions
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                if ( mask_upweighted ) {
                    // If we will mask the upweighted regions then we need to determine the location of the center of the upweighted region in the image
                    RotationMatrix RASTR_temp_matrix;
                    float          RASTR_rotated_x, RASTR_rotated_y, RASTR_rotated_z;
                    // // generate the full rotation matrix
                    RASTR_temp_matrix.SetToEulerRotation(-(90.0 - best_psi_value[subtraction_image_counter]), -90.0, -phi); // removed the negative from all

                    RASTR_temp_matrix.RotateCoords((x_mask_center - current_image.physical_address_of_box_center_x), (y_mask_center - current_image.physical_address_of_box_center_y), (z_mask_center - current_image.physical_address_of_box_center_x), RASTR_rotated_x, RASTR_rotated_y, RASTR_rotated_z);

                    // center the masked upweighted regions to the center
                    RASTR_adjusted_x_shifts[current_counter] = -RASTR_rotated_x; //best_x_shift_value[subtraction_image_counter]
                    RASTR_adjusted_y_shifts[current_counter] = -RASTR_rotated_y; //best_y_shift_value[subtraction_image_counter]

                    float average = subtracted_RASTR_image.ReturnAverageOfRealValues( );
                    mask_parameters.Init(phi, 90.0, 90.0 - best_psi_value[subtraction_image_counter], 0.0, 0.0); //* pixel_size

                    mask_projection_image.Allocate(x_dim, y_dim, false); // false as it is in FS
                    padded_mask_projection_image.Allocate(padding_factor * x_dim, padding_factor * y_dim, false);

                    mask_projection->ExtractSlice(padded_mask_projection_image, mask_parameters);

                    padded_mask_projection_image.SwapRealSpaceQuadrants( ); // must do this step as image is not centered in the box
                    padded_mask_projection_image.BackwardFFT( );

                    //rebinarize based on the new threshold >=0.01 should be 1 and less should be 0 (This should be done before shifting as shifting may affect the threshold used)
                    padded_mask_projection_image.Binarise(0.01f);
                    padded_mask_projection_image.ClipInto(&mask_projection_image);
                    //subtracted_RASTR_image.QuickAndDirtyWriteSlice("subtracted_RASTR_images.mrc", current_counter + 1);
                    float filter_edge = 40.0;
                    float mask_volume_in_voxels;
                    if ( filter_radius == 0.0 )
                        filter_radius = pixel_size;

                    //multiply the mask by the mask subtracted image with the upweighted regions
                    mask_volume_in_voxels = subtracted_RASTR_image.ApplyMask(mask_projection_image, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, average, true);
                    // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //masked subtracted before centering
                    //subtracted_RASTR_image.QuickAndDirtyWriteSlice("masked_subtracted_RASTR_images.mrc", current_counter + 1);
                    // if user specified a masked upweighted region should be generated then it can be centered (not aligned) or (aligned not centered) or (centered and aligned) or (saved as is not aligned not centered)
                    // if the user specified that the upweighted regions should be centered before saving them
                    if ( center_upweighted == true && align_upweighted == false ) {
                        subtracted_RASTR_image.PhaseShift(RASTR_adjusted_x_shifts[current_counter], RASTR_adjusted_y_shifts[current_counter]);

#pragma omp critical
                        subtracted_RASTR_image.WriteSlice(&my_output_RASTR_filename, current_counter + 1);
                    }
                    else if ( align_upweighted == true && center_upweighted == false ) {
                        // rotate the masked upweighted regions to be aligned with Y axis
                        subtracted_RASTR_image.Rotate2DInPlace(best_psi_value[subtraction_image_counter], FLT_MAX);
#pragma omp critical
                        subtracted_RASTR_image.WriteSlice(&my_output_RASTR_filename, current_counter + 1);
                    }
                    else if ( (align_upweighted == true && center_upweighted == true) ) {
                        subtracted_RASTR_image.Rotate2DInPlace(best_psi_value[subtraction_image_counter], FLT_MAX);
                        // I guess since the images will be aligned top down then no need for y shift?
                        subtracted_RASTR_image.PhaseShift(RASTR_adjusted_x_shifts[current_counter], 0.0);

#pragma omp critical
                        subtracted_RASTR_image.WriteSlice(&my_output_RASTR_filename, current_counter + 1);
                    }
                    else { // save the unaligned uncentered masked upweighted regions

#pragma omp critical
                        subtracted_RASTR_image.WriteSlice(&my_output_RASTR_filename, current_counter + 1);
                    }

                    mask_projection_image.Deallocate( );
                    padded_mask_projection_image.Deallocate( );
                    subtracted_RASTR_image.Deallocate( );
                }
                else { // if the subtracted images will not be masked then we will save the full image as centered aligned
                    // align the tube vertically
                    subtracted_RASTR_image.Rotate2DInPlace(best_psi_value[subtraction_image_counter], FLT_MAX);
                    // shift the tube to the center (not the upweighted region)
                    subtracted_RASTR_image.PhaseShift(adjusted_x_shifts[subtraction_image_counter], 0);

                    if ( ! my_output_RASTR_filename.IsOpen( ) ) {
                        my_output_RASTR_filename.OpenFile(RASTR_output_filename.ToStdString( ), true);
                        if ( ! my_output_RASTR_filename.IsOpen( ) ) {
                            wxPrintf("ERROR: Could not open '%s' for writing\n", RASTR_output_filename.ToStdString( ));
                            DEBUG_ABORT;
                        }
                    }

#pragma omp critical
                    subtracted_RASTR_image.WriteSlice(&my_output_RASTR_filename, current_counter + 1);
                    subtracted_RASTR_image.Deallocate( );
                }

                if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                    mask_subtract_progress->Update(number_of_input_images * model_counter + subtraction_image_counter + 1);
            }
        }
        delete mask_subtract_progress;
        delete mask_projection;
        delete[] RASTR_adjusted_x_shifts;
        delete[] RASTR_adjusted_y_shifts;

        //Add a header file for the saved subtracted images
        my_output_RASTR_filename.my_header.SetDimensionsVolume(x_dim, y_dim, number_of_input_images * number_of_models);
        my_output_RASTR_filename.my_header.SetPixelSize(my_input_file.ReturnPixelSize( ));
        my_output_RASTR_filename.WriteHeader( );
    }

    // write the parameters file
    cisTEMParameters RASTR_output_params;

    RASTR_output_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);

    if ( RASTR == true ) {
        RASTR_output_params.PreallocateMemoryAndBlank(number_of_input_images * number_of_models);

        //#pragma omp for ordered schedule(static, 1)
        for ( long model_counter = 0; model_counter < number_of_models; model_counter++ ) {
            for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                long current_counter                                                  = number_of_input_images * model_counter + image_counter; // This is for the position in the stack only
                RASTR_output_params.all_parameters[current_counter].position_in_stack = current_counter + 1;
                if ( mask_upweighted ) {
                    if ( align_upweighted ) {
                        RASTR_output_params.all_parameters[current_counter].psi = 90.0; // make the angle 90.0 as it is aligned
                    }
                    else {
                        RASTR_output_params.all_parameters[current_counter].psi = 90.0 - best_psi_value[image_counter]; // use the psi from extract slices
                    }

                    if ( center_upweighted == true ) { //if centered then no shift is needed
                        RASTR_output_params.all_parameters[current_counter].x_shift = 0.0;
                        RASTR_output_params.all_parameters[current_counter].y_shift = 0.0;
                    }
                    else { // since they are masked so we need the RASTR adjusted shift to know the exact location of the upweghted region
                        RASTR_output_params.all_parameters[current_counter].x_shift = -RASTR_adjusted_x_shifts[current_counter];
                        RASTR_output_params.all_parameters[current_counter].y_shift = -RASTR_adjusted_y_shifts[current_counter];
                    }
                }
                else { // if not masked then we are saving the aligned centered image
                    RASTR_output_params.all_parameters[current_counter].psi = 90.0; // make the angle 90.0 as it is aligned
                    // the tube is centered but the upweighted regions are not
                    RASTR_output_params.all_parameters[current_counter].x_shift = 0.0; // -adjusted_x_shifts[current_counter]?
                    RASTR_output_params.all_parameters[current_counter].y_shift = 0.0; // -adjusted_y_shifts[current_counter]
                }

                RASTR_output_params.all_parameters[current_counter].theta = 90.0f;
                RASTR_output_params.all_parameters[current_counter].phi   = model_counter * 360.0 / number_of_models;

                RASTR_output_params.all_parameters[current_counter].defocus_1                          = ctf_parameters_stack[image_counter].defocus_1;
                RASTR_output_params.all_parameters[current_counter].defocus_2                          = ctf_parameters_stack[image_counter].defocus_2;
                RASTR_output_params.all_parameters[current_counter].defocus_angle                      = ctf_parameters_stack[image_counter].astigmatism_angle;
                RASTR_output_params.all_parameters[current_counter].phase_shift                        = ctf_parameters_stack[image_counter].additional_phase_shift;
                RASTR_output_params.all_parameters[current_counter].image_is_active                    = 1;
                RASTR_output_params.all_parameters[current_counter].occupancy                          = 100.0f;
                RASTR_output_params.all_parameters[current_counter].logp                               = -1000.0f;
                RASTR_output_params.all_parameters[current_counter].sigma                              = 10.0f;
                RASTR_output_params.all_parameters[current_counter].pixel_size                         = pixel_size;
                RASTR_output_params.all_parameters[current_counter].microscope_voltage_kv              = ctf_parameters_stack[image_counter].acceleration_voltage;
                RASTR_output_params.all_parameters[current_counter].microscope_spherical_aberration_mm = ctf_parameters_stack[image_counter].spherical_aberration;
                RASTR_output_params.all_parameters[current_counter].amplitude_contrast                 = ctf_parameters_stack[image_counter].amplitude_contrast;
                RASTR_output_params.all_parameters[current_counter].beam_tilt_x                        = 0.0f;
                RASTR_output_params.all_parameters[current_counter].beam_tilt_y                        = 0.0f;
                RASTR_output_params.all_parameters[current_counter].image_shift_x                      = 0.0f;
                RASTR_output_params.all_parameters[current_counter].image_shift_y                      = 0.0f;
            }
        }

        RASTR_output_params.WriteTocisTEMStarFile(RASTR_output_star_filename.ToStdString( ));
    }
    if ( adjusted_y_shifts != nullptr )
        delete[] adjusted_y_shifts;
    if ( adjusted_x_shifts != nullptr )
        delete[] adjusted_x_shifts;
    delete[] ctf_parameters_stack;
    if ( image_stack != nullptr )
        delete[] image_stack;
    if ( image_stack_filtered_masked != nullptr )
        delete[] image_stack_filtered_masked;

    return true;
}

/// try the column sum but add a declaration at the begining of the code
std::vector<float> sum_image_columns(Image* current_image) {
    std::vector<float> column_sum(current_image->logical_x_dimension, 0.0);

    long pixel_counter = 0;

    for ( int i = 0; i < current_image->logical_x_dimension; i++ ) {
        for ( int j = 0; j < current_image->logical_y_dimension; j++ ) {
            long pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
            column_sum[i] += current_image->real_values[pixel_coord_xy];
            pixel_counter++;
        }
        pixel_counter += current_image->padding_jump_value;
    }

    return column_sum;
}

//// try this method but add the declaration at the begining
float sum_image_columns_float(Image* current_image) { // Tim's method to calculate the best rotation based on the row sum
    std::vector<float> column_sum(current_image->logical_x_dimension, 0.0);

    long pixel_counter = 0;

    for ( int i = 0; i < current_image->logical_x_dimension; i++ ) {
        for ( int j = 0; j < current_image->logical_y_dimension; j++ ) {
            long pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
            column_sum[i] += current_image->real_values[pixel_coord_xy];
            pixel_counter++;
        }
        pixel_counter += current_image->padding_jump_value;
    }

    float max_value = *std::max_element(column_sum.begin( ), column_sum.end( ),
                                        [](float a, float b) { return std::abs(a) < std::abs(b); });

    return abs(max_value);
}

void divide_by_ctf_sum_of_squares(Image& current_image, std::vector<float>& ctf_sum_of_squares) {
    // normalize by sum of squared CTFs (voxel by voxel)
    long pixel_counter = 0;

    for ( int j = 0; j <= current_image.physical_upper_bound_complex_y; j++ ) {
        for ( int i = 0; i <= current_image.physical_upper_bound_complex_x; i++ ) {
            if ( ctf_sum_of_squares[pixel_counter] != 0.0 )
                current_image.complex_values[pixel_counter] /= sqrtf(ctf_sum_of_squares[pixel_counter]);
            pixel_counter++;
        }
    }
}

void InitializeCTFSumOfSquares(int numBins, Image& current_image, std::vector<std::vector<float>>* ctf_sum_of_squares) {
    int number_of_pixels = current_image.real_memory_allocated / 2;
    // Allocate and initialize images for each bin
    for ( int bin_counter = 0; bin_counter < numBins; ++bin_counter ) {
        // Allocate memory for CTF sum of squares based on the image size
        // it needs to be dynamically allocated as image size will differ from one input to another
        (*ctf_sum_of_squares)[bin_counter] = std::vector<float>(number_of_pixels, 0.0f);
        //ZeroFloatArray(ctf_sum_of_squares[bin_counter], number_of_pixels / 2);
    }
}

void ApplyCTFAndReturnCTFSumOfSquares(Image& image, CTF ctf_to_apply, bool absolute, bool apply_beam_tilt, bool apply_envelope, std::vector<float>& ctf_sum_of_squares) {
    MyDebugAssertTrue(image.is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(image.is_in_real_space == false, "Image not in Fourier space");
    MyDebugAssertTrue(image.logical_z_dimension == 1, "Volumes not supported");

    //std::vector<float> squared_ctf_values; // Vector to store squared CTF values

    long pixel_counter = 0;

    float y_coord_sq;
    float x_coord_sq;

    float y_coord;
    float x_coord;

    float frequency_squared;
    float azimuth;
    float ctf_value;

    for ( int j = 0; j <= image.physical_upper_bound_complex_y; j++ ) {
        y_coord    = image.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * image.fourier_voxel_size_y;
        y_coord_sq = pow(y_coord, 2.0);

        for ( int i = 0; i <= image.physical_upper_bound_complex_x; i++ ) {
            x_coord    = i * image.fourier_voxel_size_x;
            x_coord_sq = pow(x_coord, 2);

            // Compute the azimuth
            if ( i == 0 && j == 0 ) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2(y_coord, x_coord);
            }

            // Compute the square of the frequency
            frequency_squared = x_coord_sq + y_coord_sq;

            if ( apply_envelope ) {
                ctf_value = ctf_to_apply.EvaluateWithEnvelope(frequency_squared, azimuth);
            }
            else {
                ctf_value = ctf_to_apply.Evaluate(frequency_squared, azimuth);
            }

            if ( absolute ) {
                ctf_value = fabsf(ctf_value);
            }

            // Apply CTF to the image if needed
            image.complex_values[pixel_counter] *= ctf_value;

            if ( apply_beam_tilt && (ctf_to_apply.GetBeamTiltX( ) != 0.0f || ctf_to_apply.GetBeamTiltY( ) != 0.0f) ) {
                image.complex_values[pixel_counter] *= ctf_to_apply.EvaluateBeamTiltPhaseShift(frequency_squared, azimuth);
            }

            // Add the squared CTF value to the input CTF sum of squares vector
            // but check first if the vector is empty to avoid memory problems
            // if (ctf_sum_of_squares.empty()) {
            //     ctf_sum_of_squares[pixel_counter] = powf(ctf_value, 2);
            // } else {
            ctf_sum_of_squares[pixel_counter] += powf(ctf_value, 2);
            // }

            pixel_counter++;
        }
    }

    //return ctf_sum_of_squares;
}

void sum_image_direction(Image* current_image, int dim) {
    // image must be in real-space
    Image directional_image_sum;
    directional_image_sum.Allocate(current_image->logical_x_dimension, current_image->logical_y_dimension, true);
    directional_image_sum.SetToConstant(0.0);

    // x-direction
    if ( dim == 1 ) {

        long pixel_coord_y  = 0;
        long pixel_coord_xy = 0;
        long pixel_counter  = 0;

        // sum columns of my_image_sum (NxM) and store in array (1xN)
        for ( int j = 0; j < current_image->logical_y_dimension; j++ ) {
            for ( int i = 0; i < current_image->logical_x_dimension; i++ ) {
                pixel_coord_y  = current_image->ReturnReal1DAddressFromPhysicalCoord(0, j, 0);
                pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                directional_image_sum.real_values[pixel_coord_y] += current_image->real_values[pixel_coord_xy];
                pixel_counter++;
            }
            pixel_counter += current_image->padding_jump_value;
        }

        // repeat column sum into my_vertical_sum
        pixel_counter = 0;
        for ( int j = 0; j < directional_image_sum.logical_y_dimension; j++ ) {
            for ( int i = 0; i < directional_image_sum.logical_x_dimension; i++ ) {
                pixel_coord_y                                     = directional_image_sum.ReturnReal1DAddressFromPhysicalCoord(0, j, 0);
                pixel_coord_xy                                    = directional_image_sum.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                directional_image_sum.real_values[pixel_coord_xy] = directional_image_sum.real_values[pixel_coord_y];
                pixel_counter++;
            }
            pixel_counter += directional_image_sum.padding_jump_value;
        }

        directional_image_sum.DivideByConstant(directional_image_sum.logical_x_dimension);
    }
    // y-direction
    else {

        long pixel_coord_x  = 0;
        long pixel_coord_xy = 0;
        long pixel_counter  = 0;

        // sum columns of my_image_sum (NxM) and store in array (1xM)
        for ( int i = 0; i < current_image->logical_x_dimension; i++ ) {
            for ( int j = 0; j < current_image->logical_y_dimension; j++ ) {
                pixel_coord_x  = current_image->ReturnReal1DAddressFromPhysicalCoord(i, 0, 0);
                pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                directional_image_sum.real_values[pixel_coord_x] += current_image->real_values[pixel_coord_xy];
                pixel_counter++;
            }
            pixel_counter += current_image->padding_jump_value;
        }

        // repeat column sum into my_vertical_sum
        pixel_counter = 0;
        for ( int i = 0; i < directional_image_sum.logical_x_dimension; i++ ) {
            for ( int j = 0; j < directional_image_sum.logical_y_dimension; j++ ) {
                pixel_coord_x                                     = directional_image_sum.ReturnReal1DAddressFromPhysicalCoord(i, 0, 0);
                pixel_coord_xy                                    = directional_image_sum.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                directional_image_sum.real_values[pixel_coord_xy] = directional_image_sum.real_values[pixel_coord_x];
                pixel_counter++;
            }
            pixel_counter += directional_image_sum.padding_jump_value;
        }

        directional_image_sum.DivideByConstant(directional_image_sum.logical_y_dimension);
    }

    // copy to current iamge
    current_image->CopyFrom(&directional_image_sum);
    directional_image_sum.Deallocate( );
}

void apply_ctf(Image* current_image, CTF ctf_to_apply, float* ctf_sum_of_squares, bool absolute, bool do_fill_sum_of_squares) {
    float y_coord_sq;
    float x_coord_sq;

    float y_coord;
    float x_coord;

    float frequency_squared;
    float azimuth;
    float ctf_value;

    long pixel_counter = 0;

    for ( int j = 0; j <= current_image->physical_upper_bound_complex_y; j++ ) {
        y_coord    = current_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * current_image->fourier_voxel_size_y;
        y_coord_sq = powf(y_coord, 2.0);

        for ( int i = 0; i <= current_image->physical_upper_bound_complex_x; i++ ) {
            x_coord    = i * current_image->fourier_voxel_size_x;
            x_coord_sq = powf(x_coord, 2.0);

            // Compute the azimuth
            if ( i == 0 && j == 0 ) {
                azimuth = 0.0;
            }
            else {
                azimuth = atan2f(y_coord, x_coord);
            }

            // Compute the square of the frequency
            frequency_squared = x_coord_sq + y_coord_sq;
            ctf_value         = ctf_to_apply.Evaluate(frequency_squared, azimuth);

            // phase-flip
            if ( absolute )
                ctf_value = fabsf(ctf_value);

            current_image->complex_values[pixel_counter] *= ctf_value;
            if ( do_fill_sum_of_squares ) {
                ctf_sum_of_squares[pixel_counter] += powf(ctf_value, 2);
            }
            pixel_counter++;
        }
    }
}

// Function to ensure the angle is within the range [0, 360)
float angle_within360(float angle) {
    if ( angle < 0.0 ) {
        angle += 360.0;
        return angle_within360(angle);
    }
    else if ( angle >= 360.0 ) {
        angle -= 360.0;
        return angle_within360(angle);
    }
    else {
        return angle;
    }
}

// calculates the average of real values on the vertical edges
float ReturnAverageOfRealValuesOnVerticalEdges(Image* current_image) {
    double sum;
    long   number_of_pixels;
    int    pixel_counter;
    int    line_counter;
    int    plane_counter;
    long   address;

    sum              = 0.0;
    number_of_pixels = 0;
    address          = 0;

    if ( current_image->logical_z_dimension == 1 ) {
        // Two-dimensional image
        for ( line_counter = 0; line_counter < current_image->logical_y_dimension; line_counter++ ) {
            sum += current_image->real_values[address];
            address += current_image->logical_x_dimension - 1;
            sum += current_image->real_values[address];
            address += current_image->padding_jump_value + 1;
            number_of_pixels += 2;
        }
    }
    else {
        // Three-dimensional volume
        for ( plane_counter = 0; plane_counter < current_image->logical_z_dimension; plane_counter++ ) {
            for ( line_counter = 0; line_counter < current_image->logical_y_dimension; line_counter++ ) {
                if ( line_counter == 0 || line_counter == current_image->logical_y_dimension - 1 ) {
                    // First and last line of that section
                    for ( pixel_counter = 0; pixel_counter < current_image->logical_x_dimension; pixel_counter++ ) {
                        sum += current_image->real_values[address];
                        address++;
                    }
                    address += current_image->padding_jump_value;
                    number_of_pixels += current_image->logical_x_dimension;
                }
                else {
                    // All other lines (only count first and last pixel)
                    sum += current_image->real_values[address];
                    address += current_image->logical_x_dimension - 1;
                    sum += current_image->real_values[address];
                    address += current_image->padding_jump_value + 1;
                    number_of_pixels += 2;
                }
            }
        }
    }

    return sum / float(number_of_pixels);
}

void create_black_sphere_mask(Image* mask_file, int x_sphere_center, int y_sphere_center, int z_spehere_center, float radius) {

    int   boxsize = mask_file->logical_x_dimension;
    int   i, j, k;
    int   dx, dy, dz;
    float d;
    // initialize the mask file to be 1.0
    mask_file->SetToConstant(1.0);

    long pixel_counter = 0;

    for ( k = 0; k < mask_file->logical_z_dimension; k++ ) {
        for ( j = 0; j < mask_file->logical_y_dimension; j++ ) {
            for ( i = 0; i < mask_file->logical_x_dimension; i++ ) {
                dx = i - x_sphere_center;
                dy = j - y_sphere_center;
                dz = k - z_spehere_center;
                d  = sqrtf(dx * dx + dy * dy + dz * dz);
                if ( d < radius ) {
                    mask_file->real_values[pixel_counter] = 0.0;
                }
                else {
                    mask_file->real_values[pixel_counter] = 1.0;
                }
                pixel_counter++;
            }
            pixel_counter += mask_file->padding_jump_value;
        }
    }
}

void create_white_sphere_mask(Image* mask_file, int x_sphere_center, int y_sphere_center, int z_spehere_center, float radius) {

    int   boxsize = mask_file->logical_x_dimension;
    int   i, j, k;
    int   dx, dy, dz;
    float d;
    // initialize the mask to 0.0
    mask_file->SetToConstant(0.0);

    long pixel_counter = 0;
    for ( k = 0; k < mask_file->logical_z_dimension; k++ ) {
        for ( j = 0; j < mask_file->logical_y_dimension; j++ ) {
            for ( i = 0; i < mask_file->logical_x_dimension; i++ ) {
                dx = i - x_sphere_center;
                dy = j - y_sphere_center;
                dz = k - z_spehere_center;
                d  = sqrtf(dx * dx + dy * dy + dz * dz);
                if ( d < radius ) {
                    mask_file->real_values[pixel_counter] = 1.0;
                }
                else {
                    mask_file->real_values[pixel_counter] = 0.0;
                }
                pixel_counter++;
            }
            pixel_counter += mask_file->padding_jump_value;
        }
    }
    //mask_file->QuickAndDirtyWriteSlices("make_white_sphere_mask_inside_function.mrc", 1, mask_file->logical_z_dimension);
}

void save_all_columns_sum_to_file(
        const std::vector<std::vector<float>>& all_columns_sum,
        const std::string&                     filename) {
    std::ofstream out_file(filename);
    if ( ! out_file.is_open( ) ) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        //return;
    }

    out_file << std::fixed << std::setprecision(2); // Set float precision to 2 decimal places

    for ( const auto& row : all_columns_sum ) {
        for ( size_t i = 0; i < row.size( ); ++i ) {
            out_file << row[i];
            if ( i < row.size( ) - 1 )
                out_file << ", ";
        }
        out_file << '\n';
    }

    out_file.close( );
}

// Detects the two strongest outer-edge peaks in a 1D intensity profile.
// Returns indices of the best peak pair (sorted low->high), or an empty vector if none found.
std::pair<int, int> findOuterTubeEdges(const std::vector<float>& cols, float min_tube_diameter, float max_tube_diameter) {
    int n = cols.size( );
    if ( n < 3 )
        return {-1, -1}; // need at least 3 points to form a peak

    // 1) Normalize the 1D profile
    float              minVal = *std::min_element(cols.begin( ), cols.end( ));
    float              maxVal = *std::max_element(cols.begin( ), cols.end( ));
    std::vector<float> norm(n);
    for ( int i = 0; i < n; ++i )
        norm[i] = cols[i] - minVal;

    float normMax = *std::max_element(norm.begin( ), norm.end( ));
    if ( normMax <= 0.0f )
        return {-1, -1};

    // Inverted profile for negative peaks
    std::vector<float> normInv(n);
    for ( int i = 0; i < n; ++i )
        normInv[i] = normMax - norm[i];

    // 2) Detect peaks
    std::vector<std::pair<int, float>> posPeaks;
    std::vector<std::pair<int, float>> negPeaks;

    for ( int i = 1; i < n - 1; ++i ) {
        if ( norm[i] > norm[i - 1] && norm[i] > norm[i + 1] ) {
            posPeaks.emplace_back(i, norm[i]);
        }
        if ( normInv[i] > normInv[i - 1] && normInv[i] > normInv[i + 1] ) {
            negPeaks.emplace_back(i, normInv[i]);
        }
    }
    // // debugging and printing the scores
    // std::cerr << "posPeaks (idx,val): ";
    // for ( auto& p : posPeaks )
    //     std::cerr << "(" << p.first << "," << p.second << ") ";
    // std::cerr << "\n";
    // std::cerr << "negPeaks (idx,val): ";
    // for ( auto& p : negPeaks )
    //     std::cerr << "(" << p.first << "," << p.second << ") ";
    // std::cerr << "\n";

    // helper function to find the best pair of peaks based on their height and distance between peaks
    auto bestPair = [&](const std::vector<std::pair<int, float>>& peaks)
            -> std::pair<float, std::pair<int, int>> {
        float               bestScore = -std::numeric_limits<float>::infinity( );
        std::pair<int, int> bestIdx   = {-1, -1};

        // Adding gap penalty and out of range penalty so that we would favor more the peaks within the range, but also if nothing was found within range, out of range peaks are saved and returned
        const float IDEAL_GAP           = min_tube_diameter;
        const float GAP_PENALTY         = 0.1f; // e.g. 0.1 points lost per pixel of gap deviation
        const float OUT_OF_RANGE_FACTOR = 10.0f; // scale factor for out-of-range penalty- changed that from 2 to 10 to heavily penalize out of range to favor in range more

        for ( size_t a = 0; a < peaks.size( ); ++a ) {
            for ( size_t b = a + 1; b < peaks.size( ); ++b ) {
                int i   = peaks[a].first;
                int j   = peaks[b].first;
                int gap = j - i;

                float sumAmp = peaks[a].second + peaks[b].second;
                float score  = sumAmp - GAP_PENALTY * std::fabs(gap - IDEAL_GAP);

                // scale penalty by how far out of range the gap is
                if ( gap < min_tube_diameter ) {
                    score -= OUT_OF_RANGE_FACTOR * (min_tube_diameter - gap);
                }
                else if ( gap > max_tube_diameter ) {
                    score -= OUT_OF_RANGE_FACTOR * (gap - max_tube_diameter);
                }

                if ( score > bestScore ) {
                    bestScore = score;
                    bestIdx   = {i, j};
                }
            }
        }

        return std::make_pair(bestScore, bestIdx);
    };

    // 3) Find best pair among positive peaks and among negative peaks.
    auto [scorePos, bestPos] = bestPair(posPeaks);
    auto [scoreNeg, bestNeg] = bestPair(negPeaks);

    std::pair<int, int> bestPairIdx = {-1, -1};

    // 4) If no valid pairs exist at all, return -1
    if ( scorePos == -std::numeric_limits<float>::infinity( ) &&
         scoreNeg == -std::numeric_limits<float>::infinity( ) ) {
        return {-1, -1};
    }

    // 5) keeping the values of the best negative peaks as reference
    bestPairIdx     = bestNeg;
    float bestScore = scoreNeg;

    // find the highest negative peaks within the range of the expected diameter
    // then find the positive peak before the first negative peak and the positive peak after the second negative peak and those should be the outer edges
    if ( bestNeg.first != -1 && bestNeg.second != -1 ) {
        int iNeg = bestNeg.first;
        int jNeg = bestNeg.second;
        if ( iNeg > jNeg )
            std::swap(iNeg, jNeg); // enforce left->right

        // Find last positive BEFORE iNeg
        int   posBefore = -1;
        float ampBefore = 0;
        for ( auto it = posPeaks.rbegin( ); it != posPeaks.rend( ); ++it ) {
            if ( it->first < iNeg ) {
                posBefore = it->first;
                ampBefore = it->second;
                break;
            }
        }

        // Find first positive AFTER jNeg
        int   posAfter = -1;
        float ampAfter = 0;
        for ( auto& p : posPeaks ) {
            if ( p.first > jNeg ) {
                posAfter = p.first;
                ampAfter = p.second;
                break;
            }
        }
        const float IDEAL_GAP           = min_tube_diameter;
        const float GAP_PENALTY         = 0.1f; // e.g. 0.1 points lost per pixel of gap deviation
        const float OUT_OF_RANGE_FACTOR = 10.0f; // scale factor for out-of-range penalty

        // Step 3: Only refine if both positives exist and are ordered
        if ( posAfter != -1 && posBefore != -1 && posAfter < posBefore ) {
            int   gap    = posBefore - posAfter;
            float sumAmp = ampAfter + ampBefore;
            float score  = sumAmp - GAP_PENALTY * std::fabs(gap - IDEAL_GAP);

            if ( gap < min_tube_diameter )
                score -= OUT_OF_RANGE_FACTOR * (min_tube_diameter - gap);
            else if ( gap > max_tube_diameter )
                score -= OUT_OF_RANGE_FACTOR * (gap - max_tube_diameter);

            // Step 4: Replace if adjacency score is better
            if ( score > bestScore ) {
                bestScore   = score;
                bestPairIdx = {posAfter, posBefore};
            }
        }
    }
    // Final: enforce sorted order before returning
    if ( bestPairIdx.first > bestPairIdx.second )
        std::swap(bestPairIdx.first, bestPairIdx.second);

    return std::make_pair(bestPairIdx.first, bestPairIdx.second);
}
