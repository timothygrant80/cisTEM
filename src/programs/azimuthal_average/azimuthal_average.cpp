#include "../../core/core_headers.h"

class
        AzimuthalAverage : public MyApp {

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

void azimuthal_alignment(Image* input_stack, Image* input_stack_times_ctf, ImageFile* my_input_file, ctf_parameters* ctf_parameters_stack, bool phase_flip_only, bool if_input_ctf_values_from_text_file, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float start_angle_for_peak_search, float end_angle_for_peak_search, float rotation_step_size, float max_rotation_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, bool refine_locally, float refinement_factor, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, float* psi_angles, float* ctf_sum_of_squares, bool use_memory, bool do_fill_sum_of_squares);
void apply_ctf(Image* current_image, CTF ctf_to_apply, float* ctf_sum_of_squares, bool absolute, bool do_fill_sum_of_squares);
void divide_by_ctf_sum_of_squares(Image* current_image, float* ctf_sum_of_squares); // const float
void sum_image_direction(Image* current_image, int dim);

//void sum_image_direction(Image *current_image, Image *directional_image_sum, int dim);
void  scale_and_subtract_reference(Image* input_stack, ImageFile* my_input_file, int number_of_images, float pixel_size, float* x_shifts, float* y_shifts, float* psi_angles, Image* current_volume, Image* masked_volume, int number_of_models, ctf_parameters* ctf_parameters_stack, bool absolute, float padding_factor, int max_threads, bool use_memory);
void  normalize_image(Image* input_image, float pixel_size, float mask_falloff);
void  invert_mask(Image* mask_file);
void  writeToStarFile(ctf_parameters* ctf_parameters_stack, int number_of_images, int number_of_models, float pixel_size, float* x_shifts, float* y_shifts, float* psi_angles);
float ReturnAverageOfRealValuesOnVerticalEdges(Image* current_image);
float ReturnDifferenceOfSquares(Image* first_image, Image* second_image);

IMPLEMENT_APP(AzimuthalAverage)

// override the DoInteractiveUserInput

void AzimuthalAverage::DoInteractiveUserInput( ) {
    // intial parameters
    int         new_z_size = 1;
    std::string text_filename;
    float       pixel_size;

    // mask parameters
    bool        apply_mask = false;
    std::string input_mask_filename;
    int         number_of_models  = 4;
    float       cosine_edge       = 10.0;
    float       outside_weight    = 0.0;
    float       filter_radius     = 0.0;
    float       outside_value     = 0.0;
    bool        use_outside_value = false;
    bool        use_memory        = false;

    // ctf parameters
    float acceleration_voltage;
    float spherical_aberration;
    float amplitude_contrast;
    float defocus_1              = 0.0;
    float defocus_2              = 0.0;
    float astigmatism_angle      = 0.0;
    float additional_phase_shift = 0.0;
    bool  input_ctf_values_from_text_file;
    bool  phase_flip_only;

    // alignment expert options
    int max_iterations = 5;
    // translation (Angstroms)
    float minimum_shift_in_angstroms         = 2.0;
    float maximum_shift_in_angstroms         = 80.0;
    float termination_threshold_in_angstroms = 1.0;
    // rotation (degrees)
    float psi_min                          = -180.0;
    float psi_max                          = 180.0;
    float psi_step                         = 5.0;
    float termination_threshold_in_degrees = 0.25;
    bool  local_refine                     = true;
    /*
	float psi_min = -5.0;
	float psi_max = 5.0; 
	float psi_step = 0.25;
	float termination_threshold_in_degrees = 0.25;
	bool local_refine = false;
	*/
    float refinement_factor = 20.0;
    // image processing
    float bfactor_in_angstroms                 = 1500;
    bool  should_mask_central_cross            = false;
    int   horizontal_mask_size                 = 1;
    int   vertical_mask_size                   = 1;
    float exposure_per_frame                   = 0.0;
    int   number_of_frames_for_running_average = 1;
    float padding                              = 2.0;

    bool set_expert_options;
    int  max_threads;

    UserInput* my_input = new UserInput("AzimuthalAverage", 1.0);

    std::string input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of input stack", "input_stack.mrc", true);
    apply_mask                 = my_input->GetYesNoFromUser("Apply mask to 3D azimuthal average?", "Mask out region of interest", "NO");
    use_memory                 = my_input->GetYesNoFromUser("Allocate images to memory?", "Choice between memory allocation or using functions; no is recommended for systems with limited memory.", "NO");

    // get mask properties from user
    if ( apply_mask == true ) {
        input_mask_filename = my_input->GetFilenameFromUser("Input mask file name", "The mask to be applied to the 3D azimuthal average model", "my_mask.mrc", true);
        number_of_models    = my_input->GetIntFromUser("Number of models to generate", "3D azimuthal average model is rotated in increments of (360/n) degrees", "4", 1);
        cosine_edge         = my_input->GetFloatFromUser("Width of cosine edge (A)", "Width of the smooth edge to add to the mask in Angstroms", "10.0", 0.0);
        outside_weight      = my_input->GetFloatFromUser("Weight of density outside mask", "Factor to multiply density outside of the mask", "0.0", 0.0, 1.0);
        filter_radius       = my_input->GetFloatFromUser("Low-pass filter outside mask (A)", "Low-pass filter to be applied to the density outside the mask", "0.0", 0.0);
        outside_value       = my_input->GetFloatFromUser("Outside mask value", "Value used to set density outside the mask", "0.0", 0.0);
        use_outside_value   = my_input->GetYesNoFromUser("Use outside mask value", "Should the density outside the mask be set to the user-provided value", "No");
    }

    // get CTF from user
    pixel_size           = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (keV)", "Acceleration voltage, in keV", "300.0", 0.0, 500.0);
    spherical_aberration = my_input->GetFloatFromUser("Spherical aberration (mm)", "Objective lens spherical aberration", "2.7", 0.0);
    amplitude_contrast   = my_input->GetFloatFromUser("Amplitude contrast", "Fraction of total contrast attributed to amplitude contrast", "0.07", 0.0);

    input_ctf_values_from_text_file = my_input->GetYesNoFromUser("Use a text file to input defocus values?", "If yes, a text file with one line per image is required", "NO");

    if ( input_ctf_values_from_text_file == true ) {
        text_filename = my_input->GetFilenameFromUser("File containing defocus values", "should have 3 or 4 values per line", "my_defocus.txt", true);
    }
    else {
        defocus_1              = my_input->GetFloatFromUser("Underfocus 1 (A)", "In Angstroms, the objective lens underfocus along the first axis", "1.2");
        defocus_2              = my_input->GetFloatFromUser("Underfocus 2 (A)", "In Angstroms, the objective lens underfocus along the second axis", "1.2");
        astigmatism_angle      = my_input->GetFloatFromUser("Astigmatism angle", "Angle between the first axis and the x axis of the image", "0.0");
        additional_phase_shift = my_input->GetFloatFromUser("Additional phase shift (rad)", "Additional phase shift relative to undiffracted beam, as introduced for example by a phase plate", "0.0");
    }

    phase_flip_only = my_input->GetYesNoFromUser("Phase Flip Only", "If Yes, only phase flipping is performed", "NO");

    set_expert_options = my_input->GetYesNoFromUser("Set Expert Options?", "Set these for more control, hopefully not needed", "NO");

    // set alignment options from user
    if ( set_expert_options == true ) {
        max_iterations                     = my_input->GetIntFromUser("Maximum number of iterations", "Alignment will stop at this number, even if the threshold shift is not reached", "5", 0);
        minimum_shift_in_angstroms         = my_input->GetFloatFromUser("Minimum shift for initial search (A)", "Initial search will be limited to between the inner and outer radii.", "2.0", 0.0);
        maximum_shift_in_angstroms         = my_input->GetFloatFromUser("Outer radius shift limit (A)", "The maximum shift of each alignment step will be limited to this value.", "80.0", minimum_shift_in_angstroms);
        termination_threshold_in_angstroms = my_input->GetFloatFromUser("Termination shift threshold (A)", "Alignment will iterate until the maximum shift is below this value", "1", 0.0);

        psi_min                          = my_input->GetFloatFromUser("Minimum rotation for initial search (degrees)", "Initial search will be limited to between the inner and outer radii.", "-180.0", -180.0);
        psi_max                          = my_input->GetFloatFromUser("Outer radius rotation limit (degrees)", "The maximum rotation of each alignment step will be limited to this value.", "180.0", 180.0);
        psi_step                         = my_input->GetFloatFromUser("Rotation step size (degrees)", "The step size of each rotation will be limited to this value.", "5.0", 0.0);
        termination_threshold_in_degrees = my_input->GetFloatFromUser("Termination rotation threshold (degrees)", "Alignment will iterate until the maximum rotation is below this value", "0.25", 0.0);

        bfactor_in_angstroms = my_input->GetFloatFromUser("B-factor to apply to images (A^2)", "This B-Factor will be used to filter the reference prior to alignment", "1500", 0.0);

        local_refine      = my_input->GetYesNoFromUser("Local refinment of alignment parameters?", "If Yes, in-plane rotation angle is refined", "YES");
        refinement_factor = my_input->GetFloatFromUser("Rotation refinement factor", "The original step size of each rotation will be divided by this number.", "20.0", 1.0);

        padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    }

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    my_current_job.Reset(38);
    my_current_job.ManualSetArguments("tbtiffffbffffbtffffbiffffffffbiifibffib", input_filename.c_str( ),
                                      apply_mask,
                                      input_mask_filename.c_str( ),
                                      number_of_models,
                                      cosine_edge,
                                      outside_weight,
                                      filter_radius,
                                      outside_value,
                                      use_outside_value,
                                      pixel_size,
                                      acceleration_voltage,
                                      spherical_aberration,
                                      amplitude_contrast,
                                      input_ctf_values_from_text_file,
                                      text_filename.c_str( ),
                                      defocus_1,
                                      defocus_2,
                                      astigmatism_angle,
                                      additional_phase_shift,
                                      phase_flip_only,
                                      max_iterations,
                                      minimum_shift_in_angstroms,
                                      maximum_shift_in_angstroms,
                                      termination_threshold_in_angstroms,
                                      psi_min,
                                      psi_max,
                                      psi_step,
                                      termination_threshold_in_degrees,
                                      bfactor_in_angstroms,
                                      should_mask_central_cross,
                                      horizontal_mask_size,
                                      vertical_mask_size,
                                      exposure_per_frame,
                                      number_of_frames_for_running_average,
                                      local_refine,
                                      refinement_factor,
                                      padding,
                                      max_threads,
                                      use_memory);
}

// override the do calculation method which will be what is actually run..

bool AzimuthalAverage::DoCalculation( ) {
    CTF current_ctf;

    // get the arguments for this job..
    std::string input_filename                       = my_current_job.arguments[0].ReturnStringArgument( );
    bool        apply_mask                           = my_current_job.arguments[1].ReturnBoolArgument( );
    std::string input_mask_filename                  = my_current_job.arguments[2].ReturnStringArgument( );
    int         number_of_models                     = my_current_job.arguments[3].ReturnIntegerArgument( );
    float       cosine_edge                          = my_current_job.arguments[4].ReturnFloatArgument( );
    float       outside_weight                       = my_current_job.arguments[5].ReturnFloatArgument( );
    float       filter_radius                        = my_current_job.arguments[6].ReturnFloatArgument( );
    float       outside_value                        = my_current_job.arguments[7].ReturnFloatArgument( );
    bool        use_outside_value                    = my_current_job.arguments[8].ReturnBoolArgument( );
    float       pixel_size                           = my_current_job.arguments[9].ReturnFloatArgument( );
    float       acceleration_voltage                 = my_current_job.arguments[10].ReturnFloatArgument( );
    float       spherical_aberration                 = my_current_job.arguments[11].ReturnFloatArgument( );
    float       amplitude_contrast                   = my_current_job.arguments[12].ReturnFloatArgument( );
    bool        input_ctf_values_from_text_file      = my_current_job.arguments[13].ReturnBoolArgument( );
    std::string text_filename                        = my_current_job.arguments[14].ReturnStringArgument( );
    float       defocus_1                            = my_current_job.arguments[15].ReturnFloatArgument( );
    float       defocus_2                            = my_current_job.arguments[16].ReturnFloatArgument( );
    float       astigmatism_angle                    = my_current_job.arguments[17].ReturnFloatArgument( );
    float       additional_phase_shift               = my_current_job.arguments[18].ReturnFloatArgument( );
    bool        phase_flip_only                      = my_current_job.arguments[19].ReturnBoolArgument( );
    int         max_iterations                       = my_current_job.arguments[20].ReturnIntegerArgument( );
    float       minimum_shift_in_angstroms           = my_current_job.arguments[21].ReturnFloatArgument( );
    float       maximum_shift_in_angstroms           = my_current_job.arguments[22].ReturnFloatArgument( );
    float       termination_threshold_in_angstroms   = my_current_job.arguments[23].ReturnFloatArgument( );
    float       psi_min                              = my_current_job.arguments[24].ReturnFloatArgument( );
    float       psi_max                              = my_current_job.arguments[25].ReturnFloatArgument( );
    float       psi_step                             = my_current_job.arguments[26].ReturnFloatArgument( );
    float       termination_threshold_in_degrees     = my_current_job.arguments[27].ReturnFloatArgument( );
    float       bfactor_in_angstroms                 = my_current_job.arguments[28].ReturnFloatArgument( );
    bool        should_mask_central_cross            = my_current_job.arguments[29].ReturnBoolArgument( );
    int         horizontal_mask_size                 = my_current_job.arguments[30].ReturnIntegerArgument( );
    int         vertical_mask_size                   = my_current_job.arguments[31].ReturnIntegerArgument( );
    float       exposure_per_frame                   = my_current_job.arguments[32].ReturnFloatArgument( );
    int         number_of_frames_for_running_average = my_current_job.arguments[33].ReturnIntegerArgument( );
    bool        local_refine                         = my_current_job.arguments[34].ReturnBoolArgument( );
    float       refinement_factor                    = my_current_job.arguments[35].ReturnFloatArgument( );
    float       padding_factor                       = my_current_job.arguments[36].ReturnFloatArgument( );
    int         max_threads                          = my_current_job.arguments[37].ReturnIntegerArgument( );
    bool        use_memory                           = my_current_job.arguments[38].ReturnBoolArgument( );

    // The Files
    ImageFile        my_input_file(input_filename, false);
    MRCFile*         my_mask_file;
    NumericTextFile* input_text;
    long             number_of_input_images = my_input_file.ReturnNumberOfSlices( );
    Image            current_image;

    if ( apply_mask == true ) {
        if ( ! DoesFileExist(input_mask_filename) ) {
            SendError(wxString::Format("Error: Mask %s not found\n", input_mask_filename));
            exit(-1);
        }

        my_mask_file = new MRCFile(input_mask_filename, false);
    }

    if ( input_ctf_values_from_text_file == true ) {
        input_text = new NumericTextFile(text_filename, OPEN_TO_READ);
        if ( input_text->number_of_lines != number_of_input_images ) {
            SendError("Error: Number of lines in defocus text file != number of images!");
            DEBUG_ABORT;
        }

        if ( input_text->records_per_line != 3 && input_text->records_per_line != 4 ) {
            SendError("Error: Expect 3 or 4 records per line in defocus text file");
            DEBUG_ABORT;
        }
    }

    // CTF object
    // temporary array to hold CTF parameters read from file
    float temp_float[5];
    // store CTF parameters for each image in the stack
    ctf_parameters* ctf_parameters_stack = new ctf_parameters[number_of_input_images];

    current_ctf.Init(acceleration_voltage, spherical_aberration, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.5, 0.0, pixel_size, additional_phase_shift);

    // Profiling
    wxDateTime overall_start = wxDateTime::Now( );
    wxDateTime overall_finish;
    wxDateTime read_frames_start;
    wxDateTime read_frames_finish;
    wxDateTime first_alignment_start;
    wxDateTime first_alignment_finish;
    wxDateTime main_alignment_start;
    wxDateTime main_alignment_finish;
    wxDateTime subtract_start;
    wxDateTime subtract_finish;

    long pixel_counter; // not used
    long image_counter;

    wxPrintf("\nReading Images...\n");
    // read in image stack and CTF
    read_frames_start = wxDateTime::Now( );

    // rotational and translational alignment of images
    Image* image_stack;
    if ( use_memory )
        image_stack = new Image[number_of_input_images];
    else
        image_stack = nullptr;

    // input stack times ctf
    Image* image_stack_times_ctf;
    if ( use_memory )
        image_stack_times_ctf = new Image[number_of_input_images];
    else
        image_stack_times_ctf = nullptr;

    // downweighted input stack
    //Image *image_stack_subtracted = new Image[number_of_input_images*number_of_models];
    // allocate arrays for the ctf sum of squares
    float* ctf_sum_of_squares;
    // Arrays to hold the shifts and rotations
    float* x_shifts   = new float[number_of_input_images];
    float* y_shifts   = new float[number_of_input_images];
    float* psi_angles = new float[number_of_input_images];
    float  min_shift_in_pixels;
    float  max_shift_in_pixels;
    float  termination_threshold_in_pixels;
    float  unitless_bfactor;

    // read CTF from file into array
    if ( input_ctf_values_from_text_file == true ) {
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {

            input_text->ReadLine(temp_float);

            if ( input_text->records_per_line == 3 ) {
                ctf_parameters_stack[image_counter].acceleration_voltage          = acceleration_voltage;
                ctf_parameters_stack[image_counter].spherical_aberration          = spherical_aberration;
                ctf_parameters_stack[image_counter].amplitude_contrast            = amplitude_contrast;
                ctf_parameters_stack[image_counter].defocus_1                     = temp_float[0];
                ctf_parameters_stack[image_counter].defocus_2                     = temp_float[1];
                ctf_parameters_stack[image_counter].astigmatism_angle             = temp_float[2];
                ctf_parameters_stack[image_counter].lowest_frequency_for_fitting  = 0.0;
                ctf_parameters_stack[image_counter].highest_frequency_for_fitting = 0.5;
                ctf_parameters_stack[image_counter].astigmatism_tolerance         = 0.0;
                ctf_parameters_stack[image_counter].pixel_size                    = pixel_size;
                ctf_parameters_stack[image_counter].additional_phase_shift        = 0.0;
            }
            else {
                ctf_parameters_stack[image_counter].acceleration_voltage          = acceleration_voltage;
                ctf_parameters_stack[image_counter].spherical_aberration          = spherical_aberration;
                ctf_parameters_stack[image_counter].amplitude_contrast            = amplitude_contrast;
                ctf_parameters_stack[image_counter].defocus_1                     = temp_float[0];
                ctf_parameters_stack[image_counter].defocus_2                     = temp_float[1];
                ctf_parameters_stack[image_counter].astigmatism_angle             = temp_float[2];
                ctf_parameters_stack[image_counter].lowest_frequency_for_fitting  = 0.0;
                ctf_parameters_stack[image_counter].highest_frequency_for_fitting = 0.5;
                ctf_parameters_stack[image_counter].astigmatism_tolerance         = 0.0;
                ctf_parameters_stack[image_counter].pixel_size                    = pixel_size;
                ctf_parameters_stack[image_counter].additional_phase_shift        = temp_float[3];
            }
        }
    }

    // read image stack and FFT
    if ( use_memory ) {
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            // Read from disk
            image_stack[image_counter].ReadSlice(&my_input_file, image_counter + 1);

            // Normalize images
            normalize_image(&image_stack[image_counter], pixel_size, 10.0);

            // FT
            image_stack[image_counter].ForwardFFT( );
            image_stack[image_counter].ZeroCentralPixel( );

            // Init shifts
            x_shifts[image_counter]   = 0.0;
            y_shifts[image_counter]   = 0.0;
            psi_angles[image_counter] = 0.0;
        }
    }

    else {
        // Init shifts
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            x_shifts[image_counter]   = 0.0;
            y_shifts[image_counter]   = 0.0;
            psi_angles[image_counter] = 0.0;
        }
    }

    bool do_fill_sum_of_squares = true;

    // apply CTF to copy of image stack
    // allocate arrays for the ctf sum of squares now that we know size of each image
    if ( use_memory ) {
        ctf_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];
        ZeroFloatArray(ctf_sum_of_squares, image_stack[0].real_memory_allocated / 2);
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            image_stack_times_ctf[image_counter].CopyFrom(&image_stack[image_counter]);

            if ( input_ctf_values_from_text_file )
                current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
            apply_ctf(&image_stack_times_ctf[image_counter], current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);
        }
    }

    else {
        current_image.ReadSlice(&my_input_file, 1);
        ctf_sum_of_squares = new float[current_image.real_memory_allocated / 2];
        ZeroFloatArray(ctf_sum_of_squares, current_image.real_memory_allocated / 2);

        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            current_image.ReadSlice(&my_input_file, image_counter + 1);
            normalize_image(&current_image, pixel_size, 10.0);
            current_image.ForwardFFT( );
            current_image.ZeroCentralPixel( );
            // read CTF from file
            if ( input_ctf_values_from_text_file )
                current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);

            apply_ctf(&current_image, current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);
        }
    }

    // Now we have sum_of_squares, don't need to update again (only relevant when not using memory)
    do_fill_sum_of_squares = false;

    read_frames_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_to_read = read_frames_finish.Subtract(read_frames_start);
    wxPrintf(" Read frames                : %s\n", time_to_read.Format( ));

    wxPrintf("\nAligning Images...\n\n");
    // convert shifts to pixels..
    min_shift_in_pixels             = minimum_shift_in_angstroms / pixel_size;
    max_shift_in_pixels             = maximum_shift_in_angstroms / pixel_size;
    termination_threshold_in_pixels = termination_threshold_in_angstroms / pixel_size;
    if ( min_shift_in_pixels <= 1.01 )
        min_shift_in_pixels = 1.01; // we always want to ignore the central peak initially.
    // calculate the bfactor
    unitless_bfactor = bfactor_in_angstroms / pow(pixel_size, 2);

    // now do the actual refinement
    main_alignment_start = wxDateTime::Now( );
    azimuthal_alignment(image_stack, image_stack_times_ctf, &my_input_file, ctf_parameters_stack, phase_flip_only, input_ctf_values_from_text_file, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, psi_min, psi_max, psi_step, termination_threshold_in_degrees, pixel_size, number_of_frames_for_running_average, local_refine, refinement_factor, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, psi_angles, ctf_sum_of_squares, use_memory, do_fill_sum_of_squares);
    main_alignment_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_to_align = main_alignment_finish.Subtract(main_alignment_start);
    wxPrintf(" Main alignment                : %s\n", time_to_align.Format( ));

    // we should be finished with alignment, now we need to make the final sum
    wxPrintf("\nAveraging Images...\n\n");
    ProgressBar* my_progress = new ProgressBar(number_of_input_images);

    // CTF-corrected average equal to sum( X_i * CTF_i ) / sqrt( sum( CTF_i * CTF_i ) )
    Image sum_image;

    if ( use_memory )
        sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
    else
        sum_image.Allocate(my_input_file.ReturnXSize( ), my_input_file.ReturnYSize( ), false);
    sum_image.SetToConstant(0.0);

    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        if ( use_memory ) {
            // write to disk
            image_stack[image_counter].BackwardFFT( );
            //image_stack[image_counter].QuickAndDirtyWriteSlice("my_aligned_frames.mrc", image_counter + 1);

            sum_image.AddImage(&image_stack_times_ctf[image_counter]);
        }

        // Have to apply the latest (and last/best) shifts to the sum outside of alignment function
        else {
            current_image.ReadSlice(&my_input_file, image_counter + 1);
            normalize_image(&current_image, pixel_size, 10.0);
            current_image.ForwardFFT( );
            current_image.ZeroCentralPixel( );
            if ( input_ctf_values_from_text_file )
                current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
            apply_ctf(&current_image, current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);

            current_image.BackwardFFT( );

            // Rotate the image
            current_image.Rotate2DInPlace(psi_angles[image_counter], 0.0);
            current_image.ForwardFFT( );

            // Shift the image
            current_image.PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 0.0);

            // Have the image in needed state, apply operation
            sum_image.AddImage(&current_image);
        }
        my_progress->Update(image_counter + 1);
    }

    // vertically-averaged CTF-corrected sum
    divide_by_ctf_sum_of_squares(&sum_image, ctf_sum_of_squares);
    sum_image.BackwardFFT( );

    // Align vertically, then average rotationally
    // Helps eliminate the asymmetry from the tubules, and reduces low res blurring at the center of the image
    sum_image_direction(&sum_image, 2);
    sum_image.QuickAndDirtyWriteSlice(wxString::Format("my_aligned_sum_%i.mrc", max_iterations).ToStdString( ), 1);
    sum_image.ApplyRampFilter( );
    sum_image.AverageRotationally( );

    delete my_progress;

    // rotationally averaged 3D reconstruction
    Image my_volume;

    my_volume.Allocate(padding_factor * sum_image.logical_x_dimension, padding_factor * sum_image.logical_y_dimension, padding_factor * sum_image.logical_x_dimension, true);
    my_volume.SetToConstant(0.0);

    // Fill in volume with info from rotational average
    float edge_value = sum_image.ReturnAverageOfRealValuesOnEdges( );
    sum_image.Resize(my_volume.logical_x_dimension, my_volume.logical_y_dimension, 1, edge_value);
    long pixel_coord_xy  = 0;
    long pixel_coord_xyz = 0;
    long volume_counter  = 0;
    for ( int z = 0; z < my_volume.logical_z_dimension; z++ ) {
        for ( int y = 0; y < my_volume.logical_y_dimension; y++ ) {
            for ( int x = 0; x < my_volume.logical_x_dimension; x++ ) {
                pixel_coord_xy = sum_image.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                //pixel_coord_xyz = my_volume.ReturnReal1DAddressFromPhysicalCoord(x, y, z);
                //my_volume.real_values[pixel_coord_xyz] = sum_image.real_values[pixel_coord_xy];
                my_volume.real_values[volume_counter] = sum_image.real_values[pixel_coord_xy];
                volume_counter++;
            }
            volume_counter += sum_image.padding_jump_value;
        }
    }

    // print out 2D rotational average

    sum_image.Resize(my_volume.logical_x_dimension / padding_factor, my_volume.logical_y_dimension / padding_factor, 1);
    sum_image.QuickAndDirtyWriteSlice("my_rotationally_averaged_sum.mrc", 1);

    // scale and subtract masked 3D AA projection from unaligned input stack
    wxPrintf("\nScaling and Subtracting References...\n");
    subtract_start = wxDateTime::Now( );

    // generate masked 3D AA models
    Image my_masked_volume;
    Image my_mask;

    my_masked_volume.CopyFrom(&my_volume);

    float filter_edge = 40.0;
    float mask_volume_in_voxels;
    if ( apply_mask == true ) {
        my_mask.Allocate(my_volume.logical_x_dimension, my_volume.logical_y_dimension, my_volume.logical_z_dimension, true);
        my_mask.ReadSlices(my_mask_file, 1, my_mask_file->ReturnNumberOfSlices( ));

        wxPrintf("\nMasking Volume...\n");

        if ( ! my_volume.HasSameDimensionsAs(&my_mask) ) {
            wxPrintf("\nVolume and mask file have different dimensions\n");
            DEBUG_ABORT;
        }

        // multiply mask and 3D AA model
        invert_mask(&my_mask);

        if ( filter_radius == 0.0 )
            filter_radius = pixel_size;
        mask_volume_in_voxels = my_masked_volume.ApplyMask(my_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
    }

    my_masked_volume.QuickAndDirtyWriteSlices("my_masked_volume.mrc", 1, my_masked_volume.logical_z_dimension);

    scale_and_subtract_reference(image_stack, &my_input_file, number_of_input_images, pixel_size, x_shifts, y_shifts, psi_angles, &my_volume, &my_masked_volume, number_of_models, ctf_parameters_stack, phase_flip_only, padding_factor, max_threads, use_memory);

    // write out cisTEM star file for subtracted images
    writeToStarFile(ctf_parameters_stack, number_of_input_images, number_of_models, pixel_size, x_shifts, y_shifts, psi_angles);
    subtract_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_to_subtract = subtract_finish.Subtract(subtract_start);
    wxPrintf(" Subtract frames                : %s\n", time_to_subtract.Format( ));

    /*
	wxPrintf("\nSaving Downweighted Images...\n");
	// save downweighted image stack
	for (image_counter = 0; image_counter < number_of_models * number_of_input_images; image_counter++)
	{
		// write to disk
		image_stack_subtracted[image_counter].QuickAndDirtyWriteSlice("my_subtracted_frames_copy.mrc", image_counter + 1);
	}
*/
    // print out scaled 3D reconstruction -- also get rid of padding before writing out
    my_volume.Resize(my_volume.logical_x_dimension / padding_factor, my_volume.logical_y_dimension / padding_factor, my_volume.logical_z_dimension / padding_factor);
    my_volume.QuickAndDirtyWriteSlices("my_averaged_volume.mrc", 1, my_volume.logical_z_dimension);

    // save orthogonal views
    Image orth_image;
    orth_image.Allocate(my_volume.logical_x_dimension * 3, my_volume.logical_y_dimension * 2, 1, true);
    my_volume.CreateOrthogonalProjectionsImage(&orth_image);
    orth_image.QuickAndDirtyWriteSlice("my_orthogonal_views.mrc", 1);

    // clean-up
    if ( input_ctf_values_from_text_file == true )
        delete input_text;
    if ( apply_mask == true )
        delete my_mask_file;
    ;
    //delete my_progress;
    delete[] x_shifts;
    delete[] y_shifts;
    delete[] psi_angles;
    delete[] image_stack;
    //delete[] image_stack_subtracted;
    delete[] image_stack_times_ctf;
    delete[] ctf_sum_of_squares;
    delete[] ctf_parameters_stack;
    wxPrintf("\n\n");

    overall_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_total = overall_finish.Subtract(overall_start);
    wxPrintf(" Overall time                : %s\n", time_total.Format( ));

    wxPrintf("\nAzimuthalAverage: Normal termination\n\n");

    return true;
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

void divide_by_ctf_sum_of_squares(Image* current_image, float* ctf_sum_of_squares) {
    // normalize by sum of squared CTFs (voxel by voxel)
    long pixel_counter = 0;

    for ( int j = 0; j <= current_image->physical_upper_bound_complex_y; j++ ) {
        for ( int i = 0; i <= current_image->physical_upper_bound_complex_x; i++ ) {
            if ( ctf_sum_of_squares[pixel_counter] != 0.0 )
                current_image->complex_values[pixel_counter] /= sqrtf(ctf_sum_of_squares[pixel_counter]);
            pixel_counter++;
        }
    }
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

// rotational and translational alignment of tubes
// function similar to unblur_refine_alignment in unblur.cpp with a few differences
// 1. Does not do smoothing, 2. Does not subtract current image from sum 3. Does not calculate running average (must be set to 1)
void azimuthal_alignment(Image* input_stack, Image* input_stack_times_ctf, ImageFile* my_input_file, ctf_parameters* ctf_parameters_stack, bool phase_flip_only, bool if_input_ctf_values_from_text_file, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float start_angle_for_peak_search, float end_angle_for_peak_search, float rotation_step_size, float max_rotation_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, bool refine_locally, float refinement_factor, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, float* psi_angles, float* ctf_sum_of_squares, bool use_memory, bool do_fill_sum_of_squares) {
    long pixel_counter;
    long image_counter;
    int  running_average_counter;
    int  start_frame_for_average;
    int  end_frame_for_average;
    long iteration_counter;

    CTF current_ctf;

    int number_of_middle_image    = number_of_images / 2;
    int running_average_half_size = (number_of_frames_for_running_average - 1) / 2;
    if ( running_average_half_size < 1 )
        running_average_half_size = 1;

    // rotational alignment parameters
    float psi;
    // initial refinement on coarse angular grid
    int number_of_rotations = int((end_angle_for_peak_search - start_angle_for_peak_search) / rotation_step_size + 0.5);
    if ( number_of_rotations < 1 )
        number_of_rotations = 1;
    rotation_step_size = (end_angle_for_peak_search - start_angle_for_peak_search) / number_of_rotations;
    if ( rotation_step_size < max_rotation_convergence_threshold )
        rotation_step_size = max_rotation_convergence_threshold;
    // another refinement on a finer angular grid
    int number_of_local_rotations = int((2 * rotation_step_size) / (rotation_step_size / refinement_factor) + 0.5);
    if ( number_of_local_rotations < 1 )
        number_of_local_rotations = 1;
    float local_rotation_step_size = (2 * rotation_step_size) / number_of_local_rotations;
    if ( local_rotation_step_size < max_rotation_convergence_threshold )
        local_rotation_step_size = max_rotation_convergence_threshold;

    float* current_psi_angles = new float[number_of_images];

    float average_image_psi_rotation;
    float middle_image_psi_rotation;

    float max_rotation;
    float total_rotation;

    // translational alignment parameters
    float* current_x_shifts = new float[number_of_images];
    float* current_y_shifts = new float[number_of_images];

    float average_image_x_shift;
    float average_image_y_shift;
    float middle_image_x_shift;
    float middle_image_y_shift;

    float max_shift;
    float total_shift;

    float best_inplane_score = -FLT_MAX;
    float best_inplane_values[3];

    if ( IsOdd(savitzy_golay_window_size) == false )
        savitzy_golay_window_size++;
    if ( savitzy_golay_window_size < 5 )
        savitzy_golay_window_size = 5;

    Image  sum_of_images;
    Image  sum_of_images_temp;
    Image* running_average_stack;
    Image* buffer_stack;
    if ( use_memory )
        buffer_stack = new Image[number_of_images];

    Image* stack_for_alignment; // pointer that can be switched between running average stack and image stack if necessary
    Peak   current_peak;

    Curve x_shifts_curve;
    Curve y_shifts_curve;

    if ( use_memory ) {
        sum_of_images.Allocate(input_stack_times_ctf[0].logical_x_dimension, input_stack_times_ctf[0].logical_y_dimension, false);
        sum_of_images.SetToConstant(0.0);
    }
    else {
        sum_of_images.Allocate(my_input_file->ReturnXSize( ), my_input_file->ReturnYSize( ), false);
        sum_of_images.SetToConstant(0.0);
    }

    // Only assign a stack if using memory; otherwise un-needed
    if ( use_memory ) {
        if ( number_of_frames_for_running_average > 1 ) {
            running_average_stack = new Image[number_of_images];

            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                running_average_stack[image_counter].Allocate(input_stack_times_ctf[image_counter].logical_x_dimension, input_stack_times_ctf[image_counter].logical_y_dimension, 1, false);
            }

            stack_for_alignment = running_average_stack;
        }
        else
            stack_for_alignment = input_stack_times_ctf;
    }

    // prepare the initial sum which is vertically-averaged CTF-corrected sum

    Image current_image;
    for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        if ( use_memory ) {
            sum_of_images.AddImage(&input_stack_times_ctf[image_counter]);
        }

        // Now we normalize and apply ctf to each image before summing for initial sum
        else {

            // Read in and set up
            current_image.ReadSlice(my_input_file, image_counter + 1);
            normalize_image(&current_image, pixel_size, 10.0);
            current_image.ForwardFFT( );
            current_image.ZeroCentralPixel( );
            if ( if_input_ctf_values_from_text_file )
                current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
            apply_ctf(&current_image, current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);

            // Add to the sum
            sum_of_images.AddImage(&current_image);
        }

        current_x_shifts[image_counter]   = 0;
        current_y_shifts[image_counter]   = 0;
        current_psi_angles[image_counter] = 0;
    }

    // vertical average
    divide_by_ctf_sum_of_squares(&sum_of_images, ctf_sum_of_squares);
    sum_of_images.BackwardFFT( );

    sum_image_direction(&sum_of_images, 2);
    sum_of_images.ApplyRampFilter( );
    sum_of_images.AverageRotationally( );
    sum_image_direction(&sum_of_images, 2);
    sum_of_images.ForwardFFT( );

    // print unaligned sum
    sum_of_images.QuickAndDirtyWriteSlice("my_aligned_sum_0.mrc", 1);

    // perform the main alignment loop until we reach a max shift less than wanted, or max iterations

    for ( iteration_counter = 1; iteration_counter <= max_iterations; iteration_counter++ ) {
        //	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
        max_shift    = -FLT_MAX;
        max_rotation = -FLT_MAX;

        float average_image_psi_rotation = 0.;
        float average_image_x_shift      = 0.;
        float average_image_y_shift      = 0.;

        // make the current running average if necessary
        if ( number_of_frames_for_running_average > 1 ) {
#pragma omp          parallel for default(shared) num_threads(max_threads) private(image_counter, start_frame_for_average, end_frame_for_average, running_average_counter)
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                         start_frame_for_average = image_counter - running_average_half_size;
                         end_frame_for_average   = image_counter + running_average_half_size;

                         if ( start_frame_for_average < 0 ) {
                             end_frame_for_average -= start_frame_for_average; // add it to the right
                             start_frame_for_average = 0;
                }

                         if ( end_frame_for_average >= number_of_images ) {
                             start_frame_for_average -= (end_frame_for_average - (number_of_images - 1));
                             end_frame_for_average = number_of_images - 1;
                }

                         if ( start_frame_for_average < 0 )
                    start_frame_for_average = 0;
                if ( end_frame_for_average >= number_of_images )
                    end_frame_for_average = number_of_images - 1;
                if ( use_memory )
                    running_average_stack[image_counter].SetToConstant(0.0f);

                for ( running_average_counter = start_frame_for_average; running_average_counter <= end_frame_for_average; running_average_counter++ ) {
                             if ( use_memory )
                        running_average_stack[image_counter].AddImage(&input_stack_times_ctf[running_average_counter]);
                }
            }
        }

        // do no subtract current image from sum
#pragma omp parallel num_threads(max_threads) default(none) shared(current_psi_angles, current_x_shifts, current_y_shifts, ctf_parameters_stack, if_input_ctf_values_from_text_file, pixel_size, start_angle_for_peak_search, number_of_rotations, refine_locally, number_of_local_rotations, local_rotation_step_size, input_stack_times_ctf, my_input_file, phase_flip_only, stack_for_alignment, sum_of_images, number_of_images, inner_radius_for_peak_search, outer_radius_for_peak_search, rotation_step_size, unitless_bfactor, mask_central_cross, width_of_vertical_line, width_of_horizontal_line, use_memory, ctf_sum_of_squares, do_fill_sum_of_squares) private(image_counter, sum_of_images_temp, current_peak, psi, best_inplane_score, best_inplane_values, current_image, current_ctf)
        { // for omp

            if ( use_memory ) {
                         sum_of_images_temp.Allocate(input_stack_times_ctf[0].logical_x_dimension, input_stack_times_ctf[0].logical_y_dimension, false);
            }
            else {
                         sum_of_images_temp.Allocate(my_input_file->ReturnXSize( ), my_input_file->ReturnYSize( ), false);
            }
#pragma omp for
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                /*
			// prepare the sum reference by subtracting out the current image, applying a bfactor and masking central cross
			sum_of_images_temp.CopyFrom(&sum_of_images);
			//sum_of_images_temp.SubtractImage(&stack_for_alignment[image_counter]); // turned-off
			sum_of_images_temp.ApplyBFactor(unitless_bfactor);

			if (mask_central_cross == true)
			{
				sum_of_images_temp.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
			}

			sum_of_images_temp.BackwardFFT();
			// debug by writing out sum_of_images_temp
			//sum_of_images_temp.QuickAndDirtyWriteSlice("my_rotated_sum.mrc", 1);
			*/

                //best_inplane_score = - std::numeric_limits<float>::max();
                best_inplane_score = -FLT_MAX;

                // Not using memory and already in image loop; just read it in now for this iteration, get it to proper state.
                if ( ! use_memory ) {
#pragma omp critical
                    current_image.ReadSlice(my_input_file, image_counter + 1);
                    normalize_image(&current_image, pixel_size, 10.0);
                    current_image.ForwardFFT( );
                    current_image.ZeroCentralPixel( );
                    if ( if_input_ctf_values_from_text_file )
                        current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
                    apply_ctf(&current_image, current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);
                }

                // loop over rotations
                for ( int psi_i = 0; psi_i <= number_of_rotations; psi_i++ ) {
                    // re-copy sum_of_images
                    sum_of_images_temp.CopyFrom(&sum_of_images);
                    sum_of_images_temp.ApplyBFactor(unitless_bfactor);

                    if ( mask_central_cross == true ) {
                        sum_of_images_temp.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
                    }

                    sum_of_images_temp.BackwardFFT( ); // sum_of_images was in complex space; temp needs to be in real space for rotation

                    psi = start_angle_for_peak_search + psi_i * rotation_step_size;
                    sum_of_images_temp.Rotate2DInPlace(psi, 0.0); // 0.0 pixel size is mask with radius of half-box size by default
                    sum_of_images_temp.ForwardFFT( );

                    // compute the cross correlation function and find the peak
                    if ( use_memory ) {
                        sum_of_images_temp.CalculateCrossCorrelationImageWith(&stack_for_alignment[image_counter]);
                    }
                    else {
                        sum_of_images_temp.CalculateCrossCorrelationImageWith(&current_image);
                    }

                    current_peak = sum_of_images_temp.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

                    // sum_of_images_temp is in real space here

                    // find (x,y,psi) that gives the highest score
                    if ( current_peak.value > best_inplane_score ) {
                        best_inplane_score     = current_peak.value;
                        best_inplane_values[0] = -psi; // negative psi because rotating image in stack is in opposite direction
                        best_inplane_values[1] = current_peak.x;
                        best_inplane_values[2] = current_peak.y;
                    }

                    // debug scores
                    //wxPrintf("returning, psi = %f, best score = %f, current score = %f\n", psi, best_inplane_score, current_peak.value);
                }

                // update the shifts..
                current_psi_angles[image_counter] = best_inplane_values[0];
                current_x_shifts[image_counter]   = best_inplane_values[1];
                current_y_shifts[image_counter]   = best_inplane_values[2];
                //current_y_shifts[image_counter] = 0.;

                // do a local refinement of in-plane rotation angle
                if ( refine_locally == true ) {
                    best_inplane_score = -FLT_MAX;
                    for ( int psi_i = 0; psi_i <= number_of_local_rotations; psi_i++ ) {
                        // re-copy sum_of_images
                        sum_of_images_temp.CopyFrom(&sum_of_images);
                        sum_of_images_temp.ApplyBFactor(unitless_bfactor);

                        if ( mask_central_cross == true ) {
                            sum_of_images_temp.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
                        }

                        sum_of_images_temp.BackwardFFT( );

                        psi = -current_psi_angles[image_counter] - rotation_step_size + psi_i * local_rotation_step_size;
                        sum_of_images_temp.Rotate2DInPlace(psi, 0.0); // 0.0 pixel size is mask with radius of half-box size by default
                        sum_of_images_temp.ForwardFFT( );

                        // compute the cross correlation function and find the peak
                        if ( use_memory ) {
                            sum_of_images_temp.CalculateCrossCorrelationImageWith(&stack_for_alignment[image_counter]);
                        }
                        else {
                            sum_of_images_temp.CalculateCrossCorrelationImageWith(&current_image);
                        }
                        current_peak = sum_of_images_temp.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

                        // sum_of_images_temp is in real space here

                        // find (x,y,psi) that gives the highest score
                        if ( current_peak.value > best_inplane_score ) {
                            best_inplane_score     = current_peak.value;
                            best_inplane_values[0] = -psi; // negative psi because rotating image in stack is in opposite direction
                            best_inplane_values[1] = current_peak.x;
                            best_inplane_values[2] = current_peak.y;
                        }
                    }

                    // update the shifts..
                    current_psi_angles[image_counter] = best_inplane_values[0];
                    current_x_shifts[image_counter]   = best_inplane_values[1];
                    current_y_shifts[image_counter]   = best_inplane_values[2];
                    //current_y_shifts[image_counter] = 0.;
                }

            } // end omp for
            sum_of_images_temp.Deallocate( );
        } // end omp

        // smooth the shifts (do not smooth shifts)
        /*
		x_shifts_curve.ClearData();
		y_shifts_curve.ClearData();

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			x_shifts_curve.AddPoint(image_counter, x_shifts[image_counter] + current_x_shifts[image_counter]);
			y_shifts_curve.AddPoint(image_counter, y_shifts[image_counter] + current_y_shifts[image_counter]);

			wxPrintf("Before = %li : %f, %f\n", image_counter, x_shifts[image_counter] + current_x_shifts[image_counter], y_shifts[image_counter] + current_y_shifts[image_counter]);
		}

		if (inner_radius_for_peak_search != 0) // in this case, weird things can happen (+1/-1 flips), we want to really smooth it. use a polynomial.  This should only affect the first round..
		{
			if (x_shifts_curve.number_of_points > 2)
			{
				x_shifts_curve.FitPolynomialToData(4);
				y_shifts_curve.FitPolynomialToData(4);

				// copy back

				for (image_counter = 0; image_counter < number_of_images; image_counter++)
				{
					current_x_shifts[image_counter] = x_shifts_curve.polynomial_fit[image_counter] - x_shifts[image_counter];
					current_y_shifts[image_counter] = y_shifts_curve.polynomial_fit[image_counter] - y_shifts[image_counter];
					wxPrintf("After poly = %li : %f, %f\n", image_counter, x_shifts_curve.polynomial_fit[image_counter], y_shifts_curve.polynomial_fit[image_counter]);
				}
			}
		}
		else
		{
			if (savitzy_golay_window_size < x_shifts_curve.number_of_points) // when the input movie is dodgy (very few frames), the fitting won't work
			{
				x_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);
				y_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);

				// copy them back..

				for (image_counter = 0; image_counter < number_of_images; image_counter++)
				{
					current_x_shifts[image_counter] = x_shifts_curve.savitzky_golay_fit[image_counter] - x_shifts[image_counter];
					current_y_shifts[image_counter] = y_shifts_curve.savitzky_golay_fit[image_counter] - y_shifts[image_counter];
					wxPrintf("After SG = %li : %f, %f\n", image_counter, x_shifts_curve.savitzky_golay_fit[image_counter], y_shifts_curve.savitzky_golay_fit[image_counter]);
				}
			}
		}
*/

        // do not subtract shift of the middle image from all images to keep things centred around it (want to keep tubes aligned vertically)

        //middle_image_x_shift = current_x_shifts[number_of_middle_image];
        //middle_image_y_shift = current_y_shifts[number_of_middle_image];
        //middle_image_psi_rotation = current_psi_angles[number_of_middle_image];

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            //current_x_shifts[image_counter] -= middle_image_x_shift;
            //current_y_shifts[image_counter] -= middle_image_y_shift;
            //current_psi_angles[image_counter] -= middle_image_psi_rotation;

            average_image_x_shift += fabs(current_x_shifts[image_counter]);
            average_image_y_shift += fabs(current_y_shifts[image_counter]);
            average_image_psi_rotation += fabs(current_psi_angles[image_counter]);

            total_shift    = sqrt(pow(current_x_shifts[image_counter], 2) + pow(current_y_shifts[image_counter], 2));
            total_rotation = current_psi_angles[image_counter];

            if ( total_shift > max_shift )
                max_shift = total_shift;

            if ( total_rotation > max_rotation )
                max_rotation = total_rotation;
        }

        // average shift and rotation
        average_image_x_shift /= number_of_images;
        average_image_y_shift /= number_of_images;
        average_image_psi_rotation /= number_of_images;

        // actually rotate and shift the images, also add the subtracted shifts to the overall shifts
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter, current_image)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            // recopy original image stack if we have to do another round and apply current rotation and shift
            if ( use_memory ) {
                buffer_stack[image_counter].CopyFrom(&input_stack_times_ctf[image_counter]);

                // rotate first
                buffer_stack[image_counter].BackwardFFT( );
                buffer_stack[image_counter].Rotate2DInPlace(current_psi_angles[image_counter], 0.0); // 0.0 pixel size is mask with radius of half-box size
                buffer_stack[image_counter].ForwardFFT( );

                // then shift
                buffer_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);
            }

            // store current shifts and rotations
            x_shifts[image_counter]   = current_x_shifts[image_counter];
            y_shifts[image_counter]   = current_y_shifts[image_counter];
            psi_angles[image_counter] = current_psi_angles[image_counter];
        }

        // check to see if the convergence criteria have been reached and return if so
        //if (iteration_counter >= max_iterations || (max_shift <= max_shift_convergence_threshold && max_rotation <= max_rotation_convergence_threshold))
        if ( iteration_counter >= max_iterations ) {
            wxPrintf("returning, iteration = %li, max_shift = %f, max_rotation = %f, average_shift = %f, average_rotation = %f\n", iteration_counter, max_shift, max_rotation, average_image_x_shift, average_image_psi_rotation);

            // shift and rotate the final image stack,
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter, current_image)
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                // rotate and shift input_stack * CTF
                if ( use_memory ) {
                    // rotate first
                    input_stack_times_ctf[image_counter].BackwardFFT( );
                    input_stack_times_ctf[image_counter].Rotate2DInPlace(current_psi_angles[image_counter], 0.0); // 0.0 pixel size is mask with radius of half-box size
                    input_stack_times_ctf[image_counter].ForwardFFT( );
                    // then shift
                    input_stack_times_ctf[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);
                }
            }

            delete[] current_x_shifts;
            delete[] current_y_shifts;
            delete[] current_psi_angles;
            if ( use_memory )
                delete[] buffer_stack;

            if ( number_of_frames_for_running_average > 1 )
                delete[] running_average_stack;

            return;
        }
        else {
            wxPrintf("Not. returning, iteration = %li, max_shift = %f, max_rotation = %f, average_shift = %f, average_rotation = %f\n", iteration_counter, max_shift, max_rotation, average_image_x_shift, average_image_psi_rotation);
        }

        // going to be doing another round so we need to make the new sum..

        sum_of_images.SetToConstant(0.0);

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            if ( use_memory )
                sum_of_images.AddImage(&buffer_stack[image_counter]);

            // Skipped applying shifts and rotations to images until they're being summed
            else {
                current_image.ReadSlice(my_input_file, image_counter + 1);
                normalize_image(&current_image, pixel_size, 10.0);
                current_image.ForwardFFT( );
                current_image.ZeroCentralPixel( );
                if ( if_input_ctf_values_from_text_file )
                    current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
                apply_ctf(&current_image, current_ctf, ctf_sum_of_squares, phase_flip_only, do_fill_sum_of_squares);

                current_image.BackwardFFT( );

                // Rotate the image.
                current_image.Rotate2DInPlace(current_psi_angles[image_counter], 0.0);
                current_image.ForwardFFT( );

                // Shift the image
                current_image.PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

                // Have the image in needed state, apply operation
                sum_of_images.AddImage(&current_image);
            }
        }
        // vertically-averaged CTF-corrected sum
        divide_by_ctf_sum_of_squares(&sum_of_images, ctf_sum_of_squares);
        sum_of_images.BackwardFFT( );

        sum_of_images.ApplyRampFilter( );
        sum_of_images.AverageRotationally( );
        sum_image_direction(&sum_of_images, 2);
        sum_of_images.ForwardFFT( );

        // print out current alignment reference
        sum_of_images.QuickAndDirtyWriteSlice(wxString::Format("my_aligned_sum_%ld.mrc", iteration_counter).ToStdString( ), 1);

    } // end alignment cycle
}

// projection generated from Central Slice Theorem (i.e., Fourier space) and then subtracts unaligned input stack
void scale_and_subtract_reference(Image* input_stack, ImageFile* my_input_file, int number_of_images, float pixel_size, float* x_shifts, float* y_shifts, float* psi_angles, Image* current_volume, Image* masked_volume, int number_of_models, ctf_parameters* ctf_parameters_stack, bool absolute, float padding_factor, int max_threads, bool use_memory) {
    CTF                 current_ctf;
    Image               projection_3d;
    Image               projection_image;
    Image               padded_projection_image;
    Image               masked_projection_3d;
    Image               masked_projection_image;
    Image               masked_padded_projection_image;
    Image               subtracted_image;
    AnglesAndShifts     my_parameters;
    ReconstructedVolume input_3d;
    ReconstructedVolume masked_3d;

    double scale_factors[number_of_images];

    Image proj_two; // side-view
    proj_two.Allocate(current_volume->logical_x_dimension, current_volume->logical_y_dimension, true);
    proj_two.SetToConstant(0.0);

    long pix_counter = 0;

    for ( int k = 0; k < current_volume->logical_z_dimension; k++ ) {
        for ( int j = 0; j < current_volume->logical_y_dimension; j++ ) {
            for ( int i = 0; i < current_volume->logical_x_dimension; i++ ) {

                proj_two.real_values[proj_two.ReturnReal1DAddressFromPhysicalCoord(j, k, 0)] += current_volume->real_values[pix_counter];

                pix_counter++;
            }

            pix_counter += current_volume->padding_jump_value;
        }
    }

    // large mask radius to CorrectSinc entire image
    float mask_radius = FLT_MAX;

    input_3d.InitWithDimensions(current_volume->logical_x_dimension, current_volume->logical_y_dimension, current_volume->logical_z_dimension, pixel_size);
    input_3d.density_map->CopyFrom(current_volume);

    //input_3d.density_map->Resize(padding_factor * current_volume->logical_x_dimension, padding_factor * current_volume->logical_y_dimension, padding_factor * current_volume->logical_z_dimension, input_3d.density_map->ReturnAverageOfRealValuesOnEdges());
    input_3d.mask_radius = mask_radius;
    //input_3d.density_map->CorrectSinc(mask_radius / pixel_size);	// is this called twice?
    input_3d.PrepareForProjections(0.0, 2.0 * pixel_size);
    projection_3d.CopyFrom(input_3d.density_map);

    // calculate average scale factor between unmasked volume and input image stack
    long  pixel_counter;
    long  image_counter;
    float average; // per image average values at edges
    float sum_of_pixelwise_product; // per image
    float sum_of_squares; // per image
    float scale_factor; // per image scale factor
    float sum_of_scale_factors         = 0.0; // average scale factor
    float sum_of_scale_factors_squares = 0.0; // variance scale factor
    float scale_factor_variance;
    int   i, j;

    Image current_image;

#pragma omp parallel num_threads(max_threads) default(none) shared(proj_two, input_stack, my_input_file, use_memory, number_of_images, pixel_size, psi_angles, x_shifts, y_shifts, current_volume, projection_3d, ctf_parameters_stack, absolute, sum_of_scale_factors, sum_of_scale_factors_squares, padding_factor, scale_factors) private(i, j, pixel_counter, sum_of_pixelwise_product, sum_of_squares, scale_factor, image_counter, average, my_parameters, padded_projection_image, projection_image, current_ctf, current_image)
    { // start omp

        // project padded unmasked 3D volume using central slice theorem
        projection_image.Allocate(current_volume->logical_x_dimension / padding_factor, current_volume->logical_y_dimension / padding_factor, true);
        //padded_projection_image.Allocate(projection_3d.logical_x_dimension, projection_3d.logical_y_dimension, false);
        padded_projection_image.Allocate(current_volume->logical_x_dimension * padding_factor, current_volume->logical_y_dimension * padding_factor, false);

#pragma omp for reduction(+ \
                          : sum_of_scale_factors, sum_of_scale_factors_squares)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            // angles and shifts are negative to align projection with original input stack
            my_parameters.Init(0.0, 90.0, 90.0 - psi_angles[image_counter], -x_shifts[image_counter] * pixel_size, -y_shifts[image_counter] * pixel_size);
            //my_parameters.Init(0.0, 90.0, 90.0, -x_shifts[image_counter] * pixel_size, -y_shifts[image_counter] * pixel_size);
            projection_3d.ExtractSlice(padded_projection_image, my_parameters);
            //padded_projection_image.complex_values[0] = projection_3d.complex_values[0]; // sets central pixel to zero.
            padded_projection_image.ZeroCentralPixel( );
            //padded_projection_image.DivideByConstant(sqrt(padded_projection_image.ReturnSumOfSquares()));

            // apply ctf
            current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);

            if ( absolute == true )
                padded_projection_image.ApplyCTFPhaseFlip(current_ctf);
            else
                padded_projection_image.ApplyCTF(current_ctf);

            padded_projection_image.PhaseShift(my_parameters.ReturnShiftX( ) / pixel_size, my_parameters.ReturnShiftY( ) / pixel_size);
            padded_projection_image.SwapRealSpaceQuadrants( );
            padded_projection_image.BackwardFFT( );

            // de-pad projection
            //padded_projection_image.ChangePixelSize(&projection_image, 1.0, 0.001f);
            //projection_image.CopyFrom(&padded_projection_image);
            padded_projection_image.ClipInto(&projection_image);
            //projection_image.ForwardFFT();

            if ( image_counter == 0 ) {
                projection_image.QuickAndDirtyWriteSlice("my_proj_times_ctf_one.mrc", 1);
                //projection_image.QuickAndDirtyWriteSlice("frames_to_be_subtracted.mrc", image_counter + 1); // Check alignment for all images
            }

            // subtract mean value at edges to set zero-mean of noise
            // it would better to calculate noise average from vertical stripes from left and right edges of images
            //average = projection_image.ReturnAverageOfRealValues(projection_image.physical_address_of_box_center_x - (10.0 / 2.0), true);
            //average = projection_image.ReturnAverageOfRealValuesOnEdges();
            average = ReturnAverageOfRealValuesOnVerticalEdges(&projection_image);
            projection_image.AddConstant(-average);
            //input_stack[image_counter].AddConstant(-input_stack[image_counter].ReturnAverageOfRealValuesOnEdges());
            //input_stack[image_counter].DivideByConstant(sqrt(input_stack[image_counter].ReturnSumOfSquares()));

            // least squares scale factor = A*B / A*A
            // here A = projection and B = micrograph
            // init values to zero
            pixel_counter            = 0;
            sum_of_pixelwise_product = 0.0;
            sum_of_squares           = 0.0;
            scale_factor             = 0.0;

// Read in from stack
#pragma omp critical
            if ( ! use_memory ) {
                current_image.ReadSlice(my_input_file, image_counter + 1);
                normalize_image(&current_image, pixel_size, 10.0);
            }

#pragma omp critical
            {
                for ( j = 0; j < projection_image.logical_y_dimension; j++ ) {
                    for ( i = 0; i < projection_image.logical_x_dimension; i++ ) {

                        /*				if (i > (60 + my_parameters.ReturnShiftX()/pixel_size) && i < (180 + my_parameters.ReturnShiftX()/pixel_size))
				{
					sum_of_pixelwise_product += projection_image.real_values[pixel_counter] * input_stack[image_counter].real_values[pixel_counter];
					sum_of_squares += projection_image.real_values[pixel_counter] * projection_image.real_values[pixel_counter];
				}
*/
                        if ( use_memory )
                            sum_of_pixelwise_product += projection_image.real_values[pixel_counter] * input_stack[image_counter].real_values[pixel_counter];
                        else
                            sum_of_pixelwise_product += projection_image.real_values[pixel_counter] * current_image.real_values[pixel_counter];

                        sum_of_squares += projection_image.real_values[pixel_counter] * projection_image.real_values[pixel_counter];
                        pixel_counter++;
                    }

                    pixel_counter += projection_image.padding_jump_value;
                }

                scale_factor                 = sum_of_pixelwise_product / sum_of_squares;
                scale_factors[image_counter] = scale_factor;

                sum_of_scale_factors += scale_factor;
                sum_of_scale_factors_squares += powf(scale_factor, 2);
            }
        } // end omp for
        projection_image.Deallocate( );
        padded_projection_image.Deallocate( );

    } // end omp

    // print average scale factor between images and projection
    sum_of_scale_factors /= number_of_images;
    sum_of_scale_factors_squares /= number_of_images;
    scale_factor_variance = fabsf(float(sum_of_scale_factors_squares - powf(sum_of_scale_factors, 2)));
    wxPrintf("mean scale factor = %f, variance scale factor = %f\n", sum_of_scale_factors, scale_factor_variance);

    // multiply reference by average scale factor
    //current_volume->MultiplyByConstant(sum_of_scale_factors);
    //masked_volume->MultiplyByConstant(sum_of_scale_factors);

    // generate masked 3D AA models
    float phi;
    long  model_counter    = 0;
    int   images_processed = 0;
    float diff             = 0.0;

    // pad 3D masked volume
    masked_3d.InitWithDimensions(masked_volume->logical_x_dimension, masked_volume->logical_y_dimension, masked_volume->logical_z_dimension, pixel_size);
    masked_3d.density_map->CopyFrom(masked_volume);

    //masked_3d.density_map->Resize(padding_factor * masked_volume->logical_x_dimension, padding_factor * masked_volume->logical_y_dimension, padding_factor * masked_volume->logical_z_dimension, masked_3d.density_map->ReturnAverageOfRealValuesOnEdges());
    masked_3d.mask_radius = mask_radius;
    //masked_3d.density_map->CorrectSinc(mask_radius / pixel_size);
    masked_3d.PrepareForProjections(0.0, 2.0 * pixel_size);
    masked_projection_3d.CopyFrom(masked_3d.density_map);

#pragma omp parallel    num_threads(max_threads) default(none) shared(input_stack, my_input_file, use_memory, number_of_images, pixel_size, psi_angles, x_shifts, y_shifts, images_processed, masked_volume, masked_projection_3d, number_of_models, ctf_parameters_stack, absolute, padding_factor, sum_of_scale_factors, scale_factors) private(subtracted_image, image_counter, model_counter, phi, my_parameters, masked_padded_projection_image, masked_projection_image, current_ctf, diff)
    { // start omp

        masked_projection_image.Allocate(masked_volume->logical_x_dimension / padding_factor, masked_volume->logical_y_dimension / padding_factor, true);
        //masked_padded_projection_image.Allocate(masked_projection_3d.logical_x_dimension, masked_projection_3d.logical_y_dimension, false);
        masked_padded_projection_image.Allocate(masked_volume->logical_x_dimension, masked_volume->logical_y_dimension, false);
        subtracted_image.Allocate(masked_volume->logical_x_dimension / padding_factor, masked_volume->logical_y_dimension / padding_factor, true);

        // ICC doesn't accept that collapse was perfectly nested; instead, we parallelize only the inner loop
        // NOTE: this is more effective the smaller number_of_models is relative to the the number of specified threads
        for ( model_counter = 0; model_counter < number_of_models; model_counter++ ) {
#pragma omp for ordered schedule(static, 1)
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                   // angle of azimuthal projection
                phi = model_counter * 360.0 / number_of_models;

                if ( use_memory )
                    subtracted_image.CopyFrom(&input_stack[image_counter]);
                else {
   #pragma omp critical
                    subtracted_image.ReadSlice(my_input_file, image_counter + 1);
                    normalize_image(&subtracted_image, pixel_size, 10.0);
                }

                my_parameters.Init(phi, 90.0, 90.0 - psi_angles[image_counter], -x_shifts[image_counter] * pixel_size, -y_shifts[image_counter] * pixel_size);
                // extract slice from masked 3D AA model
                masked_projection_3d.ExtractSlice(masked_padded_projection_image, my_parameters);
                //masked_padded_projection_image.complex_values[0] = masked_projection_3d.complex_values[0]; // sets central pixel to zero.
                masked_padded_projection_image.ZeroCentralPixel( );

                // apply ctf
                current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
                if ( absolute == true )
                    masked_padded_projection_image.ApplyCTFPhaseFlip(current_ctf);
                else
                    masked_padded_projection_image.ApplyCTF(current_ctf);

                masked_padded_projection_image.PhaseShift(my_parameters.ReturnShiftX( ) / pixel_size, my_parameters.ReturnShiftY( ) / pixel_size);
                masked_padded_projection_image.SwapRealSpaceQuadrants( );
                masked_padded_projection_image.BackwardFFT( );

                // de-pad projection
                //masked_padded_projection_image.ChangePixelSize(&masked_projection_image, 1.0, 0.001f);
                //masked_projection_image.CopyFrom(&masked_padded_projection_image);
                masked_padded_projection_image.ClipInto(&masked_projection_image);
                //masked_projection_image.ForwardFFT();

                // multiply reference by average of per image scale factor
                masked_projection_image.MultiplyByConstant(scale_factors[image_counter]); // 9.0 (good), 10.0, 12.0 (good), 13.0 (good) 15.0 (starting to invert), 16.0 (inverse of projection from here)
                /*
			if (image_counter == 0)
			{
				diff = ReturnDifferenceOfSquares(&masked_projection_image, &subtracted_image);
				wxPrintf("diff = %f\n", diff);
			}
			*/
                // subtract projection from image
                subtracted_image.SubtractImage(&masked_projection_image);

// save downweighted images
//subtracted_stack[number_of_images * model_counter + image_counter].CopyFrom(&subtracted_image);

// print out downweighted images
#pragma omp ordered
                subtracted_image.QuickAndDirtyWriteSlice("my_subtracted_frames.mrc", number_of_images * model_counter + image_counter + 1);

#pragma omp atomic
                images_processed++;
            }
        } // end omp for

        masked_projection_image.Deallocate( );
        masked_padded_projection_image.Deallocate( );
        subtracted_image.Deallocate( );

    } // end omp

    wxPrintf("Number of images processed = %i\n", images_processed);

    // clean-up
    projection_3d.Deallocate( );
    masked_projection_3d.Deallocate( );
}

void normalize_image(Image* input_image, float pixel_size, float mask_falloff) {
    // Normalize background variance and average
    float variance;
    float average;

    // subtract mean value from each image pixel to get a zero-mean
    // divide each pixel value by standard deviation to have unit-variance
    variance = input_image->ReturnVarianceOfRealValues(input_image->physical_address_of_box_center_x - (mask_falloff / pixel_size), 0.0, 0.0, 0.0, true);
    average  = input_image->ReturnAverageOfRealValues(input_image->physical_address_of_box_center_x - (mask_falloff / pixel_size), true);

    if ( variance == 0.0f ) {
        input_image->SetToConstant(0.0f);
    }
    else {
        input_image->AddMultiplyConstant(-average, 1.0 / sqrtf(variance));
    }
}

void invert_mask(Image* mask_file) {
    // inverts binarized mask pixel values (i.e., 0 changes to 1 and 1 changes to 0)
    for ( long pixel_counter = 0; pixel_counter < mask_file->real_memory_allocated; pixel_counter++ ) {
        if ( mask_file->real_values[pixel_counter] == 0 )
            mask_file->real_values[pixel_counter] = 1.0;
        else
            mask_file->real_values[pixel_counter] = 0.0;
    }
}

void writeToStarFile(ctf_parameters* ctf_parameters_stack, int number_of_images, int number_of_models, float pixel_size, float* x_shifts, float* y_shifts, float* psi_angles) {
    long image_counter;
    long model_counter;
    long current_image;

    cisTEMParameters output_params;
    output_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);
    output_params.PreallocateMemoryAndBlank(number_of_images * number_of_models);

    current_image = 0;
    for ( model_counter = 0; model_counter < number_of_models; model_counter++ ) {
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            output_params.all_parameters[current_image].position_in_stack                  = current_image + 1;
            output_params.all_parameters[current_image].psi                                = 90.0 + psi_angles[image_counter];
            output_params.all_parameters[current_image].theta                              = 90.0f;
            output_params.all_parameters[current_image].phi                                = model_counter * 360.0 / number_of_models;
            output_params.all_parameters[current_image].x_shift                            = x_shifts[image_counter] * pixel_size;
            output_params.all_parameters[current_image].y_shift                            = y_shifts[image_counter] * pixel_size;
            output_params.all_parameters[current_image].defocus_1                          = ctf_parameters_stack[image_counter].defocus_1;
            output_params.all_parameters[current_image].defocus_2                          = ctf_parameters_stack[image_counter].defocus_2;
            output_params.all_parameters[current_image].defocus_angle                      = ctf_parameters_stack[image_counter].astigmatism_angle;
            output_params.all_parameters[current_image].phase_shift                        = ctf_parameters_stack[image_counter].additional_phase_shift;
            output_params.all_parameters[current_image].image_is_active                    = 1;
            output_params.all_parameters[current_image].occupancy                          = 100.0f;
            output_params.all_parameters[current_image].logp                               = -1000.0f;
            output_params.all_parameters[current_image].sigma                              = 10.0f;
            output_params.all_parameters[current_image].pixel_size                         = pixel_size;
            output_params.all_parameters[current_image].microscope_voltage_kv              = ctf_parameters_stack[image_counter].acceleration_voltage;
            output_params.all_parameters[current_image].microscope_spherical_aberration_mm = ctf_parameters_stack[image_counter].spherical_aberration;
            output_params.all_parameters[current_image].amplitude_contrast                 = ctf_parameters_stack[image_counter].amplitude_contrast;
            output_params.all_parameters[current_image].beam_tilt_x                        = 0.0f;
            output_params.all_parameters[current_image].beam_tilt_y                        = 0.0f;
            output_params.all_parameters[current_image].image_shift_x                      = 0.0f;
            output_params.all_parameters[current_image].image_shift_y                      = 0.0f;

            current_image++;
        }
    }

    output_params.WriteTocisTEMStarFile("my_subtracted_frames_params.star");
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

// calculates the pixelwise difference of squares between two images
float ReturnDifferenceOfSquares(Image* first_image, Image* second_image) {
    float difference_of_squares = 0.0;
    long  pixel_counter         = 0;

    for ( int j = 0; j < first_image->logical_y_dimension; j++ ) {
        for ( int i = 0; i < first_image->logical_x_dimension; i++ ) {
            difference_of_squares += powf(first_image->real_values[pixel_counter] - second_image->real_values[pixel_counter], 2);
            pixel_counter++;
        }

        pixel_counter += first_image->padding_jump_value;
    }

    /*
	for (pixel_counter = 0; pixel_counter < first_image->real_memory_allocated; pixel_counter++)
		{
			difference_of_squares += powf(first_image->real_values[pixel_counter] - second_image->real_values[pixel_counter], 2);
		}
*/

    return difference_of_squares;
}