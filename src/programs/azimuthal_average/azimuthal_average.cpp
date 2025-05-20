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

void azimuthal_alignment(Image* input_stack, Image* input_stack_times_ctf, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float start_angle_for_peak_search, float end_angle_for_peak_search, float rotation_step_size, float max_rotation_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, float* psi_angles, float* ctf_sum_of_squares);
void apply_ctf(Image* current_image, CTF ctf_to_apply, float* ctf_sum_of_squares, bool absolute);
void divide_by_ctf_sum_of_squares(Image* current_image, float* ctf_sum_of_squares); // const float
void sum_image_direction(Image* current_image, int dim);
//void sum_image_direction(Image *current_image, Image *directional_image_sum, int dim);
void average_rotationally(Image* current_image, Image* current_volume);
void scale_reference(Image* input_stack, int number_of_images, Image* current_volume, ctf_parameters* ctf_parameters_stack, bool absolute, int max_threads);
void normalize_image(Image* input_image, float pixel_size, float mask_falloff);

IMPLEMENT_APP(AzimuthalAverage)

// override the DoInteractiveUserInput

void AzimuthalAverage::DoInteractiveUserInput( ) {
    // intial parameters
    int         new_z_size = 1;
    std::string text_filename;
    float       pixel_size;

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
    float psi_min                          = -5.0;
    float psi_max                          = 5.0;
    float psi_step                         = 0.25;
    float termination_threshold_in_degrees = 0.25;
    // image processing
    float bfactor_in_angstroms                 = 1500;
    bool  should_mask_central_cross            = false;
    int   horizontal_mask_size                 = 1;
    int   vertical_mask_size                   = 1;
    float exposure_per_frame                   = 0.0;
    int   number_of_frames_for_running_average = 1;

    bool set_expert_options;
    int  max_threads;

    UserInput* my_input = new UserInput("AzimuthalAverage", 1.0);

    std::string input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of stack to be added to", "input_stack1.mrc", true);
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

    if ( set_expert_options == true ) {
        max_iterations                     = my_input->GetIntFromUser("Maximum number of iterations", "Alignment will stop at this number, even if the threshold shift is not reached", "5", 0);
        minimum_shift_in_angstroms         = my_input->GetFloatFromUser("Minimum shift for initial search (A)", "Initial search will be limited to between the inner and outer radii.", "2.0", 0.0);
        maximum_shift_in_angstroms         = my_input->GetFloatFromUser("Outer radius shift limit (A)", "The maximum shift of each alignment step will be limited to this value.", "80.0", minimum_shift_in_angstroms);
        termination_threshold_in_angstroms = my_input->GetFloatFromUser("Termination shift threshold (A)", "Alignment will iterate until the maximum shift is below this value", "1", 0.0);

        psi_min                          = my_input->GetFloatFromUser("Minimum rotation for initial search (degrees)", "Initial search will be limited to between the inner and outer radii.", "-5.0", -10.0);
        psi_max                          = my_input->GetFloatFromUser("Outer radius rotation limit (degrees)", "The maximum rotation of each alignment step will be limited to this value.", "5.0", psi_min);
        psi_step                         = my_input->GetFloatFromUser("Rotation step size (degrees)", "The step size of each rotation will be limited to this value.", "0.25", 0.0);
        termination_threshold_in_degrees = my_input->GetFloatFromUser("Termination rotation threshold (degrees)", "Alignment will iterate until the maximum rotation is below this value", "0.25", 0.0);

        bfactor_in_angstroms = my_input->GetFloatFromUser("B-factor to apply to images (A^2)", "This B-Factor will be used to filter the reference prior to alignment", "1500", 0.0);
    }

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    my_current_job.Reset(27);
    my_current_job.ManualSetArguments("tffffffffbtbiffffffffbiifii", input_filename.c_str( ),
                                      pixel_size,
                                      acceleration_voltage,
                                      spherical_aberration,
                                      amplitude_contrast,
                                      defocus_1,
                                      defocus_2,
                                      astigmatism_angle,
                                      additional_phase_shift,
                                      input_ctf_values_from_text_file,
                                      text_filename.c_str( ),
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
                                      max_threads);
}

// override the do calculation method which will be what is actually run..

bool AzimuthalAverage::DoCalculation( ) {
    CTF current_ctf;

    // get the arguments for this job..
    std::string input_filename                       = my_current_job.arguments[0].ReturnStringArgument( );
    float       pixel_size                           = my_current_job.arguments[1].ReturnFloatArgument( );
    float       acceleration_voltage                 = my_current_job.arguments[2].ReturnFloatArgument( );
    float       spherical_aberration                 = my_current_job.arguments[3].ReturnFloatArgument( );
    float       amplitude_contrast                   = my_current_job.arguments[4].ReturnFloatArgument( );
    float       defocus_1                            = my_current_job.arguments[5].ReturnFloatArgument( );
    float       defocus_2                            = my_current_job.arguments[6].ReturnFloatArgument( );
    float       astigmatism_angle                    = my_current_job.arguments[7].ReturnFloatArgument( );
    float       additional_phase_shift               = my_current_job.arguments[8].ReturnFloatArgument( );
    bool        input_ctf_values_from_text_file      = my_current_job.arguments[9].ReturnBoolArgument( );
    std::string text_filename                        = my_current_job.arguments[10].ReturnStringArgument( );
    bool        phase_flip_only                      = my_current_job.arguments[11].ReturnBoolArgument( );
    int         max_iterations                       = my_current_job.arguments[12].ReturnIntegerArgument( );
    float       minimum_shift_in_angstroms           = my_current_job.arguments[13].ReturnFloatArgument( );
    float       maximum_shift_in_angstroms           = my_current_job.arguments[14].ReturnFloatArgument( );
    float       termination_threshold_in_angstroms   = my_current_job.arguments[15].ReturnFloatArgument( );
    float       psi_min                              = my_current_job.arguments[16].ReturnFloatArgument( );
    float       psi_max                              = my_current_job.arguments[17].ReturnFloatArgument( );
    float       psi_step                             = my_current_job.arguments[18].ReturnFloatArgument( );
    float       termination_threshold_in_degrees     = my_current_job.arguments[19].ReturnFloatArgument( );
    float       bfactor_in_angstroms                 = my_current_job.arguments[20].ReturnFloatArgument( );
    bool        should_mask_central_cross            = my_current_job.arguments[21].ReturnBoolArgument( );
    int         horizontal_mask_size                 = my_current_job.arguments[22].ReturnIntegerArgument( );
    int         vertical_mask_size                   = my_current_job.arguments[23].ReturnIntegerArgument( );
    float       exposure_per_frame                   = my_current_job.arguments[24].ReturnFloatArgument( );
    int         number_of_frames_for_running_average = my_current_job.arguments[25].ReturnIntegerArgument( );
    int         max_threads                          = my_current_job.arguments[26].ReturnIntegerArgument( );

    // The Files
    ImageFile        my_input_file(input_filename, false);
    NumericTextFile* input_text;
    long             number_of_input_images = my_input_file.ReturnNumberOfSlices( );

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

    // user input already
    //float input_pixel_size = my_input_file.ReturnPixelSize();

    // Profiling
    wxDateTime overall_start = wxDateTime::Now( );
    wxDateTime overall_finish;
    wxDateTime read_frames_start;
    wxDateTime read_frames_finish;
    wxDateTime first_alignment_start;
    wxDateTime first_alignment_finish;
    wxDateTime main_alignment_start;
    wxDateTime main_alignment_finish;

    long pixel_counter; // not used
    long image_counter;

    // rotational and translational alignment of images
    Image* image_stack = new Image[number_of_input_images];
    // input stack times ctf
    Image* image_stack_times_ctf = new Image[number_of_input_images];
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

    // read CTF from file
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

    // read in image stack and FFT
    read_frames_start = wxDateTime::Now( );

    // read image stack

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

    // apply CTF to copy of image stack

    // allocate arrays for the ctf sum of squares
    ctf_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];
    ZeroFloatArray(ctf_sum_of_squares, image_stack[0].real_memory_allocated / 2);

    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        image_stack_times_ctf[image_counter].CopyFrom(&image_stack[image_counter]);

        // read CTF from file
        if ( input_ctf_values_from_text_file == true ) {
            current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
        }

        apply_ctf(&image_stack_times_ctf[image_counter], current_ctf, ctf_sum_of_squares, phase_flip_only);
    }

    read_frames_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_to_read = read_frames_finish.Subtract(read_frames_start);
    wxPrintf(" Read frames                : %s\n", time_to_read.Format( ));

    // convert shifts to pixels..
    min_shift_in_pixels             = minimum_shift_in_angstroms / pixel_size;
    max_shift_in_pixels             = maximum_shift_in_angstroms / pixel_size;
    termination_threshold_in_pixels = termination_threshold_in_angstroms / pixel_size;
    if ( min_shift_in_pixels <= 1.01 )
        min_shift_in_pixels = 1.01; // we always want to ignore the central peak initially.
    // calculate the bfactor
    unitless_bfactor = bfactor_in_angstroms / pow(pixel_size, 2);

    // do the initial refinement (only 1 round - with the min shift)
    //first_alignment_start = wxDateTime::Now();
    //azimuthal_alignment(image_stack_times_ctf, number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, min_shift_in_pixels, max_shift_in_pixels, termination_threshold_in_pixels, psi_min, psi_max, psi_step, termination_threshold_in_degrees, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, psi_angles, ctf_sum_of_squares);
    //first_alignment_finish = wxDateTime::Now();

    // now do the actual refinement
    main_alignment_start = wxDateTime::Now( );
    azimuthal_alignment(image_stack, image_stack_times_ctf, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, psi_min, psi_max, psi_step, termination_threshold_in_degrees, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, psi_angles, ctf_sum_of_squares);
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
    sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
    sum_image.SetToConstant(0.0);

    for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
        // write to disk
        image_stack[image_counter].BackwardFFT( );
        image_stack[image_counter].QuickAndDirtyWriteSlice("my_aligned_frames.mrc", image_counter + 1);

        sum_image.AddImage(&image_stack_times_ctf[image_counter]);

        my_progress->Update(image_counter + 1);
    }

    // vertically-averaged CTF-corrected sum
    divide_by_ctf_sum_of_squares(&sum_image, ctf_sum_of_squares);
    sum_image.BackwardFFT( );
    sum_image_direction(&sum_image, 2);
    sum_image.QuickAndDirtyWriteSlice("my_aligned_sum.mrc", 1);
    delete my_progress;

    // rotationally averaged 3D reconstruction
    // assume volume is square/cube (x_size = y_size = z_size)
    Image my_volume;
    my_volume.Allocate(sum_image.logical_x_dimension, sum_image.logical_y_dimension, sum_image.logical_x_dimension, true);
    my_volume.SetToConstant(0.0);

    average_rotationally(&sum_image, &my_volume);

    // print out 2D rotational average
    sum_image.QuickAndDirtyWriteSlice("my_rotationally_averaged_sum.mrc", 1);

    // get scale factor to multiply reference with
    wxPrintf("\nScaling Reference...\n");
    scale_reference(image_stack, number_of_input_images, &my_volume, ctf_parameters_stack, phase_flip_only, max_threads);

    // print out 3D reconstruction
    my_volume.QuickAndDirtyWriteSlices("my_averaged_volume.mrc", 1, my_volume.logical_z_dimension);

    // save orthogonal views
    Image orth_image;
    orth_image.Allocate(my_volume.logical_x_dimension * 3, my_volume.logical_y_dimension * 2, 1, true);
    my_volume.CreateOrthogonalProjectionsImage(&orth_image);
    orth_image.QuickAndDirtyWriteSlice("my_orthogonal_views.mrc", 1);

    // clean-up
    if ( input_ctf_values_from_text_file == true )
        delete input_text;
    //delete my_progress;
    delete[] x_shifts;
    delete[] y_shifts;
    delete[] psi_angles;
    delete[] image_stack;
    delete[] image_stack_times_ctf;
    delete[] ctf_sum_of_squares;
    delete[] ctf_parameters_stack;
    wxPrintf("\n\n");

    overall_finish = wxDateTime::Now( );

    // Timings
    wxPrintf("\nTimings\n");
    wxTimeSpan time_total = overall_finish.Subtract(overall_start);
    wxPrintf(" Overal time                : %s\n", time_total.Format( ));

    return true;
}

void apply_ctf(Image* current_image, CTF ctf_to_apply, float* ctf_sum_of_squares, bool absolute) {
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
            ctf_sum_of_squares[pixel_counter] += powf(ctf_value, 2);
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

/*
void sum_image_direction(Image *current_image, Image *directional_image_sum, int dim)
{
	// x-direction
	if (dim == 1)
	{

		long pixel_coord_y = 0;
		long pixel_coord_xy = 0;
		long pixel_counter = 0;

		// sum columns of my_image_sum (NxM) and store in array (1xN)
		for (int j = 0; j < current_image->logical_y_dimension; j++)
		{
			for (int i = 0; i < current_image->logical_x_dimension; i++)
			{
				pixel_coord_y = current_image->ReturnReal1DAddressFromPhysicalCoord(0, j, 0);
				pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
				directional_image_sum->real_values[pixel_coord_y] += current_image->real_values[pixel_coord_xy];
				pixel_counter++;
			}
			pixel_counter += current_image->padding_jump_value;
		}

		// repeat column sum into my_vertical_sum
		pixel_counter = 0;
		for (int j = 0; j < directional_image_sum->logical_y_dimension; j++)
		{
			for (int i = 0; i < directional_image_sum->logical_x_dimension; i++)
			{
				pixel_coord_y = directional_image_sum->ReturnReal1DAddressFromPhysicalCoord(0, j, 0);
				pixel_coord_xy = directional_image_sum->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
				directional_image_sum->real_values[pixel_coord_xy] = directional_image_sum->real_values[pixel_coord_y];
				pixel_counter++;
			}
			pixel_counter += directional_image_sum->padding_jump_value;
		}
	}
	// y-direction
	else
	{

		long pixel_coord_x = 0;
		long pixel_coord_xy = 0;
		long pixel_counter = 0;

		// sum columns of my_image_sum (NxM) and store in array (1xN)
		for (int i = 0; i < current_image->logical_x_dimension; i++)
		{
			for (int j = 0; j < current_image->logical_y_dimension; j++)
			{
				pixel_coord_x = current_image->ReturnReal1DAddressFromPhysicalCoord(i, 0, 0);
				pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
				directional_image_sum->real_values[pixel_coord_x] += current_image->real_values[pixel_coord_xy];
				pixel_counter++;
			}
			pixel_counter += current_image->padding_jump_value;
		}

		// repeat column sum into my_vertical_sum
		pixel_counter = 0;
		for (int i = 0; i < directional_image_sum->logical_x_dimension; i++)
		{
			for (int j = 0; j < directional_image_sum->logical_y_dimension; j++)
			{
				pixel_coord_x = directional_image_sum->ReturnReal1DAddressFromPhysicalCoord(i, 0, 0);
				pixel_coord_xy = directional_image_sum->ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
				directional_image_sum->real_values[pixel_coord_xy] = directional_image_sum->real_values[pixel_coord_x];
				pixel_counter++;
			}
			pixel_counter += directional_image_sum->padding_jump_value;
		}
	}
}
*/

// rotational and translational alignment of tubes
// function similar to unblur_refine_alignment in unblur.cpp with a few differences
// 1. Does not do smoothing, 2. Does not shift in y (vertically), 3. Does not subtract current image from sum 4. Does not calculate running average (must be set to 1)
void azimuthal_alignment(Image* input_stack, Image* input_stack_times_ctf, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float start_angle_for_peak_search, float end_angle_for_peak_search, float rotation_step_size, float max_rotation_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, float* psi_angles, float* ctf_sum_of_squares) {
    long pixel_counter;
    long image_counter;
    int  running_average_counter;
    int  start_frame_for_average;
    int  end_frame_for_average;
    long iteration_counter;

    int number_of_middle_image    = number_of_images / 2;
    int running_average_half_size = (number_of_frames_for_running_average - 1) / 2;
    if ( running_average_half_size < 1 )
        running_average_half_size = 1;

    // rotational alignment parameters
    float psi;

    int number_of_rotations = int((end_angle_for_peak_search - start_angle_for_peak_search) / rotation_step_size + 0.5);
    if ( number_of_rotations < 1 )
        number_of_rotations = 1;
    rotation_step_size = (end_angle_for_peak_search - start_angle_for_peak_search) / number_of_rotations;
    if ( rotation_step_size < 0.25 )
        rotation_step_size = 0.25;

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
    Image* buffer_stack = new Image[number_of_images];

    Image* stack_for_alignment; // pointer that can be switched between running average stack and image stack if necessary
    Peak   current_peak;

    sum_of_images.Allocate(input_stack_times_ctf[0].logical_x_dimension, input_stack_times_ctf[0].logical_y_dimension, false);
    sum_of_images.SetToConstant(0.0);

    if ( number_of_frames_for_running_average > 1 ) {
        running_average_stack = new Image[number_of_images];

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            running_average_stack[image_counter].Allocate(input_stack_times_ctf[image_counter].logical_x_dimension, input_stack_times_ctf[image_counter].logical_y_dimension, 1, false);
        }

        stack_for_alignment = running_average_stack;
    }
    else
        stack_for_alignment = input_stack_times_ctf;

    // prepare the initial sum which is vertically-averaged CTF-corrected sum

    for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        sum_of_images.AddImage(&input_stack_times_ctf[image_counter]);
        current_x_shifts[image_counter]   = 0;
        current_y_shifts[image_counter]   = 0;
        current_psi_angles[image_counter] = 0;
    }

    // vertical average
    divide_by_ctf_sum_of_squares(&sum_of_images, ctf_sum_of_squares);
    sum_of_images.BackwardFFT( );
    sum_image_direction(&sum_of_images, 2);
    sum_of_images.ForwardFFT( );

    // print unaligned sum
    sum_of_images.QuickAndDirtyWriteSlice("my_unaligned_sum.mrc", 1);

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
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter, start_frame_for_average, end_frame_for_average, running_average_counter)
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
                running_average_stack[image_counter].SetToConstant(0.0f);

                for ( running_average_counter = start_frame_for_average; running_average_counter <= end_frame_for_average; running_average_counter++ ) {
                    running_average_stack[image_counter].AddImage(&input_stack_times_ctf[running_average_counter]);
                }
            }
        }

// do no subtract current image from sum and do not shift in y-direction
#pragma omp parallel default(shared) num_threads(max_threads) private(image_counter, sum_of_images_temp, current_peak, psi, best_inplane_score, best_inplane_values)
        { // for omp

            sum_of_images_temp.Allocate(input_stack_times_ctf[0].logical_x_dimension, input_stack_times_ctf[0].logical_y_dimension, false);

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
                // loop over rotations
                for ( int psi_i = 0; psi_i <= number_of_rotations; psi_i++ ) {
                    // re-copy sum_of_images
                    sum_of_images_temp.CopyFrom(&sum_of_images);
                    sum_of_images_temp.ApplyBFactor(unitless_bfactor);

                    if ( mask_central_cross == true ) {
                        sum_of_images_temp.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
                    }

                    sum_of_images_temp.BackwardFFT( );

                    psi = start_angle_for_peak_search + psi_i * rotation_step_size;
                    sum_of_images_temp.Rotate2DInPlace(psi, 0.0); // 0.0 pixel size is mask with radius of half-box size by default
                    // debug by writing out sum_of_images_temp
                    //if (psi_i == 0) sum_of_images_temp.QuickAndDirtyWriteSlice("my_rotated_sum.mrc", 1);
                    sum_of_images_temp.ForwardFFT( );

                    // compute the cross correlation function and find the peak
                    sum_of_images_temp.CalculateCrossCorrelationImageWith(&stack_for_alignment[image_counter]);
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
                //current_y_shifts[image_counter] = best_inplane_values[2]; // do not shift in y-direction
                current_y_shifts[image_counter] = 0.;
            }

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
			if (x_shifts_curve.NumberOfPoints( ) > 2)
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
			if (savitzy_golay_window_size < x_shifts_curve.NumberOfPoints( )) // when the input movie is dodgy (very few frames), the fitting won't work
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
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            // recopy original image stack if we have to do another round and apply current rotation and shift
            buffer_stack[image_counter].CopyFrom(&input_stack_times_ctf[image_counter]);

            // rotate first
            buffer_stack[image_counter].BackwardFFT( );
            buffer_stack[image_counter].Rotate2DInPlace(current_psi_angles[image_counter], 0.0); // 0.0 pixel size is mask with radius of half-box size
            buffer_stack[image_counter].ForwardFFT( );

            // then shift
            buffer_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

            x_shifts[image_counter] += current_x_shifts[image_counter];
            y_shifts[image_counter] += current_y_shifts[image_counter];
            psi_angles[image_counter] += current_psi_angles[image_counter];
        }

        // check to see if the convergence criteria have been reached and return if so

        if ( iteration_counter >= max_iterations || (max_shift <= max_shift_convergence_threshold && max_rotation <= max_rotation_convergence_threshold) ) {
            wxPrintf("returning, iteration = %li, max_shift = %f, max_rotation = %f, average_shift = %f, average_rotation = %f\n", iteration_counter, max_shift, max_rotation, average_image_x_shift, average_image_psi_rotation);

// shift and rotate the final image stack,
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                // rotate and shift input_stack * CTF

                // rotate first
                input_stack_times_ctf[image_counter].BackwardFFT( );
                input_stack_times_ctf[image_counter].Rotate2DInPlace(current_psi_angles[image_counter], 0.0); // 0.0 pixel size is mask with radius of half-box size
                input_stack_times_ctf[image_counter].ForwardFFT( );
                // then shift
                input_stack_times_ctf[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

                // rotate and shift input_stack

                // rotate first
                input_stack[image_counter].BackwardFFT( );
                input_stack[image_counter].Rotate2DInPlace(current_psi_angles[image_counter], 0.0); // 0.0 pixel size is mask with radius of half-box size
                input_stack[image_counter].ForwardFFT( );
                // then shift
                input_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);
            }

            delete[] current_x_shifts;
            delete[] current_y_shifts;
            delete[] current_psi_angles;
            delete[] buffer_stack;

            if ( number_of_frames_for_running_average > 1 ) {
                delete[] running_average_stack;
            }
            return;
        }
        else {
            wxPrintf("Not. returning, iteration = %li, max_shift = %f, max_rotation = %f, average_shift = %f, average_rotation = %f\n", iteration_counter, max_shift, max_rotation, average_image_x_shift, average_image_psi_rotation);
        }

        // going to be doing another round so we need to make the new sum..

        sum_of_images.SetToConstant(0.0);

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            sum_of_images.AddImage(&buffer_stack[image_counter]);
        }
        // vertically-averaged CTF-corrected sum
        divide_by_ctf_sum_of_squares(&sum_of_images, ctf_sum_of_squares);
        sum_of_images.BackwardFFT( );
        sum_image_direction(&sum_of_images, 2);
        sum_of_images.ForwardFFT( );

        // print out current alignment reference
        if ( iteration_counter == 1 ) {
            sum_of_images.QuickAndDirtyWriteSlice("my_aligned_sum_1.mrc", 1);
        }
        if ( iteration_counter == 2 ) {
            sum_of_images.QuickAndDirtyWriteSlice("my_aligned_sum_2.mrc", 1);
        }
        if ( iteration_counter == 3 ) {
            sum_of_images.QuickAndDirtyWriteSlice("my_aligned_sum_3.mrc", 1);
        }
        //
    } // end alignment cycle
}

// computes the 1D rotational average and sets the corner values to the average value at edge-1
void average_rotationally(Image* current_image, Image* current_volume) {
    /*
	// check if image is in real space
	MyDebugAssertTrue(current_volume->is_in_real_space,"Cannot rotationally average a complex image");
	// check if image is allocated
	MyDebugAssertTrue(current_volume->is_in_memory,"Cannot rotationally average a unallocated image");
	*/

    // max radius in real space is sqrt(2)*0.5*logical_dimension
    long  number_of_rings = current_image->logical_x_dimension;
    float edge_value      = current_image->ReturnAverageOfRealValues(std::min(current_image->physical_address_of_box_center_x - 2, current_image->physical_address_of_box_center_y - 2), true);

    double* ring_axis   = new double[number_of_rings];
    double* ring_values = new double[number_of_rings];
    double* ring_weight = new double[number_of_rings];

    long central_x_pixel = current_image->physical_address_of_box_center_x;
    long central_y_pixel = current_image->physical_address_of_box_center_y;

    double radius;
    double difference;
    long   index_of_bin;
    long   counter;
    long   x;
    long   y;
    long   z;

    // intialize values and weights (number of bins run from 0 to N-1)

    for ( counter = 0; counter < number_of_rings; counter++ ) {
        ring_axis[counter]   = 0.0 + counter * (current_image->ReturnMaximumDiagonalRadius( ) - 0.0) / float(number_of_rings - 1);
        ring_values[counter] = 0;
        ring_weight[counter] = 0;
    }

    // edge radius in real space is 0.5*logical_dimension
    long edge_bin = long((0.5 * current_image->logical_x_dimension - ring_axis[0]) / (ring_axis[1] - ring_axis[0]));

    // now go through and work out the average;

    counter = 0;

    for ( y = 0; y < current_image->logical_y_dimension; y++ ) {
        for ( x = 0; x < current_image->logical_x_dimension; x++ ) {
            radius       = sqrtf(powf(double(central_x_pixel - x), 2) + powf(double(central_y_pixel - y), 2));
            index_of_bin = long((radius - ring_axis[0]) / (ring_axis[1] - ring_axis[0]));
            if ( index_of_bin >= edge_bin ) {
                ring_values[index_of_bin] += current_image->real_values[counter];
                //ring_values[index_of_bin] += edge_value;
                ring_weight[index_of_bin] += 1;
            }
            else {
                difference = (radius - ring_axis[index_of_bin]) / (ring_axis[index_of_bin + 1] - ring_axis[index_of_bin]);
                ring_values[index_of_bin] += current_image->real_values[counter] * (1 - difference);
                ring_values[index_of_bin + 1] += current_image->real_values[counter] * difference;
                ring_weight[index_of_bin] += (1 - difference);
                ring_weight[index_of_bin + 1] += difference;
            }

            counter++;
        }
        counter += current_image->padding_jump_value;
    }

    // divide by number of members..

    for ( counter = 0; counter < number_of_rings; counter++ ) {
        if ( ring_weight[counter] != 0.0 )
            ring_values[counter] /= ring_weight[counter];
    }

    // put the data back into the image

    counter = 0;

    for ( y = 0; y < current_image->logical_y_dimension; y++ ) {
        for ( x = 0; x < current_image->logical_x_dimension; x++ ) {
            radius       = sqrtf(powf(double(central_x_pixel - x), 2) + powf(double(central_y_pixel - y), 2));
            index_of_bin = long((radius - ring_axis[0]) / (ring_axis[1] - ring_axis[0]));

            if ( index_of_bin >= edge_bin ) {
                // set corner values to average at edge
                current_image->real_values[counter] = ring_values[edge_bin - 1];
                //current_image->real_values[counter] = edge_value;
            }
            else {
                difference                          = (radius - ring_axis[index_of_bin]) / (ring_axis[index_of_bin + 1] - ring_axis[index_of_bin]);
                current_image->real_values[counter] = (ring_values[index_of_bin] * (1 - difference)) + (ring_values[index_of_bin + 1] * difference);
            }

            counter++;
        }
        counter += current_image->padding_jump_value;
    }

    // put the data back into the image in the z-direction

    long pixel_coord_xy  = 0;
    long pixel_coord_xyz = 0;
    counter              = 0;

    for ( z = 0; z < current_volume->logical_z_dimension; z++ ) {
        for ( y = 0; y < current_volume->logical_y_dimension; y++ ) {
            for ( x = 0; x < current_volume->logical_x_dimension; x++ ) {
                pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                //pixel_coord_xyz = current_volume->ReturnReal1DAddressFromPhysicalCoord(x, y, z);
                //current_volume->real_values[pixel_coord_xyz] = current_image->real_values[pixel_coord_xy];
                current_volume->real_values[counter] = current_image->real_values[pixel_coord_xy];
                counter++;
            }
            counter += current_volume->padding_jump_value;
        }
    }

    delete[] ring_axis;
    delete[] ring_values;
    delete[] ring_weight;

    /*
	Image psf;
	psf.Allocate(current_image->logical_x_dimension, current_image->logical_y_dimension, true);

	for (int pixel_counter = 0; pixel_counter < current_image->real_memory_allocated; pixel_counter++)
	{
		psf.real_values[pixel_counter] = current_image->real_values[pixel_counter];
	}
	
	Curve psf_radial_average;
	Curve psf_radial_count;
	psf_radial_average.SetupXAxis(0.0, psf.ReturnMaximumDiagonalRadius(), psf.logical_x_dimension);
	psf_radial_count.SetupXAxis(0.0, psf.ReturnMaximumDiagonalRadius(), psf.logical_x_dimension);
	psf.Compute1DRotationalAverage(psf_radial_average, psf_radial_count);
*/

    /*	
	// in this method, the average extendes beyond the edge of the box
	// Pixel values in the image are replaced with the radial average from the image
	current_image->AverageRadially();

	// put the data back into the image in the z-direction

	long pixel_coord_xy = 0;
	long pixel_coord_xyz = 0;
	long counter = 0;
	long x;
	long y;
	long z;

	for (z = 0; z < current_volume->logical_z_dimension; z++)
	{
		for (y = 0; y < current_volume->logical_y_dimension; y++)
		{
			for (x = 0; x < current_volume->logical_x_dimension; x++)
			{
				pixel_coord_xy = current_image->ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
				//pixel_coord_xyz = current_volume->ReturnReal1DAddressFromPhysicalCoord(x, y, z);
				//current_volume->real_values[pixel_coord_xyz] = current_image->real_values[pixel_coord_xy];
				current_volume->real_values[counter] = current_image->real_values[pixel_coord_xy];
				counter++;
			}
			counter += current_volume->padding_jump_value;
		}
	}
*/
}

// takes projection and multiply by CTF of each image and then subtract image (removes low resolution features)
void scale_reference(Image* input_stack, int number_of_images, Image* current_volume, ctf_parameters* ctf_parameters_stack, bool absolute, int max_threads) {
    CTF    current_ctf;
    Image  buffer_proj;
    Image  proj_two; // side-view
    Image* buffer_stack = new Image[number_of_images];

    proj_two.Allocate(current_volume->logical_x_dimension, current_volume->logical_y_dimension, true);
    proj_two.SetToConstant(0.0);

    long pixel_counter = 0;
    long image_counter;

    for ( int k = 0; k < current_volume->logical_z_dimension; k++ ) {
        for ( int j = 0; j < current_volume->logical_y_dimension; j++ ) {
            for ( int i = 0; i < current_volume->logical_x_dimension; i++ ) {

                proj_two.real_values[proj_two.ReturnReal1DAddressFromPhysicalCoord(j, k, 0)] += current_volume->real_values[pixel_counter];

                pixel_counter++;
            }

            pixel_counter += current_volume->padding_jump_value;
        }
    }

    proj_two.ForwardFFT( );
    proj_two.ZeroCentralPixel( );

// copy original image stack
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
    for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        buffer_stack[image_counter].CopyFrom(&input_stack[image_counter]);
    }

    buffer_proj.Allocate(current_volume->logical_x_dimension, current_volume->logical_y_dimension, false);

    float sum_of_pixelwise_product;
    float sum_of_squares;
    float scale_factor; // per image scale factor
    float sum_of_scale_factors         = 0.; // average scale factor
    float sum_of_scale_factors_squares = 0.; // variance scale factor
    float scale_factor_variance;
    float average;

    for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        // initialize sums
        pixel_counter            = 0;
        sum_of_pixelwise_product = 0.;
        sum_of_squares           = 0.;
        scale_factor             = 0.;

        // copy projection
        buffer_proj.CopyFrom(&proj_two);

        // apply ctf
        current_ctf.Init(ctf_parameters_stack[image_counter].acceleration_voltage, ctf_parameters_stack[image_counter].spherical_aberration, ctf_parameters_stack[image_counter].amplitude_contrast, ctf_parameters_stack[image_counter].defocus_1, ctf_parameters_stack[image_counter].defocus_2, ctf_parameters_stack[image_counter].astigmatism_angle, ctf_parameters_stack[image_counter].lowest_frequency_for_fitting, ctf_parameters_stack[image_counter].highest_frequency_for_fitting, ctf_parameters_stack[image_counter].astigmatism_tolerance, ctf_parameters_stack[image_counter].pixel_size, ctf_parameters_stack[image_counter].additional_phase_shift);
        if ( absolute == true )
            buffer_proj.ApplyCTFPhaseFlip(current_ctf);
        else
            buffer_proj.ApplyCTF(current_ctf);

        buffer_proj.BackwardFFT( );

        if ( image_counter == 0 ) {
            buffer_proj.QuickAndDirtyWriteSlice("my_proj_times_ctf_one.mrc", 1);
        }

        // subtract mean value at edges to set zero-mean of noise
        // it would better to calculate noise average from vertical stripes from left and right edges of images
        average = buffer_proj.ReturnAverageOfRealValues(buffer_proj.physical_address_of_box_center_x - 10.0 / 2.0, true);
        buffer_proj.AddConstant(-average);
        //buffer_proj.AddConstant(-buffer_proj.ReturnAverageOfRealValuesOnEdges());
        //buffer_stack[image_counter].AddConstant(-buffer_stack[image_counter].ReturnAverageOfRealValuesOnEdges());

        // least squares scale factor = A*B / A*A (projection of B onto A)
        // here A = projection and B = micrograph
        for ( int j = 0; j < buffer_proj.logical_y_dimension; j++ ) {
            for ( int i = 0; i < buffer_proj.logical_x_dimension; i++ ) {

                sum_of_pixelwise_product += buffer_proj.real_values[pixel_counter] * buffer_stack[image_counter].real_values[pixel_counter];

                //sum_of_squares += buffer_stack[image_counter].real_values[pixel_counter] * buffer_stack[image_counter].real_values[pixel_counter];
                sum_of_squares += buffer_proj.real_values[pixel_counter] * buffer_proj.real_values[pixel_counter];

                pixel_counter++;
            }

            pixel_counter += buffer_proj.padding_jump_value;
        }

        scale_factor = sum_of_pixelwise_product / sum_of_squares;

        sum_of_scale_factors += scale_factor;
        sum_of_scale_factors_squares += powf(scale_factor, 2);

        // multiply by scale factor pixelwise
        buffer_proj.MultiplyByConstant(scale_factor);
        //buffer_stack[image_counter].MultiplyByConstant(1./scale_factor);

        // subtract projection from image
        buffer_stack[image_counter].SubtractImage(&buffer_proj);

        // print out sf*X_i - Proj*CTF_i
        buffer_stack[image_counter].QuickAndDirtyWriteSlice("my_subtracted_frames.mrc", image_counter + 1);

        // multiply image stack by per image scale factor
        //input_stack[image_counter].MultiplyByConstant(1./scale_factor);
    }

    // print average scale factor between images and projection
    sum_of_scale_factors /= number_of_images;
    sum_of_scale_factors_squares /= number_of_images;
    scale_factor_variance = fabsf(float(sum_of_scale_factors_squares - powf(sum_of_scale_factors, 2)));
    wxPrintf("mean scale factor = %f, variance scale factor = %f\n", sum_of_scale_factors, scale_factor_variance);

    /*
	// multiply image stack by average scale factor
	for (image_counter = 0; image_counter < number_of_images; image_counter++)
	{
		input_stack[image_counter].MultiplyByConstant(1./sum_of_scale_factors);
	}
	*/

    // multiply reference by average scale factor
    current_volume->MultiplyByConstant(sum_of_scale_factors);

    // clean-up
    buffer_proj.Deallocate( );
    delete[] buffer_stack;
}

void normalize_image(Image* input_image, float pixel_size, float mask_falloff) {
    // Normalize background variance and average
    float variance;
    float average;

    // subtract mean value from each image pixel to get a zero-mean
    // divide each pixel value by standard deviation to have unit-variance
    variance = input_image->ReturnVarianceOfRealValues(input_image->physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
    average  = input_image->ReturnAverageOfRealValues(input_image->physical_address_of_box_center_x - mask_falloff / pixel_size, true);
    if ( variance == 0.0f )
        input_image->SetToConstant(0.0f);
    else
        input_image->AddMultiplyConstant(-average, 1.0 / sqrtf(variance));
}
