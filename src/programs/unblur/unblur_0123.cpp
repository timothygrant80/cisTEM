#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <string>
#include <fstream>
#include <iostream>

#include "dlib/dlib/optimization.h"
#include <vector>
// #include "./splinefuncs.h"
#include <dlib/dlib/matrix.h>
#include "../niko_test/utilities.h"

// #include "../niko_test/bicubicspline.h"
// #include "../niko_test/cubicspline.h"
// #include "../niko_test/movieframespline.h"

// The timing that unblur originally tracks is always on, by direct reference to cistem_timer::StopWatch
// The profiling for development is under conrtol of --enable-profiling.
#ifdef PROFILING
using namespace cistem_timer;
#else
// #define PRINT_VERBOSE
using namespace cistem_timer_noop;
#endif

class
        UnBlurApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

// models
using namespace dlib;
using namespace std;

typedef matrix<double, 3, 1>  input_vector;
typedef matrix<double, 18, 1> param_vector_quadratic;
typedef matrix<double, 9, 1>  param_vector_linear;
typedef matrix<double, 0, 1>  column_vector;
MovieFrameSpline              Spline3dx, Spline3dy;
bicubicsplinestack            ccmap_stack;

// typedef matrix<double, 18, 1> param_vector;

// ----------------------------------------------------------------------------------------

double model_quadratic(
        const input_vector&           input,
        const param_vector_quadratic& params) {
    const double c0 = params(0);
    const double c1 = params(1);
    const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7);
    const double c8 = params(8), c9 = params(9), c10 = params(10), c11 = params(11), c12 = params(12), c13 = params(13);
    const double c14 = params(14), c15 = params(15), c16 = params(16), c17 = params(17);

    const double x = input(0);
    const double y = input(1);
    const double t = input(2);

    const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * pow(x, 2) * t + c7 * pow(x, 2) * pow(t, 2) + c8 * pow(x, 2) * pow(t, 3) + c9 * y * t + c10 * y * pow(t, 2) + c11 * y * pow(t, 3) + c12 * pow(y, 2) * t + c13 * pow(y, 2) * pow(t, 2) + c14 * pow(y, 2) * pow(t, 3) + c15 * x * y * t + c16 * x * y * pow(t, 2) + c17 * x * y * pow(t, 3);

    return temp;
}

double residual_quadratic(
        const std::pair<input_vector, double>& data,
        const param_vector_quadratic&          params) {
    return model_quadratic(data.first, params) - data.second;
}

param_vector_quadratic residual_derivative_quadratic(
        const std::pair<input_vector, double>& data,
        const param_vector_quadratic&          params) {
    param_vector_quadratic der;

    const double c0 = params(0);
    const double c1 = params(1);
    const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7);
    const double c8 = params(8), c9 = params(9), c10 = params(10), c11 = params(11), c12 = params(12), c13 = params(13);
    const double c14 = params(14), c15 = params(15), c16 = params(16), c17 = params(17);

    const double x = data.first(0);
    const double y = data.first(1);
    const double t = data.first(2);

    const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * pow(x, 2) * t + c7 * pow(x, 2) * pow(t, 2) + c8 * pow(x, 2) * pow(t, 3) + c9 * y * t + c10 * y * pow(t, 2) + c11 * y * pow(t, 3) + c12 * pow(y, 2) * t + c13 * pow(y, 2) * pow(t, 2) + c14 * pow(y, 2) * pow(t, 3) + c15 * x * y * t + c16 * x * y * pow(t, 2) + c17 * x * y * pow(t, 3);
    der(0)            = t;
    der(1)            = pow(t, 2);
    der(2)            = pow(t, 3);
    der(3)            = x * t;
    der(4)            = x * pow(t, 2);
    der(5)            = x * pow(t, 3);
    der(6)            = pow(x, 2) * t;
    der(7)            = pow(x, 2) * pow(t, 2);
    der(8)            = pow(x, 2) * pow(t, 3);
    der(9)            = y * t;
    der(10)           = y * pow(t, 2);
    der(11)           = y * pow(t, 3);
    der(12)           = pow(y, 2) * t;
    der(13)           = pow(y, 2) * pow(t, 2);
    der(14)           = pow(y, 2) * pow(t, 3);
    der(15)           = x * y * t;
    der(16)           = x * y * pow(t, 2);
    der(17)           = x * y * pow(t, 3);
    return der;
}

double model_linear(
        const input_vector&        input,
        const param_vector_linear& params) {
    const double c0 = params(0);
    const double c1 = params(1);
    const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7), c8 = params(8);
    //add  params(9) to params(17) equals to 0??
    const double x = input(0);
    const double y = input(1);
    const double t = input(2);

    const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * y * t + c7 * y * pow(t, 2) + c8 * y * pow(t, 3);

    return temp;
}

double residual_linear(
        const std::pair<input_vector, double>& data,
        const param_vector_linear&             params) {
    return model_linear(data.first, params) - data.second;
}

// This function is the derivative of the residual() function with respect to the parameters.
param_vector_linear residual_derivative_linear(
        const std::pair<input_vector, double>& data,
        const param_vector_linear&             params) {
    param_vector_linear der;

    const double c0 = params(0);
    const double c1 = params(1);
    const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7);
    const double c8 = params(8);
    //add  params(9) to params(17) equals to 0??
    //add  der(9) to der(17) equals to 0??
    const double x = data.first(0);
    const double y = data.first(1);
    const double t = data.first(2);

    const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * y * t + c7 * y * pow(t, 2) + c8 * y * pow(t, 3);
    der(0)            = t;
    der(1)            = pow(t, 2);
    der(2)            = pow(t, 3);
    der(3)            = x * t;
    der(4)            = x * pow(t, 2);
    der(5)            = x * pow(t, 3);
    der(6)            = y * t;
    der(7)            = y * pow(t, 2);
    der(8)            = y * pow(t, 3);

    return der;
}

double minfunc3dSplineObjectxy(matrix<double> join_1d) {
    double         result, result1, result2;
    int            len, halflen;
    matrix<double> tmp;
    len       = join_1d.size( );
    halflen   = len / 2;
    int count = 0;
    // cout << range(0, halflen) << endl;
    // cout << range(halflen, len - 1) << endl;
    tmp = colm(join_1d, range(0, halflen - 1));
    tmp.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        tmp(i) = join_1d(count);
        count++;
    }
    result1 = Spline3dx.OptimizationKnotObejctFast(tmp);
    for ( int i = 0; i < halflen; i++ ) {
        tmp(i) = join_1d(count);
        count++;
    }
    result2 = Spline3dy.OptimizationKnotObejctFast(tmp);
    result  = result1 + result2;
    return result;
}

double minfunc3dSplineObjectxyControlPoints(matrix<double> control_1d) {
    double         result, result1, result2;
    int            len, halflen;
    matrix<double> tmp;
    len       = control_1d.size( );
    halflen   = len / 2;
    int count = 0;
    // cout << range(0, halflen) << endl;
    // cout << range(halflen, len - 1) << endl;
    tmp = colm(control_1d, range(0, halflen - 1));
    tmp.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        tmp(i) = control_1d(count);
        count++;
    }
    // wxPrintf("before update\n");
    result1 = Spline3dx.OptimizationKnotObejctFromControlPoints(tmp);
    for ( int i = 0; i < halflen; i++ ) {
        tmp(i) = control_1d(count);
        count++;
    }
    result2 = Spline3dy.OptimizationKnotObejctFromControlPoints(tmp);
    result  = result1 + result2;
    return result;
}

double minfunc3dSplineCCLossObject(matrix<double> join_1d) {
    double          loss;
    matrix<double>* shiftsx;
    matrix<double>* shiftsy;
    int             len, halflen;
    matrix<double>  knot_on_spline_for_x, knot_on_spline_for_y;
    len       = join_1d.size( );
    halflen   = len / 2;
    int count = 0;

    // pass some params:
    int image_no    = Spline3dx.frame_no;
    int patch_x_num = Spline3dx.Mapping_Mat_Col_no;
    int patch_y_num = Spline3dx.Mapping_Mat_Row_no;

    int ccmap_patch_dim      = ccmap_stack.m;
    int half_ccmap_patch_dim = ccmap_patch_dim / 2;

    // tmp = colm(join_1d, range(0, halflen - 1));
    knot_on_spline_for_x.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_x(i) = join_1d(count);
        count++;
    }

    shiftsx = Spline3dx.KnotToInterp(knot_on_spline_for_x);

    knot_on_spline_for_y.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_y(i) = join_1d(count);
        count++;
    }
    shiftsy = Spline3dy.KnotToInterp(knot_on_spline_for_y);

    loss = 0;
    for ( int img_ind = 0; img_ind < image_no; img_ind++ ) {
        for ( int i = 0; i < patch_y_num; i++ ) {
            for ( int j = 0; j < patch_x_num; j++ ) {
                int patch_ind = i * patch_x_num + j;
                loss -= ccmap_stack.spline_stack[patch_ind * image_no + img_ind].ApplySplineFunc(shiftsx[img_ind](i, j) + half_ccmap_patch_dim, shiftsy[img_ind](i, j) + half_ccmap_patch_dim);
            }
        }
    }
    loss = loss;
    return loss;
}

double minfunc3dSplineCCLossObjectControlPoints(matrix<double> control_1d) {
    double          loss;
    matrix<double>* shiftsx;
    matrix<double>* shiftsy;
    int             len, halflen;
    matrix<double>  control_on_spline_for_x, control_on_spline_for_y;
    len       = control_1d.size( );
    halflen   = len / 2;
    int count = 0;

    // pass some params:
    int image_no    = Spline3dx.frame_no;
    int patch_x_num = Spline3dx.Mapping_Mat_Col_no;
    int patch_y_num = Spline3dx.Mapping_Mat_Row_no;

    int ccmap_patch_dim      = ccmap_stack.m;
    int half_ccmap_patch_dim = ccmap_patch_dim / 2;

    // tmp = colm(join_1d, range(0, halflen - 1));
    control_on_spline_for_x.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_x(i) = control_1d(count);
        count++;
    }

    // shiftsx = Spline3dx.KnotToInterp(knot_on_spline_for_x);
    shiftsx = Spline3dx.ControlToInterp(control_on_spline_for_x);

    control_on_spline_for_y.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_y(i) = control_1d(count);
        count++;
    }
    shiftsy = Spline3dy.ControlToInterp(control_on_spline_for_y);

    loss = 0;
    for ( int img_ind = 0; img_ind < image_no; img_ind++ ) {
        for ( int i = 0; i < patch_y_num; i++ ) {
            for ( int j = 0; j < patch_x_num; j++ ) {
                int patch_ind = i * patch_x_num + j;
                loss -= ccmap_stack.spline_stack[patch_ind * image_no + img_ind].ApplySplineFunc(shiftsx[img_ind](i, j) + half_ccmap_patch_dim, shiftsy[img_ind](i, j) + half_ccmap_patch_dim);
            }
        }
    }
    loss = loss;
    return loss;
}

void    unblur_refine_alignment(Image* input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, StopWatch& profile_timing_refinement_method, bool reverse_shift);
Image** patch_trimming_independent(Image* input_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, wxString outpath, bool use_coord_file = false, wxString coordinates_filename = " ");
void    patch_trimming(Image* input_stack, Image** patch_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, wxString outpath, int max_threads, bool use_coord_file, wxString coordinates_filename = " ");
void    split_and_update(matrix<double> Join1d, int joinsize);
void    apply_fitting_quadratic(Image* input_stack, int number_of_images, param_vector_quadratic params_x, param_vector_quadratic params_y);
void    apply_fitting_linear(Image* input_stack, int number_of_images, param_vector_linear params_x, param_vector_linear params_y);
void    apply_fitting_quadratic_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, param_vector_quadratic params_x, param_vector_quadratic params_y);
void    apply_fitting_linear_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, param_vector_linear params_x, param_vector_linear params_y);
// void    apply_fitting_spline_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d, int max_threads);
// void apply_fitting_spline_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d_R0, column_vector Join1d_R1, int max_threads);
void apply_fitting_spline_sup(int output_x_size, int output_y_size, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d_R0, column_vector Join1d_R1, int max_threads);
void apply_fitting_spline_sup_control(int output_x_size, int output_y_size, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Control1d_R0, column_vector Control1d_R1, int max_threads);
void dosefilter(Image* image_stack, int first_frame, int last_frame, float* dose_filter_sum_of_squares, ElectronDose* my_electron_dose, StopWatch profile_timing, float exposure_per_frame, float pre_exposure_amount, int max_threads);
void write_shifts(int patch_no_x, int patch_no_y, int image_no, std::string output_path, std::string shift_file_prefx, std::string shift_file_prefy);

IMPLEMENT_APP(UnBlurApp)

// override the DoInteractiveUserInput

void UnBlurApp::DoInteractiveUserInput( ) {
    std::string input_filename;
    std::string output_filename;
    std::string aligned_frames_filename;
    std::string output_shift_text_file;
    float       original_pixel_size                = 1;
    float       minimum_shift_in_angstroms         = 2;
    float       maximum_shift_in_angstroms         = 80;
    bool        should_dose_filter                 = true;
    bool        should_restore_power               = true;
    float       termination_threshold_in_angstroms = 1;
    int         max_iterations                     = 20;
    float       bfactor_in_angstroms               = 1500;
    bool        should_mask_central_cross          = true;
    int         horizontal_mask_size               = 1;
    int         vertical_mask_size                 = 1;
    float       exposure_per_frame                 = 0.0;
    float       acceleration_voltage               = 300.0;
    float       pre_exposure_amount                = 0.0;
    bool        movie_is_gain_corrected            = true;
    wxString    gain_filename                      = "";
    bool        movie_is_dark_corrected            = true;
    wxString    dark_filename                      = "";
    float       output_binning_factor              = 1;

    bool     set_expert_options;
    bool     correct_mag_distortion;
    float    mag_distortion_angle;
    float    mag_distortion_major_scale;
    float    mag_distortion_minor_scale;
    bool     patchcorrection;
    int      patch_num_x;
    int      patch_num_y;
    int      distortion_model;
    wxString outputpath = "";
    int      first_frame;
    int      last_frame;
    int      number_of_frames_for_running_average;
    int      max_threads;
    int      eer_frames_per_image = 0;
    int      eer_super_res_factor = 1;

    bool save_aligned_frames;

    UserInput* my_input = new UserInput("Unblur", 2.0);

    input_filename         = my_input->GetFilenameFromUser("Input stack filename", "The input file, containing your raw movie frames", "my_movie.mrc", true);
    output_filename        = my_input->GetFilenameFromUser("Output aligned sum", "The output file, containing a weighted sum of the aligned input frames", "my_aligned_sum.mrc", false);
    output_shift_text_file = my_input->GetFilenameFromUser("Output shift text file", "The output text file, containing shifts in angstroms", "my_shifts.txt", false);

    original_pixel_size   = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    output_binning_factor = my_input->GetFloatFromUser("Output binning factor", "Output images will be binned (downsampled) by this factor relative to the input images", "1", 1);
    should_dose_filter    = my_input->GetYesNoFromUser("Apply Exposure filter?", "Apply an exposure-dependent filter to frames before summing them", "yes");

    if ( should_dose_filter == true ) {
        acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (kV)", "Acceleration voltage during imaging", "300.0");
        exposure_per_frame   = my_input->GetFloatFromUser("Exposure per frame (e/A^2)", "Exposure per frame, in electrons per square Angstrom", "1.0", 0.0);
        pre_exposure_amount  = my_input->GetFloatFromUser("Pre-exposure amount (e/A^2)", "Amount of pre-exposure prior to the first frame, in electrons per square Angstrom", "0.0", 0.0);
    }
    else {
        exposure_per_frame   = 0.0;
        acceleration_voltage = 300.0;
        pre_exposure_amount  = 0.0;
    }

    set_expert_options = my_input->GetYesNoFromUser("Set Expert Options?", "Set these for more control, hopefully not needed", "no");

    if ( set_expert_options == true ) {
        minimum_shift_in_angstroms         = my_input->GetFloatFromUser("Minimum shift for initial search (A)", "Initial search will be limited to between the inner and outer radii.", "2.0", 0.0);
        maximum_shift_in_angstroms         = my_input->GetFloatFromUser("Outer radius shift limit (A)", "The maximum shift of each alignment step will be limited to this value.", "80.0", minimum_shift_in_angstroms);
        bfactor_in_angstroms               = my_input->GetFloatFromUser("B-factor to apply to images (A^2)", "This B-Factor will be used to filter the reference prior to alignment", "1500", 0.0);
        vertical_mask_size                 = my_input->GetIntFromUser("Half-width of vertical Fourier mask", "The vertical line mask will be twice this size. The central cross mask helps\nreduce problems by line artefacts from the detector", "1", 1);
        horizontal_mask_size               = my_input->GetIntFromUser("Half-width of horizontal Fourier mask", "The horizontal line mask will be twice this size. The central cross mask helps\nreduce problems by line artefacts from the detector", "1", 1);
        termination_threshold_in_angstroms = my_input->GetFloatFromUser("Termination shift threshold (A)", "Alignment will iterate until the maximum shift is below this value", "1", 0.0);
        max_iterations                     = my_input->GetIntFromUser("Maximum number of iterations", "Alignment will stop at this number, even if the threshold shift is not reached", "20", 0);

        if ( should_dose_filter == true ) {
            should_restore_power = my_input->GetYesNoFromUser("Restore Noise Power?", "Restore the power of the noise to the level it would be without exposure filtering", "yes");
        }

        movie_is_dark_corrected = my_input->GetYesNoFromUser("Input stack is dark-subtracted?", "The input frames are already dark substracted", "yes");

        if ( ! movie_is_dark_corrected ) {
            dark_filename = my_input->GetFilenameFromUser("Dark image filename", "The filename of the camera's dark reference image", "my_dark_reference.dm4", true);
        }

        movie_is_gain_corrected = my_input->GetYesNoFromUser("Input stack is gain-corrected?", "The input frames are already gain-corrected", "yes");

        if ( ! movie_is_gain_corrected ) {
            gain_filename = my_input->GetFilenameFromUser("Gain image filename", "The filename of the camera's gain reference image", "my_gain_reference.dm4", true);
        }

        first_frame = my_input->GetIntFromUser("First frame to use for sum", "You can use this to ignore the first n frames", "1", 1);
        last_frame  = my_input->GetIntFromUser("Last frame to use for sum (0 for last frame)", "You can use this to ignore the last n frames", "0", 0);

        number_of_frames_for_running_average = my_input->GetIntFromUser("Number of frames for running average", "use a running average of frames, useful for low SNR frames, must be odd", "1", 1);

        save_aligned_frames = my_input->GetYesNoFromUser("Save Aligned Frames?", "If yes, save the aligned frames", "no");
        if ( save_aligned_frames == true ) {
            aligned_frames_filename = my_input->GetFilenameFromUser("Output aligned frames filename", "The output file, containing your aligned movie frames", "my_aligned_frames.mrc", false);
        }
        else {
            aligned_frames_filename = "";
        }

        if ( FilenameExtensionMatches(input_filename, "eer") ) {
            eer_frames_per_image = my_input->GetIntFromUser("Number of EER frames per image", "If the input movie is in EER format, we will average EER frames together so that each frame image for alignment has a reasonable exposure", "25", 1);
            eer_super_res_factor = my_input->GetIntFromUser("EER super resolution factor", "Choose between 1 (no supersampling), 2, or 4 (image pixel size will be 4 times smaller that the camera phyiscal pixel)", "1", 1, 4);
        }
        else {
            eer_frames_per_image = 0;
            eer_super_res_factor = 1;
        }
    }
    else {
        minimum_shift_in_angstroms           = original_pixel_size * output_binning_factor + 0.001;
        maximum_shift_in_angstroms           = 100.0;
        bfactor_in_angstroms                 = 1500.0;
        vertical_mask_size                   = 1;
        horizontal_mask_size                 = 1;
        termination_threshold_in_angstroms   = original_pixel_size * output_binning_factor / 2;
        max_iterations                       = 20;
        should_restore_power                 = true;
        movie_is_gain_corrected              = true;
        movie_is_dark_corrected              = true;
        gain_filename                        = "";
        dark_filename                        = "";
        first_frame                          = 1;
        last_frame                           = 0;
        number_of_frames_for_running_average = 1;
        save_aligned_frames                  = false;
        aligned_frames_filename              = "";
        eer_frames_per_image                 = 0;
        eer_super_res_factor                 = 1;
    }

    correct_mag_distortion = my_input->GetYesNoFromUser("Correct Magnification Distortion?", "If yes, a magnification distortion can be corrected", "no");

    if ( correct_mag_distortion == true ) {
        mag_distortion_angle       = my_input->GetFloatFromUser("Distortion Angle (Degrees)", "The distortion angle in degrees", "0.0");
        mag_distortion_major_scale = my_input->GetFloatFromUser("Major Scale", "The major axis scale factor", "1.0", 0.0);
        mag_distortion_minor_scale = my_input->GetFloatFromUser("Minor Scale", "The minor axis scale factor", "1.0", 0.0);
        ;
    }
    else {
        mag_distortion_angle       = 0.0;
        mag_distortion_major_scale = 1.0;
        mag_distortion_minor_scale = 1.0;
    }

    patchcorrection = my_input->GetYesNoFromUser("Use Patchbased Distortion Correction?", "If yes, distortion correction will be conducted", "yes");

    if ( patchcorrection == true ) {
        patch_num_x      = my_input->GetIntFromUser("Number of Patches along X Axis", "input integer", "6");
        patch_num_y      = my_input->GetIntFromUser("Number of Patches along Y Axis", "input integer", "4");
        outputpath       = my_input->GetStringFromUser("Output patch and patch shift path", "output movie patchs path", "/data/outpatch/");
        distortion_model = my_input->GetIntFromUser("1 for linear, 2 for quadratic model", "input integer", "2");
    }
    else {
        patch_num_x      = 1;
        patch_num_y      = 1;
        distortion_model = 2;
    }

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    // this are defaulted to off in the interactive version for now
    bool        write_out_amplitude_spectrum = false;
    std::string amplitude_spectrum_filename  = "/dev/null";
    bool        write_out_small_sum_image    = false;
    std::string small_sum_image_filename     = "/dev/null";

    my_current_job.ManualSetArguments("ttfffbbfifbiifffbsbsfbfffbtbtiiiibttiisbiii", input_filename.c_str( ),
                                      output_filename.c_str( ),
                                      original_pixel_size,
                                      minimum_shift_in_angstroms,
                                      maximum_shift_in_angstroms,
                                      should_dose_filter,
                                      should_restore_power,
                                      termination_threshold_in_angstroms,
                                      max_iterations,
                                      bfactor_in_angstroms,
                                      should_mask_central_cross,
                                      horizontal_mask_size,
                                      vertical_mask_size,
                                      acceleration_voltage,
                                      exposure_per_frame,
                                      pre_exposure_amount,
                                      movie_is_gain_corrected,
                                      gain_filename.ToStdString( ).c_str( ),
                                      movie_is_dark_corrected,
                                      dark_filename.ToStdString( ).c_str( ),
                                      output_binning_factor,
                                      correct_mag_distortion,
                                      mag_distortion_angle,
                                      mag_distortion_major_scale,
                                      mag_distortion_minor_scale,
                                      write_out_amplitude_spectrum,
                                      amplitude_spectrum_filename.c_str( ),
                                      write_out_small_sum_image,
                                      small_sum_image_filename.c_str( ),
                                      first_frame,
                                      last_frame,
                                      number_of_frames_for_running_average,
                                      max_threads,
                                      save_aligned_frames,
                                      aligned_frames_filename.c_str( ),
                                      output_shift_text_file.c_str( ),
                                      eer_frames_per_image,
                                      eer_super_res_factor,
                                      outputpath.ToUTF8( ).data( ),
                                      patchcorrection,
                                      patch_num_x,
                                      patch_num_y,
                                      distortion_model);
}

// overide the do calculation method which will be what is actually run..

bool UnBlurApp::DoCalculation( ) {
    int  pre_binning_factor;
    long image_counter;
    int  pixel_counter;

    float unitless_bfactor;

    float pixel_size;
    float min_shift_in_pixels;
    float max_shift_in_pixels;
    float termination_threshold_in_pixels;

    Image sum_image;
    Image sum_image_no_dose_filter;

    // get the arguments for this job..

    std::string input_filename                       = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename                      = my_current_job.arguments[1].ReturnStringArgument( );
    float       original_pixel_size                  = my_current_job.arguments[2].ReturnFloatArgument( );
    float       minumum_shift_in_angstroms           = my_current_job.arguments[3].ReturnFloatArgument( );
    float       maximum_shift_in_angstroms           = my_current_job.arguments[4].ReturnFloatArgument( );
    bool        should_dose_filter                   = my_current_job.arguments[5].ReturnBoolArgument( );
    bool        should_restore_power                 = my_current_job.arguments[6].ReturnBoolArgument( );
    float       termination_threshold_in_angstoms    = my_current_job.arguments[7].ReturnFloatArgument( );
    int         max_iterations                       = my_current_job.arguments[8].ReturnIntegerArgument( );
    float       bfactor_in_angstoms                  = my_current_job.arguments[9].ReturnFloatArgument( );
    bool        should_mask_central_cross            = my_current_job.arguments[10].ReturnBoolArgument( );
    int         horizontal_mask_size                 = my_current_job.arguments[11].ReturnIntegerArgument( );
    int         vertical_mask_size                   = my_current_job.arguments[12].ReturnIntegerArgument( );
    float       acceleration_voltage                 = my_current_job.arguments[13].ReturnFloatArgument( );
    float       exposure_per_frame                   = my_current_job.arguments[14].ReturnFloatArgument( );
    float       pre_exposure_amount                  = my_current_job.arguments[15].ReturnFloatArgument( );
    bool        movie_is_gain_corrected              = my_current_job.arguments[16].ReturnBoolArgument( );
    wxString    gain_filename                        = my_current_job.arguments[17].ReturnStringArgument( );
    bool        movie_is_dark_corrected              = my_current_job.arguments[18].ReturnBoolArgument( );
    wxString    dark_filename                        = my_current_job.arguments[19].ReturnStringArgument( );
    float       output_binning_factor                = my_current_job.arguments[20].ReturnFloatArgument( );
    bool        correct_mag_distortion               = my_current_job.arguments[21].ReturnBoolArgument( );
    float       mag_distortion_angle                 = my_current_job.arguments[22].ReturnFloatArgument( );
    float       mag_distortion_major_scale           = my_current_job.arguments[23].ReturnFloatArgument( );
    float       mag_distortion_minor_scale           = my_current_job.arguments[24].ReturnFloatArgument( );
    bool        write_out_amplitude_spectrum         = my_current_job.arguments[25].ReturnBoolArgument( );
    std::string amplitude_spectrum_filename          = my_current_job.arguments[26].ReturnStringArgument( );
    bool        write_out_small_sum_image            = my_current_job.arguments[27].ReturnBoolArgument( );
    std::string small_sum_image_filename             = my_current_job.arguments[28].ReturnStringArgument( );
    int         first_frame                          = my_current_job.arguments[29].ReturnIntegerArgument( );
    int         last_frame                           = my_current_job.arguments[30].ReturnIntegerArgument( );
    int         number_of_frames_for_running_average = my_current_job.arguments[31].ReturnIntegerArgument( );
    int         max_threads                          = my_current_job.arguments[32].ReturnIntegerArgument( );
    bool        saved_aligned_frames                 = my_current_job.arguments[33].ReturnBoolArgument( );
    std::string aligned_frames_filename              = my_current_job.arguments[34].ReturnStringArgument( );
    std::string output_shift_text_file               = my_current_job.arguments[35].ReturnStringArgument( );
    int         eer_frames_per_image                 = my_current_job.arguments[36].ReturnIntegerArgument( );
    int         eer_super_res_factor                 = my_current_job.arguments[37].ReturnIntegerArgument( );
    wxString    outputpath                           = my_current_job.arguments[38].ReturnStringArgument( );
    bool        patch_track                          = my_current_job.arguments[39].ReturnBoolArgument( );
    int         patch_num_x                          = my_current_job.arguments[40].ReturnIntegerArgument( );
    int         patch_num_y                          = my_current_job.arguments[41].ReturnIntegerArgument( );
    int         distortion_model                     = my_current_job.arguments[42].ReturnIntegerArgument( );

    if ( is_running_locally == false )
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...

    //my_current_job.PrintAllArguments();

    // StopWatch objects
    cistem_timer::StopWatch unblur_timing;
    StopWatch               profile_timing;
    StopWatch               profile_timing_refinement_method;

    float temp_float[2];

    if ( IsOdd(number_of_frames_for_running_average == false) )
        SendError("Error: number of frames for running average must be odd");

    // The Files

    if ( ! DoesFileExist(input_filename) ) {
        SendError(wxString::Format("Error: Input movie %s not found\n", input_filename));
        exit(-1);
    }
    ImageFile input_file;
    bool      input_file_is_valid = input_file.OpenFile(input_filename, false, false, false, eer_super_res_factor, eer_frames_per_image);
    if ( ! input_file_is_valid ) {
        SendInfo(wxString::Format("Input movie %s seems to be corrupt. Unblur results may not be meaningful.\n", input_filename));
    }
    else {
        wxPrintf("Input file looks OK, proceeding\n");
    }
    //MRCFile output_file(output_filename, true); changed to quick and dirty write as the file is only used once, and this way it is not created until it is actually written, which is cleaner for cancelled / crashed jobs

    ImageFile gain_file;
    ImageFile dark_file;

    if ( ! movie_is_gain_corrected ) {
        gain_file.OpenFile(gain_filename.ToStdString( ), false);
    }

    if ( ! movie_is_dark_corrected ) {
        dark_file.OpenFile(dark_filename.ToStdString( ), false);
    }

    long number_of_input_images = input_file.ReturnNumberOfSlices( );

    if ( last_frame == 0 )
        last_frame = number_of_input_images;

    if ( first_frame > number_of_input_images ) {
        SendError(wxString::Format("(%s) First frame is greater than total number of frames, using frame 1 instead.", input_filename));
        first_frame = 1;
    }

    if ( last_frame > number_of_input_images ) {
        SendError(wxString::Format("(%s) Specified last frame is greater than total number of frames.. using last frame instead.", input_filename));
        last_frame = number_of_input_images;
    }

    long slice_byte_size;

    Image* unbinned_image_stack; // We will allocate this later depending on if we are binning or not.
    Image* raw_image_stack; //stores the super resolution frames. used in distortion correction.
    Image* image_stack = new Image[number_of_input_images];
    Image* running_average_stack; // we will allocate this later if necessary;
    if ( patch_track ) {
        raw_image_stack = new Image[number_of_input_images];
    }
    Image* counting_image_stack;
    Image* unbinned_counting_stack;
    if ( patch_track ) {
        counting_image_stack = new Image[number_of_input_images];
    }

    Image gain_image;
    Image dark_image;

    // output sizes..

    int output_x_size;
    int output_y_size;

    if ( output_binning_factor > 1.0001 ) {
        output_x_size = myroundint(float(input_file.ReturnXSize( )) / output_binning_factor);
        output_y_size = myroundint(float(input_file.ReturnYSize( )) / output_binning_factor);
    }
    else {
        output_x_size = input_file.ReturnXSize( );
        output_y_size = input_file.ReturnYSize( );
    }

    // work out the output pixel size..

    float x_bin_factor       = float(input_file.ReturnXSize( )) / float(output_x_size);
    float y_bin_factor       = float(input_file.ReturnYSize( )) / float(output_y_size);
    float average_bin_factor = (x_bin_factor + y_bin_factor) / 2.0;

    float output_pixel_size = original_pixel_size * float(average_bin_factor);

    // change if we need to correct for the distortion..

    if ( correct_mag_distortion == true ) {
        output_pixel_size = ReturnMagDistortionCorrectedPixelSize(output_pixel_size, mag_distortion_major_scale, mag_distortion_minor_scale);
    }

    // Arrays to hold the shifts..

    float* x_shifts = new float[number_of_input_images];
    float* y_shifts = new float[number_of_input_images];

    // Arrays to hold the 1D dose filter, and 1D restoration filter..

    // float* dose_filter;
    float* dose_filter_sum_of_squares;

    int first_frame_to_preprocess;
    int last_frame_to_preprocess;
    int number_of_preprocess_blocks;
    int preprocess_block_counter;
    int total_processed;

    profile_timing.start("allocate electron dose");
    // Electron dose object for if dose filtering..

    ElectronDose* my_electron_dose;

    if ( should_dose_filter == true )
        my_electron_dose = new ElectronDose(acceleration_voltage, output_pixel_size);
    profile_timing.lap("allocate electron dose");
    // some quick checks..

    /*
	if (number_of_input_images <= 2)
	{
		SendError(wxString::Format("Error: Movie (%s) contains less than 3 frames.. Terminating.", input_filename));
		wxSleep(10);
		exit(-1);
	}
	*/

    // Read in dark/gain reference
    if ( ! movie_is_gain_corrected ) {
        gain_image.ReadSlice(&gain_file, 1);
    }
    if ( ! movie_is_dark_corrected ) {
        dark_image.ReadSlice(&dark_file, 1);
    }

    // Read in, gain-correct, FFT and resample all the images..

    unblur_timing.start("read frames");
    // big loop over number of threads, so that far large number of frames you might save some memory..

    // read in frames, non threaded..

    number_of_preprocess_blocks = int(ceilf(float(number_of_input_images) / float(max_threads)));

    first_frame_to_preprocess = 1;
    last_frame_to_preprocess  = max_threads;
    total_processed           = 0;

    for ( preprocess_block_counter = 0; preprocess_block_counter < number_of_preprocess_blocks; preprocess_block_counter++ ) {
        profile_timing.start("read in frames");
        for ( image_counter = first_frame_to_preprocess; image_counter <= last_frame_to_preprocess; image_counter++ ) {
            // Read from disk
            image_stack[image_counter - 1].ReadSlice(&input_file, image_counter);
        }
        profile_timing.lap("read in frames");

#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = first_frame_to_preprocess; image_counter <= last_frame_to_preprocess; image_counter++ ) {
            // Dark correction
            if ( ! movie_is_dark_corrected ) {
                profile_timing.start("dark correct");
                if ( ! image_stack[image_counter - 1].HasSameDimensionsAs(&dark_image) ) {
                    SendError(wxString::Format("Error: location %li of input file (%s) does not have same dimensions as the dark image (%s)", image_counter, input_filename, dark_filename));
                    wxSleep(10);
                    exit(-1);
                }
                //if (image_counter == 0) SendInfo(wxString::Format("Info: multiplying %s by gain %s\n",input_filename,gain_filename.ToStdString()));
                image_stack[image_counter - 1].SubtractImage(&dark_image);
                profile_timing.lap("dark correct");
            }

            // Gain correction
            if ( ! movie_is_gain_corrected ) {
                profile_timing.start("gain correct");
                if ( ! image_stack[image_counter - 1].HasSameDimensionsAs(&gain_image) ) {
                    SendError(wxString::Format("Error: location %li of input file (%s) does not have same dimensions as the gain image (%s)", image_counter, input_filename, gain_filename));
                    wxSleep(10);
                    exit(-1);
                }
                counting_image_stack[image_counter - 1].CopyFrom(&image_stack[image_counter - 1]);
                //if (image_counter == 0) SendInfo(wxString::Format("Info: multiplying %s by gain %s\n",input_filename,gain_filename.ToStdString()));
                image_stack[image_counter - 1].MultiplyPixelWise(gain_image);
                profile_timing.lap("gain correct");
            }

            profile_timing.start("replace outliers");
            image_stack[image_counter - 1].ReplaceOutliersWithMean(12);
            // counting_image_stack[image_counter - 1].ReplaceOutliersWithMean(12);

            profile_timing.lap("replace outliers");

            if ( correct_mag_distortion == true ) {
                profile_timing.start("correct mag distortion");
                image_stack[image_counter - 1].CorrectMagnificationDistortion(mag_distortion_angle, mag_distortion_major_scale, mag_distortion_minor_scale);
                profile_timing.lap("correct mag distortion");
            }

            // FT
            profile_timing.start("forward FFT");
            image_stack[image_counter - 1].ForwardFFT(true);
            image_stack[image_counter - 1].ZeroCentralPixel( );
            counting_image_stack[image_counter - 1].ForwardFFT(true);
            counting_image_stack[image_counter - 1].ZeroCentralPixel( );
            profile_timing.lap("forward FFT");

            // Resize the FT (binning)
            if ( output_binning_factor > 1.0001 ) {
                profile_timing.start("resize");
                raw_image_stack[image_counter - 1].Allocate(image_stack[image_counter - 1].logical_x_dimension, image_stack[image_counter - 1].logical_y_dimension, 1, false);
                raw_image_stack[image_counter - 1].CopyFrom(&image_stack[image_counter - 1]);
                image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);
                counting_image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);
                profile_timing.lap("resize");
            }

            // Init shifts
            x_shifts[image_counter - 1] = 0.0;
            y_shifts[image_counter - 1] = 0.0;
        } // end omp block

        first_frame_to_preprocess += max_threads;
        last_frame_to_preprocess += max_threads;

        if ( first_frame_to_preprocess > number_of_input_images )
            first_frame_to_preprocess = number_of_input_images;
        if ( last_frame_to_preprocess > number_of_input_images )
            last_frame_to_preprocess = number_of_input_images;
    }
    input_file.CloseFile( );

    unblur_timing.lap("read frames");
    // if we are binning - choose a binning factor..

    pre_binning_factor = int(myround(5. / output_pixel_size));
    if ( pre_binning_factor < 1 )
        pre_binning_factor = 1;

    //	wxPrintf("Prebinning factor = %i\n", pre_binning_factor);

    // if we are going to be binning, we need to allocate the unbinned array..

    if ( pre_binning_factor > 1 ) {
        unbinned_image_stack    = image_stack;
        image_stack             = new Image[number_of_input_images];
        pixel_size              = output_pixel_size * pre_binning_factor;
        unbinned_counting_stack = counting_image_stack;
    }
    else {
        pixel_size = output_pixel_size;
    }

    // convert shifts to pixels..

    min_shift_in_pixels             = minumum_shift_in_angstroms / pixel_size;
    max_shift_in_pixels             = maximum_shift_in_angstroms / pixel_size;
    termination_threshold_in_pixels = termination_threshold_in_angstoms / pixel_size;

    // calculate the bfactor

    unitless_bfactor = bfactor_in_angstoms / pow(pixel_size, 2);
    wxPrintf("unitless_bfactor: %f\n", unitless_bfactor);
    if ( min_shift_in_pixels <= 1.01 )
        min_shift_in_pixels = 1.01; // we always want to ignore the central peak initially.

    if ( pre_binning_factor > 1 ) {
        profile_timing.start("make prebinned stack");
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            image_stack[image_counter].Allocate(unbinned_image_stack[image_counter].logical_x_dimension / pre_binning_factor, unbinned_image_stack[image_counter].logical_y_dimension / pre_binning_factor, 1, false);
            unbinned_image_stack[image_counter].ClipInto(&image_stack[image_counter]);
            // counting_image_stack[image_counter].Allocate(unbinned_counting_stack[image_counter].logical_x_dimension / pre_binning_factor, unbinned_counting_stack[image_counter].logical_y_dimension / pre_binning_factor, 1, false);
            // unbinned_counting_stack[image_counter].ClipInto(&counting_image_stack[image_counter]);
            //image_stack[image_counter].QuickAndDirtyWriteSlice("binned.mrc", image_counter + 1);
        }
        profile_timing.lap("make prebinned stack");
        // for the binned images, we don't want to insist on a super low termination factor.

        if ( termination_threshold_in_pixels < 1 && pre_binning_factor > 1 )
            termination_threshold_in_pixels = 1;
    }
    //added for patch tracking correction
    // bool patch_track = true;
    int total_rounds;
    if ( patch_track )
        total_rounds = 2;
    else
        total_rounds = 1;
    for ( int round_index = 0; round_index < total_rounds; round_index++ ) {
        if ( round_index == 1 ) {
            // patches
            // int     patch_num_x = 6, patch_num_y = 4;
            int     patch_num             = patch_num_x * patch_num_y;
            int     output_stack_box_size = 512;
            Image** unbinned_patch_stack;
            Image** patch_stack = new Image*[patch_num];
            for ( int i = 0; i < patch_num; i++ ) {
                patch_stack[i] = new Image[number_of_input_images];
            }
            float **patch_shift_x, **patch_shift_y;
            Allocate2DFloatArray(patch_shift_x, patch_num, int(number_of_input_images));
            Allocate2DFloatArray(patch_shift_y, patch_num, int(number_of_input_images));
            //init shifts
            for ( int patch_counter = 0; patch_counter < patch_num; patch_counter++ ) {
                for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    patch_shift_x[patch_counter][image_counter] = 0.0;
                    patch_shift_y[patch_counter][image_counter] = 0.0;
                }
            }

            // The image_stack is binned.
            // if the does filter is active, the current image_stack is already does filtered
            wxPrintf("size of the image stack in patch tracking: %i,%i\n", image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension);
            // the unbinned_image_stack is also the same as image_stack, which is binned.
            // wxPrintf("size of the image stack in patch tracking: %i,%i\n", unbinned_image_stack[0].logical_x_dimension, unbinned_image_stack[0].logical_y_dimension);

            // patch_stack = patch_trimming_independent(image_stack, number_of_input_images, patch_num_x, patch_num_y, output_stack_box_size, outputpath, false);
            unblur_timing.start("Trimming Patches");
            // use the stack that's not dose filtered -================!!! need to be refined
            //             if(should_dose_filter == true){

            //                    raw_image_stack[image_counter - 1].CopyFrom(&image_stack[image_counter - 1]);
            //                 image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);

            //                 delete[] image_stack;
            //                 image_stack = unbinned_image_stack;
            //                 pixel_size  = output_pixel_size;

            //                 // Adjust the shifts, then phase shift the original images
            //                 profile_timing.start("apply shifts 1");
            // #pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
            //                 for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            //                     x_shifts[image_counter] *= pre_binning_factor;
            //                     y_shifts[image_counter] *= pre_binning_factor;

            //                     image_stack[image_counter].PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 0.0);
            //                 }
            //             }
            patch_trimming(image_stack, patch_stack, number_of_input_images, patch_num_x, patch_num_y, output_stack_box_size, outputpath, max_threads, false);
            if ( image_stack[0].is_in_real_space ) {
                wxPrintf("after trimming image stack is in real space\n");
            }
            unblur_timing.lap("Trimming Patches");
            // if ( pre_binning_factor > 1 ) {

            //     unbinned_patch_stack = patch_stack;
            //     patch_stack          = new Image*[patch_num];
            //     for ( int i = 0; i < patch_num; i++ ) {
            //         patch_stack[i] = new Image[number_of_input_images];
            //     }
            //     //check the allocated dimension, if it is correct
            //     for ( int patch_counter = 0; patch_counter < patch_num; patch_counter++ ) {
            //         for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            //             patch_stack[patch_counter][image_counter].Allocate(unbinned_patch_stack[patch_counter][image_counter].logical_x_dimension / pre_binning_factor, unbinned_patch_stack[patch_counter][image_counter].logical_y_dimension / pre_binning_factor, 1, false);
            //             unbinned_patch_stack[patch_counter][image_counter].ClipInto(&patch_stack[patch_counter][image_counter]);
            //         }
            //     }
            // }
            // start
            // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch/";
            // wxString      patch_shift = outputpath + "patch_shift.txt";
            std::ofstream xoFile;

            // xoFile.open(patch_shift.c_str( ));
            // /*
            unblur_timing.start("Patch Alignment");
            for ( int patch_counter = 0; patch_counter < patch_num; patch_counter++ ) {
                wxPrintf("aligning patch %d\n", patch_counter);
                if ( patch_stack[patch_counter][0].is_in_real_space ) {
                    for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                        patch_stack[patch_counter][image_counter].ForwardFFT(true);
                        // patch_stack[patch_counter][image_counter].ZeroCentralPixel( );
                    }
                }
                unblur_refine_alignment(patch_stack[patch_counter], number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, patch_shift_x[patch_counter], patch_shift_y[patch_counter], profile_timing_refinement_method, false);
                // for ( int i = 0; i < number_of_input_images; i++ ) {
                //     wxPrintf("patch %i, shifts, %f, %f\n", patch_counter, patch_shift_x[patch_counter][i], patch_shift_y[patch_counter][i]);
                // }

                unblur_refine_alignment(patch_stack[patch_counter], number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, patch_shift_x[patch_counter], patch_shift_y[patch_counter], profile_timing_refinement_method, false);
                // for ( int i = 0; i < number_of_input_images; i++ ) {
                //     wxPrintf("patch %i, shifts, %f, %f\n", patch_counter, patch_shift_x[patch_counter][i], patch_shift_y[patch_counter][i]);
                // }
                unblur_refine_alignment(patch_stack[patch_counter], number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, patch_shift_x[patch_counter], patch_shift_y[patch_counter], profile_timing_refinement_method, false);
                // unblur_refine_alignment(patch_stack[patch_counter], number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average * 3, myroundint(5.0f / exposure_per_frame), max_threads, patch_shift_x[patch_counter], patch_shift_y[patch_counter], profile_timing_refinement_method, true);

                //write out the patch shift info
                wxString patch_shift;
                patch_shift = wxString::Format(outputpath + "%04i_" + "shift.txt", patch_counter);
                xoFile.open(patch_shift.c_str( ));
                if ( xoFile.is_open( ) ) {
                    // wxPrintf("files are open\n");
                    // float myarray[10][5760];
                    for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                        xoFile << patch_shift_x[patch_counter][image_counter] << '\t';
                        xoFile << patch_shift_y[patch_counter][image_counter] << '\t';
                        xoFile << '\n';
                    }
                }
                xoFile.close( );
            }
            unblur_timing.lap("Patch Alignment");
            // */
            // for ( int i = 0; i < patch_num; ++i ) {
            //     delete[] patch_stack[i]; // each i-th pointer must be deleted first
            // }
            // delete[] patch_stack; // now delete pointer array
            unblur_timing.start("Distortion Fix");
            wxPrintf("modeling the distortion \n");
            //patch shift fitting-------------------------------------------
            if ( distortion_model == 2 || distortion_model == 1 ) {

                float patch_locations[patch_num][2];
                int   image_dim_x = image_stack[0].logical_x_dimension;
                int   image_dim_y = image_stack[0].logical_y_dimension;
                int   step_size_x = myroundint(float(image_dim_x) / float(patch_num_x) / 2);
                int   step_size_y = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
                for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
                    for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
                        patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x;
                        // patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
                        patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1] = patch_y_ind * step_size_y * 2 + step_size_y;

                        // wxPrintf("patch locations: %f, %f\n", patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0], patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1]);
                    }
                }
                // FixOutliers(patch_peaksx, patch_peaksy, patch_locations_x, patch_locations_y, patch_num_x, patch_num_y, step_size_x, step_size_y, number_of_input_images);

                input_vector                                 input;
                std::vector<std::pair<input_vector, double>> data_x, data_y;
                // for ( int image_index = 0; image_index < number_of_input_images; image_index++ ) {
                //     for ( int patch_index = 0; patch_index < patch_num; patch_index++ ) {
                //         float time;
                //         time     = image_index;
                //         input(0) = patch_locations[patch_index][0];
                //         input(1) = patch_locations[patch_index][1];
                //         input(2) = time;
                //         // double outputx = shiftsx[image_index][patch_index];
                //         // double outputy = shiftsy[image_index][patch_index];
                //         // the following is to set the first image has no shift
                //         double outputx = patch_shift_x[patch_index][image_index] - patch_shift_x[patch_index][0];
                //         double outputy = patch_shift_y[patch_index][image_index] - patch_shift_y[patch_index][0];
                //         data_x.push_back(std::make_pair(input, outputx));
                //         data_y.push_back(std::make_pair(input, outputy));
                //     }
                // }
                matrix<double>* patch_peaksx;
                matrix<double>* patch_peaksy;
                patch_peaksx = new matrix<double>[number_of_input_images];
                patch_peaksy = new matrix<double>[number_of_input_images];
                for ( int i = 0; i < number_of_input_images; i++ ) {
                    patch_peaksx[i].set_size(patch_num_y, patch_num_x);
                    patch_peaksy[i].set_size(patch_num_y, patch_num_x);
                }
                matrix<double> patch_locations_x;
                matrix<double> patch_locations_y;
                patch_locations_x.set_size(patch_num_x, 1);
                patch_locations_y.set_size(patch_num_y, 1);
                for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
                    patch_locations_x(patch_x_ind) = patch_x_ind * step_size_x * 2 + step_size_x;
                }
                for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
                    // patch_locations_y(patch_y_ind) = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
                    patch_locations_y(patch_y_ind) = patch_y_ind * step_size_y * 2 + step_size_y;
                }
                for ( int patch_index_y = 0; patch_index_y < patch_num_y; patch_index_y++ ) {
                    for ( int patch_index_x = 0; patch_index_x < patch_num_x; patch_index_x++ ) {
                        int patch_index = patch_index_y * patch_num_x + patch_index_x;
                        for ( int image_ind = 0; image_ind < number_of_input_images; image_ind++ ) {
                            patch_peaksx[image_ind](patch_index_y, patch_index_x) = patch_shift_x[patch_index][image_ind];
                            patch_peaksy[image_ind](patch_index_y, patch_index_x) = patch_shift_y[patch_index][image_ind];
                        }
                    }
                }
                std::vector<int> outlier_index;
                outlier_index = FixOutliers(patch_peaksx, patch_peaksy, patch_locations_x, patch_locations_y, patch_num_x, patch_num_y, step_size_x, step_size_y, number_of_input_images);
                delete[] patch_peaksx;
                delete[] patch_peaksy;
                for ( int image_index = 0; image_index < number_of_input_images; image_index++ ) {
                    int outlier_ind = 0;
                    for ( int patch_index = 0; patch_index < patch_num; patch_index++ ) {
                        if ( patch_index == outlier_index[outlier_ind] ) {
                            outlier_ind++;
                            continue;
                        }
                        float time;
                        time     = image_index;
                        input(0) = patch_locations[patch_index][0];
                        input(1) = patch_locations[patch_index][1];
                        input(2) = time;
                        // double outputx = shiftsx[image_index][patch_index];
                        // double outputy = shiftsy[image_index][patch_index];
                        // the following is to set the first image has no shift
                        double outputx = patch_shift_x[patch_index][image_index] - patch_shift_x[patch_index][0];
                        double outputy = patch_shift_y[patch_index][image_index] - patch_shift_y[patch_index][0];
                        data_x.push_back(std::make_pair(input, outputx));
                        data_y.push_back(std::make_pair(input, outputy));
                    }
                }
                if ( distortion_model == 2 ) {

                    param_vector_quadratic x_quad, y_quad;
                    x_quad = 1;
                    y_quad = 1;

                    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                                           residual_quadratic,
                                           residual_derivative_quadratic,
                                           data_x,
                                           x_quad);
                    wxPrintf("x fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x_quad(0), x_quad(1), x_quad(2), x_quad(3), x_quad(4), x_quad(5), x_quad(6), x_quad(7), x_quad(8), x_quad(9), x_quad(10), x_quad(11), x_quad(12), x_quad(13), x_quad(14), x_quad(15), x_quad(16), x_quad(17));

                    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                                           residual_quadratic,
                                           residual_derivative_quadratic,
                                           data_y,
                                           y_quad);
                    wxPrintf("y fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", y_quad(0), y_quad(1), y_quad(2), y_quad(3), y_quad(4), y_quad(5), y_quad(6), y_quad(7), y_quad(8), y_quad(9), y_quad(10), y_quad(11), y_quad(12), y_quad(13), y_quad(14), y_quad(15), y_quad(16), y_quad(17));

                    // //apply fitting-------------------------------------------
                    for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                        // image_stack[image_counter].BackwardFFT( );
                        raw_image_stack[image_counter].BackwardFFT( );
                    }
                    // apply_fitting_quadratic(image_stack, number_of_input_images, x, y);
                    apply_fitting_quadratic_sup(image_stack, raw_image_stack, output_binning_factor, number_of_input_images, x_quad, y_quad);
                }
                else if ( distortion_model == 1 ) {
                    param_vector_linear x_linear, y_linear;
                    x_linear = 1;
                    y_linear = 1;

                    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                                           residual_linear,
                                           residual_derivative_linear,
                                           data_x,
                                           x_linear);
                    wxPrintf("x fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x_linear(0), x_linear(1), x_linear(2), x_linear(3), x_linear(4), x_linear(5), x_linear(6), x_linear(7), x_linear(8));

                    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                                           residual_linear,
                                           residual_derivative_linear,
                                           data_y,
                                           y_linear);
                    wxPrintf("y fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", y_linear(0), y_linear(1), y_linear(2), y_linear(3), y_linear(4), y_linear(5), y_linear(6), y_linear(7), y_linear(8));

                    // //apply fitting-------------------------------------------
                    for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                        // image_stack[image_counter].BackwardFFT( );
                        raw_image_stack[image_counter].BackwardFFT( );
                    }
                    // apply_fitting_linear(image_stack, number_of_input_images, x, y);
                    apply_fitting_linear_sup(image_stack, raw_image_stack, output_binning_factor, number_of_input_images, x_linear, y_linear);
                }
                for ( int i = 0; i < patch_num; ++i ) {
                    delete[] patch_stack[i]; // each i-th pointer must be deleted first
                }
                delete[] patch_stack; // now delete pointer array
                patch_stack = NULL;
                Deallocate2DFloatArray(patch_shift_x, patch_num);
                Deallocate2DFloatArray(patch_shift_y, patch_num);
            }
            else if ( distortion_model == 3 ) {
                // prepare for spline  ---------------------------------------------------
                matrix<double> patch_locations_x;
                matrix<double> patch_locations_y;
                int            image_dim_x = image_stack[0].logical_x_dimension;
                int            image_dim_y = image_stack[0].logical_y_dimension;
                int            step_size_x = myroundint(float(image_dim_x) / float(patch_num_x) / 2);
                int            step_size_y = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
                patch_locations_x.set_size(patch_num_x, 1);
                patch_locations_y.set_size(patch_num_y, 1);
                for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
                    patch_locations_x(patch_x_ind) = patch_x_ind * step_size_x * 2 + step_size_x;
                }
                for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
                    // patch_locations_y(patch_y_ind) = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
                    // patch_locations_y(patch_y_ind) = patch_y_ind * step_size_y * 2 + step_size_y;
                    patch_locations_y(patch_num_y - patch_y_ind - 1) = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
                }

                matrix<double>* patch_peaksx;
                matrix<double>* patch_peaksy;
                patch_peaksx = new matrix<double>[number_of_input_images];
                patch_peaksy = new matrix<double>[number_of_input_images];
                for ( int i = 0; i < number_of_input_images; i++ ) {
                    patch_peaksx[i].set_size(patch_num_y, patch_num_x);
                    patch_peaksy[i].set_size(patch_num_y, patch_num_x);
                }
                for ( int patch_index_y = 0; patch_index_y < patch_num_y; patch_index_y++ ) {
                    for ( int patch_index_x = 0; patch_index_x < patch_num_x; patch_index_x++ ) {
                        int patch_index = patch_index_y * patch_num_x + patch_index_x;
                        for ( int image_ind = 0; image_ind < number_of_input_images; image_ind++ ) {
                            patch_peaksx[image_ind](patch_index_y, patch_index_x) = patch_shift_x[patch_index][image_ind];
                            patch_peaksy[image_ind](patch_index_y, patch_index_x) = patch_shift_y[patch_index][image_ind];
                        }
                    }
                }
                Deallocate2DFloatArray(patch_shift_x, patch_num);
                Deallocate2DFloatArray(patch_shift_y, patch_num);
                // set the first image as reference image
                for ( int i = 1; i < number_of_input_images; i++ ) {
                    patch_peaksy[i] = patch_peaksy[i] - patch_peaksy[0];
                    patch_peaksx[i] = patch_peaksx[i] - patch_peaksx[0];
                }
                patch_peaksy[0] = patch_peaksy[0] - patch_peaksy[0];
                patch_peaksx[0] = patch_peaksx[0] - patch_peaksx[0];

                //flip the patch regarding y so that the y is in ascending order
                // for ( int i = 0; i < number_of_input_images; i++ ) {
                //     patch_peaksy[i] = flipud(patch_peaksy[i]);
                //     patch_peaksx[i] = flipud(patch_peaksx[i]);
                // }
                // patch_locations_y = flipud(patch_locations_y);

                //spline modeling:
                std::vector<int> outlier_index;
                outlier_index = FixOutliers(patch_peaksx, patch_peaksy, patch_locations_x, patch_locations_y, patch_num_x, patch_num_y, step_size_x, step_size_y, number_of_input_images);
                // prepare for spline end ---------------------------------------------------

                // initialize the 3d splines  ---------------------------------------------------
                double total_dose  = exposure_per_frame * number_of_input_images;
                double sample_dose = 4; //8; //4; //10; //4;

                // double knot_on_z      = 2;
                // double knot_z_dis     = ceil(total_dose / (knot_on_z - 1));
                double         knot_z_dis = sample_dose;
                double         knot_on_z  = ceil(total_dose / sample_dose) + 1;
                double         knot_on_x  = 8; //8;
                double         knot_x_dis = ceil(image_dim_x / (knot_on_x - 1));
                double         knot_on_y  = 5; //5;
                double         knot_y_dis = ceil(image_dim_y / (knot_on_y - 1));
                column_vector  Join1d, Join1d_ccmap;
                column_vector  Control1d, Control1d_ccmap;
                matrix<double> z;
                z.set_size(number_of_input_images, 1);
                for ( int i = 0; i < number_of_input_images; i++ ) {
                    z(i) = (i + 1) * exposure_per_frame;
                }
                Spline3dx.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
                Spline3dy.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
                // Spline3dx.Update3DSplineInterpMapping(patch_locations_x, patch_locations_y, z);
                // Spline3dy.Update3DSplineInterpMapping(patch_locations_x, patch_locations_y, z);

                Spline3dx.Update3DSplineInterpMappingControl(patch_locations_x, patch_locations_y, z);
                Spline3dy.Update3DSplineInterpMappingControl(patch_locations_x, patch_locations_y, z);
                // Update3DSplineInterpMappingControl
                Spline3dx.UpdateDiscreteValues(patch_peaksx);
                Spline3dy.UpdateDiscreteValues(patch_peaksy);
                delete[] patch_peaksx;
                delete[] patch_peaksy;
                // initialize the 3d splines end ---------------------------------------------------

                // start the fitting ============================================
                // /*
                // initialize the parameters
                // Join1d = ones_matrix<double>(knot_on_x * knot_on_y * knot_on_z * 2, 1);
                Join1d = ones_matrix<double>(knot_on_x * knot_on_y * (knot_on_z + 1) * 2, 1);
                // Control1d = ones_matrix<double>((knot_on_x + 2) * (knot_on_y + 2) * (knot_on_z + 1) * 2, 1);
                Control1d = ones_matrix<double>(knot_on_x * knot_on_y * (knot_on_z)*2, 1);

                char          buffer[200];
                std::string   shift_file_pref = "_shift";
                std::string   lossfile;
                std::ofstream file;
                std::snprintf(buffer, sizeof(buffer), "%s%s%s", outputpath.ToStdString( ).c_str( ), shift_file_pref.c_str( ), "loss_record.txt");
                // shift_file = oss.str( );
                lossfile = buffer;
                cout << "file " << lossfile << endl;

                file.open(lossfile.c_str( ));
                std::streambuf* coutBuf0 = std::cout.rdbuf( );
                std::cout.rdbuf(file.rdbuf( ));

                // #1 initialize based on the alignment -----
                int maxsize = knot_on_x * knot_on_y * (knot_on_z + 1) * 2;
                wxPrintf("max memory size is set to: %d\n", maxsize * 2);
                auto   start1        = std::chrono::high_resolution_clock::now( );
                double initial_error = minfunc3dSplineObjectxyControlPoints(Control1d);
                double stop_cri      = initial_error / 100000;
                std::cout << " initial error, " << initial_error << endl;
                std::cout << " stop criteria, " << stop_cri << endl;
                find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 4), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                                                       objective_delta_stop_strategy(stop_cri).be_verbose( ),
                                                       minfunc3dSplineObjectxyControlPoints, Control1d, -1);
                // find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 4), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                //                                        objective_delta_stop_strategy(1e-4).be_verbose( ),
                //                                        minfunc3dSplineObjectxy, Join1d, -1);
                // before apply the stop criteria
                // find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 4), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                //                                        objective_delta_stop_strategy(1e-4).be_verbose( ),
                //                                        minfunc3dSplineObjectxyControlPoints, Control1d, -1);
                auto stop1     = std::chrono::high_resolution_clock::now( );
                auto duration1 = std::chrono::duration_cast<std::chrono::minutes>(stop1 - start1);
                std::cout << "Lap time " << duration1.count( ) << " minutes\n";
                // write_joins(outputpath.ToStdString( ), "Joins_R0", Join1d);
                write_joins(outputpath.ToStdString( ), "Control_R0", Control1d);
                write_shifts(patch_num_x, patch_num_y, number_of_input_images, outputpath.ToStdString( ), "_shiftx_R0", "_shifty_R0");
                // */
                // /*
                // #2  use the crosscorrelation map for loss function -----
                // Join1d = read_joins("Joins_R0", outputpath.ToStdString( ), knot_on_x * knot_on_y * knot_on_z * 2);
                // split_and_update(Join1d, Join1d.size( ));
                int     quater_patch_dim = 64 / 2;
                Image** counting_patch   = new Image*[patch_num];
                for ( int i = 0; i < patch_num; i++ ) {
                    counting_patch[i] = new Image[number_of_input_images];
                }
                patch_trimming(image_stack, patch_stack, number_of_input_images, patch_num_x, patch_num_y, output_stack_box_size / 2, outputpath, max_threads, false);
                patch_trimming(counting_image_stack, counting_patch, number_of_input_images, patch_num_x, patch_num_y, output_stack_box_size / 2, outputpath, max_threads, false);
                if ( patch_stack[0][0].is_in_real_space ) {
                    for ( int patch_counter = 0; patch_counter < patch_num; patch_counter++ ) {
                        for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                            patch_stack[patch_counter][image_counter].ForwardFFT(true);
                            counting_patch[patch_counter][image_counter].ForwardFFT(true);
                            // patch_stack[patch_counter][image_counter].ZeroCentralPixel( );
                        }
                    }
                }
                // for ( int image_ind = 0; image_ind < number_of_input_images; image_ind++ ) {
                //     for ( int i = 0; i < patch_num_y; i++ ) {
                //         for ( int j = 0; j < patch_num_x; j++ ) {
                //             int patch_ind = i * patch_num_x + j;
                //             patch_stack[patch_ind][image_ind].PhaseShift(Spline3dx.smooth_interp[image_ind](i, j), Spline3dy.smooth_interp[image_ind](i, j), 0.0);
                //             counting_patch[patch_ind][image_ind].PhaseShift(Spline3dx.smooth_interp[image_ind](i, j), Spline3dy.smooth_interp[image_ind](i, j), 0.0);

                //             // yoFile << shifted_map_y[i][j] << '\t';
                //         }
                //     }
                // }
                //-----AIC calculation --- start
                // Image** ref_stack = new Image*[patch_num];
                // for ( int i = 0; i < patch_num; i++ ) {
                //     ref_stack[i] = new Image[number_of_input_images];
                // }
                float           diff_square, diff_counting_square;
                NumericTextFile AICRecord(outputpath.ToStdString( ) + "AIC_records.txt", OPEN_TO_WRITE, 11);

                for ( int sample = sample_dose; sample < sample_dose + 1; sample++ ) {
                    for ( int knot_x = 4; knot_x < 9; knot_x++ ) {
                        for ( int knot_y = 4; knot_y < 7; knot_y++ ) {
                            // Spline3dx.Deallocate( );
                            // Spline3dy.Deallocate( );
                            knot_on_x = knot_x;
                            knot_on_y = knot_y;
                            wxPrintf("sample dose %d\n", sample);
                            wxPrintf("pixel size %f\n", pixel_size);
                            double total_dose = exposure_per_frame * number_of_input_images;
                            // double sample_dose = 6; //10; //4;
                            double sample_dose = double(sample);

                            // double knot_on_z      = 2;
                            // double knot_z_dis     = ceil(total_dose / (knot_on_z - 1));
                            double knot_z_dis = sample_dose;
                            double knot_on_z  = ceil(total_dose / sample_dose) + 1;
                            // double         knot_on_x  = 6; //8;
                            double knot_x_dis = ceil(image_dim_x / (knot_on_x - 1));
                            // double         knot_on_y  = 4; //5;
                            double         knot_y_dis = ceil(image_dim_y / (knot_on_y - 1));
                            column_vector  Contro_1d_search;
                            matrix<double> z;
                            z.set_size(number_of_input_images, 1);
                            for ( int i = 0; i < number_of_input_images; i++ ) {
                                z(i) = (i + 1) * exposure_per_frame;
                            }
                            Spline3dx.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
                            Spline3dy.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
                            // Spline3dx.Update3DSplineInterpMapping(patch_locations_x, patch_locations_y, z);
                            // Spline3dy.Update3DSplineInterpMapping(patch_locations_x, patch_locations_y, z);
                            Spline3dx.Update3DSplineInterpMappingControl(patch_locations_x, patch_locations_y, z);
                            Spline3dy.Update3DSplineInterpMappingControl(patch_locations_x, patch_locations_y, z);

                            int maxsize         = knot_on_x * knot_on_y * knot_on_z * 2;
                            Contro_1d_search    = zeros_matrix<double>(knot_on_x * knot_on_y * knot_on_z * 2, 1);
                            double search_error = minfunc3dSplineObjectxyControlPoints(Contro_1d_search);
                            wxPrintf("initial loss: %f \n", search_error);
                            double search_converge_Cri = search_error / 100000;
                            auto   start2              = std::chrono::high_resolution_clock::now( );
                            find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 2), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                                                                   objective_delta_stop_strategy(search_converge_Cri).be_verbose( ),
                                                                   minfunc3dSplineObjectxyControlPoints, Contro_1d_search, -100, 1e-2);
                            auto stop2     = std::chrono::high_resolution_clock::now( );
                            auto duration2 = std::chrono::duration_cast<std::chrono::minutes>(stop2 - start2);
                            std::cout << "Lap time " << duration2.count( ) << " minutes\n";
                            double likelihood;
                            double AIC;
                            double k;
                            double sigma_square;
                            // sigma_square = total_dose / double(number_of_input_images) * pixel_size * pixel_size;
                            // sigma_square = 0.009;
                            sigma_square = exposure_per_frame * pixel_size * pixel_size;

                            wxPrintf("pixel size %f\n", pixel_size);
                            k = knot_on_x * knot_on_y * knot_on_z * 2;
                            // likelihood = -minfunc3dSplineCCLossObject(Join1d_ccmap) * (-1) / sigma_square;
                            for ( int image_ind = 0; image_ind < number_of_input_images; image_ind++ ) {
                                for ( int i = 0; i < patch_num_y; i++ ) {
                                    for ( int j = 0; j < patch_num_x; j++ ) {
                                        int patch_ind = i * patch_num_x + j;
                                        patch_stack[patch_ind][image_ind].PhaseShift(Spline3dx.smooth_interp[image_ind](i, j), Spline3dy.smooth_interp[image_ind](i, j), 0.0);
                                        counting_patch[patch_ind][image_ind].PhaseShift(Spline3dx.smooth_interp[image_ind](i, j), Spline3dy.smooth_interp[image_ind](i, j), 0.0);

                                        // yoFile << shifted_map_y[i][j] << '\t';
                                    }
                                }
                            }
                            diff_square          = CalculateDiffSquare(patch_stack, unitless_bfactor, patch_num, number_of_input_images, true, outputpath.ToStdString( ), "Ref_Stack");
                            diff_counting_square = CalculateDiffSquare(counting_patch, unitless_bfactor, patch_num, number_of_input_images, true, outputpath.ToStdString( ), "Ref_Stack_counting");
                            likelihood           = -diff_square / 2 / sigma_square;
                            wxPrintf("diff square and couting diff square %f %f\n", diff_square, diff_counting_square);
                            double likelihood_counting = -diff_counting_square / 2 / sigma_square;
                            // likelihood  = 0;

                            AIC                 = k - likelihood;
                            double AIC_counting = k - likelihood_counting;
                            wxPrintf("current sigmasquare k, diffsquare, like, AIC are %f, %f, %f, %f \n ", sigma_square, k, diff_square, likelihood, AIC);
                            wxPrintf("couting mode k,diffsquare, like, AIC are %f, %f, %f, %f \n ", sigma_square, k, diff_counting_square, likelihood_counting, AIC_counting);
                            double temp_array[11];

                            temp_array[0]  = knot_on_x;
                            temp_array[1]  = knot_on_y;
                            temp_array[2]  = sample_dose;
                            temp_array[3]  = sigma_square;
                            temp_array[4]  = k;
                            temp_array[5]  = diff_square;
                            temp_array[6]  = likelihood;
                            temp_array[7]  = AIC;
                            temp_array[8]  = diff_counting_square;
                            temp_array[9]  = likelihood_counting;
                            temp_array[10] = AIC_counting;
                            AICRecord.WriteLine(temp_array);
                            for ( int image_ind = 0; image_ind < number_of_input_images; image_ind++ ) {
                                for ( int i = 0; i < patch_num_y; i++ ) {
                                    for ( int j = 0; j < patch_num_x; j++ ) {
                                        int patch_ind = i * patch_num_x + j;
                                        patch_stack[patch_ind][image_ind].PhaseShift(-Spline3dx.smooth_interp[image_ind](i, j), -Spline3dy.smooth_interp[image_ind](i, j), 0.0);
                                        counting_patch[patch_ind][image_ind].PhaseShift(-Spline3dx.smooth_interp[image_ind](i, j), -Spline3dy.smooth_interp[image_ind](i, j), 0.0);

                                        // yoFile << shifted_map_y[i][j] << '\t';
                                    }
                                }
                            }
                        }
                    }
                }
                AICRecord.Close( );
                std::cout.rdbuf(coutBuf0);
                file.close( );
                for ( int i = 0; i < patch_num; ++i ) {
                    delete[] counting_patch[i]; // each i-th pointer must be deleted first
                }
                delete[] counting_patch; // now delete pointer array
                delete[] counting_image_stack;

                //-----AIC calculation --- end

                ccmap_stack.InitializeSplineStack(quater_patch_dim, quater_patch_dim, patch_num * number_of_input_images, 1, 1);
                Generate_CoeffSpline(ccmap_stack, patch_stack, unitless_bfactor, patch_num, number_of_input_images, true, outputpath.ToStdString( ), "CCMapBfactor_R1");

                Join1d_ccmap = zeros_matrix<double>(knot_on_x * knot_on_y * (knot_on_z + 1) * 2, 1);
                // Control1d_ccmap = zeros_matrix<double>((knot_on_x + 2) * (knot_on_y + 2) * (knot_on_z + 1) * 2, 1);
                Control1d_ccmap = zeros_matrix<double>(knot_on_x * knot_on_y * (knot_on_z)*2, 1);

                // double ccmap_error = minfunc3dSplineCCLossObject(Join1d_ccmap);
                double ccmap_error = minfunc3dSplineCCLossObjectControlPoints(Control1d_ccmap);
                wxPrintf("initial loss: %f \n", ccmap_error);
                double stop_cri_map = abs(ccmap_error / 100000);
                std::cout << " initial error, " << ccmap_error << endl;
                std::cout << " stop criteria, " << stop_cri_map << endl;
                auto start2 = std::chrono::high_resolution_clock::now( );
                find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 2), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                                                       objective_delta_stop_strategy(stop_cri_map).be_verbose( ),
                                                       minfunc3dSplineCCLossObjectControlPoints, Control1d_ccmap, -100, 1e-1);
                // find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 2), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                //                                        gradient_norm_stop_strategy(stop_cri_map, 10).be_verbose( ),
                //                                        minfunc3dSplineCCLossObjectControlPoints, Control1d_ccmap, -100, 1e-1);
                // find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 2), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                //                                        objective_delta_stop_strategy(1e-6).be_verbose( ),
                //                                        minfunc3dSplineCCLossObject, Join1d_ccmap, -100, 1e-1);
                // following is before using the gradient stop strategy
                // find_min_using_approximate_derivatives(lbfgs_search_strategy(maxsize * 2), // when it's 10, the result is not correct, when it's 1000, result is close, 10000 give the best. Remains to figure out why.
                //                                        objective_delta_stop_strategy(1e-6).be_verbose( ),
                //                                        minfunc3dSplineCCLossObjectControlPoints, Control1d_ccmap, -100, 1e-1);
                auto stop2     = std::chrono::high_resolution_clock::now( );
                auto duration2 = std::chrono::duration_cast<std::chrono::minutes>(stop2 - start2);
                std::cout << "Lap time " << duration2.count( ) << " minutes\n";
                std::cout.rdbuf(coutBuf0);
                file.close( );
                // write_joins(outputpath.ToStdString( ), "Joins_R1", Join1d_ccmap);
                write_joins(outputpath.ToStdString( ), "Control_R1", Control1d_ccmap);
                write_shifts(patch_num_x, patch_num_y, number_of_input_images, outputpath.ToStdString( ), "_shiftx_R1", "_shifty_R1");
                ccmap_stack.FreeSplineStack( );
                // */
                // end the fitting ============================================

                // Control1d       = read_joins("Control_R0", outputpath.ToStdString( ), knot_on_x * knot_on_y * knot_on_z * 2);
                // Control1d_ccmap = read_joins("Control_R1", outputpath.ToStdString( ), knot_on_x * knot_on_y * knot_on_z * 2);
                // Join1d       = read_joins("Joins_R0", outputpath.ToStdString( ), knot_on_x * knot_on_y * knot_on_z * 2);
                // Join1d_ccmap = read_joins("Joins_R1", outputpath.ToStdStri
                for ( int i = 0; i < patch_num; ++i ) {
                    delete[] patch_stack[i]; // each i-th pointer must be deleted first
                }
                delete[] patch_stack; // now delete pointer array
                patch_stack = NULL;
                for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    // image_stack[image_counter].BackwardFFT( );
                    raw_image_stack[image_counter].BackwardFFT( );
                }
                wxPrintf("correcting distortion\n");
                delete[] image_stack;
                image_stack = nullptr;
                // apply_fitting_spline_sup(image_stack, raw_image_stack, output_binning_factor, number_of_input_images, Join1d, Join1d_ccmap, max_threads);
                // apply_fitting_spline_sup(output_x_size, output_y_size, raw_image_stack, output_binning_factor, number_of_input_images, Join1d, Join1d_ccmap, max_threads);
                apply_fitting_spline_sup_control(output_x_size, output_y_size, raw_image_stack, output_binning_factor, number_of_input_images, Control1d, Control1d_ccmap, max_threads);
                image_stack = raw_image_stack;
                Spline3dx.Deallocate( );
                Spline3dy.Deallocate( );
            }
            unblur_timing.lap("Distortion Fix");
            // wxPrintf("writing the fixed frames \n");
            // Image sum_image_fixed;
            // sum_image_fixed.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, true);
            // sum_image_fixed.SetToConstant(0.0);
            // for ( int image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
            //     sum_image_fixed.AddImage(&image_stack[image_counter]);

            //     if ( saved_aligned_frames == true ) {
            //         image_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "distortion_corrected_frames.mrc", image_counter + 1);
            //     }
            // }
            // // sum_image_fixed.BackwardFFT( );
            // sum_image_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "distortion_corrected_image.mrc", 1);

            profile_timing.start("patch-based correction cleanup");
            // for ( int i = 0; i < patch_num; ++i ) {
            //     delete[] patch_stack[i]; // each i-th pointer must be deleted first
            // }
            // delete[] patch_stack; // now delete pointer array
            // patch_stack = NULL;
            // delete[] raw_image_stack;
            // Deallocate2DFloatArray(patch_shift_x, patch_num);
            // Deallocate2DFloatArray(patch_shift_y, patch_num);

            profile_timing.lap("patch-based correction cleanup");
            //patches end
        }
        if ( round_index == 0 ) {
            // do the initial refinement (only 1 round - with the min shift)
            unblur_timing.start("whole frame initial refine");
            profile_timing.start("initial refine");
            //SendInfo(wxString::Format("Doing first alignment on %s\n",input_filename));
            unblur_refine_alignment(image_stack, number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, min_shift_in_pixels, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, profile_timing_refinement_method, false);
            unblur_timing.lap("whole frame initial refine");
            profile_timing.lap("initial refine");

            // now do the actual refinement..
            unblur_timing.start("whole frame main refine");
            profile_timing.start("main refine");
            //SendInfo(wxString::Format("Doing main alignment on %s\n",input_filename));
            unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, profile_timing_refinement_method, false);
            unblur_timing.lap("whole frame main refine");
            profile_timing.lap("main refine");

            // if we have been using pre-binning, we need to do a refinment on the unbinned data..
            unblur_timing.start("whole frame final refine");
            if ( pre_binning_factor > 1 ) {
                // we don't need the binned images anymore..

                delete[] image_stack;
                image_stack = unbinned_image_stack;
                pixel_size  = output_pixel_size;

                // Adjust the shifts, then phase shift the original images
                profile_timing.start("apply shifts 1");
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
                for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    x_shifts[image_counter] *= pre_binning_factor;
                    y_shifts[image_counter] *= pre_binning_factor;

                    image_stack[image_counter].PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 0.0);
                    //         #pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
                    // for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    raw_image_stack[image_counter].PhaseShift(x_bin_factor * x_shifts[image_counter], y_bin_factor * y_shifts[image_counter], 0.0);
                    // }
                }
                profile_timing.lap("apply shifts 1");

                // convert parameters to pixels with new pixel size..

                min_shift_in_pixels             = minumum_shift_in_angstroms / output_pixel_size;
                max_shift_in_pixels             = maximum_shift_in_angstroms / output_pixel_size;
                termination_threshold_in_pixels = termination_threshold_in_angstoms / output_pixel_size;

                // recalculate the bfactor

                unitless_bfactor = bfactor_in_angstoms / pow(output_pixel_size, 2);

                // do the refinement..
                //SendInfo(wxString::Format("Doing final unbinned alignment on %s\n",input_filename));
                profile_timing.start("final refine");
                unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, output_pixel_size, number_of_frames_for_running_average, myroundint(5.0f / exposure_per_frame), max_threads, x_shifts, y_shifts, profile_timing_refinement_method, false);
                profile_timing.lap("final refine");
                // if allocated delete the binned stack, and swap the unbinned to image_stack - so that no matter what is happening we can just use image_stack
            }
            unblur_timing.lap("whole frame final refine");

            // we should be finished with alignment, now we just need to make the final sum..

            // fill the result..

            profile_timing.start("fill result");
            float* result_array = new float[number_of_input_images * 2];

            if ( is_running_locally == true ) {
                NumericTextFile shifts_file(output_shift_text_file, OPEN_TO_WRITE, 2);
                shifts_file.WriteCommentLine("X/Y Shifts for file %s\n", input_filename.c_str( ));

                for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    temp_float[0] = x_shifts[image_counter] * output_pixel_size;
                    temp_float[1] = y_shifts[image_counter] * output_pixel_size;
                    shifts_file.WriteLine(temp_float);
#ifdef PRINT_VERBOSE
                    wxPrintf("image #%li = %f, %f\n", image_counter, result_array[image_counter], result_array[image_counter + number_of_input_images]);
#endif
                }
            }
            else {
                for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                    result_array[image_counter]                          = x_shifts[image_counter] * output_pixel_size;
                    result_array[image_counter + number_of_input_images] = y_shifts[image_counter] * output_pixel_size;
                }
            }
            wxPrintf("mark 1\n");
            my_result.SetResult(number_of_input_images * 2, result_array);
            wxPrintf("mark 2\n");
            profile_timing.lap("fill result");
            wxPrintf("mark 3\n");
            delete[] result_array;
            delete[] x_shifts;
            delete[] y_shifts;
        }
        // write the final corrected image and frames
        // if ( round_index == total_rounds - 1 ) {
        if ( image_stack[0].is_in_real_space ) {
            for ( image_counter = 0; image_counter < number_of_input_images; image_counter++ ) {
                image_stack[image_counter].ForwardFFT(true);
            }
        }

        sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
        sum_image.SetToConstant(0.0);
        if ( should_dose_filter == true ) {
            if ( write_out_amplitude_spectrum == true ) {
                profile_timing.start("amplitude spectrum");
                sum_image_no_dose_filter.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
                sum_image_no_dose_filter.SetToConstant(0.0);

                for ( image_counter = first_frame - 1; image_counter < last_frame; image_counter++ ) {
                    sum_image_no_dose_filter.AddImage(&image_stack[image_counter]);
                }
                profile_timing.lap("amplitude spectrum");
            }
            profile_timing.start("setup dose filter");
            dose_filter_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];
            ZeroFloatArray(dose_filter_sum_of_squares, image_stack[0].real_memory_allocated / 2);
            profile_timing.lap("setup dose filter");
            dosefilter(image_stack, first_frame, last_frame, dose_filter_sum_of_squares, my_electron_dose, profile_timing, exposure_per_frame, pre_exposure_amount, max_threads);

            profile_timing.start("final sum");
            for ( image_counter = first_frame - 1; image_counter < last_frame; image_counter++ ) {
                sum_image.AddImage(&image_stack[image_counter]);

                if ( saved_aligned_frames == true ) {
                    // wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( )
                    // outputpath.ToStdString( ) + "distortion_corrected_frames.mrc";
                    image_stack[image_counter].QuickAndDirtyWriteSlice(wxString::Format(outputpath + "Unblur_Frames_Round_%04i.mrc", round_index).ToStdString( ), image_counter + 1);
                }
            }
            profile_timing.lap("final sum");
            // }
            // else // just add them
            // {
            profile_timing.start("final sum");
            for ( image_counter = first_frame - 1; image_counter < last_frame; image_counter++ ) {
                sum_image.AddImage(&image_stack[image_counter]);

                if ( saved_aligned_frames == true ) {
                    // image_stack[image_counter].QuickAndDirtyWriteSlice(aligned_frames_filename, image_counter + 1);
                    image_stack[image_counter].QuickAndDirtyWriteSlice(wxString::Format(outputpath + "Unblur_Frames_Round_%04i.mrc", round_index).ToStdString( ), image_counter + 1);
                }
            }
            profile_timing.lap("final sum");
            // }

            // if we are restoring the power - do it here..

            if ( should_dose_filter == true && should_restore_power == true ) {
                profile_timing.start("restore power");
                for ( pixel_counter = 0; pixel_counter < sum_image.real_memory_allocated / 2; pixel_counter++ ) {
                    if ( dose_filter_sum_of_squares[pixel_counter] != 0 ) {
                        sum_image.complex_values[pixel_counter] /= sqrtf(dose_filter_sum_of_squares[pixel_counter]);
                    }
                }
                profile_timing.lap("restore power");
            }

            // do we need to write out the amplitude spectra

            if ( write_out_amplitude_spectrum == true ) {
                profile_timing.start("write out amplitude spectrum");
                Image current_power_spectrum;
                current_power_spectrum.Allocate(sum_image.logical_x_dimension, sum_image.logical_y_dimension, true);

                //		if (should_dose_filter == true) sum_image_no_dose_filter.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);
                //	else sum_image.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);

                sum_image.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);

                // Set origin of amplitude spectrum to 0.0
                current_power_spectrum.real_values[current_power_spectrum.ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum.physical_address_of_box_center_x, current_power_spectrum.physical_address_of_box_center_y, current_power_spectrum.physical_address_of_box_center_z)] = 0.0;

                // Forward Transform
                current_power_spectrum.ForwardFFT( );

                // make it square

                int micrograph_square_dimension = std::max(sum_image.logical_x_dimension, sum_image.logical_y_dimension);
                if ( IsOdd((micrograph_square_dimension)) )
                    micrograph_square_dimension++;

                if ( sum_image.logical_x_dimension != micrograph_square_dimension || sum_image.logical_y_dimension != micrograph_square_dimension ) {
                    Image current_input_image_square;
                    current_input_image_square.Allocate(micrograph_square_dimension, micrograph_square_dimension, false);
                    current_power_spectrum.ClipInto(&current_input_image_square, 0);
                    current_power_spectrum.Consume(&current_input_image_square);
                }

                // how big will the amplitude spectra have to be in total to have the central 512x512 be a 2.8 angstrom Nyquist?

                // this is the (in the amplitudes real space) scale factor to make the nyquist 2.8 (inverse as real space)

                float pixel_size_scale_factor;
                if ( output_pixel_size < 1.4 )
                    pixel_size_scale_factor = 1.4 / output_pixel_size;
                else
                    pixel_size_scale_factor = 1.0;

                // this is the scale factor to make the box 512
                float box_size_scale_factor = 512.0 / float(micrograph_square_dimension);

                // overall scale factor

                float overall_scale_factor = pixel_size_scale_factor * box_size_scale_factor;

                {
                    Image scaled_spectrum;
                    scaled_spectrum.Allocate(myroundint(micrograph_square_dimension * overall_scale_factor), myroundint(micrograph_square_dimension * overall_scale_factor), false);
                    current_power_spectrum.ClipInto(&scaled_spectrum, 0);
                    scaled_spectrum.BackwardFFT( );
                    current_power_spectrum.Allocate(512, 512, 1, true);
                    scaled_spectrum.ClipInto(&current_power_spectrum, scaled_spectrum.ReturnAverageOfRealValuesOnEdges( ));
                }

                // now we need to filter it

                float average;
                float sigma;

                current_power_spectrum.ComputeAverageAndSigmaOfValuesInSpectrum(float(current_power_spectrum.logical_x_dimension) * 0.5, float(current_power_spectrum.logical_x_dimension), average, sigma, 12);
                current_power_spectrum.DivideByConstant(sigma);
                current_power_spectrum.SetMaximumValueOnCentralCross(average / sigma + 10.0);
                current_power_spectrum.ForwardFFT( );
                current_power_spectrum.CosineMask(0, 0.05, true);
                current_power_spectrum.BackwardFFT( );
                current_power_spectrum.SetMinimumAndMaximumValues(average - 1.0, average + 3.0);
                //current_power_spectrum.CosineRingMask(0.05,0.45, 0.05);
                //average_spectrum->QuickAndDirtyWriteSlice("dbg_average_spectrum_before_conv.mrc",1);
                current_power_spectrum.QuickAndDirtyWriteSlice(amplitude_spectrum_filename, 1, true);
                profile_timing.lap("write out amplitude spectrum");
            }

            //  Shall we write out a scaled image?

            if ( write_out_small_sum_image == true ) {
                profile_timing.start("write out small sum image");
                // work out a good size..
                int   largest_dimension = std::max(sum_image.logical_x_dimension, sum_image.logical_y_dimension);
                float scale_factor      = float(SCALED_IMAGE_SIZE) / float(largest_dimension);

                if ( scale_factor < 1.0 ) {
                    Image buffer_image;
                    buffer_image.Allocate(myroundint(sum_image.logical_x_dimension * scale_factor), myroundint(sum_image.logical_y_dimension * scale_factor), 1, false);
                    sum_image.ClipInto(&buffer_image);
                    buffer_image.QuickAndDirtyWriteSlice(small_sum_image_filename, 1, true);
                }
                profile_timing.lap("write out small sum image");
            }

            // now we just need to write out the final sum..
            profile_timing.start("write out sum image");
            // MRCFile output_file(output_filename, true);
            MRCFile output_file(wxString::Format(outputpath + "Unblur_Image_Round_%04i.mrc", round_index).ToStdString( ), true);
            sum_image.BackwardFFT( );
            sum_image.WriteSlice(&output_file, 1); // I made this change as the file is only used once, and this way it is not created until it is actually written, which is cleaner for cancelled / crashed jobs
            output_file.SetPixelSize(output_pixel_size);
            EmpiricalDistribution density_distribution;
            sum_image.UpdateDistributionOfRealValues(&density_distribution);
            output_file.SetDensityStatistics(density_distribution.GetMinimum( ), density_distribution.GetMaximum( ), density_distribution.GetSampleMean( ), sqrtf(density_distribution.GetSampleVariance( )));
            output_file.CloseFile( );
            profile_timing.lap("write out sum image");

            // profile_timing.start("whole frame alignment cleanup");

            // delete[] image_stack;
            wxPrintf("mark 4\n");
            // if ( should_dose_filter == true ) {
            //     delete my_electron_dose;
            //     delete[] dose_filter_sum_of_squares;
            // }
            // profile_timing.lap("whole frame alignment cleanup");
            // unblur_timing.print_times( );
            // profile_timing.print_times( );
            // profile_timing_refinement_method.print_times( );
            // }
        }
    }

    if ( should_dose_filter == true ) {
        delete my_electron_dose;
        delete[] dose_filter_sum_of_squares;
    }
    wxPrintf("mark 5\n");
    unblur_timing.print_times( );
    profile_timing.print_times( );
    profile_timing_refinement_method.print_times( );
    wxPrintf("mark 6\n");
    delete[] image_stack;
    return true;
}

void unblur_refine_alignment(Image* input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, StopWatch& profile_timing_refinement_method, bool reverse_shift) {

    profile_timing_refinement_method.mark_entry_or_exit_point( );

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

    float* current_x_shifts = new float[number_of_images];
    float* current_y_shifts = new float[number_of_images];

    float middle_image_x_shift;
    float middle_image_y_shift;

    float max_shift;
    float total_shift;

    if ( IsOdd(savitzy_golay_window_size) == false )
        savitzy_golay_window_size++;
    if ( savitzy_golay_window_size < 5 )
        savitzy_golay_window_size = 5;

    Image  sum_of_images;
    Image  sum_of_images_minus_current;
    Image* running_average_stack;

    Image* stack_for_alignment; // pointer that can be switched between running average stack and image stack if necessary
    Peak   my_peak;

    Curve x_shifts_curve;
    Curve y_shifts_curve;

    sum_of_images.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
    sum_of_images.SetToConstant(0.0);

    profile_timing_refinement_method.start("allocate running average");
    if ( number_of_frames_for_running_average > 1 ) {
        running_average_stack = new Image[number_of_images];

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            running_average_stack[image_counter].Allocate(input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, false);
        }

        stack_for_alignment = running_average_stack;
    }
    else
        stack_for_alignment = input_stack;
    profile_timing_refinement_method.lap("allocate running average");

    // prepare the initial sum
    profile_timing_refinement_method.start("prepare initial sum");
    for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        sum_of_images.AddImage(&input_stack[image_counter]);
        current_x_shifts[image_counter] = 0;
        current_y_shifts[image_counter] = 0;
    }
    profile_timing_refinement_method.lap("prepare initial sum");
    // perform the main alignment loop until we reach a max shift less than wanted, or max iterations

    for ( iteration_counter = 1; iteration_counter <= max_iterations; iteration_counter++ ) {
        //	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
        max_shift = -FLT_MAX;

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
                    running_average_stack[image_counter].AddImage(&input_stack[running_average_counter]);
                }
            }
        }

#pragma omp parallel default(shared) num_threads(max_threads) private(image_counter, sum_of_images_minus_current, my_peak)
        { // for omp

            sum_of_images_minus_current.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);

#pragma omp for
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                // prepare the sum reference by subtracting out the current image, applying a bfactor and masking central cross
                profile_timing_refinement_method.start("prepare sum");
                sum_of_images_minus_current.CopyFrom(&sum_of_images);
                sum_of_images_minus_current.SubtractImage(&stack_for_alignment[image_counter]);
                sum_of_images_minus_current.ApplyBFactor(unitless_bfactor);

                if ( mask_central_cross == true ) {
                    sum_of_images_minus_current.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
                }
                profile_timing_refinement_method.lap("prepare sum");
                // compute the cross correlation function and find the peak
                profile_timing_refinement_method.start("compute cross correlation");
                sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&stack_for_alignment[image_counter]);
                profile_timing_refinement_method.lap("compute cross correlation");
                profile_timing_refinement_method.start("find peak");
                my_peak = sum_of_images_minus_current.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);
                profile_timing_refinement_method.lap("find peak");
                // update the shifts..

                current_x_shifts[image_counter] = my_peak.x;
                current_y_shifts[image_counter] = my_peak.y;
            }
            profile_timing_refinement_method.start("deallocate sum minus");
            sum_of_images_minus_current.Deallocate( );
            profile_timing_refinement_method.lap("deallocate sum minus");
        } // end omp
        // smooth the shifts
        profile_timing_refinement_method.start("smooth shifts");
        x_shifts_curve.ClearData( );
        y_shifts_curve.ClearData( );

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            x_shifts_curve.AddPoint(image_counter, x_shifts[image_counter] + current_x_shifts[image_counter]);
            y_shifts_curve.AddPoint(image_counter, y_shifts[image_counter] + current_y_shifts[image_counter]);

#ifdef PRINT_VERBOSE
            wxPrintf("Before = %li : %f, %f\n", image_counter, x_shifts[image_counter] + current_x_shifts[image_counter], y_shifts[image_counter] + current_y_shifts[image_counter]);
#endif
        }

        if ( inner_radius_for_peak_search != 0 ) // in this case, weird things can happen (+1/-1 flips), we want to really smooth it. use a polynomial.  This should only affect the first round..
        {
            if ( x_shifts_curve.number_of_points > 2 ) {
                x_shifts_curve.FitPolynomialToData(4);
                y_shifts_curve.FitPolynomialToData(4);

                // copy back
                for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                    current_x_shifts[image_counter] = x_shifts_curve.polynomial_fit[image_counter] - x_shifts[image_counter];
                    current_y_shifts[image_counter] = y_shifts_curve.polynomial_fit[image_counter] - y_shifts[image_counter];

#ifdef PRINT_VERBOSE
                    wxPrintf("After poly = %li : %f, %f\n", image_counter, x_shifts_curve.polynomial_fit[image_counter], y_shifts_curve.polynomial_fit[image_counter]);
#endif
                }
            }
        }
        else {
            if ( savitzy_golay_window_size < x_shifts_curve.number_of_points ) // when the input movie is dodgy (very few frames), the fitting won't work
            {
                x_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);
                y_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);

                // copy them back..

                for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                    current_x_shifts[image_counter] = x_shifts_curve.savitzky_golay_fit[image_counter] - x_shifts[image_counter];
                    current_y_shifts[image_counter] = y_shifts_curve.savitzky_golay_fit[image_counter] - y_shifts[image_counter];

#ifdef PRINT_VERBOSE
                    wxPrintf("After SG = %li : %f, %f\n", image_counter, x_shifts_curve.savitzky_golay_fit[image_counter], y_shifts_curve.savitzky_golay_fit[image_counter]);
#endif
                }
            }
        }
        profile_timing_refinement_method.lap("smooth shifts");

        // subtract shift of the middle image from all images to keep things centred around it

        middle_image_x_shift = current_x_shifts[number_of_middle_image];
        middle_image_y_shift = current_y_shifts[number_of_middle_image];

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            current_x_shifts[image_counter] -= middle_image_x_shift;
            current_y_shifts[image_counter] -= middle_image_y_shift;

            total_shift = sqrt(pow(current_x_shifts[image_counter], 2) + pow(current_y_shifts[image_counter], 2));
            if ( total_shift > max_shift )
                max_shift = total_shift;
        }

        // actually shift the images, also add the subtracted shifts to the overall shifts

#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            profile_timing_refinement_method.start("shift image");
            input_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);
            x_shifts[image_counter] += current_x_shifts[image_counter];
            y_shifts[image_counter] += current_y_shifts[image_counter];
            profile_timing_refinement_method.lap("shift image");
        }

        // check to see if the convergence criteria have been reached and return if so

        if ( iteration_counter >= max_iterations || max_shift <= max_shift_convergence_threshold ) {
            profile_timing_refinement_method.start("cleanup");
            wxPrintf("returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
            delete[] current_x_shifts;
            delete[] current_y_shifts;

            if ( number_of_frames_for_running_average > 1 ) {
                delete[] running_average_stack;
            }
            profile_timing_refinement_method.lap("cleanup");
            profile_timing_refinement_method.mark_entry_or_exit_point( );
            return;
        }
        else {
            wxPrintf("Not. returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
        }

        // going to be doing another round so we need to make the new sum..
        profile_timing_refinement_method.start("remake sum");
        sum_of_images.SetToConstant(0.0);

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            sum_of_images.AddImage(&input_stack[image_counter]);
        }
        profile_timing_refinement_method.lap("remake sum");
    }

    if ( reverse_shift ) {
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            input_stack[image_counter].PhaseShift(-x_shifts[image_counter], -y_shifts[image_counter], 0.0);
        }
    }
    // wxPrintf("shift the images\n");
    profile_timing_refinement_method.mark_entry_or_exit_point( );
}

// void patch_trimming(Image* input_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, bool use_coord_file = false, wxString coordinates_filename = "");
// Image** patch_trimming(Image* input_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, bool use_coord_file = false, wxString coordinates_filename = " ");
// void patch_trimming(Image* input_stack, Image** patch_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, wxString outpath, int max_threads, bool use_coord_file, wxString coordinates_filename) {

//     // Image current_image;
//     // Image    box;
//     // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_24_16_0626/";
//     // int my_x, my_y;
//     // int x_at_centertlt, y_at_centertlt;
//     // int          number_of_patchgroups = input_coos_file->number_of_lines;
//     // float        temp_angle[1];
//     int   number_of_patchgroups = patch_num_x * patch_num_y;
//     float temp_array[number_of_patchgroups][2];
//     // MRCFile  input_stack(input_imgstack.ToStdString( ), false);

//     MRCFile* patch = new MRCFile[number_of_patchgroups];
//     // Image**  patch_stack = new Image*[number_of_patchgroups];
//     // for ( int i = 0; i < number_of_patchgroups; i++ ) {
//     //     patch_stack[i] = new Image[number_of_images];
//     // }

//     // void Deallocate2DFloatArray(float**& array, int dim1) {
//     //     for ( int i = 0; i < dim1; ++i ) {
//     //         delete[] array[i]; // each i-th pointer must be deleted first
//     //     }
//     //     delete[] array; // now delete pointer array
//     //     array = NULL;
//     // }

//     int    image_dim_x;
//     int    image_dim_y;
//     Image* input_stack_real_space = new Image[number_of_images];
//     image_dim_x                   = input_stack[0].logical_x_dimension;
//     image_dim_y                   = input_stack[0].logical_y_dimension;

//     if ( ! input_stack[0].is_in_real_space ) {
//         wxPrintf("do backward fft\n");
//         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//             input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
//             input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
//             input_stack_real_space[image_counter].BackwardFFT( );
//         }
//     }
//     else {
//         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//             input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
//             input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
//         }
//     }

//     // box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);

//     //write a if statement to judge if the number of coordinates in the coord file equals to image_no
//     if ( use_coord_file ) {
//         NumericTextFile* input_coos_file;
//         input_coos_file = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
//         for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
//             input_coos_file->ReadLine(temp_array[patch_counter]);
//             patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
//         }
//     }
//     else {
//         int step_size_x   = myroundint(float(image_dim_x) / float(patch_num_x) / 2); // check whether myroundint is correct
//         int step_size_y   = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
//         int patch_counter = 0;
//         for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
//             for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
//                 temp_array[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x - image_dim_x / 2;
//                 // temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y - image_dim_y / 2;
//                 temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = patch_y_ind * step_size_y * 2 + step_size_y - image_dim_y / 2;

//                 patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
//                 patch_counter += 1;
//             }
//         }
//     }
//     wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
//     // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
// #pragma omp parallel for default(shared) num_threads(max_threads)
//     for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

//         // float image_mean = input_stack_real_space[image_counter].ReturnAverageOfRealValues( );

//         // float image_mean = 0;
//         // wxPrintf("trimming image %i\n", image_counter);
//         wxPrintf("current patch: %i\n", patch_counter);
//         // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
//         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {

//             float image_mean = input_stack_real_space[image_counter].ReturnAverageOfRealValues( );
//             int   my_x       = temp_array[patch_counter][0];
//             int   my_y       = temp_array[patch_counter][1];

//             patch_stack[patch_counter][image_counter].Allocate(output_stack_box_size, output_stack_box_size, 1, true);

//             input_stack_real_space[image_counter].ClipInto(&patch_stack[patch_counter][image_counter], image_mean, false, 1.0, int(my_x), int(my_y), 0);
//             patch_stack[patch_counter][image_counter].WriteSlice(&patch[patch_counter], image_counter + 1);
//         }
//         patch[patch_counter].CloseFile( );
//     }

//     // delete input_coos_file;
//     delete[] patch;
//     delete[] input_stack_real_space;
//     wxPrintf("Done Patch Trimming\n");
//     // return patch_stack;
// }
void patch_trimming(Image* input_stack, Image** patch_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, wxString outpath, int max_threads, bool use_coord_file, wxString coordinates_filename) {

    // Image current_image;
    // Image    box;
    // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_24_16_0626/";
    // int my_x, my_y;
    // int x_at_centertlt, y_at_centertlt;
    // int          number_of_patchgroups = input_coos_file->number_of_lines;
    // float        temp_angle[1];
    int   number_of_patchgroups = patch_num_x * patch_num_y;
    float temp_array[number_of_patchgroups][2];
    // MRCFile  input_stack(input_imgstack.ToStdString( ), false);

    MRCFile* patch = new MRCFile[number_of_patchgroups];
    // Image**  patch_stack = new Image*[number_of_patchgroups];
    // for ( int i = 0; i < number_of_patchgroups; i++ ) {
    //     patch_stack[i] = new Image[number_of_images];
    // }

    // void Deallocate2DFloatArray(float**& array, int dim1) {
    //     for ( int i = 0; i < dim1; ++i ) {
    //         delete[] array[i]; // each i-th pointer must be deleted first
    //     }
    //     delete[] array; // now delete pointer array
    //     array = NULL;
    // }

    int image_dim_x;
    int image_dim_y;
    // Image* input_stack_real_space = new Image[number_of_images];
    image_dim_x = input_stack[0].logical_x_dimension;
    image_dim_y = input_stack[0].logical_y_dimension;

    if ( ! input_stack[0].is_in_real_space ) {
        wxPrintf("do backward fft\n");
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            // input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
            // input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
            input_stack[image_counter].BackwardFFT( );
        }
    }
    // else {
    //     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //         input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
    //         input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
    //     }
    // }

    // box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);

    //write a if statement to judge if the number of coordinates in the coord file equals to image_no
    if ( use_coord_file ) {
        NumericTextFile* input_coos_file;
        input_coos_file = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
        for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
            input_coos_file->ReadLine(temp_array[patch_counter]);
            patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
        }
    }
    else {
        int step_size_x   = myroundint(float(image_dim_x) / float(patch_num_x) / 2); // check whether myroundint is correct
        int step_size_y   = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
        int patch_counter = 0;
        for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
            for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
                temp_array[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x - image_dim_x / 2;
                // temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y - image_dim_y / 2;
                // temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = patch_y_ind * step_size_y * 2 + step_size_y - image_dim_y / 2;
                // temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y - image_dim_y / 2;
                temp_array[patch_num_x * (patch_num_y - patch_y_ind - 1) + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y - image_dim_y / 2;

                patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
                patch_counter += 1;
            }
        }
    }
    wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
    // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
#pragma omp parallel for default(shared) num_threads(max_threads)
    for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

        // float image_mean = input_stack_real_space[image_counter].ReturnAverageOfRealValues( );

        // float image_mean = 0;
        // wxPrintf("trimming image %i\n", image_counter);
        wxPrintf("current patch: %i\n", patch_counter);
        // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {

            float image_mean = input_stack[image_counter].ReturnAverageOfRealValues( );
            int   my_x       = temp_array[patch_counter][0];
            int   my_y       = temp_array[patch_counter][1];

            patch_stack[patch_counter][image_counter].Allocate(output_stack_box_size, output_stack_box_size, 1, true);

            input_stack[image_counter].ClipInto(&patch_stack[patch_counter][image_counter], image_mean, false, 1.0, int(my_x), int(my_y), 0);
            patch_stack[patch_counter][image_counter].WriteSlice(&patch[patch_counter], image_counter + 1);
        }
        patch[patch_counter].CloseFile( );
    }

    // delete input_coos_file;
    delete[] patch;
    // delete[] input_stack_real_space;
    wxPrintf("Done Patch Trimming\n");
    // return patch_stack;
}

Image** patch_trimming_independent(Image* input_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, wxString outpath, bool use_coord_file, wxString coordinates_filename) {

    // Image current_image;
    // Image    box;
    // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_24_16_0626/";
    int my_x, my_y;
    int x_at_centertlt, y_at_centertlt;
    // int          number_of_patchgroups = input_coos_file->number_of_lines;
    // float        temp_angle[1];
    int   number_of_patchgroups = patch_num_x * patch_num_y;
    float temp_array[number_of_patchgroups][2];
    // MRCFile  input_stack(input_imgstack.ToStdString( ), false);

    MRCFile* patch       = new MRCFile[number_of_patchgroups];
    Image**  patch_stack = new Image*[number_of_patchgroups];
    for ( int i = 0; i < number_of_patchgroups; i++ ) {
        patch_stack[i] = new Image[number_of_images];
    }

    // void Deallocate2DFloatArray(float**& array, int dim1) {
    //     for ( int i = 0; i < dim1; ++i ) {
    //         delete[] array[i]; // each i-th pointer must be deleted first
    //     }
    //     delete[] array; // now delete pointer array
    //     array = NULL;
    // }

    int    image_dim_x;
    int    image_dim_y;
    Image* input_stack_real_space = new Image[number_of_images];
    image_dim_x                   = input_stack[0].logical_x_dimension;
    image_dim_y                   = input_stack[0].logical_y_dimension;

    if ( ! input_stack[0].is_in_real_space ) {
        // wxPrintf("do backward fft\n");
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
            input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
            input_stack_real_space[image_counter].BackwardFFT( );
        }
    }
    else {
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            input_stack_real_space[image_counter].Allocate(image_dim_x, image_dim_y, 1, false);
            input_stack_real_space[image_counter].CopyFrom(&input_stack[image_counter]);
        }
    }

    // box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);

    //write a if statement to judge if the number of coordinates in the coord file equals to image_no
    if ( use_coord_file ) {
        NumericTextFile* input_coos_file;
        input_coos_file = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
        for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

            input_coos_file->ReadLine(temp_array[patch_counter]);
            patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
        }
    }
    else {
        int step_size_x   = myroundint(float(image_dim_x) / float(patch_num_x) / 2); // check whether myroundint is correct
        int step_size_y   = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
        int patch_counter = 0;
        for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
            for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
                temp_array[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x - image_dim_x / 2;
                temp_array[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y - image_dim_y / 2;
                patch[patch_counter].OpenFile(wxString::Format(outpath + "%04i.mrc", patch_counter).ToStdString( ), true);
                patch_counter += 1;
            }
        }
    }
    wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
    // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

        // float image_mean = input_stack_real_space[image_counter].ReturnAverageOfRealValues( );

        // float image_mean = 0;
        // wxPrintf("trimming image %i\n", image_counter);
        wxPrintf("current patch: %i\n", patch_counter);
        // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {

            float image_mean = input_stack_real_space[image_counter].ReturnAverageOfRealValues( );
            x_at_centertlt   = temp_array[patch_counter][0];
            y_at_centertlt   = temp_array[patch_counter][1];

            patch_stack[patch_counter][image_counter].Allocate(output_stack_box_size, output_stack_box_size, 1, true);

            my_x = x_at_centertlt;
            my_y = y_at_centertlt;

            input_stack_real_space[image_counter].ClipInto(&patch_stack[patch_counter][image_counter], image_mean, false, 1.0, int(my_x), int(my_y), 0);
            patch_stack[patch_counter][image_counter].WriteSlice(&patch[patch_counter], image_counter + 1);
        }
        patch[patch_counter].CloseFile( );
    }

    // delete input_coos_file;
    delete[] patch;
    wxPrintf("Done Patch Trimming");
    return patch_stack;
}

void split_and_update(matrix<double> Join1d, int joinsize) {
    int            count = 0;
    matrix<double> knot_on_spline_for_x, knot_on_spline_for_y;
    int            halflen = joinsize / 2;
    // int            halflen = knot_on_x * knot_on_y * knot_on_z;
    knot_on_spline_for_x.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_x(i) = Join1d(count);
        count++;
    }

    // Spline3dx.Update3DSpline1dInput(knot_on_spline_for_x);
    Spline3dx.Update3DSpline1dInput(knot_on_spline_for_x);

    knot_on_spline_for_y.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_y(i) = Join1d(count);
        count++;
    }
    // Spline3dy.Update3DSpline1dInput(knot_on_spline_for_y);
    Spline3dy.Update3DSpline1dInput(knot_on_spline_for_y);
};

void apply_fitting_quadratic_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, param_vector_quadratic params_x, param_vector_quadratic params_y) {
    // image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);
    int   super_dim_x     = super_res_stack[0].logical_x_dimension;
    int   super_dim_y     = super_res_stack[0].logical_y_dimension;
    int   image_dim_x     = myroundint(super_res_stack[0].logical_x_dimension / output_binning_factor);
    int   image_dim_y     = myroundint(super_res_stack[0].logical_y_dimension / output_binning_factor);
    float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
    float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
    int   totalpixels     = super_dim_x * super_dim_y;

    input_vector input;
    float        time;

    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];
    float* shifted_map_x  = new float[totalpixels];
    float* shifted_map_y  = new float[totalpixels];

    // initialize the pixel coordinates
    for ( int i = 0; i < super_dim_y; i++ ) {
        for ( int j = 0; j < super_dim_x; j++ ) {
            original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
            original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
        }
    }

    Image tmp_sup_res;
    tmp_sup_res.Allocate(super_res_stack[0].logical_x_dimension, super_res_stack[0].logical_y_dimension, 1, true);
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        wxPrintf("correcting frame %i \n ", image_counter + 1);
        time     = image_counter;
        input(2) = time;
        for ( int pix = 0; pix < totalpixels; pix++ ) {
            input(0)           = original_map_x[pix];
            input(1)           = original_map_y[pix];
            shifted_map_x[pix] = model_quadratic(input, params_x) * x_binning_float + original_map_x[pix] * x_binning_float;
            shifted_map_y[pix] = model_quadratic(input, params_y) * y_binning_float + original_map_y[pix] * y_binning_float;
        }

        tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
        tmp_sup_res.SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        // super_res_stack[image_counter].BackwardFFT();
        super_res_stack[image_counter].Distortion(&tmp_sup_res, shifted_map_x, shifted_map_y);
        tmp_sup_res.ForwardFFT(true);
        tmp_sup_res.ZeroCentralPixel( );
        tmp_sup_res.Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
        input_stack[image_counter].CopyFrom(&tmp_sup_res);
    }

    delete[] original_map_x;
    delete[] original_map_y;
    delete[] shifted_map_x;
    delete[] shifted_map_y;
    original_map_x = nullptr;
    original_map_y = nullptr;
    shifted_map_x  = nullptr;
    shifted_map_y  = nullptr;
}

void apply_fitting_quadratic(Image* input_stack, int number_of_images, param_vector_quadratic params_x, param_vector_quadratic params_y) {
    // image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);

    int    image_dim_x    = input_stack[0].logical_x_dimension;
    int    image_dim_y    = input_stack[0].logical_y_dimension;
    int    totalpixels    = image_dim_x * image_dim_y;
    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];
    float* shifted_map_x  = new float[totalpixels];
    float* shifted_map_y  = new float[totalpixels];
    // Image* distorted_stack;
    // distorted_stack = new Image[number_of_images];
    Image tmp_img;
    tmp_img.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, 1, true);
    // initialize the pixel coordinates
    for ( int i = 0; i < image_dim_y; i++ ) {
        for ( int j = 0; j < image_dim_x; j++ ) {
            original_map_x[i * image_dim_x + j] = j;
            original_map_y[i * image_dim_x + j] = i;
        }
    }
    input_vector input;
    float        time;
    // wxString     outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_24_16_0626/";
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        wxPrintf("correcting frame %i \n ", image_counter + 1);
        time     = image_counter;
        input(2) = time;
        for ( int pix = 0; pix < totalpixels; pix++ ) {
            input(0)           = original_map_x[pix];
            input(1)           = original_map_y[pix];
            shifted_map_x[pix] = model_quadratic(input, params_x) + original_map_x[pix];
            shifted_map_y[pix] = model_quadratic(input, params_y) + original_map_y[pix];
        }
        // distorted_stack[image_counter].Allocate(input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, true);
        // distorted_stack[image_counter].SetToConstant(input_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        // input_stack[image_counter].Distortion(&distorted_stack[image_counter], shifted_map_x, shifted_map_y);
        tmp_img.CopyFrom(&input_stack[image_counter]);
        input_stack[image_counter].SetToConstant(input_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        tmp_img.Distortion(&input_stack[image_counter], shifted_map_x, shifted_map_y);
    }
    // delete[] input_stack;
    // input_stack = distorted_stack; //??? why this doesn't work, what's different from copyfrom

    // Image sum_image_fixed, distorted_fixed;
    // sum_image_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
    // distorted_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
    // sum_image_fixed.SetToConstant(0.0);
    // distorted_fixed.SetToConstant(0.0);
    // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //     sum_image_fixed.AddImage(&input_stack[image_counter]);
    //     distorted_fixed.AddImage(&distorted_stack[image_counter]);
    //     input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "original_frames.mrc", image_counter + 1);
    //     distorted_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "fixed_frames.mrc", image_counter + 1);
    // }
    // // sum_image_fixed.BackwardFFT( );
    // // distorted_fixed.BackwardFFT( );
    // sum_image_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "original.mrc", 1);
    // distorted_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "fixed.mrc", 1);
    // delete[] input_stack;
    delete[] original_map_x;
    delete[] original_map_y;
    delete[] shifted_map_x;
    delete[] shifted_map_y;
}

void apply_fitting_spline_sup(int output_x_size, int output_y_size, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d_R0, column_vector Join1d_R1, int max_threads) {
    int   super_dim_x     = super_res_stack[0].logical_x_dimension;
    int   super_dim_y     = super_res_stack[0].logical_y_dimension;
    int   image_dim_x     = output_x_size;
    int   image_dim_y     = output_y_size;
    float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
    float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
    int   totalpixels     = super_dim_x * super_dim_y;

    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];

    int              joinsize = Join1d_R0.size( );
    MovieFrameSpline spline3dx_R0, spline3dy_R0;
    wxPrintf("Join size: %d\n", joinsize);
    // split_and_update(Join1d_R0, joinsize);
    // spline3dx_R0.CopyFrom(Spline3dx);
    // spline3dy_R0.CopyFrom(Spline3dy); //check copyfrom

    spline3dx_R0.Initialize(Spline3dx.knot_no_z, Spline3dx.knot_no_y, Spline3dx.knot_no_x, Spline3dx.frame_no, image_dim_x, image_dim_y, Spline3dx.spline_patch_dimz, Spline3dx.spline_patch_dimx, Spline3dx.spline_patch_dimy);
    spline3dy_R0.Initialize(Spline3dy.knot_no_z, Spline3dy.knot_no_y, Spline3dy.knot_no_x, Spline3dy.frame_no, image_dim_x, image_dim_y, Spline3dy.spline_patch_dimz, Spline3dy.spline_patch_dimx, Spline3dy.spline_patch_dimy);
    spline3dx_R0.Update3DSplineInterpMapping(Spline3dx.x, Spline3dx.y, Spline3dx.z);
    spline3dy_R0.Update3DSplineInterpMapping(Spline3dy.x, Spline3dy.y, Spline3dy.z);
    wxPrintf("initialized spline 3d for 0 round\n");
    int            count = 0;
    matrix<double> knot_on_spline_for_x, knot_on_spline_for_y;
    int            halflen = joinsize / 2;
    // int            halflen = knot_on_x * knot_on_y * knot_on_z;
    knot_on_spline_for_x.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_x(i) = Join1d_R0(count);
        count++;
    }

    // Spline3dx.Update3DSpline1dInput(knot_on_spline_for_x);
    wxPrintf("before update3d spline x\n");
    spline3dx_R0.Update3DSpline1dInput(knot_on_spline_for_x);
    wxPrintf("done update3d spline x\n");
    knot_on_spline_for_y.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_y(i) = Join1d_R0(count);
        count++;
    }
    // Spline3dy.Update3DSpline1dInput(knot_on_spline_for_y);
    spline3dy_R0.Update3DSpline1dInput(knot_on_spline_for_y);

    count = 0;
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_x(i) = Join1d_R1(count);
        count++;
    }
    Spline3dx.Update3DSpline1dInput(knot_on_spline_for_x);
    for ( int i = 0; i < halflen; i++ ) {
        knot_on_spline_for_y(i) = Join1d_R1(count);
        count++;
    }
    Spline3dy.Update3DSpline1dInput(knot_on_spline_for_y);

    // split_and_update(Join1d_R1, joinsize);
    // initialize the pixel coordinates
    for ( int i = 0; i < super_dim_y; i++ ) {
        for ( int j = 0; j < super_dim_x; j++ ) {
            original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
            original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
        }
    }

    Image  tmp_sup_res;
    float* shifted_map_x;
    float* shifted_map_y;
    // int   pix;

#pragma omp parallel for num_threads(max_threads) private(tmp_sup_res, shifted_map_x, shifted_map_y)
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        shifted_map_x = new float[totalpixels];
        shifted_map_y = new float[totalpixels];
        wxPrintf("generating interpolation map %i \n ", image_counter + 1);
        for ( int i = 0; i < super_dim_y; i++ ) {
            for ( int j = 0; j < super_dim_x; j++ ) {
                int pix            = i * super_dim_x + j;
                shifted_map_x[pix] = j + Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * x_binning_float;
                shifted_map_y[pix] = i + Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * y_binning_float;
                shifted_map_x[pix] = shifted_map_x[pix] + spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * x_binning_float;
                shifted_map_y[pix] = shifted_map_y[pix] + spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * y_binning_float;
            }
        }
        // if ( image_counter == 74 ) {
        //     for ( int i = 0; i < super_dim_y; i += 1000 ) {
        //         for ( int j = 0; j < super_dim_x; j += 1000 ) {
        //             int pix = i * super_dim_x + j;
        //             wxPrintf("%f %f\t", shifted_map_x[pix], shifted_map_y[pix]);
        //         }
        //         wxPrintf("\n");
        //     }
        // }
        // wxPrintf("spline shift\n");
        // if ( image_counter == 74 ) {
        //     for ( int i = 0; i < super_dim_y; i += 1000 ) {
        //         for ( int j = 0; j < super_dim_x; j += 1000 ) {
        //             int pix = i * super_dim_x + j;
        //             wxPrintf("%f %f\t", Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter), Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter));
        //             wxPrintf("%f %f\n", spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter), spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter));
        //         }
        //         wxPrintf("\n");
        //     }
        // }
        wxPrintf("correcting frame %i \n ", image_counter + 1);
        tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
        super_res_stack[image_counter].SetToConstant(tmp_sup_res.ReturnAverageOfRealValuesOnEdges( ));
        tmp_sup_res.Distortion(&super_res_stack[image_counter], shifted_map_x, shifted_map_y);
        delete[] shifted_map_x;
        delete[] shifted_map_y;
        shifted_map_x = nullptr;
        shifted_map_y = nullptr;
        tmp_sup_res.Deallocate( );
    }
    wxPrintf("distortion finished\n");

    delete[] original_map_x;
    delete[] original_map_y;
    original_map_x = nullptr;
    original_map_y = nullptr;
#pragma omp parallel for num_threads(max_threads)
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        super_res_stack[image_counter].ForwardFFT(true);
        super_res_stack[image_counter].ZeroCentralPixel( );
        super_res_stack[image_counter].Resize(image_dim_x, image_dim_y, 1);
        super_res_stack[image_counter].BackwardFFT( );
    }
    spline3dx_R0.Deallocate( );
    spline3dy_R0.Deallocate( );
    Spline3dx.Deallocate( );
    Spline3dy.Deallocate( );
};

void apply_fitting_spline_sup_control(int output_x_size, int output_y_size, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Control1d_R0, column_vector Control1d_R1, int max_threads) {
    int   super_dim_x     = super_res_stack[0].logical_x_dimension;
    int   super_dim_y     = super_res_stack[0].logical_y_dimension;
    int   image_dim_x     = output_x_size;
    int   image_dim_y     = output_y_size;
    float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
    float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
    int   totalpixels     = super_dim_x * super_dim_y;

    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];

    int              controlsize = Control1d_R0.size( );
    MovieFrameSpline spline3dx_R0, spline3dy_R0;
    wxPrintf("Control Points size: %d\n", controlsize);
    // split_and_update(Join1d_R0, joinsize);
    // spline3dx_R0.CopyFrom(Spline3dx);
    // spline3dy_R0.CopyFrom(Spline3dy); //check copyfrom

    spline3dx_R0.Initialize(Spline3dx.knot_no_z, Spline3dx.knot_no_y, Spline3dx.knot_no_x, Spline3dx.frame_no, image_dim_x, image_dim_y, Spline3dx.spline_patch_dimz, Spline3dx.spline_patch_dimx, Spline3dx.spline_patch_dimy);
    spline3dy_R0.Initialize(Spline3dy.knot_no_z, Spline3dy.knot_no_y, Spline3dy.knot_no_x, Spline3dy.frame_no, image_dim_x, image_dim_y, Spline3dy.spline_patch_dimz, Spline3dy.spline_patch_dimx, Spline3dy.spline_patch_dimy);
    spline3dx_R0.Update3DSplineInterpMappingControl(Spline3dx.x, Spline3dx.y, Spline3dx.z);
    spline3dy_R0.Update3DSplineInterpMappingControl(Spline3dy.x, Spline3dy.y, Spline3dy.z);
    wxPrintf("initialized spline 3d for 0 round\n");
    int            count = 0;
    matrix<double> control_on_spline_for_x, control_on_spline_for_y;
    int            halflen = controlsize / 2;
    // int            halflen = knot_on_x * knot_on_y * knot_on_z;
    control_on_spline_for_x.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_x(i) = Control1d_R0(count);
        count++;
    }

    // Spline3dx.Update3DSpline1dInput(knot_on_spline_for_x);
    wxPrintf("before update3d spline x\n");
    spline3dx_R0.Update3DSpline1dInputControlPoints(control_on_spline_for_x);
    wxPrintf("done update3d spline x\n");
    control_on_spline_for_y.set_size(halflen, 1);
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_y(i) = Control1d_R0(count);
        count++;
    }
    // Spline3dy.Update3DSpline1dInput(knot_on_spline_for_y);
    spline3dy_R0.Update3DSpline1dInputControlPoints(control_on_spline_for_y);

    count = 0;
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_x(i) = Control1d_R1(count);
        count++;
    }
    Spline3dx.Update3DSpline1dInputControlPoints(control_on_spline_for_x);
    for ( int i = 0; i < halflen; i++ ) {
        control_on_spline_for_y(i) = Control1d_R1(count);
        count++;
    }
    Spline3dy.Update3DSpline1dInputControlPoints(control_on_spline_for_y);

    // split_and_update(Join1d_R1, joinsize);
    // initialize the pixel coordinates
    for ( int i = 0; i < super_dim_y; i++ ) {
        for ( int j = 0; j < super_dim_x; j++ ) {
            original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
            original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
        }
    }

    Image  tmp_sup_res;
    float* shifted_map_x;
    float* shifted_map_y;

#pragma omp parallel for num_threads(max_threads) private(tmp_sup_res, shifted_map_x, shifted_map_y)
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        shifted_map_x = new float[totalpixels];
        shifted_map_y = new float[totalpixels];
        wxPrintf("generating interpolation map %i \n ", image_counter + 1);
        for ( int i = 0; i < super_dim_y; i++ ) {
            for ( int j = 0; j < super_dim_x; j++ ) {
                int pix            = i * super_dim_x + j;
                shifted_map_x[pix] = j + Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * x_binning_float;
                shifted_map_y[pix] = i + Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * y_binning_float;
                shifted_map_x[pix] = shifted_map_x[pix] + spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * x_binning_float;
                shifted_map_y[pix] = shifted_map_y[pix] + spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter) * y_binning_float;
            }
        }

        wxPrintf("correcting frame %i \n ", image_counter + 1);
        tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
        super_res_stack[image_counter].SetToConstant(tmp_sup_res.ReturnAverageOfRealValuesOnEdges( ));
        wxPrintf("before distortion %i \n ", image_counter + 1);
        tmp_sup_res.Distortion(&super_res_stack[image_counter], shifted_map_x, shifted_map_y);
        delete[] shifted_map_x;
        delete[] shifted_map_y;
        shifted_map_x = nullptr;
        shifted_map_y = nullptr;
        tmp_sup_res.Deallocate( );
    }
    wxPrintf("distortion finished\n");

    delete[] original_map_x;
    delete[] original_map_y;
    original_map_x = nullptr;
    original_map_y = nullptr;
#pragma omp parallel for num_threads(max_threads)
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        super_res_stack[image_counter].ForwardFFT(true);
        super_res_stack[image_counter].ZeroCentralPixel( );
        super_res_stack[image_counter].Resize(image_dim_x, image_dim_y, 1);
        super_res_stack[image_counter].BackwardFFT( );
    }
    spline3dx_R0.Deallocate( );
    spline3dy_R0.Deallocate( );
    // Spline3dx.Deallocate( );
    // Spline3dy.Deallocate( );
};

// 12042023
// void apply_fitting_spline_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d_R0, column_vector Join1d_R1, int max_threads) {
//     int   super_dim_x     = super_res_stack[0].logical_x_dimension;
//     int   super_dim_y     = super_res_stack[0].logical_y_dimension;
//     int   image_dim_x     = myroundint(super_res_stack[0].logical_x_dimension / output_binning_factor);
//     int   image_dim_y     = myroundint(super_res_stack[0].logical_y_dimension / output_binning_factor);
//     float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
//     float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
//     int   totalpixels     = super_dim_x * super_dim_y;

//     float* original_map_x = new float[totalpixels];
//     float* original_map_y = new float[totalpixels];

//     int              joinsize = Join1d_R0.size( );
//     MovieFrameSpline spline3dx_R0, spline3dy_R0;
//     split_and_update(Join1d_R0, joinsize);
//     spline3dx_R0.CopyFrom(Spline3dx);
//     spline3dy_R0.CopyFrom(Spline3dy);
//     split_and_update(Join1d_R1, joinsize);

//     // initialize the pixel coordinates
//     for ( int i = 0; i < super_dim_y; i++ ) {
//         for ( int j = 0; j < super_dim_x; j++ ) {
//             original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
//             original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
//         }
//     }

//     Image  tmp_sup_res;
//     float* shifted_map_x;
//     float* shifted_map_y;
//     // int   pix;

// #pragma omp parallel for num_threads(max_threads) private(tmp_sup_res, shifted_map_x, shifted_map_y)
//     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//         // for ( int image_counter = 0; image_counter < 10; image_counter++ ) {

//         // tmp_sup_res.Allocate(super_res_stack[0].logical_x_dimension, super_res_stack[0].logical_y_dimension, true);
//         // float* shifted_map_x = new float[totalpixels];
//         // float* shifted_map_y = new float[totalpixels];
//         shifted_map_x = new float[totalpixels];
//         shifted_map_y = new float[totalpixels];
//         wxPrintf("generating interpolation map %i \n ", image_counter + 1);
//         for ( int i = 0; i < super_dim_y; i++ ) {
//             for ( int j = 0; j < super_dim_x; j++ ) {
//                 int pix            = i * super_dim_x + j;
//                 shifted_map_x[pix] = j + Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//                 shifted_map_y[pix] = i + Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//                 shifted_map_x[pix] = shifted_map_x[pix] + spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//                 shifted_map_y[pix] = shifted_map_y[pix] + spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//             }
//         }
//         wxPrintf("correcting frame %i \n ", image_counter + 1);
//         tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
//         tmp_sup_res.SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//         super_res_stack[image_counter].Distortion(&tmp_sup_res, shifted_map_x, shifted_map_y);
//         tmp_sup_res.ForwardFFT(true);
//         tmp_sup_res.ZeroCentralPixel( );
//         tmp_sup_res.Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
//         tmp_sup_res.BackwardFFT( );
//         input_stack[image_counter].CopyFrom(&tmp_sup_res);
//         // input_stack[image_counter].ForwardFFT(true);
//         // tmp_sup_res.ClipInto(&input_stack[image_counter]);
//         // input_stack[image_counter].BackwardFFT( );
//         // input_stack[image_counter].SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//         // super_res_stack[image_counter].Distortion(&input_stack[image_counter], shifted_map_x, shifted_map_y);
//         // input_stack[image_counter].ForwardFFT(true);
//         // input_stack[image_counter].ZeroCentralPixel( );
//         // input_stack[image_counter].Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
//         delete[] shifted_map_x;
//         delete[] shifted_map_y;
//         shifted_map_x = nullptr;
//         shifted_map_y = nullptr;
//         tmp_sup_res.Deallocate( );
//         // tmp_sup_res = nullptr;
//     }
//     tmp_sup_res = nullptr;
//     // // #pragma omp parallel for num_threads(max_threads) private(tmp_sup_res, shifted_map_x, shifted_map_y)
//     // shifted_map_x = new float[totalpixels];
//     // shifted_map_y = new float[totalpixels];
//     // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//     //     // tmp_sup_res.Allocate(super_res_stack[0].logical_x_dimension, super_res_stack[0].logical_y_dimension, true);
//     //     // float* shifted_map_x = new float[totalpixels];
//     //     // float* shifted_map_y = new float[totalpixels];

//     //     wxPrintf("generating interpolation map %i \n ", image_counter + 1);
//     //     for ( int i = 0; i < super_dim_y; i++ ) {
//     //         for ( int j = 0; j < super_dim_x; j++ ) {
//     //             int pix            = i * super_dim_x + j;
//     //             shifted_map_x[pix] = j + Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//     //             shifted_map_y[pix] = i + Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//     //             shifted_map_x[pix] = shifted_map_x[pix] + spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//     //             shifted_map_y[pix] = shifted_map_y[pix] + spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//     //         }
//     //     }
//     //     wxPrintf("correcting frame %i \n ", image_counter + 1);
//     //     tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
//     //     tmp_sup_res.SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//     //     super_res_stack[image_counter].Distortion(&tmp_sup_res, shifted_map_x, shifted_map_y);
//     //     tmp_sup_res.ForwardFFT(true);
//     //     tmp_sup_res.ZeroCentralPixel( );
//     //     // tmp_sup_res.Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
//     //     // input_stack[image_counter].CopyFrom(&tmp_sup_res);

//     //     tmp_sup_res.ClipInto(&input_stack[image_counter]);
//     //     // input_stack[image_counter].SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//     //     // super_res_stack[image_counter].Distortion(&input_stack[image_counter], shifted_map_x, shifted_map_y);
//     //     // input_stack[image_counter].ForwardFFT(true);
//     //     // input_stack[image_counter].ZeroCentralPixel( );
//     //     // input_stack[image_counter].Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
//     // }
//     // delete[] shifted_map_x;
//     // delete[] shifted_map_y;
//     // shifted_map_x = nullptr;
//     // shifted_map_y = nullptr;
//     // tmp_sup_res.Deallocate( );
//     // tmp_sup_res = nullptr;
//     delete[] original_map_x;
//     delete[] original_map_y;
//     original_map_x = nullptr;
//     original_map_y = nullptr;
// };

// void apply_fitting_spline_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, column_vector Join1d, int max_threads) {
//     int   super_dim_x     = super_res_stack[0].logical_x_dimension;
//     int   super_dim_y     = super_res_stack[0].logical_y_dimension;
//     int   image_dim_x     = myroundint(super_res_stack[0].logical_x_dimension / output_binning_factor);
//     int   image_dim_y     = myroundint(super_res_stack[0].logical_y_dimension / output_binning_factor);
//     float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
//     float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
//     int   totalpixels     = super_dim_x * super_dim_y;

//     float* original_map_x = new float[totalpixels];
//     float* original_map_y = new float[totalpixels];

//     int joinsize = Join1d.size( );
//     split_and_update(Join1d, joinsize);

//     // initialize the pixel coordinates
//     for ( int i = 0; i < super_dim_y; i++ ) {
//         for ( int j = 0; j < super_dim_x; j++ ) {
//             original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
//             original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
//         }
//     }

//     Image tmp_sup_res;
//     float time;
//     // int   pix;

// #pragma omp parallel for num_threads(max_threads) private(tmp_sup_res)
//     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//         float* shifted_map_x = new float[totalpixels];
//         float* shifted_map_y = new float[totalpixels];
//         tmp_sup_res.Allocate(super_res_stack[0].logical_x_dimension, super_res_stack[0].logical_y_dimension, true);
//         wxPrintf("generating interpolation map %i \n ", image_counter + 1);
//         for ( int i = 0; i < super_dim_y; i++ ) {
//             for ( int j = 0; j < super_dim_x; j++ ) {
//                 int pix            = i * super_dim_x + j;
//                 shifted_map_x[pix] = j + Spline3dx.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//                 shifted_map_y[pix] = i + Spline3dy.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_counter);
//             }
//         }
//         wxPrintf("correcting frame %i \n ", image_counter + 1);
//         // tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
//         tmp_sup_res.SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//         super_res_stack[image_counter].Distortion(&tmp_sup_res, shifted_map_x, shifted_map_y);
//         tmp_sup_res.ForwardFFT(true);
//         tmp_sup_res.ZeroCentralPixel( );
//         tmp_sup_res.Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
//         input_stack[image_counter].CopyFrom(&tmp_sup_res);
//         delete[] shifted_map_x;
//         delete[] shifted_map_y;
//         tmp_sup_res.Deallocate( );
//     }

//     delete[] original_map_x;
//     delete[] original_map_y;
// };

void apply_fitting_linear_sup(Image* input_stack, Image* super_res_stack, float output_binning_factor, int number_of_images, param_vector_linear params_x, param_vector_linear params_y) {
    // image_stack[image_counter - 1].Resize(myroundint(image_stack[image_counter - 1].logical_x_dimension / output_binning_factor), myroundint(image_stack[image_counter - 1].logical_y_dimension / output_binning_factor), 1);
    int   super_dim_x     = super_res_stack[0].logical_x_dimension;
    int   super_dim_y     = super_res_stack[0].logical_y_dimension;
    int   image_dim_x     = myroundint(super_res_stack[0].logical_x_dimension / output_binning_factor);
    int   image_dim_y     = myroundint(super_res_stack[0].logical_y_dimension / output_binning_factor);
    float x_binning_float = super_res_stack[0].logical_x_dimension / image_dim_x;
    float y_binning_float = super_res_stack[0].logical_y_dimension / image_dim_y;
    int   totalpixels     = super_dim_x * super_dim_y;

    input_vector input;
    float        time;

    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];
    float* shifted_map_x  = new float[totalpixels];
    float* shifted_map_y  = new float[totalpixels];

    // initialize the pixel coordinates
    for ( int i = 0; i < super_dim_y; i++ ) {
        for ( int j = 0; j < super_dim_x; j++ ) {
            original_map_x[i * super_dim_x + j] = float(j) / x_binning_float;
            original_map_y[i * super_dim_x + j] = float(i) / y_binning_float;
        }
    }

    Image tmp_sup_res;
    tmp_sup_res.Allocate(super_res_stack[0].logical_x_dimension, super_res_stack[0].logical_y_dimension, 1, true);
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        wxPrintf("correcting frame %i \n ", image_counter + 1);
        time     = image_counter;
        input(2) = time;
        for ( int pix = 0; pix < totalpixels; pix++ ) {
            input(0)           = original_map_x[pix];
            input(1)           = original_map_y[pix];
            shifted_map_x[pix] = model_linear(input, params_x) * x_binning_float + original_map_x[pix] * x_binning_float;
            shifted_map_y[pix] = model_linear(input, params_y) * y_binning_float + original_map_y[pix] * y_binning_float;
        }

        tmp_sup_res.CopyFrom(&super_res_stack[image_counter]);
        tmp_sup_res.SetToConstant(super_res_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        // super_res_stack[image_counter].BackwardFFT();
        super_res_stack[image_counter].Distortion(&tmp_sup_res, shifted_map_x, shifted_map_y);
        tmp_sup_res.ForwardFFT(true);
        tmp_sup_res.ZeroCentralPixel( );
        tmp_sup_res.Resize(image_dim_x, image_dim_y, 1, super_res_stack[image_counter].ReturnAverageOfRealValues( ));
        input_stack[image_counter].CopyFrom(&tmp_sup_res);
    }

    delete[] original_map_x;
    delete[] original_map_y;
    delete[] shifted_map_x;
    delete[] shifted_map_y;
    original_map_x = nullptr;
    original_map_y = nullptr;
    shifted_map_x  = nullptr;
    shifted_map_y  = nullptr;
}

void apply_fitting_linear(Image* input_stack, int number_of_images, param_vector_linear params_x, param_vector_linear params_y) {

    int    image_dim_x    = input_stack[0].logical_x_dimension;
    int    image_dim_y    = input_stack[0].logical_y_dimension;
    int    totalpixels    = image_dim_x * image_dim_y;
    float* original_map_x = new float[totalpixels];
    float* original_map_y = new float[totalpixels];
    float* shifted_map_x  = new float[totalpixels];
    float* shifted_map_y  = new float[totalpixels];
    // Image* distorted_stack;
    // distorted_stack = new Image[number_of_images];
    Image tmp_img;
    tmp_img.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, 1, true);
    // initialize the pixel coordinates
    for ( int i = 0; i < image_dim_y; i++ ) {
        for ( int j = 0; j < image_dim_x; j++ ) {
            original_map_x[i * image_dim_x + j] = j;
            original_map_y[i * image_dim_x + j] = i;
        }
    }
    input_vector input;
    float        time;
    // wxString     outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_24_16_0626/";
    // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_linear_0626/";
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        wxPrintf("correcting frame %i \n ", image_counter + 1);
        time     = image_counter;
        input(2) = time;
        for ( int pix = 0; pix < totalpixels; pix++ ) {
            input(0)           = original_map_x[pix];
            input(1)           = original_map_y[pix];
            shifted_map_x[pix] = model_linear(input, params_x) + original_map_x[pix];
            shifted_map_y[pix] = model_linear(input, params_y) + original_map_y[pix];
        }
        // distorted_stack[image_counter].Allocate(input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, true);
        // distorted_stack[image_counter].SetToConstant(input_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        // input_stack[image_counter].Distortion(&distorted_stack[image_counter], shifted_map_x, shifted_map_y);

        // input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "beforedistortion.mrc", image_counter + 1);

        tmp_img.CopyFrom(&input_stack[image_counter]);
        // tmp_img.QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "tmpbeforedistortion.mrc", image_counter + 1);
        // input_stack[image_counter].SetToConstant(input_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
        tmp_img.Distortion(&input_stack[image_counter], shifted_map_x, shifted_map_y);
        // input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "afterdistortion.mrc", image_counter + 1);

        // // following is to write the shifts for each pixel------
        // // if ( image_counter == 69 ) {
        // wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch_linear_0626/";
        // // wxString shifted_mapx_file = outputpath + "shifted_x.txt";
        // // wxString shifted_mapy_file = outputpath + "shifted_y.txt";
        // wxString      shifted_mapx_file = wxString::Format(outputpath + "%04i_" + "shifted_x.txt", image_counter);
        // wxString      shifted_mapy_file = wxString::Format(outputpath + "%04i_" + "shifted_y.txt", image_counter);
        // std::ofstream xoFile, yoFile;
        // tmp_img.QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "tmpbeforedistortion.mrc", 1);
        // input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "afterdistortion.mrc", 1);
        // // wxPrintf("1\n");
        // // for ( int i = 0; i < 10; i++ ) {
        // //     for ( int j = 0; j < 10; j++ ) {
        // //         shifted_map_x[i][j] = j;
        // //         shifted_map_y[i][j] = i;
        // //     }
        // // }

        // xoFile.open(shifted_mapx_file.c_str( ));
        // yoFile.open(shifted_mapy_file.c_str( ));
        // if ( xoFile.is_open( ) && yoFile.is_open( ) ) {
        //     wxPrintf("files are open\n");
        //     // float myarray[10][5760];
        //     for ( int i = 0; i < image_dim_y; i++ ) {
        //         for ( int j = 0; j < image_dim_x; j++ ) {
        //             xoFile << shifted_map_x[i * image_dim_x + j] - original_map_x[i * image_dim_x + j] << '\t';
        //             yoFile << shifted_map_y[i * image_dim_x + j] - original_map_y[i * image_dim_x + j] << '\t';
        //         }
        //         xoFile << '\n';
        //         yoFile << '\n';
        //     }
        // }
        // xoFile.close( );
        // yoFile.close( );
        // // }
        // // // above is to write the shifts for each pixel------
    }
    // delete[] input_stack;
    // input_stack = distorted_stack; //??? why this doesn't work, what's different from copyfrom

    // Image sum_image_fixed, distorted_fixed;
    // sum_image_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
    // distorted_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
    // sum_image_fixed.SetToConstant(0.0);
    // distorted_fixed.SetToConstant(0.0);
    // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //     sum_image_fixed.AddImage(&input_stack[image_counter]);
    //     distorted_fixed.AddImage(&distorted_stack[image_counter]);
    //     input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "original_frames.mrc", image_counter + 1);
    //     distorted_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "fixed_frames.mrc", image_counter + 1);
    // }
    // // sum_image_fixed.BackwardFFT( );
    // // distorted_fixed.BackwardFFT( );
    // sum_image_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "original.mrc", 1);
    // distorted_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "fixed.mrc", 1);
    // delete[] input_stack;
    delete[] original_map_x;
    delete[] original_map_y;
    delete[] shifted_map_x;
    delete[] shifted_map_y;
}

// ----------------------------------------------------------------------------------------
void dosefilter(Image* image_stack, int first_frame, int last_frame, float* dose_filter_sum_of_squares, ElectronDose* my_electron_dose, StopWatch profile_timing, float exposure_per_frame, float pre_exposure_amount, int max_threads) {
    float* dose_filter;
    // allocate arrays for the filter, and the sum of squares..
    // profile_timing.start("setup dose filter");
    // dose_filter_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];
    // ZeroFloatArray(dose_filter_sum_of_squares, image_stack[0].real_memory_allocated / 2);
    // profile_timing.lap("setup dose filter");

    // We don't want any copying of the timer, so just let them all have a pointer, only thread zero will do anything with it.
    StopWatch* shared_ptr;
    shared_ptr = &profile_timing;
// #pragma omp parallel default(shared) num_threads(max_threads) shared(shared_ptr) private(image_counter, dose_filter, pixel_counter)
#pragma omp parallel default(shared) num_threads(max_threads) shared(shared_ptr) private(dose_filter)

    { // for omp

        shared_ptr->start("setup dose filter");
        dose_filter = new float[image_stack[0].real_memory_allocated / 2];
        ZeroFloatArray(dose_filter, image_stack[0].real_memory_allocated / 2);
        float* thread_dose_filter_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];
        ZeroFloatArray(thread_dose_filter_sum_of_squares, image_stack[0].real_memory_allocated / 2);
        shared_ptr->lap("setup dose filter");

#pragma omp for
        for ( int image_counter = first_frame - 1; image_counter < last_frame; image_counter++ ) {
            shared_ptr->start("calc dose filter");
            my_electron_dose->CalculateDoseFilterAs1DArray(&image_stack[image_counter], dose_filter, (image_counter * exposure_per_frame) + pre_exposure_amount, ((image_counter + 1) * exposure_per_frame) + pre_exposure_amount);
            shared_ptr->lap("calc dose filter");
            // filter the image, and also calculate the sum of squares..

            shared_ptr->start("apply dose filter");
            for ( int pixel_counter = 0; pixel_counter < image_stack[image_counter].real_memory_allocated / 2; pixel_counter++ ) {
                image_stack[image_counter].complex_values[pixel_counter] *= dose_filter[pixel_counter];
                thread_dose_filter_sum_of_squares[pixel_counter] += powf(dose_filter[pixel_counter], 2);
                //if (image_counter == 65) wxPrintf("%f\n", dose_filter[pixel_counter]);
            }
            shared_ptr->lap("apply dose filter");
        }

        delete[] dose_filter;

        // copy the local sum of squares to global
        shared_ptr->start("copy dose filter sum of squares");
#pragma omp critical
        {

            for ( int pixel_counter = 0; pixel_counter < image_stack[0].real_memory_allocated / 2; pixel_counter++ ) {
                dose_filter_sum_of_squares[pixel_counter] += thread_dose_filter_sum_of_squares[pixel_counter];
            }
        }
        shared_ptr->lap("copy dose filter sum of squares");

        delete[] thread_dose_filter_sum_of_squares;

    } // end omp section
}

void write_shifts(int patch_no_x, int patch_no_y, int image_no, std::string output_path, std::string shift_file_prefx, std::string shift_file_prefy) {
    std::ofstream      xoFile, yoFile;
    char               buffer[200];
    std::string        shift_file, shift_filex, shift_filey;
    std::ostringstream oss;
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        int shift_file_index = image_ind;
        oss << "%s%04i%s%s" << output_path.c_str( ), shift_file_index, shift_file_prefx.c_str( ), "ccmap.txt";
        std::snprintf(buffer, sizeof(buffer), "%s%04i%s%s", output_path.c_str( ), shift_file_index, shift_file_prefx.c_str( ), "ccmap.txt");
        // shift_file = oss.str( );
        shift_filex = buffer;
        // cout << "file " << shift_filex << endl;
        oss << "%s%04i%s%s" << output_path.c_str( ), shift_file_index, shift_file_prefy.c_str( ), "ccmap.txt";
        std::snprintf(buffer, sizeof(buffer), "%s%04i%s%s", output_path.c_str( ), shift_file_index, shift_file_prefy.c_str( ), "ccmap.txt");
        // shift_file = oss.str( );
        shift_filey = buffer;
        // cout << "file " << shift_filey << endl;
        xoFile.open(shift_filex.c_str( ));
        yoFile.open(shift_filey.c_str( ));
        // yoFile.open(shifted_mapy_file.c_str( ));
        if ( xoFile.is_open( ) && yoFile.is_open( ) ) {
            // wxPrintf("files are open\n");
            // cout << "files are open" << endl;
            // float myarray[10][5760];
            for ( int i = 0; i < patch_no_y; i++ ) {
                for ( int j = 0; j < patch_no_x; j++ ) {
                    xoFile << Spline3dx.smooth_interp[image_ind](i, j) << '\t';
                    yoFile << Spline3dy.smooth_interp[image_ind](i, j) << '\t';
                    // yoFile << shifted_map_y[i][j] << '\t';
                }
                xoFile << '\n';
                yoFile << '\n';
            }
        }
        xoFile.close( );
        yoFile.close( );
    }
}