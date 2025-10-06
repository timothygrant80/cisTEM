

#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <string>
#include <fstream>
#include <iostream>

// #include <vector>
#include <dlib/dlib/matrix.h>
#include "utilities.h"

// The timing that unblur originally tracks is always on, by direct reference to cistem_timer::StopWatch
// The profiling for development is under conrtol of --enable-profiling.
#ifdef CISTEM_PROFILING
using namespace cistem_timer;
#else
// #define PRINT_VERBOSE
using namespace cistem_timer_noop;
#endif

class
        ShiftFieldGeneration : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

using namespace dlib;
using namespace std;
typedef matrix<double, 0, 1> column_vector;

IMPLEMENT_APP(ShiftFieldGeneration)
void find_patch_locations(int image_dim_x, int image_dim_y, int patch_num_x, int patch_num_y, matrix<double>& patch_locations_x, matrix<double>& patch_locations_y);
void WriteShiftField(const std::string& file_path, float* shift_field, int image_dim_x, int image_dim_y);

void ShiftFieldGeneration::DoInteractiveUserInput( ) {
    // UserInput* my_input = new UserInput("TrimStack", 1.0);
    UserInput* my_input   = new UserInput("ShiftFieldGeneration", 1.0);
    wxString   outputpath = "";

    float       exposure_per_frame;
    int         total_number_of_frames;
    int         target_frame;
    int         image_dim_x;
    int         image_dim_y;
    int         image_dim_x_sup          = 0;
    int         image_dim_y_sup          = 0;
    std::string control_file_name_R0     = "";
    std::string control_file_name_R1     = "";
    std::string knot_parameter_file_name = "";

    bool generate_shiftfiled_for_supreso_frames = false;

    total_number_of_frames = my_input->GetIntFromUser("Total number of frames of the movie", "number of frames in the movie (integer)", "1", 1);
    target_frame           = my_input->GetIntFromUser("Frame  index you want to generate the shift field (start from 1)", "frame index (integer)", "1", 1);
    exposure_per_frame     = my_input->GetFloatFromUser("Exposure per frame (e/A^2)", "Exposure per frame, in electrons per square Angstrom", "1.0", 0.0);

    image_dim_x = my_input->GetIntFromUser("Image width along x", "Image width of your output micrograph (integer)", "1", 1);
    image_dim_y = my_input->GetIntFromUser("Image height along y", "Image height of your output micrograph (integer)", "1", 1);

    control_file_name_R0     = my_input->GetFilenameFromUser("Spline parameter filepath round 1", "The filepath of spline model parameters of 1st round of refinement", "Control_R0.txt", true);
    control_file_name_R1     = my_input->GetFilenameFromUser("Spline parameter filepath round 2", "The filepath of spline model parameters of 2nd round of refinement", "Control_R1.txt", true);
    knot_parameter_file_name = my_input->GetFilenameFromUser("Knot number filepath", "The filename of file saving the knot number and sample dose", "shift_loss.txt", true);

    outputpath = my_input->GetStringFromUser("Output folder for saving the shift filed", "output path", "/data/outpatch/");

    generate_shiftfiled_for_supreso_frames = my_input->GetYesNoFromUser("Generate shift field for super resolution frame?", "Do you want to generate shift field for the super resolution frames", "yes");
    if ( generate_shiftfiled_for_supreso_frames == true ) {
        image_dim_x_sup = my_input->GetIntFromUser("Width of super resolution movie frame (along x)", "Image width of the super resolution movie frame (integer)", "1", 1);
        image_dim_y_sup = my_input->GetIntFromUser("Height of super resolution movie frame (along x)", "Image height of the super resolution movie frame (integer)", "1", 1);
    }

    delete my_input;
    my_current_job.ManualSetArguments("iifiibiittts",
                                      total_number_of_frames,
                                      target_frame,
                                      exposure_per_frame,
                                      image_dim_x,
                                      image_dim_y,
                                      generate_shiftfiled_for_supreso_frames,
                                      image_dim_x_sup,
                                      image_dim_y_sup,
                                      control_file_name_R0.c_str( ),
                                      control_file_name_R1.c_str( ),
                                      knot_parameter_file_name.c_str( ),
                                      outputpath.ToUTF8( ).data( ));
}

bool ShiftFieldGeneration::DoCalculation( ) {

    int         number_of_input_images                 = my_current_job.arguments[0].ReturnIntegerArgument( );
    int         target_frame                           = my_current_job.arguments[1].ReturnIntegerArgument( );
    float       exposure_per_frame                     = my_current_job.arguments[2].ReturnFloatArgument( );
    int         image_dim_x                            = my_current_job.arguments[3].ReturnIntegerArgument( );
    int         image_dim_y                            = my_current_job.arguments[4].ReturnIntegerArgument( );
    bool        generate_shiftfiled_for_supreso_frames = my_current_job.arguments[5].ReturnBoolArgument( );
    int         image_dim_x_sup                        = my_current_job.arguments[6].ReturnIntegerArgument( );
    int         image_dim_y_sup                        = my_current_job.arguments[7].ReturnIntegerArgument( );
    std::string control_file_name_R0                   = my_current_job.arguments[8].ReturnStringArgument( );
    std::string control_file_name_R1                   = my_current_job.arguments[9].ReturnStringArgument( );
    std::string knot_parameter_file_name               = my_current_job.arguments[10].ReturnStringArgument( );
    wxString    outputpath                             = my_current_job.arguments[11].ReturnStringArgument( );

    // read the knot information from the shift loss file ==================================
    MovieFrameSpline Spline3dx_R0, Spline3dy_R0;
    MovieFrameSpline Spline3dx_R1, Spline3dy_R1;
    std::ifstream    input_file_stream(knot_parameter_file_name);
    if ( ! input_file_stream.is_open( ) ) {
        wxPrintf("\nError: Unable to open file %s", knot_parameter_file_name.c_str( ));
        return false;
    }

    std::string first_line;
    if ( ! std::getline(input_file_stream, first_line) ) {
        wxPrintf("\nError: Unable to read the knot information %s\n", knot_parameter_file_name.c_str( ));
    }

    std::istringstream line_stream(first_line);
    std::string        str1, str2, str3;
    double             knot_on_x, knot_on_y, sample_dose;
    if ( line_stream >> str1 >> str2 >> str3 >> knot_on_x >> knot_on_y >> sample_dose ) {
        wxPrintf("\nExtracted knot info:\n");
        wxPrintf("%s %f\n%s % f\n%s % f\n ", str1, knot_on_x, str2, knot_on_y, str3, sample_dose);
    }
    else {
        wxPrintf("\nError: Unable to extract knot information. \n");
    }
    input_file_stream.close( );

    // initialize the knot grids =========================================

    double knot_z_dis = sample_dose;
    double knot_x_dis = ceil(image_dim_x / (knot_on_x - 1));
    double knot_y_dis = ceil(image_dim_y / (knot_on_y - 1));

    double total_dose = exposure_per_frame * number_of_input_images;
    double knot_on_z  = ceil(total_dose / sample_dose) + 1;

    if ( knot_on_z > number_of_input_images ) {
        knot_on_z = number_of_input_images;
    }

    Spline3dx_R0.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
    Spline3dy_R0.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
    Spline3dx_R1.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);
    Spline3dy_R1.Initialize(knot_on_z, knot_on_y, knot_on_x, number_of_input_images, image_dim_x, image_dim_y, knot_z_dis, knot_x_dis, knot_y_dis);

    // interpolate knot grids on each frame =========================================
    matrix<double> z;
    z.set_size(number_of_input_images, 1);
    for ( int i = 0; i < number_of_input_images; i++ ) {
        z(i) = (i + 1) * exposure_per_frame;
    }
    Spline3dx_R0.Update3DSplineInterpFrames(z);
    Spline3dy_R0.Update3DSplineInterpFrames(z);
    Spline3dx_R1.Update3DSplineInterpFrames(z);
    Spline3dy_R1.Update3DSplineInterpFrames(z);

    // assign knot parameters to splines =========================================
    column_vector Join1d_R0, Join1d_R1;
    Join1d_R0 = read_join_file(control_file_name_R0, knot_on_x, knot_on_y, knot_on_z);
    Join1d_R1 = read_join_file(control_file_name_R1, knot_on_x, knot_on_y, knot_on_z);

    int            controlsize = Join1d_R0.size( );
    matrix<double> control_on_spline_for_x_R0, control_on_spline_for_y_R0;
    matrix<double> control_on_spline_for_x_R1, control_on_spline_for_y_R1;

    int halflen = controlsize / 2;
    control_on_spline_for_x_R0.set_size(halflen, 1);
    control_on_spline_for_y_R0.set_size(halflen, 1);
    control_on_spline_for_x_R1.set_size(halflen, 1);
    control_on_spline_for_y_R1.set_size(halflen, 1);

    control_on_spline_for_x_R0 = rowm(Join1d_R0, range(0, halflen - 1));
    control_on_spline_for_y_R0 = rowm(Join1d_R0, range(halflen, controlsize - 1));
    control_on_spline_for_x_R1 = rowm(Join1d_R1, range(0, halflen - 1));
    control_on_spline_for_y_R1 = rowm(Join1d_R1, range(halflen, controlsize - 1));

    Spline3dx_R0.Update3DSpline1dInputControlPoints(control_on_spline_for_x_R0);
    Spline3dy_R0.Update3DSpline1dInputControlPoints(control_on_spline_for_y_R0);
    Spline3dx_R1.Update3DSpline1dInputControlPoints(control_on_spline_for_x_R1);
    Spline3dy_R1.Update3DSpline1dInputControlPoints(control_on_spline_for_y_R1);

    // generate shifts for each frame =========================================
    int    totalpixels;
    float* original_map_x;
    float* original_map_y;
    float* shiftx_field;
    float* shifty_field;
    float  x_binning_float;
    float  y_binning_float;
    int    x_dimension;
    int    y_dimension;

    if ( generate_shiftfiled_for_supreso_frames ) {
        x_binning_float = float(image_dim_x_sup) / float(image_dim_x);
        y_binning_float = float(image_dim_y_sup) / float(image_dim_y);
        wxPrintf("x binning %f, y binning %f\n", x_binning_float, y_binning_float);
        totalpixels = image_dim_x_sup * image_dim_y_sup;
        x_dimension = image_dim_x_sup;
        y_dimension = image_dim_y_sup;
    }
    else {
        x_binning_float = 1.0;
        y_binning_float = 1.0;
        totalpixels     = image_dim_x * image_dim_y;
        x_dimension     = image_dim_x;
        y_dimension     = image_dim_y;
    }
    shiftx_field   = new float[totalpixels];
    shifty_field   = new float[totalpixels];
    original_map_x = new float[totalpixels];
    original_map_y = new float[totalpixels];

    int image_index = target_frame - 1;
    for ( int i = 0; i < y_dimension; i++ ) {
        for ( int j = 0; j < x_dimension; j++ ) {
            // original_map_x[i * image_dim_x + j] = float(j);
            // original_map_y[i * image_dim_x + j] = float(i);
            original_map_x[i * x_dimension + j] = float(j) / x_binning_float;
            original_map_y[i * x_dimension + j] = float(i) / y_binning_float;
        }
    }
    for ( int i = 0; i < y_dimension; i++ ) {
        for ( int j = 0; j < x_dimension; j++ ) {
            int pix           = i * x_dimension + j;
            shiftx_field[pix] = Spline3dx_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_index) * x_binning_float;
            shifty_field[pix] = Spline3dy_R0.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_index) * y_binning_float;
            shiftx_field[pix] = shifty_field[pix] + Spline3dx_R1.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_index) * x_binning_float;
            shifty_field[pix] = shifty_field[pix] + Spline3dy_R1.Apply3DSplineFunc(original_map_x[pix], original_map_y[pix], image_index) * y_binning_float;
        }
    }
    wxPrintf("Writing shift field to %s\n", outputpath.c_str( ));
    if ( generate_shiftfiled_for_supreso_frames ) {
        WriteShiftField(wxString::Format(outputpath + "shiftx_field_sup_%04i.txt", image_index).ToStdString( ), shiftx_field, image_dim_x_sup, image_dim_y_sup);
        WriteShiftField(wxString::Format(outputpath + "shifty_field_sup_%04i.txt", image_index).ToStdString( ), shifty_field, image_dim_x_sup, image_dim_y_sup);
    }
    else {
        WriteShiftField(wxString::Format(outputpath + "shiftx_field_%04i.txt", image_index).ToStdString( ), shiftx_field, image_dim_x, image_dim_y);
        WriteShiftField(wxString::Format(outputpath + "shifty_field_%04i.txt", image_index).ToStdString( ), shifty_field, image_dim_x, image_dim_y);
    }
    // // use patch place to debug the spline =========================================
    // matrix<double> patch_locations_x;
    // matrix<double> patch_locations_y;
    // int patch_num_x = 12;
    // int patch_num_y = 8;

    // find_patch_locations(image_dim_x, image_dim_y, patch_num_x, patch_num_y, patch_locations_x, patch_locations_y);
    // for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
    //     wxPrintf("x loc %f\n", patch_locations_x(patch_x_ind));
    // }
    // for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
    //     wxPrintf("y loc %f\n", patch_locations_y(patch_y_ind));
    // }
    // float shx, shy;
    // target_frame = number_of_input_images - 1;
    // for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
    //     for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
    //         shx = Spline3dx_R0.Apply3DSplineFunc(patch_locations_x(patch_x_ind), patch_locations_y(patch_y_ind), target_frame);
    //         wxPrintf("%f\t", shx);
    //     }
    //     wxPrintf("\n");
    // }
    // wxPrintf("\n");
    // for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
    //     for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
    //         shy = Spline3dy_R0.Apply3DSplineFunc(patch_locations_x(patch_x_ind), patch_locations_y(patch_y_ind), target_frame);
    //         wxPrintf("%f\t", shy);
    //     }
    //     wxPrintf("\n");
    // }
    return true;
}

void find_patch_locations(int image_dim_x, int image_dim_y, int patch_num_x, int patch_num_y, matrix<double>& patch_locations_x, matrix<double>& patch_locations_y) {

    int step_size_x = myroundint(float(image_dim_x) / float(patch_num_x) / 2);
    int step_size_y = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
    patch_locations_x.set_size(patch_num_x, 1);
    patch_locations_y.set_size(patch_num_y, 1);
    for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
        patch_locations_x(patch_x_ind) = patch_x_ind * step_size_x * 2 + step_size_x;
    }
    for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
        patch_locations_y(patch_num_y - patch_y_ind - 1) = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
    }
}

void WriteShiftField(const std::string& file_path, float* shift_field, int image_dim_x, int image_dim_y) {
    std::ofstream out_file(file_path);
    out_file.precision(6);

    if ( out_file.is_open( ) ) {
        for ( int i = 0; i < image_dim_y; i++ ) {
            for ( int j = 0; j < image_dim_x; j++ ) {
                int pix = i * image_dim_x + j;
                out_file << shift_field[pix];
                if ( j < image_dim_x - 1 ) {
                    out_file << '\t';
                }
            }
            out_file << '\n';
        }
        out_file.close( );
        wxPrintf("shift_field written to %s\n", file_path.c_str( ));
    }
    else {
        wxPrintf("Error: Unable to open file %s for writing\n", file_path.c_str( ));
    }
}