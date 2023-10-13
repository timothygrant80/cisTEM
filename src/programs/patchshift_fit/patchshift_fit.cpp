#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <iostream>
#include <string>
#include <fstream>
// #include "../../core/dlib/optimization.h"
#include "dlib/dlib/optimization.h"

// #include <iostream>
#include <vector>

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
    //     UserInput* my_input = new UserInput("TrimStack", 1.0);

    //     wxString input_imgstack = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    //     // // wxString output_stack_filename = my_input->GetFilenameFromUser("Filename for output stack of particles.", "A stack of particles will be written to disk", "particles.mrc", false);
    //     wxString angle_filename = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
    //     // wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    //     int img_index = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

    //     delete my_input;

    //     my_current_job.Reset(3);
    //     my_current_job.ManualSetArguments("tti", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ), img_index);
}

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

typedef matrix<double, 3, 1>  input_vector;
typedef matrix<double, 18, 1> parameter_vector;

// ----------------------------------------------------------------------------------------

double quadraticmodel(
        const input_vector&     input,
        const parameter_vector& params) {
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

double residual_quadraticmodel(
        const std::pair<input_vector, double>& data,
        const parameter_vector&                params) {
    return quadraticmodel(data.first, params) - data.second;
}

parameter_vector residual_derivative_quadraticmodel(
        const std::pair<input_vector, double>& data,
        const parameter_vector&                params) {
    parameter_vector der;

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

bool NikoTestApp::DoCalculation( ) {
    // wxString patch_shifts;

    wxString input_path  = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/PatchMovies_gain/";
    wxString output_path = "/data/lingli/Lingli_20221028/draft_tmp";
    int      image_no    = 75;
    int      patch_num_x = 6;
    int      patch_num_y = 4;
    int      patch_no    = patch_num_x * patch_num_y;
    float**  shiftsx     = NULL;
    float**  shiftsy     = NULL;

    Allocate2DFloatArray(shiftsx, image_no, patch_no);
    Allocate2DFloatArray(shiftsy, image_no, patch_no);
    wxString         shift_file;
    NumericTextFile* shiftfile;
    float            shifts[2];
    float            patch_locations[patch_no][2];
    for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
        shift_file = wxString::Format(input_path + "%02i_" + "shift.txt", patch_index);
        shiftfile  = new NumericTextFile(shift_file, OPEN_TO_READ, 2);
        for ( int image_index = 0; image_index < image_no; image_index++ ) {
            shiftfile->ReadLine(shifts);
            shiftsx[image_index][patch_index] = shifts[0];
            shiftsy[image_index][patch_index] = shifts[1];
            // wxPrintf("shifts: %f, %f\n", shiftsx[image_index][patch_index], shiftsy[image_index][patch_index]);
        }
    }
    int image_dim_x = 5760;
    int image_dim_y = 4092;
    int step_size_x = myroundint(float(image_dim_x) / float(patch_num_x) / 2);
    int step_size_y = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
    for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
        for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
            patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x;
            patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
            wxPrintf("patch locations: %f, %f\n", patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0], patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1]);
        }
    }
    input_vector                                 input;
    std::vector<std::pair<input_vector, double>> data_x, data_y;
    for ( int image_index = 0; image_index < image_no; image_index++ ) {
        for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
            float time;
            time     = image_index;
            input(0) = patch_locations[patch_index][0];
            input(1) = patch_locations[patch_index][1];
            input(2) = time;
            // double outputx = shiftsx[image_index][patch_index];
            // double outputy = shiftsy[image_index][patch_index];
            // the following is to set the first image has no shift
            double outputx = shiftsx[image_index][patch_index] - shiftsx[0][patch_index];
            double outputy = shiftsy[image_index][patch_index] - shiftsy[0][patch_index];
            data_x.push_back(make_pair(input, outputx));
            data_y.push_back(make_pair(input, outputy));
        }
    }
    parameter_vector x, y;
    x = 1;
    y = 1;

    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                           residual_quadraticmodel,
                           residual_derivative_quadraticmodel,
                           data_x,
                           x);
    wxPrintf("x fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12), x(13), x(14), x(15), x(16), x(17));

    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
                           residual_quadraticmodel,
                           residual_derivative_quadraticmodel,
                           data_y,
                           y);
    wxPrintf("y fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", y(0), y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8), y(9), y(10), y(11), y(12), y(13), y(14), y(15), y(16), y(17));

    delete shiftfile;
    return true;
}
