#include "../../core/core_headers.h"

class
        PatchMovie : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(PatchMovie)

// override the DoInteractiveUserInput

void PatchMovie::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("TrimMovie", 1.0);

    wxString input_imgstack = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    wxString outputpath     = my_input->GetStringFromUser("Output path", "output movie patchs path", "/data/outpatch/");
    // std::string outputpath = my_input->GetStringFromUser("Output path", "output movie patchs path", "/data/outpatch/");

    bool     Use_Patch_coordinate_file = my_input->GetYesNoFromUser("Have pre asigned patch coordinate file?", "Have predifined coordinates for patches", "no");
    wxString coordinates_filename      = "";
    int      patch_num_x               = 0;
    int      patch_num_y               = 0;
    if ( Use_Patch_coordinate_file == true ) {
        wxString coordinates_filename = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    }
    else {
        patch_num_x = my_input->GetIntFromUser("Number of patches along x dimension", "Integer", "6", 0);
        patch_num_y = my_input->GetIntFromUser("Number of patches along y dimension", "Integer", "4", 0);
    }

    int output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);
    delete my_input;

    my_current_job.Reset(7);
    my_current_job.ManualSetArguments("tsbtiii", input_imgstack.ToUTF8( ).data( ), outputpath.ToUTF8( ).data( ), Use_Patch_coordinate_file, coordinates_filename.ToUTF8( ).data( ), patch_num_x, patch_num_y, output_stack_box_size);
    // my_current_job.ManualSetArguments("tsbtiii", input_imgstack.ToUTF8( ).data( ), outputpath, Use_Patch_coordinate_file, coordinates_filename.ToUTF8( ).data( ), patch_num_x, patch_num_y, output_stack_box_size);
}

bool PatchMovie::DoCalculation( ) {
    wxPrintf("Hello world4\n");

    wxString input_imgstack = my_current_job.arguments[0].ReturnStringArgument( );
    wxString outputpath     = my_current_job.arguments[1].ReturnStringArgument( );
    // std::string outpath               = my_current_job.arguments[1].ReturnStringArgument( );
    bool     use_coord_file        = my_current_job.arguments[2].ReturnBoolArgument( );
    wxString coordinates_filename  = my_current_job.arguments[3].ReturnStringArgument( );
    int      patch_num_x           = my_current_job.arguments[4].ReturnIntegerArgument( );
    int      patch_num_y           = my_current_job.arguments[5].ReturnIntegerArgument( );
    int      output_stack_box_size = my_current_job.arguments[6].ReturnIntegerArgument( );

    // wxPrintf(wxString::Format("%s", outputpath.ToStdString( )));
    // wxPrintf("%s", outputpath.ToStdString( ));
    // wxString outputpath = wxString::Format("%s", outpath).ToStdString( );
    wxPrintf("output path: %s\n", outputpath.ToStdString( ));

    // wxString outputpath = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test/";

    // tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);

    // NumericTextFile* shifts_file;
    // shifts_file = new NumericTextFile(shifts_filename, OPEN_TO_READ, 3);

    Image current_image;
    Image box;

    int my_x, my_y;
    int x_at_centertlt, y_at_centertlt;
    // int          number_of_patchgroups = input_coos_file->number_of_lines;
    // float        temp_angle[1];
    int      number_of_patchgroups = patch_num_x * patch_num_y;
    float    temp_array[number_of_patchgroups][2];
    MRCFile  input_stack(input_imgstack.ToStdString( ), false);
    MRCFile* patch    = new MRCFile[number_of_patchgroups];
    int      image_no = input_stack.ReturnNumberOfSlices( );
    int      image_dim_x;
    int      image_dim_y;
    image_dim_x = input_stack.ReturnXSize( );
    image_dim_y = input_stack.ReturnYSize( );

    ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));

    wxPrintf("image number in the stack: %i\n", image_no);

    box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);

    //write a if statement to judge if the number of coordinates in the coord file equals to image_no
    if ( use_coord_file ) {
        NumericTextFile* input_coos_file;
        input_coos_file = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
        for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

            input_coos_file->ReadLine(temp_array[patch_counter]);
            patch[patch_counter].OpenFile(wxString::Format(outputpath + "%02i.mrc", patch_counter).ToStdString( ), true);
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
                patch[patch_counter].OpenFile(wxString::Format(outputpath + "%02i.mrc", patch_counter).ToStdString( ), true);
                patch_counter += 1;
            }
        }
    }

    wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
    // float phi          = 6;
    // float orth_phi     = phi + 180;
    // float new_phi      = phi + 90;
    // float cos_orth_phi = cosf(deg_2_rad(orth_phi));
    // float sin_orth_phi = sinf(deg_2_rad(orth_phi));

    // float shifts[image_no][3];
    //---------------------------------- loading the shifts into an array -------------------------

    // for ( int i = 0; i < image_no; i++ ) {
    //     shifts_file->ReadLine(shifts[i]);
    //     // wxPrintf("angle %i ; % g\n", i, tilts[i]);
    // }

    for ( long image_counter = 0; image_counter < image_no; image_counter++ ) {
        current_image.ReadSlice(&input_stack, image_counter + 1);
        // current_image.TaperEdges( );
        // float cos_edge = 0.1 * wxMin(current_image.logical_x_dimension, current_image.logical_y_dimension);
        // current_image.CosineRectangularMask(current_image.logical_x_dimension / 2.0 - float(cos_edge / 2.0), current_image.logical_y_dimension / 2.0 - float(cos_edge / 2.0), 1, float(cos_edge));

        float image_mean = current_image.ReturnAverageOfRealValues( );
        // wxPrintf("the mean of the current image is %f \n", image_mean);

        // tilt_angle_file->ReadLine(temp_angle);
        // my_image.crop( );

        for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

            x_at_centertlt = temp_array[patch_counter][0];
            y_at_centertlt = temp_array[patch_counter][1];
            // // my_x           = int(x_at_centertlt * cosf(PI * temp_angle[0] / 180.0));
            // // my_y           = y_at_centertlt;

            // // the following is to put the axis to the new crop coordinate center
            // float proj_len = x_at_centertlt * cos_orth_phi + y_at_centertlt * sin_orth_phi;
            // float proj_x   = proj_len * cos_orth_phi;
            // float proj_y   = proj_len * sin_orth_phi;
            // float diff_len = sqrt(pow(proj_x, 2) + pow(proj_y, 2)) * (1 - cosf(deg_2_rad(temp_angle[0])));
            // // float diff_x   = diff_len * cos_orth_phi; this is incorrect
            // // float diff_y   = diff_len * sin_orth_phi;
            // float diff_x = diff_len * proj_x / abs(proj_len);
            // float diff_y = diff_len * proj_y / abs(proj_len);
            // wxPrintf("prj_x, prj_y: %g, %g", proj_x, proj_y);
            // wxPrintf("diff_x, diff_y: %g, %g\n", diff_x, diff_y);
            // my_x = myroundint(x_at_centertlt - diff_x);
            // my_y = myroundint(y_at_centertlt - diff_y);

            // // add the shifts from the coarse align results
            // my_x = myroundint(x_at_centertlt - diff_x - shifts[image_counter][1]);
            // my_y = myroundint(y_at_centertlt - diff_y - shifts[image_counter][2]);
            // end coordinate transform

            my_x = x_at_centertlt;
            my_y = y_at_centertlt;
            // wxPrintf("x and y %d, %d \n", my_x, my_y);

            current_image.ClipInto(&box, image_mean, false, 1.0, int(my_x), int(my_y), 0);
            box.WriteSlice(&patch[patch_counter], image_counter + 1);
        }
        my_progress->Update(image_counter + 1);
    }

    delete my_progress;
    // delete input_coos_file;
    delete[] patch;

    return true;
}
