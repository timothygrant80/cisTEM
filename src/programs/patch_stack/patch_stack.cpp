#include "../../core/core_headers.h"

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
    UserInput* my_input = new UserInput("TrimStack", 1.0);

    wxString input_imgstack        = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    wxString angle_filename        = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
    wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    int      output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("ttti", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ), coordinates_filename.ToUTF8( ).data( ), output_stack_box_size);
}

bool NikoTestApp::DoCalculation( ) {
    wxPrintf("Hello world4\n");

    wxString input_imgstack        = my_current_job.arguments[0].ReturnStringArgument( );
    wxString angle_filename        = my_current_job.arguments[1].ReturnStringArgument( );
    wxString coordinates_filename  = my_current_job.arguments[2].ReturnStringArgument( );
    int      output_stack_box_size = my_current_job.arguments[3].ReturnIntegerArgument( );

    NumericTextFile *input_coos_file, *tilt_angle_file;
    input_coos_file = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
    tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);

    Image current_image;
    Image box;

    int          my_x, my_y;
    int          x_at_centertlt, y_at_centertlt;
    int          number_of_patchgroups = input_coos_file->number_of_lines;
    float        temp_angle[1];
    float        temp_array[number_of_patchgroups][2];
    MRCFile      input_stack(input_imgstack.ToStdString( ), false);
    MRCFile*     patch       = new MRCFile[number_of_patchgroups];
    int          image_no    = input_stack.ReturnNumberOfSlices( );
    ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));

    wxPrintf("image number in the stack: %i\n", image_no);

    box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);

    //write a if statement to judge if the number of coordinates in the coord file equals to image_no
    for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
        input_coos_file->ReadLine(temp_array[patch_counter]);
        patch[patch_counter].OpenFile(wxString::Format("%i.mrc", patch_counter).ToStdString( ), true);
    }

    wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
    for ( long image_counter = 0; image_counter < image_no; image_counter++ ) {
        current_image.ReadSlice(&input_stack, image_counter + 1);
        float image_mean = current_image.ReturnAverageOfRealValues( );

        tilt_angle_file->ReadLine(temp_angle);
        // my_image.crop( );

        for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

            x_at_centertlt = temp_array[patch_counter][0];
            y_at_centertlt = temp_array[patch_counter][1];
            my_x           = int(x_at_centertlt * cosf(PI * temp_angle[0] / 180.0));
            my_y           = y_at_centertlt;
            current_image.ClipInto(&box, image_mean, false, 1.0, int(my_x), int(my_y), 0);
            box.WriteSlice(&patch[patch_counter], image_counter + 1);
        }
        my_progress->Update(image_counter + 1);
    }

    delete my_progress;
    delete input_coos_file;
    delete[] patch;

    return true;
}
