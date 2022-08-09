#include "../../core/core_headers.h"

class
        MakeOrthViews : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeOrthViews)

// override the DoInteractiveUserInput

void MakeOrthViews::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("MakeOrthViews", 1.0);

    wxString input_volume = my_input->GetFilenameFromUser("Input image/volume file name", "Name of input image volume", "input.mrc", true);
    wxString output_image = my_input->GetFilenameFromUser("Output orth views image file name", "Name of output image ", "orth.mrc", false);

    delete my_input;

    my_current_job.Reset(2);
    my_current_job.ManualSetArguments("tt", input_volume.ToUTF8( ).data( ), output_image.ToUTF8( ).data( ));
}

// override the do calculation method which will be what is actually run..

bool MakeOrthViews::DoCalculation( ) {

    wxString input_volume = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_image = my_current_job.arguments[1].ReturnStringArgument( );

    MRCFile input3d_file(input_volume.ToStdString( ), false);
    MRCFile output_file(output_image.ToStdString( ), true);

    Image my_input_volume;
    Image my_orth_views_image;

    wxPrintf("\nMaking orth views image...\n");

    my_input_volume.ReadSlices(&input3d_file, 1, input3d_file.ReturnNumberOfSlices( ));
    my_orth_views_image.Allocate(my_input_volume.logical_x_dimension * 3, my_input_volume.logical_y_dimension * 2, 1, true);
    my_input_volume.CreateOrthogonalProjectionsImage(&my_orth_views_image);
    my_orth_views_image.WriteSlice(&output_file, 1);

    return true;
}
