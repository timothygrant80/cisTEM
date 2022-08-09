#include "../../core/core_headers.h"

class
        InvertHand : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(InvertHand)

// override the DoInteractiveUserInput

void InvertHand::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("InvertHand", 1.0);

    wxString input_volume  = my_input->GetFilenameFromUser("Input volume file name", "Name of input image file", "input.mrc", true);
    wxString output_volume = my_input->GetFilenameFromUser("Output inverted volume file name", "Name of output image with mask applied", "output.mrc", false);

    delete my_input;

    //	my_current_job.Reset(9);
    my_current_job.ManualSetArguments("tt", input_volume.ToUTF8( ).data( ), output_volume.ToUTF8( ).data( ));
}

// override the do calculation method which will be what is actually run..

bool InvertHand::DoCalculation( ) {

    wxString input_volume  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_volume = my_current_job.arguments[1].ReturnStringArgument( );

    MRCFile input3d_file(input_volume.ToStdString( ), false);
    MRCFile output_file(output_volume.ToStdString( ), true);

    Image my_image;

    if ( input3d_file.ReturnZSize( ) > 1 )
        wxPrintf("\nInverting Volume Handedness...\n");

    my_image.ReadSlices(&input3d_file, 1, input3d_file.ReturnNumberOfSlices( ));
    my_image.InvertHandedness( );
    my_image.WriteSlices(&output_file, 1, input3d_file.ReturnNumberOfSlices( ));

    return true;
}
