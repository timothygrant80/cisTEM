#include "../../core/core_headers.h"

class
        CropAndAddNoiseApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CropAndAddNoiseApp)

// override the DoInteractiveUserInput

void CropAndAddNoiseApp::DoInteractiveUserInput( ) {
    wxString   input_image;
    wxString   output_image;
    UserInput* my_input = new UserInput("CropAndAddNoiseApp", 1.05);

    input_image  = my_input->GetFilenameFromUser("Input image", "The input particle image stack, containing the 2D images for each particle in the dataset", "input.mrc", true);
    output_image = my_input->GetFilenameFromUser("Output image", "The input particle image stack, containing the 2D images for each particle in the dataset", "output.mrc", false);

    delete my_input;

    my_current_job.ManualSetArguments("tt", input_image.ToUTF8( ).data( ), output_image.ToUTF8( ).data( ));
}

// override the do calculation method which will be what is actually run..

bool CropAndAddNoiseApp::DoCalculation( ) {
    wxString input_image  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_image = my_current_job.arguments[1].ReturnStringArgument( );

    Image test_image;
    test_image.QuickAndDirtyReadSlice(input_image.ToUTF8( ).data( ), 1);
    std::tuple<int, int> crop_location = test_image.CropAndAddGaussianNoiseToDarkAreas( );
    wxPrintf("%s %s %i %i \n", input_image.mb_str( ), output_image.mb_str( ), std::get<0>(crop_location), std::get<1>(crop_location));
    test_image.QuickAndDirtyWriteSlice(output_image.ToUTF8( ).data( ), 1, true);
    return true;
}
