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

    UserInput* my_input = new UserInput("RemoveOutlierPixels", 1.0);

    std::string input_filename           = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);
    std::string output_filename          = my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false);
    float       num_sigmas               = my_input->GetFloatFromUser("Number of standard deviations", "Pixels more than this number of standard deviations above or below the mean will be reset to the mean", "6.0", 0.0f);
    bool        zero_float_and_normalize = my_input->GetYesNoFromUser("Also zero-float and normalize?", "After outlier pixels have been removed, zero-float and normalize images", "no");

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("ttfb", input_filename.c_str( ), output_filename.c_str( ), num_sigmas, zero_float_and_normalize);
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation( ) {

    std::string input_filename           = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename          = my_current_job.arguments[1].ReturnStringArgument( );
    float       num_sigmas               = my_current_job.arguments[2].ReturnFloatArgument( );
    bool        zero_float_and_normalize = my_current_job.arguments[3].ReturnBoolArgument( );

    MRCFile my_input_file(input_filename, false);
    MRCFile my_output_file(output_filename, true);

    Image my_image;

    ProgressBar my_progress(my_input_file.ReturnNumberOfSlices( ));

    for ( int image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices( ); image_counter++ ) {
        my_progress.Update(image_counter + 1);
        my_image.ReadSlice(&my_input_file, image_counter + 1);
        my_image.ReplaceOutliersWithMean(num_sigmas);
        if ( zero_float_and_normalize ) {
            my_image.ZeroFloatAndNormalize( );
        }
        my_image.WriteSlice(&my_output_file, image_counter + 1);
    }

    return true;
}
