#include "../../core/core_headers.h"

class
        FilterImages : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(FilterImages)

// override the DoInteractiveUserInput

void FilterImages::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("FilterImages", 1.0);

    std::string input_filename  = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);
    std::string output_filename = my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false);
    int         filter_type     = my_input->GetIntFromUser("Filter type: 1=highpass, 2=lowpass, 3=bandpass", "Highpass, Lowpass, or Bandpass", "1", 1, 3);
    float       sigma_one       = my_input->GetFloatFromUser("First sigma value", "distribution value from 0 to 0.5, for highpass side of bandpass", "0.25", 0, 0.5);
    float       sigma_two       = 0;
    if ( filter_type == 3 )
        sigma_two = my_input->GetFloatFromUser("Second sigma value", "for lowpass side of bandpass", "0.25", 0, 0.5);

    delete my_input;

    my_current_job.Reset(5);
    my_current_job.ManualSetArguments("ttiff", input_filename.c_str( ), output_filename.c_str( ), filter_type, sigma_one, sigma_two);
}

// override the do calculation method which will be what is actually run..

bool FilterImages::DoCalculation( ) {

    std::string input_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename = my_current_job.arguments[1].ReturnStringArgument( );
    int         filter_type     = my_current_job.arguments[2].ReturnIntegerArgument( );
    float       sigma_one       = my_current_job.arguments[3].ReturnFloatArgument( );
    float       sigma_two       = my_current_job.arguments[4].ReturnFloatArgument( );
    float       pixel_size;

    MRCFile my_input_file(input_filename, false);
    MRCFile my_output_file(output_filename, true);

    Image my_image;

    pixel_size = my_input_file.ReturnPixelSize( );

    if ( filter_type == 1 ) {
        wxPrintf("\nApplying Highpass filter...\n\n");

        my_image.ReadSlice(&my_input_file, 1);
        my_image.ForwardFFT( );
        my_image.GaussianHighPassFilter(sigma_one);
        my_image.BackwardFFT( );
        my_image.WriteSlice(&my_output_file, 1);
    }
    else if ( filter_type == 2 ) {
        wxPrintf("\nApplying Lowpass filter...\n\n");

        my_image.ReadSlice(&my_input_file, 1);
        my_image.ForwardFFT( );
        my_image.GaussianLowPassFilter(sigma_one);
        my_image.BackwardFFT( );
        my_image.WriteSlice(&my_output_file, 1);
    }
    else if ( filter_type == 3 ) {
        wxPrintf("\nApplying Bandpass filter...\n\n");

        my_image.ReadSlice(&my_input_file, 1);
        my_image.ForwardFFT( );
        my_image.GaussianHighPassFilter(sigma_one);
        my_image.GaussianLowPassFilter(sigma_two);
        my_image.BackwardFFT( );
        my_image.WriteSlice(&my_output_file, 1);

        wxPrintf("\n\n");
    }

    else {
        wxPrintf("\nNot a valid filter number, try again\n\n");
    }
    my_output_file.SetPixelSize(pixel_size);
    my_output_file.WriteHeader( );

    return true;
}
