#include "../../core/core_headers.h"

class
        ApplyGainRef : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ApplyGainRef)

// override the DoInteractiveUserInput

void ApplyGainRef::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("ApplyGainRef", 1.0);

    std::string input_dark_filename;

    std::string input_filename      = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);
    std::string output_filename     = my_input->GetFilenameFromUser("Output corrected image file name", "Filename of output image", "output.mrc", false);
    std::string input_gain_filename = my_input->GetFilenameFromUser("Input gain ref file", "Filename of gain reference", "gain.mrc", true);

    bool also_use_dark = my_input->GetYesNoFromUser("Also do dark correction?", "if yes, you can provide a dark image to be subtracted", "NO");

    if ( also_use_dark == true ) {
        input_dark_filename = my_input->GetFilenameFromUser("Input dark ref file", "Filename of dark reference", "dark.mrc", true);
    }

    bool should_resample = my_input->GetYesNoFromUser("Resample the output?", "if yes, you can resample the output", "NO");
    int  new_x_size      = 0;
    int  new_y_size      = 0;

    if ( should_resample == true ) {
        new_x_size = my_input->GetIntFromUser("Wanted New X-Size", "The image will be Fourier cropped to this size", "3838");
        new_y_size = my_input->GetIntFromUser("Wanted New Y-Size", "The image will be Fourier cropped to this size", "3710");
    }

    float num_sigmas             = 100;
    bool  should_remove_outliers = my_input->GetYesNoFromUser("Remove outlier pixels?", "If yes, outlier pixels will be removed AFTER gain correction, but prior to resampling", "NO");

    if ( should_remove_outliers == true ) {
        num_sigmas = my_input->GetFloatFromUser("Number of standard deviations", "Pixels more than this number of standard deviations above or below the mean will be reset to the mean", "12.0", 0.0f);
    }

    bool zero_float_and_normalize = my_input->GetYesNoFromUser("Also zero-float and normalize?", "After outlier pixels have been removed, zero-float and normalize images", "no");

    delete my_input;

    my_current_job.Reset(11);
    my_current_job.ManualSetArguments("tttbiibfbbt", input_filename.c_str( ),
                                      output_filename.c_str( ),
                                      input_gain_filename.c_str( ),
                                      should_resample,
                                      new_x_size,
                                      new_y_size,
                                      should_remove_outliers,
                                      num_sigmas,
                                      zero_float_and_normalize,
                                      also_use_dark,
                                      input_dark_filename.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool ApplyGainRef::DoCalculation( ) {

    std::string input_filename           = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename          = my_current_job.arguments[1].ReturnStringArgument( );
    std::string input_gain_filename      = my_current_job.arguments[2].ReturnStringArgument( );
    bool        should_resample          = my_current_job.arguments[3].ReturnBoolArgument( );
    int         new_x_size               = my_current_job.arguments[4].ReturnIntegerArgument( );
    int         new_y_size               = my_current_job.arguments[5].ReturnIntegerArgument( );
    bool        should_remove_outliers   = my_current_job.arguments[6].ReturnBoolArgument( );
    float       num_sigmas               = my_current_job.arguments[7].ReturnFloatArgument( );
    bool        zero_float_and_normalize = my_current_job.arguments[8].ReturnBoolArgument( );
    bool        also_use_dark            = my_current_job.arguments[9].ReturnBoolArgument( );
    std::string input_dark_filename      = my_current_job.arguments[10].ReturnStringArgument( );

    //wxFileName input_wx_filename(input_filename);

    ImageFile my_input_file(input_filename, false);
    MRCFile   my_output_file(output_filename, true);
    ImageFile my_gain_file(input_gain_filename, false);

    Image my_image;
    Image gain_reference;
    Image dark_reference;

    gain_reference.ReadSlice(&my_gain_file, 1);
    if ( also_use_dark == true )
        dark_reference.QuickAndDirtyReadSlice(input_dark_filename, 1);

    wxPrintf("\nCorrecting...\n\n");

    ProgressBar* my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices( ));

    for ( int image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices( ); image_counter++ ) {
        my_image.ReadSlice(&my_input_file, image_counter + 1);

        if ( also_use_dark == true ) {
            my_image.SubtractImage(&dark_reference);
        }

        my_image.MultiplyPixelWise(gain_reference);

        my_image.ReplaceOutliersWithMean(num_sigmas);

        if ( should_resample == true ) {
            my_image.ForwardFFT( );
            my_image.Resize(new_x_size, new_y_size, 1);
            my_image.BackwardFFT( );
        }

        if ( zero_float_and_normalize ) {
            my_image.ZeroFloatAndNormalize( );
        }

        my_image.WriteSlice(&my_output_file, image_counter + 1);
        my_progress->Update(image_counter + 1);
    }

    delete my_progress;
    wxPrintf("\n\nApply Gain finished cleanly!\n\n");

    return true;
}
