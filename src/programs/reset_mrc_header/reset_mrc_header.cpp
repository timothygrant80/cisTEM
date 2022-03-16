#include "../../core/core_headers.h"

class
        ResetMRCHeaderApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ResetMRCHeaderApp)

// override the DoInteractiveUserInput

void ResetMRCHeaderApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("ResetMRCHeader", 1.0);

    std::string input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of input image. Its header will be reset.", "input.mrc", true);
    float       new_pixel_size = my_input->GetFloatFromUser("New pixel size", "New pixel size, in Angstroms", "1.0", 0.0);

    delete my_input;

    my_current_job.Reset(2);
    my_current_job.ManualSetArguments("tf", input_filename.c_str( ), new_pixel_size);
}

// override the do calculation method which will be what is actually run..

bool ResetMRCHeaderApp::DoCalculation( ) {

    std::string input_filename = my_current_job.arguments[0].ReturnStringArgument( );
    float       new_pixel_size = my_current_job.arguments[1].ReturnFloatArgument( );

    MRCFile my_input_file(input_filename, false);

    Image my_image;

    EmpiricalDistribution my_distribution;

    //wxPrintf("Resetting MRC header of file %s...\n",input_filename);

    ProgressBar* my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices( ));

    for ( int image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices( ); image_counter++ ) {
        my_progress->Update(image_counter + 1);
        my_image.ReadSlice(&my_input_file, image_counter + 1);
        my_image.UpdateDistributionOfRealValues(&my_distribution);
    }

    delete my_progress;

    float std = my_distribution.GetSampleVariance( );
    if ( std > 0.0 ) {
        std = sqrt(std);
    }
    my_input_file.my_header.SetDimensionsVolume(my_image.logical_x_dimension, my_image.logical_y_dimension, my_input_file.ReturnNumberOfSlices( ));
    my_input_file.my_header.SetDensityStatistics(my_distribution.GetMinimum( ), my_distribution.GetMaximum( ), my_distribution.GetSampleMean( ), std);
    my_input_file.my_header.SetPixelSize(new_pixel_size);
    my_input_file.my_header.ResetLabels( );
    my_input_file.my_header.SetOrigin(-float(my_image.physical_address_of_box_center_x) * new_pixel_size, -float(my_image.physical_address_of_box_center_y) * new_pixel_size, -float(my_input_file.ReturnNumberOfSlices( ) / 2) * new_pixel_size);

    my_input_file.WriteHeader( );

    wxPrintf("\nAll done.\n");

    return true;
}
