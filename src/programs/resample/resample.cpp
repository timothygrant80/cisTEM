#include "../../core/core_headers.h"

class
        Resample : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(Resample)

// override the DoInteractiveUserInput

void Resample::DoInteractiveUserInput( ) {

    int new_z_size = 1;

    UserInput* my_input = new UserInput("Resample", 1.0);

    std::string input_filename       = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);
    std::string output_filename      = my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false);
    bool        is_a_volume          = my_input->GetYesNoFromUser("Is the input a volume", "Yes if it is a 3D", "NO");
    float       input_pixel_size     = my_input->GetFloatFromUser("Input pixel size in angstroms", "", "1.0", 0.0, 100.0);
    float       output_pixel_size    = my_input->GetFloatFromUser("Wanted output pixel size in angstroms", "", "1.0", 0.0, 100.0);
    float       pixel_size_tolerance = my_input->GetFloatFromUser("Resampling tolerance", "allowed tolerance in the resampled pixel size", "0.001", 0.00001, 0.1);

    delete my_input;

    my_current_job.Reset(6);
    my_current_job.ManualSetArguments("ttbfff", input_filename.c_str( ), output_filename.c_str( ), is_a_volume, input_pixel_size, output_pixel_size, pixel_size_tolerance);
}

// override the do calculation method which will be what is actually run..

bool Resample::DoCalculation( ) {

    std::string input_filename       = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename      = my_current_job.arguments[1].ReturnStringArgument( );
    bool        is_a_volume          = my_current_job.arguments[2].ReturnBoolArgument( );
    float       input_pixel_size     = my_current_job.arguments[3].ReturnFloatArgument( );
    float       output_pixel_size    = my_current_job.arguments[4].ReturnFloatArgument( );
    float       pixel_size_tolerance = my_current_job.arguments[5].ReturnFloatArgument( );

    ImageFile my_input_file(input_filename, false);
    MRCFile   my_output_file(output_filename, true);

    Image                         my_image, tmp_image;
    EmpiricalDistribution<double> my_distribution;

    float actual_factor = 0.0f;
    float wanted_factor = output_pixel_size / input_pixel_size;

    if ( is_a_volume == true ) {

        wxPrintf("\nResampling Volume...\n\n");
        my_image.ReadSlices(&my_input_file, 1, my_input_file.ReturnNumberOfSlices( ));
        tmp_image = my_image;

        actual_factor = tmp_image.ChangePixelSize(&my_image, wanted_factor, pixel_size_tolerance);

        my_image.WriteSlices(&my_output_file, 1, my_input_file.ReturnNumberOfSlices( ));
    }
    else {
        wxPrintf("\nResampling Images...\n\n");
        ProgressBar* my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices( ));

        my_image.Allocate(my_input_file.ReturnXSize( ), my_input_file.ReturnYSize( ), 1);
        for ( long image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices( ); image_counter++ ) {
            tmp_image.ReadSlice(&my_input_file, image_counter + 1);

            actual_factor = tmp_image.ChangePixelSize(&my_image, wanted_factor, pixel_size_tolerance);

            my_image.WriteSlice(&my_output_file, image_counter + 1);
            my_image.UpdateDistributionOfRealValues(&my_distribution);
            my_progress->Update(image_counter + 1);
        }

        delete my_progress;
        wxPrintf("\n\n");
    }

    wxPrintf("Wanted scaling factor (output pixel size / input pixel size) %f\n", wanted_factor);
    wxPrintf("Actual scaling factor %f\n", actual_factor);

    float std = my_distribution.GetSampleVariance( );
    if ( std > 0.0 ) {
        std = sqrt(std);
    }
    my_output_file.SetDensityStatistics(my_distribution.GetMinimum( ), my_distribution.GetMaximum( ), my_distribution.GetSampleMean( ), std);
    my_output_file.SetPixelSize(input_pixel_size * actual_factor);
    my_output_file.WriteHeader( );

    return true;
}
