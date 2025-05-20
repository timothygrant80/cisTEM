#include "../../core/core_headers.h"

class
        FFTApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(FFTApp)

// override the DoInteractiveUserInput

void FFTApp::DoInteractiveUserInput( ) {

    int new_z_size = 1;

    UserInput* my_input = new UserInput("FFTApp", 1.0);

    std::string input_filename     = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);
    std::string output_filename    = my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false);
    bool        is_a_volume        = my_input->GetYesNoFromUser("Is the input a volume", "Yes if it is a 3D", "no");
    bool        enhance_thon_rings = my_input->GetYesNoFromUser("Enhance Thon rings?", "Manipulate grey values to enhance Thon rings?", "no");

    delete my_input;

    my_current_job.Reset(6);
    my_current_job.ManualSetArguments("ttbb", input_filename.c_str( ), output_filename.c_str( ), is_a_volume, enhance_thon_rings);
}

// override the do calculation method which will be what is actually run..

bool FFTApp::DoCalculation( ) {

    std::string input_filename     = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename    = my_current_job.arguments[1].ReturnStringArgument( );
    bool        is_a_volume        = my_current_job.arguments[2].ReturnBoolArgument( );
    bool        enhance_thon_rings = my_current_job.arguments[3].ReturnBoolArgument( );
    float       pixel_size;

    ImageFile my_input_file(input_filename, false);
    MRCFile   my_output_file(output_filename, true);

    Image                         my_image;
    Image                         my_amplitude_spectrum;
    EmpiricalDistribution<double> my_distribution;

    // pixel size could be non-square/cubic but we will ignore this here and assume it is square/cubic
    pixel_size = my_input_file.ReturnPixelSize( );

    if ( is_a_volume == true ) {
        SendError("3D FT is not yet supported by this program");
        wxPrintf("\nFourier transforming Volume...\n\n");
        my_image.ReadSlices(&my_input_file, 1, my_input_file.ReturnNumberOfSlices( ));

        my_image.ForwardFFT( );
        //my_image.ComputeAmplitudeSpectrumFull3D(my_amplitude_spectrum);
        //my_image.WriteSlices(&my_output_file,1, new_z_size);
    }
    else {
        wxPrintf("\nFourier transforming Images...\n\n");
        ProgressBar* my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices( ));

        for ( long image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices( ); image_counter++ ) {
            my_image.ReadSlice(&my_input_file, image_counter + 1);

            my_amplitude_spectrum.Allocate(my_image.logical_x_dimension, my_image.logical_y_dimension, true);

            my_image.ForwardFFT( );
            my_image.ComputeAmplitudeSpectrumFull2D(&my_amplitude_spectrum);

            if ( enhance_thon_rings ) {
                my_amplitude_spectrum.SetMaximumValue(my_amplitude_spectrum.ReturnAverageOfRealValuesAtRadius(10.0));
            }

            my_amplitude_spectrum.UpdateDistributionOfRealValues(&my_distribution);

            my_amplitude_spectrum.WriteSlice(&my_output_file, image_counter + 1);
            // next line is buggy because only the last image will determine image header stats
            my_progress->Update(image_counter + 1);
        }

        delete my_progress;
        wxPrintf("\n\n");
    }

    float std = my_distribution.GetSampleVariance( );
    if ( std > 0.0 ) {
        std = sqrt(std);
    }
    my_output_file.SetDensityStatistics(my_distribution.GetMinimum( ), my_distribution.GetMaximum( ), my_distribution.GetSampleMean( ), std);

    my_output_file.SetPixelSize(pixel_size);
    my_output_file.WriteHeader( );

    return true;
}
