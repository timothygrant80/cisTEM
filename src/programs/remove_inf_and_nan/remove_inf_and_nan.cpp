#include "../../core/core_headers.h"
#include <wx/dir.h>

class
        RemoveINFandNAN : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(RemoveINFandNAN)

// override the DoInteractiveUserInput

void RemoveINFandNAN::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("RemoveINFandNAN", 1.0);

    std::string input_filename  = my_input->GetFilenameFromUser("Input file name", "Filename of input image", "input.mrc", true);
    std::string output_filename = my_input->GetFilenameFromUser("Output MRC file name", "Filename of output image which should have no infs or nans", "output.mrc", false);

    delete my_input;

    my_current_job.Reset(2);
    my_current_job.ManualSetArguments("tt", input_filename.c_str( ), output_filename.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool RemoveINFandNAN::DoCalculation( ) {

    std::string input_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_filename = my_current_job.arguments[1].ReturnStringArgument( );

    ImageFile input_file;
    MRCFile   output_file;

    Image buffer_image;
    long  pixel_counter;
    long  images_with_inf_or_nan = 0;

    bool has_bad_pixel;

    input_file.OpenFile(input_filename, false);
    output_file.OpenFile(output_filename, true);

    wxPrintf("Checking File...\n\n");

    ProgressBar* my_progress = new ProgressBar(input_file.ReturnNumberOfSlices( ));

    for ( int counter = 1; counter <= input_file.ReturnNumberOfSlices( ); counter++ ) {
        buffer_image.ReadSlice(&input_file, counter);
        has_bad_pixel = false;

        for ( pixel_counter = 0; pixel_counter < buffer_image.real_memory_allocated; pixel_counter++ ) {
            if ( std::isnan(buffer_image.real_values[pixel_counter]) != 0 || std::isinf(buffer_image.real_values[pixel_counter]) != 0 ) {
                buffer_image.real_values[pixel_counter] = 0.0;
                has_bad_pixel                           = true;
            }
        }

        if ( has_bad_pixel == true )
            images_with_inf_or_nan++;

        buffer_image.WriteSlice(&output_file, counter);
        my_progress->Update(counter);
    }

    delete my_progress;

    wxPrintf("\n\n%li Images contained inf or nan\n\n", images_with_inf_or_nan);

    return true;
}
