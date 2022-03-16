#include "../../core/core_headers.h"

class
        CombineViaMax : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CombineViaMax)

// override the DoInteractiveUserInput

void CombineViaMax::DoInteractiveUserInput( ) {

    int new_z_size = 1;

    UserInput* my_input = new UserInput("CombineViaMax", 1.0);

    std::string input_filename_one = my_input->GetFilenameFromUser("Input image file name #1", "Filename of 1st stack or volume to be combined", "input_stack1.mrc", true);
    std::string input_filename_two = my_input->GetFilenameFromUser("Input image file name #2", "Filename of 2nd stack or volume to be combined", "input_stack2.mrc", true);
    std::string output_filename    = my_input->GetFilenameFromUser("Output combined file name", "The combined result", "output.mrc", false);
    bool        use_absolute       = my_input->GetYesNoFromUser("Use Absolute?", "If yes the maximum absolute value is used to choose which to take", "YES");

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("tttb", input_filename_one.c_str( ), input_filename_two.c_str( ), output_filename.c_str( ), use_absolute);
}

// override the do calculation method which will be what is actually run..

bool CombineViaMax::DoCalculation( ) {
    long pixel_counter;

    std::string input_filename_one = my_current_job.arguments[0].ReturnStringArgument( );
    std::string input_filename_two = my_current_job.arguments[1].ReturnStringArgument( );
    std::string output_filename    = my_current_job.arguments[2].ReturnStringArgument( );
    bool        use_absolute       = my_current_job.arguments[3].ReturnBoolArgument( );

    ImageFile my_input_file_one(input_filename_one, false);
    ImageFile my_input_file_two(input_filename_two, false);
    MRCFile   my_output_file(output_filename, true);

    if ( my_input_file_one.ReturnXSize( ) != my_input_file_two.ReturnXSize( ) || my_input_file_one.ReturnYSize( ) != my_input_file_two.ReturnYSize( ) ) {
        MyPrintfRed("\n\n Error: Image dimensions are not the same\n\n");
        exit(-1);
    }

    if ( my_input_file_one.ReturnNumberOfSlices( ) != my_input_file_two.ReturnNumberOfSlices( ) ) {
        MyPrintfRed("\n\n Error: Two stack contain different number of images\n\n");
        exit(-1);
    }

    Image my_image_one;
    Image my_image_two;

    float input_pixel_size = my_input_file_one.ReturnPixelSize( );

    wxPrintf("\nCombining...\n\n");
    ProgressBar* my_progress = new ProgressBar(my_input_file_one.ReturnNumberOfSlices( ));

    for ( long image_counter = 0; image_counter < my_input_file_one.ReturnNumberOfSlices( ); image_counter++ ) {
        my_image_one.ReadSlice(&my_input_file_one, image_counter + 1);
        my_image_two.ReadSlice(&my_input_file_two, image_counter + 1);

        for ( pixel_counter = 0; pixel_counter < my_image_one.number_of_real_space_pixels; pixel_counter++ ) {
            if ( use_absolute == false )
                my_image_one.real_values[pixel_counter] = std::max(my_image_one.real_values[pixel_counter], my_image_two.real_values[pixel_counter]);
            else {
                if ( fabsf(my_image_two.real_values[pixel_counter]) > fabsf(my_image_one.real_values[pixel_counter]) )
                    my_image_one.real_values[pixel_counter] = my_image_two.real_values[pixel_counter];
            }
        }

        my_image_one.WriteSlice(&my_output_file, image_counter + 1);
        my_progress->Update(image_counter + 1);
    }

    delete my_progress;
    wxPrintf("\n\n");

    my_output_file.SetPixelSize(input_pixel_size);
    my_output_file.WriteHeader( );

    return true;
}
