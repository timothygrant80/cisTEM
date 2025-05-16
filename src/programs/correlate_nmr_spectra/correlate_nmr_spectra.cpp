#include "../../core/core_headers.h"

class
        CorrelateNMRSpectra : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CorrelateNMRSpectra)

// override the DoInteractiveUserInput

void CorrelateNMRSpectra::DoInteractiveUserInput( ) {

    int   filter_type_1  = 1;
    int   filter_type_2  = 1;
    float sigma_one      = 0;
    float sigma_two      = 0;
    float sigma_three    = 0;
    float sigma_four     = 0;
    int   x_coord_center = 0;
    int   y_coord_center = 0;
    int   box_size       = 0;
    int   diagonal_width = 0;

    UserInput* my_input = new UserInput("CorrelateNMRSpectra", 1.0);

    std::string input_filename_1  = my_input->GetFilenameFromUser("Input first NMR txt file", "Filename of first NMR txt file", "input.txt", true);
    std::string input_filename_2  = my_input->GetFilenameFromUser("Input second NMR txt file", "Filename of second NMR txt file", "input.txt", true);
    std::string output_filename_1 = my_input->GetFilenameFromUser("Output image file 1 (no extension)", "Image 1, no extension because there are multiple versions", "output", false);
    std::string output_filename_2 = my_input->GetFilenameFromUser("Output image file 2 (no extension)", "Image 2, no extension because there are multiple versions", "output", false);
    bool        crop_spectra      = my_input->GetYesNoFromUser("Crop spectra?", "Should the spectra be cropped?", "NO");
    bool        remove_diagonal   = my_input->GetYesNoFromUser("Remove diagonal from cropped spectra?", "set the diagonal equal to the average value of the image?", "NO");
    bool        filter_spectra    = my_input->GetYesNoFromUser("Filter spectra?", "Should the spectra be filtered?", "NO");

    if ( crop_spectra == true ) {
        x_coord_center = my_input->GetIntFromUser("x-coordinate of box center, center of image is 0,0", "900 for CC2d", "0");
        y_coord_center = my_input->GetIntFromUser("y-coordinate of box center, center of image is 0,0", "900 for CC2d", "0");
        box_size       = my_input->GetIntFromUser("size of cropped box", "box size in pixels, 900 for CC2d", "0");
    }

    if ( crop_spectra & remove_diagonal == true ) {
        diagonal_width = my_input->GetIntFromUser("width of diagonal", "width size in pixels, around 20 for CC2d", "0");
    }
    if ( filter_spectra == true ) {
        filter_type_1 = my_input->GetIntFromUser("Filter type for first spectra: 1=highpass, 2=lowpass, 3=bandpass", "Highpass, Lowpass, or Bandpass", "1", 1, 3);
        filter_type_2 = my_input->GetIntFromUser("Filter type for second spectra: 1=highpass, 2=lowpass, 3=bandpass", "Highpass, Lowpass, or Bandpass", "1", 1, 3);
        sigma_one     = my_input->GetFloatFromUser("First sigma value for first spectra", "distribution value from 0 to 0.5, for highpass side of bandpass", "0.25", 0, 0.5);
        sigma_two     = my_input->GetFloatFromUser("Second sigma value for first spectra", "for lowpass side of bandpass", "0.25", 0, 0.5);
        sigma_three   = my_input->GetFloatFromUser("First sigma value for second spectra", "distribution value from 0 to 0.5, for highpass side of bandpass", "0.25", 0, 0.5);
        sigma_four    = my_input->GetFloatFromUser("Second sigma value for second spectra", "for lowpass side of bandpass", "0.25", 0, 0.5);
    }

    delete my_input;

    my_current_job.Reset(17);
    my_current_job.ManualSetArguments("ttttbbbiiiiiiffff", input_filename_1.c_str( ),
                                      input_filename_2.c_str( ),
                                      output_filename_1.c_str( ),
                                      output_filename_2.c_str( ),
                                      crop_spectra,
                                      remove_diagonal,
                                      filter_spectra,
                                      x_coord_center,
                                      y_coord_center,
                                      box_size,
                                      diagonal_width,
                                      filter_type_1,
                                      filter_type_2,
                                      sigma_one,
                                      sigma_two,
                                      sigma_three,
                                      sigma_four);
}

// override the do calculation method which will be what is actually run..

bool CorrelateNMRSpectra::DoCalculation( ) {

    std::string input_filename_1  = my_current_job.arguments[0].ReturnStringArgument( );
    std::string input_filename_2  = my_current_job.arguments[1].ReturnStringArgument( );
    std::string output_filename_1 = my_current_job.arguments[2].ReturnStringArgument( );
    std::string output_filename_2 = my_current_job.arguments[3].ReturnStringArgument( );
    bool        crop_spectra      = my_current_job.arguments[4].ReturnBoolArgument( );
    bool        remove_diagonal   = my_current_job.arguments[5].ReturnBoolArgument( );
    bool        filter_spectra    = my_current_job.arguments[6].ReturnBoolArgument( );
    int         x_coord_center    = my_current_job.arguments[7].ReturnIntegerArgument( );
    int         y_coord_center    = my_current_job.arguments[8].ReturnIntegerArgument( );
    int         box_size          = my_current_job.arguments[9].ReturnIntegerArgument( );
    int         diagonal_width    = my_current_job.arguments[10].ReturnIntegerArgument( );
    int         filter_type_1     = my_current_job.arguments[11].ReturnIntegerArgument( );
    int         filter_type_2     = my_current_job.arguments[12].ReturnIntegerArgument( );
    float       sigma_one         = my_current_job.arguments[13].ReturnFloatArgument( );
    float       sigma_two         = my_current_job.arguments[14].ReturnFloatArgument( );
    float       sigma_three       = my_current_job.arguments[15].ReturnFloatArgument( );
    float       sigma_four        = my_current_job.arguments[16].ReturnFloatArgument( );
    float       pixel_size_1;
    float       pixel_size_2;

    // convert the NMR spectra to mrc files

    std::ifstream my_input_file_1(input_filename_1); // Open file using ifstream
    if ( ! my_input_file_1.is_open( ) ) {
        std::cerr << "Error opening file: " << input_filename_1 << std::endl;
        return false;
    }
    std::ifstream my_input_file_2(input_filename_2); // Open file using ifstream
    if ( ! my_input_file_2.is_open( ) ) {
        std::cerr << "Error opening file: " << input_filename_2 << std::endl;
        return false;
    }

    // Create output filenames
    std::string output_filename_1_mrc      = output_filename_1 + ".mrc";
    std::string output_filename_2_mrc      = output_filename_2 + ".mrc";
    std::string output_filename_1_filtered = output_filename_1 + "_filtered.mrc";
    std::string output_filename_2_filtered = output_filename_2 + "_filtered.mrc";
    std::string output_filename_1_cropped  = output_filename_1 + "_cropped.mrc";
    std::string output_filename_2_cropped  = output_filename_2 + "_cropped.mrc";

    MRCFile my_output_file_1(output_filename_1_mrc, true);
    MRCFile my_output_file_2(output_filename_2_mrc, true);
    MRCFile my_output_file_1_filtered(output_filename_1_filtered, true);
    MRCFile my_output_file_2_filtered(output_filename_2_filtered, true);
    MRCFile my_output_file_1_cropped(output_filename_1_cropped, true);
    MRCFile my_output_file_2_cropped(output_filename_2_cropped, true);

    Image my_image_1;
    Image my_image_2;
    Image cropped_image_1;
    Image cropped_image_2;
    Image rotation_image_1;
    Image rotation_image_2;
    long  value_counter;
    long  pixel_counter;
    int   padding_jump_value = 0;
    int   x, y;
    float value;

    std::vector<int>   x_pos_1;
    std::vector<int>   y_pos_1;
    std::vector<float> value_1;

    std::vector<int>   x_pos_2;
    std::vector<int>   y_pos_2;
    std::vector<float> value_2;

    int max_x_1 = -INT_MAX;
    int max_y_1 = -INT_MAX;
    int max_x_2 = -INT_MAX;
    int max_y_2 = -INT_MAX;

    // spectra 1 conversion

    std::string line;
    while ( std::getline(my_input_file_1, line) ) {
        if ( sscanf(line.c_str( ), "%d %d %f", &x, &y, &value) == 3 ) {
            x_pos_1.push_back(x);
            y_pos_1.push_back(y);
            value_1.push_back(value);
        }
    }

    my_input_file_1.close( ); // Close the file after reading

    // After loop, calculate max values:

    max_x_1 = *std::max_element(x_pos_1.begin( ), x_pos_1.end( ));
    max_y_1 = *std::max_element(y_pos_1.begin( ), y_pos_1.end( ));

    wxPrintf("\nImage #1 has a dimension of %i,%i\n", max_x_1, max_y_1);

    my_image_1.Allocate(max_x_1, max_y_1, 1);

    pixel_counter = 0;
    value_counter = 0;
    for ( y = 0; y < my_image_1.logical_y_dimension; y++ ) {
        for ( x = 0; x < my_image_1.logical_x_dimension; x++ ) {

            my_image_1.real_values[pixel_counter] = value_1[value_counter];
            value_counter++;
            pixel_counter++;
        }

        pixel_counter += my_image_1.padding_jump_value;
    }

    // spectra 2 conversion
    std::string line_2;
    while ( std::getline(my_input_file_2, line_2) ) {
        if ( sscanf(line_2.c_str( ), "%d %d %f", &x, &y, &value) == 3 ) {
            x_pos_2.push_back(x);
            y_pos_2.push_back(y);
            value_2.push_back(value);
        }
    }

    my_input_file_2.close( ); // Close the file after reading

    // After loop, calculate max values:

    max_x_2 = *std::max_element(x_pos_2.begin( ), x_pos_2.end( ));
    max_y_2 = *std::max_element(y_pos_2.begin( ), y_pos_2.end( ));
    wxPrintf("\nImage #2 has a dimension of %i,%i\n", max_x_2, max_y_2);

    my_image_2.Allocate(max_x_2, max_y_2, 1);

    pixel_counter = 0;
    value_counter = 0;
    for ( y = 0; y < my_image_2.logical_y_dimension; y++ ) {
        for ( x = 0; x < my_image_2.logical_x_dimension; x++ ) {

            my_image_2.real_values[pixel_counter] = value_2[value_counter];
            value_counter++;
            pixel_counter++;
        }

        pixel_counter += my_image_2.padding_jump_value;
    }

    my_image_1.WriteSlice(&my_output_file_1, 1);
    my_image_2.WriteSlice(&my_output_file_2, 1);

    // filter NMR spectra if they want to

    pixel_size_1 = my_output_file_1.ReturnPixelSize( );
    pixel_size_2 = my_output_file_2.ReturnPixelSize( );

    if ( filter_spectra ) {

        // filter first spectra

        if ( filter_type_1 == 1 ) {
            wxPrintf("\nApplying Highpass filter...\n");

            my_image_1.ForwardFFT( );
            my_image_1.GaussianHighPassFilter(sigma_one);
            my_image_1.BackwardFFT( );
            my_image_1.WriteSlice(&my_output_file_1_filtered, 1);
        }
        else if ( filter_type_1 == 2 ) {
            wxPrintf("\nApplying Lowpass filter...\n");

            my_image_1.ForwardFFT( );
            my_image_1.GaussianLowPassFilter(sigma_one);
            my_image_1.BackwardFFT( );
            my_image_1.WriteSlice(&my_output_file_1_filtered, 1);
        }
        else if ( filter_type_1 == 3 ) {
            wxPrintf("\nApplying Bandpass filter...\n");

            my_image_1.ForwardFFT( );
            my_image_1.GaussianHighPassFilter(sigma_one);
            my_image_1.GaussianLowPassFilter(sigma_two);
            my_image_1.BackwardFFT( );
            my_image_1.WriteSlice(&my_output_file_1_filtered, 1);

            wxPrintf("\n\n");
        }

        else {
            wxPrintf("\nNot a valid filter number, try again\n");
        }

        // filter second spectra

        if ( filter_type_2 == 1 ) {
            wxPrintf("\nApplying Highpass filter...\n");

            my_image_2.ForwardFFT( );
            my_image_2.GaussianHighPassFilter(sigma_three);
            my_image_2.BackwardFFT( );
            my_image_2.WriteSlice(&my_output_file_2_filtered, 1);
        }
        else if ( filter_type_2 == 2 ) {
            wxPrintf("\nApplying Lowpass filter...\n");

            my_image_2.ForwardFFT( );
            my_image_2.GaussianLowPassFilter(sigma_three);
            my_image_2.BackwardFFT( );
            my_image_2.WriteSlice(&my_output_file_2_filtered, 1);
        }
        else if ( filter_type_2 == 3 ) {
            wxPrintf("\nApplying Bandpass filter...\n");

            my_image_2.ForwardFFT( );
            my_image_2.GaussianHighPassFilter(sigma_three);
            my_image_2.GaussianLowPassFilter(sigma_four);
            my_image_2.BackwardFFT( );
            my_image_2.WriteSlice(&my_output_file_2_filtered, 1);

            wxPrintf("\n\n");
        }

        else {
            wxPrintf("\nNot a valid filter number, try again\n");
        }

        my_output_file_1_filtered.SetPixelSize(pixel_size_1);
        my_output_file_1_filtered.WriteHeader( );
        my_output_file_2_filtered.SetPixelSize(pixel_size_2);
        my_output_file_2_filtered.WriteHeader( );
    }

    // crop spectra

    if ( crop_spectra == true ) {

        cropped_image_1.Allocate(box_size, box_size, true);
        cropped_image_2.Allocate(box_size, box_size, true);
        my_image_1.ClipInto(&cropped_image_1, 0, false, 1.0, x_coord_center, y_coord_center, 0);
        my_image_2.ClipInto(&cropped_image_2, 0, false, 1.0, x_coord_center, y_coord_center, 0);
        cropped_image_1.WriteSlice(&my_output_file_1_cropped, 1);
        cropped_image_2.WriteSlice(&my_output_file_2_cropped, 1);

        my_output_file_1_cropped.SetPixelSize(pixel_size_1);
        my_output_file_1_cropped.WriteHeader( );
        my_output_file_2_cropped.SetPixelSize(pixel_size_2);
        my_output_file_2_cropped.WriteHeader( );
    }

    // remove diagonal from CC2d

    if ( remove_diagonal == true ) {

        // calculate average value to fill in the diagonal
        float average_value_1 = 0;
        float average_value_2 = 0;

        my_image_1.ZeroFloatAndNormalize( );
        my_image_2.ZeroFloatAndNormalize( );
        average_value_1 = my_image_1.ReturnAverageOfRealValues( );
        average_value_2 = my_image_2.ReturnAverageOfRealValues( );
        wxPrintf("\nAverage value of Image #1 is %f\n", average_value_1);
        wxPrintf("\nAverage value of Image #2 is %f\n", average_value_2);

        int half_diagonal_width = diagonal_width / 2;

        for ( y = 0; y < cropped_image_1.logical_y_dimension; y++ ) {
            for ( x = 0; x < cropped_image_1.logical_x_dimension; x++ ) {
                if ( std::abs(x - y) <= half_diagonal_width ) {
                    int index_1 = cropped_image_1.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                    int index_2 = cropped_image_2.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);

                    // Set values to the average for the defined diagonal region
                    cropped_image_1.real_values[index_1] = average_value_1;
                    cropped_image_2.real_values[index_2] = average_value_2;
                }
            }
        }
        // write them out to same place as if you cropped them *removing the diagonal only works if you crop correctly so it must be a response to that initial user input anyways
        cropped_image_1.WriteSlice(&my_output_file_1_cropped, 1);
        my_output_file_1_cropped.SetPixelSize(pixel_size_1);
        my_output_file_1_cropped.WriteHeader( );

        cropped_image_2.WriteSlice(&my_output_file_2_cropped, 1);
        my_output_file_2_cropped.SetPixelSize(pixel_size_2);
        my_output_file_2_cropped.WriteHeader( );
    }

    // correlate NMR spectra with CC
    if ( crop_spectra == true ) {
        wxPrintf("\nCorrelation is %f\n", cropped_image_1.NormalizedCrossCorrelation(&cropped_image_2));
    }
    else {
        wxPrintf("\nCorrelation is %f\n", my_image_1.NormalizedCrossCorrelation(&my_image_2));
    }

    // correlate NMR spectra by FSC *haven't done yet* - Colin
    return true;
}
