#include "../../core/core_headers.h"

class
        AlignNMR : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(AlignNMR)

// override the DoInteractiveUserInput

void AlignNMR::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("AlignNMR", 1.2);

    std::string input_spectrum1_filename = my_input->GetFilenameFromUser("Input Spectrum 1 ft2 filename", "Filename of input ft2 file", "spectra1.ft2", true);
    std::string input_spectrum2_filename = my_input->GetFilenameFromUser("Input Spectrum 2 ft2 filename", "Filename of input ft2 file", "spectra2.ft2", true);
    std::string shifted_spectrum         = my_input->GetFilenameFromUser("Output shifted_spectrum.mrc", "Filename of output spectrum", "output.mrc", false);

    delete my_input;

    my_current_job.Reset(3);
    my_current_job.ManualSetArguments("ttt", input_spectrum1_filename.c_str( ), input_spectrum2_filename.c_str( ), shifted_spectrum.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool AlignNMR::DoCalculation( ) {
    std::string input_spectrum1_filename = my_current_job.arguments[0].ReturnStringArgument( );
    std::string input_spectrum2_filename = my_current_job.arguments[1].ReturnStringArgument( );
    std::string shifted_spectrum         = my_current_job.arguments[2].ReturnStringArgument( );

    MRCFile my_output_file(shifted_spectrum, true);

    Peak  my_peak;
    Image spec1;
    Image spec2;
    Image shift;
    Image spec2_copy;

    // get the spectra..

    std::ifstream my_input_file1(input_spectrum1_filename); // Open file using ifstream
    if ( ! my_input_file1.is_open( ) ) {
        std::cerr << "Error opening file: " << input_spectrum1_filename << std::endl;
        return false;
    }

    std::ifstream my_input_file2(input_spectrum2_filename); // Open file using ifstream
    if ( ! my_input_file2.is_open( ) ) {
        std::cerr << "Error opening file: " << input_spectrum2_filename << std::endl;
        return false;
    }

    // use NMRPipe to create txt and hdr file

    wxPrintf("\nStarting NMRPipe conversion to text and header files, this process requires NMRPipe commands, may take some time with large spectra\n");

    std::string txt1_command = "tcsh -c \"pipe2txt.tcl " + input_spectrum1_filename + " > spectra1.txt\"";
    std::string txt2_command = "tcsh -c \"pipe2txt.tcl " + input_spectrum2_filename + " > spectra2.txt\"";

    wxPrintf("\nConverting Spectra 1 to .txt...\n");

    system(txt1_command.c_str( ));

    wxPrintf("\nSpectra 1 converted successfully\n");
    wxPrintf("\nConverting Spectra 2 to .txt...\n");

    system(txt2_command.c_str( ));
    wxPrintf("\nSpectra 2 converted successfully\n");

    std::string hdr1_command = "tcsh -c \"showhdr " + input_spectrum1_filename + " > spectra1.hdr\"";
    std::string hdr2_command = "tcsh -c \"showhdr " + input_spectrum2_filename + " > spectra2.hdr\"";

    wxPrintf("\nCreating header files...\n");
    system(hdr1_command.c_str( ));
    wxPrintf("\nheader file 1 complete\n");
    system(hdr2_command.c_str( ));
    wxPrintf("\nheader file 2 complete\n");

    // first spectrum

    long value_counter      = 0;
    long pixel_counter      = 0;
    int  padding_jump_value = 0;
    int  x, y;

    std::vector<int>   x_pos_1;
    std::vector<int>   y_pos_1;
    std::vector<float> value_1;

    int max_x_1 = -INT_MAX;
    int max_y_1 = -INT_MAX;

    std::ifstream my_txt_file1("spectra1.txt"); // Open file using ifstream
    if ( ! my_txt_file1.is_open( ) ) {
        std::cerr << "Error opening file: spectra1.txt" << std::endl;
        return false;
    }

    std::string line;
    while ( std::getline(my_txt_file1, line) ) {
        float value;
        if ( sscanf(line.c_str( ), "%d %d %f", &x, &y, &value) == 3 ) {
            x_pos_1.push_back(x);
            y_pos_1.push_back(y);
            value_1.push_back(value);
        }
    }

    my_txt_file1.close( ); // Close the file after reading

    // After loop, calculate max values:

    max_x_1 = *std::max_element(x_pos_1.begin( ), x_pos_1.end( ));
    max_y_1 = *std::max_element(y_pos_1.begin( ), y_pos_1.end( ));

    wxPrintf("\nSpectrum #1 has a dimension of %i, %i\n", max_x_1, max_y_1);

    spec1.Allocate(max_x_1, max_y_1, 1);

    pixel_counter = 0;
    value_counter = 0;
    for ( y = 0; y < spec1.logical_y_dimension; y++ ) {
        for ( x = 0; x < spec1.logical_x_dimension; x++ ) {

            spec1.real_values[pixel_counter] = value_1[value_counter];
            value_counter++;
            pixel_counter++;
        }

        pixel_counter += spec1.padding_jump_value;
    }

    // second spectrum..

    std::vector<int>   x_pos_2;
    std::vector<int>   y_pos_2;
    std::vector<float> value_2;

    int max_x_2 = -INT_MAX;
    int max_y_2 = -INT_MAX;

    std::ifstream my_txt_file2("spectra2.txt"); // Open file using ifstream
    if ( ! my_txt_file2.is_open( ) ) {
        std::cerr << "Error opening file: spectra2.txt" << std::endl;
        return false;
    }

    while ( std::getline(my_txt_file2, line) ) {
        float value;
        if ( sscanf(line.c_str( ), "%d %d %f", &x, &y, &value) == 3 ) {
            x_pos_2.push_back(x);
            y_pos_2.push_back(y);
            value_2.push_back(value);
        }
    }

    my_txt_file2.close( ); // Close the file after reading

    // After loop, calculate max values:

    max_x_2 = *std::max_element(x_pos_2.begin( ), x_pos_2.end( ));
    max_y_2 = *std::max_element(y_pos_2.begin( ), y_pos_2.end( ));

    wxPrintf("Spectrum #2 has a dimension of %i, %i\n", max_x_2, max_y_2);

    spec2.Allocate(max_x_2, max_y_2, 1);

    pixel_counter = 0;
    value_counter = 0;
    for ( y = 0; y < spec2.logical_y_dimension; y++ ) {
        for ( x = 0; x < spec2.logical_x_dimension; x++ ) {

            spec2.real_values[pixel_counter] = value_2[value_counter];
            value_counter++;
            pixel_counter++;
        }

        pixel_counter += spec2.padding_jump_value;
    }

    // get header information from spectra 1..

    std::ifstream headerFile1("spectra1.hdr");
    std::string   lineheader;
    double        spec1_sw_x = 0, spec1_sw_y = 0, spec1_obs_x = 0, spec1_obs_y = 0, spec1_origin_x = 0, spec1_origin_y = 0;

    if ( ! headerFile1 ) {
        std::cerr << "Unable to open header file!" << std::endl;
        return 1;
    }

    while ( std::getline(headerFile1, lineheader) ) {
        if ( lineheader.find("SW Hz") != std::string::npos ) {
            // Skip "SW Hz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec1_sw_x >> spec1_sw_y; // Skip "SW" and "Hz:" to get the values
        }
        if ( lineheader.find("OBS MHz") != std::string::npos ) {
            // Skip "OBS MHz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec1_obs_x >> spec1_obs_y; // Skip "OBS" and "MHz:" to get the values
        }
        if ( lineheader.find("ORIG Hz") != std::string::npos ) {
            // Skip "ORIG Hz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec1_origin_x >> spec1_origin_y; // Skip "ORIG" and "Hz:" to get the values
        }
    }

    headerFile1.close( );

    // get header information from spectra 2..

    std::ifstream headerFile2("spectra2.hdr");
    double        spec2_sw_x = 0, spec2_sw_y = 0, spec2_obs_x = 0, spec2_obs_y = 0, spec2_origin_x = 0, spec2_origin_y = 0;

    if ( ! headerFile2 ) {
        std::cerr << "Unable to open header file!" << std::endl;
        return 1;
    }

    while ( std::getline(headerFile2, lineheader) ) {
        if ( lineheader.find("SW Hz") != std::string::npos ) {
            // Skip "SW Hz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec2_sw_x >> spec2_sw_y; // Skip "SW" and "Hz:" to get the values
        }
        if ( lineheader.find("OBS MHz") != std::string::npos ) {
            // Skip "OBS MHz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec2_obs_x >> spec2_obs_y; // Skip "OBS" and "MHz:" to get the values
        }
        if ( lineheader.find("ORIG Hz") != std::string::npos ) {
            // Skip "ORIG Hz:" part explicitly
            std::string        label;
            std::istringstream iss(lineheader);
            iss >> label >> label >> spec2_origin_x >> spec2_origin_y; // Skip "ORIG" and "Hz:" to get the values
        }
    }

    headerFile2.close( );

    // Output the extracted values
    wxPrintf("SW Hz1:   X = %.4f Y = %.4f\n", spec1_sw_x, spec1_sw_y);
    wxPrintf("OBS MHz1: X = %.4f Y = %.4f\n", spec1_obs_x, spec1_obs_y);
    wxPrintf("ORIG Hz1: X = %.4f Y = %.4f\n", spec1_origin_x, spec1_origin_y);
    wxPrintf("SW Hz2:   X = %.4f Y = %.4f\n", spec2_sw_x, spec2_sw_y);
    wxPrintf("OBS MHz2: X = %.4f Y = %.4f\n", spec2_obs_x, spec2_obs_y);
    wxPrintf("ORIG Hz2: X = %.4f Y = %.4f\n", spec2_origin_x, spec2_origin_y);

    // calculate sampling..

    float spectrum1_x_sampling = (spec1_sw_x / spec1_obs_x) / float(spec1.logical_x_dimension);
    float spectrum1_y_sampling = (spec1_sw_y / spec1_obs_y) / float(spec1.logical_y_dimension);

    float spectrum2_x_sampling = (spec2_sw_x / spec2_obs_x) / float(spec2.logical_x_dimension);
    float spectrum2_y_sampling = (spec2_sw_y / spec2_obs_y) / float(spec2.logical_y_dimension);

    wxPrintf("Spectrum 1 sampling = X: %.7f PPM, Y: %.7f PPM\n", spectrum1_x_sampling, spectrum1_y_sampling);
    wxPrintf("Spectrum 2 sampling = X: %.7f PPM, Y: %.7f PPM\n", spectrum2_x_sampling, spectrum2_y_sampling);

    int new_x_dimension = spec2.logical_x_dimension;
    int new_y_dimension = spec2.logical_y_dimension;

    // resample if they are not the same dimensions in sampling

    if ( spectrum1_x_sampling != spectrum2_x_sampling ) {
        new_x_dimension = spec2.logical_x_dimension / (spectrum1_x_sampling / spectrum2_x_sampling);
        wxPrintf("Resampling X spectrum 2\n");
        if ( new_x_dimension % 2 ) {
            new_x_dimension--;
        }

        spec2.ForwardFFT( );
        spec2.Resize(new_x_dimension, spec2.logical_y_dimension, 1.0f, 0.0f);
        spec2.BackwardFFT( );
    }

    if ( spectrum1_y_sampling != spectrum2_y_sampling ) {
        new_y_dimension = spec2.logical_y_dimension / (spectrum1_y_sampling / spectrum2_y_sampling);
        wxPrintf("Resampling Y spectrum 2\n");
        if ( new_y_dimension % 2 ) {
            new_y_dimension--;
        }

        spec2.ForwardFFT( );
        spec2.Resize(spec2.logical_x_dimension, new_y_dimension, 1.0f, 0.0f);
        spec2.BackwardFFT( );
    }

    wxPrintf("New Spec 2 X dimension = %i, New Spec 2 Y dimension = %i\n", new_x_dimension, new_y_dimension);

    // find origin difference
    float spec1_x_origin_ppm = spec1_origin_x / spec1_obs_x;
    float spec1_y_origin_ppm = spec1_origin_y / spec1_obs_y;
    float spec2_x_origin_ppm = spec2_origin_x / spec2_obs_x;
    float spec2_y_origin_ppm = spec2_origin_y / spec2_obs_y;

    // calculate the center origin
    float spec1_x_center = spec1_x_origin_ppm + (int((spec1.logical_x_dimension + 1) / 2) * spectrum1_x_sampling);
    float spec2_x_center = spec2_x_origin_ppm + (int((spec2.logical_x_dimension + 1) / 2) * spectrum1_x_sampling);

    float spec1_y_center = spec1_y_origin_ppm + (int((spec1.logical_y_dimension + 1) / 2) * spectrum1_y_sampling);
    float spec2_y_center = spec2_y_origin_ppm + (int((spec2.logical_y_dimension + 1) / 2) * spectrum1_y_sampling);

    // find the center origin offset
    float center_x_origin_offset = (spec1_x_center - spec2_x_center);
    float center_y_origin_offset = (spec1_y_center - spec2_y_center);

    wxPrintf("Spectrum 1 Origin : %.4f, %.4f PPM\n", spec1_x_origin_ppm, spec1_y_origin_ppm);
    wxPrintf("Spectrum 2 Origin : %.4f, %.4f PPM\n", spec2_x_origin_ppm, spec2_y_origin_ppm);
    wxPrintf("Spectrum 1 Center Origin : %.4f, %.4f PPM\n", spec1_x_center, spec1_y_center);
    wxPrintf("Spectrum 2 Center Origin : %.4f, %.4f PPM\n", spec2_x_center, spec2_y_center);
    wxPrintf("Center Origin Offset    : %.4f, %.4f PPM\n", center_x_origin_offset, center_y_origin_offset);

    if ( spec1.HasSameDimensionsAs(&spec2) == false ) {
        wxPrintf("Sizes are different, resizing Spectrum 2 to %i, %i\n", spec1.logical_x_dimension, spec1.logical_y_dimension);
        spec2.Resize(spec1.logical_x_dimension, spec1.logical_y_dimension, spec2.logical_z_dimension);
    }

    // normalize so padding operations have average value (0) and for correlations.
    spec1.ZeroFloatAndNormalize( );
    spec2.ZeroFloatAndNormalize( );

    shift.CopyFrom(&spec1);
    spec2_copy.CopyFrom(&spec2);

    float original_score = spec1.ReturnCorrelationCoefficientUnnormalized(spec2_copy);

    // high pass filter the spectra for better correlations
    spec1.ForwardFFT( );
    spec2.ForwardFFT( );
    spec1.GaussianHighPassFilter(0.1);
    spec2.GaussianHighPassFilter(0.1);

    spec1.CalculateCrossCorrelationImageWith(&spec2);
    my_peak = spec1.FindPeakWithParabolaFit( );

    float shift_x_ppm           = -my_peak.x * spectrum1_x_sampling;
    float corrected_shift_x_ppm = shift_x_ppm + (center_x_origin_offset);
    float shift_y_ppm           = -my_peak.y * spectrum1_y_sampling;
    float corrected_shift_y_ppm = shift_y_ppm + (center_y_origin_offset);

    // shift image, calculate scores
    shift.PhaseShift(-my_peak.x, -my_peak.y);
    shift.ZeroFloatAndNormalize( );
    shift.WriteSlice(&my_output_file, 1);

    float final_score = shift.ReturnCorrelationCoefficientUnnormalized(spec2_copy);

    wxPrintf("\nShift =  %.2f, %.2f pixels (%.4f, %.4f PPM)", -my_peak.x, -my_peak.y, shift_x_ppm, shift_y_ppm, my_peak.value);
    wxPrintf("\nStarting / Final CC Score : %.2f / %.2f", original_score, final_score);
    wxPrintf("\nShift Corrected for Origin Offset = %.4f, %.4f PPM\n", corrected_shift_x_ppm, corrected_shift_y_ppm);

    return true;
}
