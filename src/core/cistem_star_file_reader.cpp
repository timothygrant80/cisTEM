#include "core_headers.h"

using c_ft = cistem::fundamental_type::Enum;

//TODO : Currently, any strings with spaces will cause the star file (but not binary) reader to break - needs to be fixed.

// ADDING A NEW COLUMN
// ----------------------
// See top of cistem_parameters.cpp for documentation describing how to add a new column

cisTEMStarFileReader::cisTEMStarFileReader( ) {
    Reset( );
}

cisTEMStarFileReader::cisTEMStarFileReader(std::string wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer, bool exclude_negative_film_numbers) {
    Reset( );

    if ( alternate_cached_parameters_pointer == NULL ) {
        cached_parameters    = new ArrayOfcisTEMParameterLines;
        using_external_array = false;
    }
    else {
        cached_parameters    = alternate_cached_parameters_pointer;
        using_external_array = true;
    }

    ReadTextFile(wanted_filename, NULL, alternate_cached_parameters_pointer, exclude_negative_film_numbers);
}

void cisTEMStarFileReader::Reset( ) {
    filename                = "";
    input_file_is_opened    = false;
    current_line_number     = -1;
    binary_file_read_buffer = NULL;
    binary_file_size        = 0;

    current_position_in_stack = 0;
    current_column            = 0;

    cached_parameters    = NULL;
    using_external_array = false;

    ResetColumnPositions( );

    parameters_that_were_read.SetAllToFalse( );
}

void cisTEMStarFileReader::ResetColumnPositions( ) {
    position_in_stack_column                  = -1;
    image_is_active_column                    = -1;
    psi_column                                = -1;
    theta_column                              = -1;
    phi_column                                = -1;
    x_shift_column                            = -1;
    y_shift_column                            = -1;
    defocus_1_column                          = -1;
    defocus_2_column                          = -1;
    defocus_angle_column                      = -1;
    phase_shift_column                        = -1;
    occupancy_column                          = -1;
    logp_column                               = -1;
    sigma_column                              = -1;
    score_column                              = -1;
    score_change_column                       = -1;
    pixel_size_column                         = -1;
    microscope_voltage_kv_column              = -1;
    microscope_spherical_aberration_mm_column = -1;
    amplitude_contrast_column                 = -1;
    beam_tilt_x_column                        = -1;
    beam_tilt_y_column                        = -1;
    image_shift_x_column                      = -1;
    image_shift_y_column                      = -1;
    best_2d_class_column                      = -1;
    beam_tilt_group_column                    = -1;
    particle_group_column                     = -1;
    assigned_subset_column                    = -1;
    pre_exposure_column                       = -1;
    total_exposure_column                     = -1;
    original_image_filename_column            = -1;
    reference_3d_filename_column              = -1;
    stack_filename_column                     = -1;
    original_x_position_column                = -1;
    original_y_position_column                = -1;
}

cisTEMStarFileReader::~cisTEMStarFileReader( ) {
    Close( );
}

void cisTEMStarFileReader::Open(std::string wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer, bool read_as_binary) {
    Close( );

    if ( cached_parameters != NULL && using_external_array == false ) {
        delete cached_parameters;
        cached_parameters = NULL;
    }

    filename = wanted_filename;

    if ( read_as_binary == false ) // read it as a text based star file
    {
        std::ifstream input_stream(wanted_filename);

        if ( input_stream.is_open( ) == true ) {
            std::string current_file_line;

            while ( std::getline(input_stream, current_file_line) ) {
                if ( ! current_file_line.empty( ) && current_file_line.back( ) == '\r' )
                    current_file_line.pop_back( );
                input_file_lines.push_back(current_file_line);
            }

            input_file_is_opened = true;
        }

        if ( input_file_is_opened == false ) {
            MyPrintWithDetails("Error: Cannot open star file (%s) for read\n", wanted_filename);
            DEBUG_ABORT;
        }
    }
    else // binary input - read the whole file into memory..
    {
        FILE* binary_file = fopen(filename.c_str( ), "rb");
        fseek(binary_file, 0, SEEK_END);
        binary_file_size = ftell(binary_file);
        fseek(binary_file, 0, SEEK_SET);

        if ( binary_file_read_buffer != NULL )
            delete[] binary_file_read_buffer;
        binary_file_read_buffer = new char[binary_file_size];
        fread(binary_file_read_buffer, 1, binary_file_size, binary_file);
        fclose(binary_file);
    }

    if ( alternate_cached_parameters_pointer == NULL ) {
        cached_parameters    = new ArrayOfcisTEMParameterLines;
        using_external_array = false;
    }
    else {
        cached_parameters    = alternate_cached_parameters_pointer;
        using_external_array = true;
    }
}

void cisTEMStarFileReader::Close( ) {
    if ( cached_parameters != NULL && using_external_array == false ) {
        delete cached_parameters;
        cached_parameters = NULL;
    }

    input_file_lines.clear( );
    input_file_is_opened = false;
    current_line_number  = -1;

    if ( binary_file_read_buffer != NULL ) {
        delete[] binary_file_read_buffer;
        binary_file_read_buffer = NULL;
        binary_file_size        = -1;
    }

    binary_file_size = 0;
}

bool cisTEMStarFileReader::ExtractParametersFromLine(std::string& wanted_line, std::string* error_string, bool exclude_negative_film_numbers) {
    /*! \brief Parse a line read from a star file with checks on numeric convertibility.
	 *
	 * 	Detailed:
	 *		Each potential variable has a default column i.d. of -1. If this is not changed, this method will not try to parse the variable,
	 *		but instead will set it's respective default.
	 */

    // extract info.

    std::vector<std::string> all_tokens = SplitString(wanted_line);
    cisTEMParameterLine      temp_parameters;

    double temp_double;
    long   temp_long;

    std::string current_token;
    std::string string_buffer;

    // start with image is active, because sometimes we can just stop..

    // image is active

    if ( image_is_active_column == -1 )
        temp_parameters.image_is_active = 1.0;
    else {
        if ( StringToLong(all_tokens[image_is_active_column], temp_long) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_is_active_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[image_is_active_column]);
            return false;
        }

        temp_parameters.image_is_active = int(temp_long);
    }

    if ( temp_parameters.image_is_active < 0 && exclude_negative_film_numbers == true )
        return true;

    // position in stack

    if ( position_in_stack_column == -1 )
        temp_double = -1;
    else if ( StringToDouble(all_tokens[position_in_stack_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[position_in_stack_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[position_in_stack_column]);
        return false;
    }

    temp_parameters.position_in_stack = int(temp_double);

    // phi

    if ( phi_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[phi_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[phi_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[phi_column]);
        return false;
    }

    temp_parameters.phi = float(temp_double);

    // theta

    if ( theta_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[theta_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[theta_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[theta_column]);
        return false;
    }

    temp_parameters.theta = float(temp_double);

    // psi

    if ( psi_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[psi_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[psi_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[psi_column]);
        return false;
    }

    temp_parameters.psi = float(temp_double);

    // xshift

    if ( x_shift_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[x_shift_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[x_shift_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[x_shift_column]);
        return false;
    }

    temp_parameters.x_shift = float(temp_double);

    // yshift

    if ( y_shift_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[y_shift_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[y_shift_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[y_shift_column]);
        return false;
    }

    temp_parameters.y_shift = float(temp_double);

    // defocus1

    if ( StringToDouble(all_tokens[defocus_1_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_1_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[defocus_1_column]);
        return false;
    }

    temp_parameters.defocus_1 = float(temp_double);

    // defocus2

    if ( StringToDouble(all_tokens[defocus_2_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_2_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[defocus_2_column]);
        return false;
    }

    temp_parameters.defocus_2 = float(temp_double);

    // defocus_angle

    if ( StringToDouble(all_tokens[defocus_angle_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_angle_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[defocus_angle_column]);
        return false;
    }

    temp_parameters.defocus_angle = float(temp_double);

    // phase_shift

    if ( phase_shift_column == -1 )
        temp_parameters.phase_shift = 0.0;
    else {
        if ( StringToDouble(all_tokens[phase_shift_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[phase_shift_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[phase_shift_column]);
            return false;
        }

        temp_parameters.phase_shift = deg_2_rad(float(temp_double));
    }

    // occupancy

    if ( occupancy_column == -1 )
        temp_parameters.occupancy = 100.0f;
    else {
        if ( StringToDouble(all_tokens[occupancy_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[occupancy_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[occupancy_column]);
            return false;
        }

        temp_parameters.occupancy = float(temp_double);
    }

    // logp

    if ( logp_column == -1 )
        temp_parameters.logp = 100.0f;
    else {
        if ( StringToDouble(all_tokens[logp_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[logp_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[logp_column]);
            return false;
        }

        temp_parameters.logp = float(temp_double);
    }

    // sigma

    if ( sigma_column == -1 )
        temp_parameters.sigma = 10.0f;
    else {
        if ( StringToDouble(all_tokens[sigma_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[sigma_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[sigma_column]);
            return false;
        }

        temp_parameters.sigma = float(temp_double);
    }

    // score

    if ( score_column == -1 )
        temp_parameters.score = 0.0f;
    else {
        if ( StringToDouble(all_tokens[score_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[score_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[score_column]);
            return false;
        }

        temp_parameters.score = float(temp_double);
        //Printf("Score = %f, image_is_active = %i, exclude negative = %s\n", temp_parameters.score, temp_parameters.image_is_active, BoolToYesNo(exclude_negative_film_numbers));
    }

    // score_change

    if ( score_change_column == -1 )
        temp_parameters.score_change = 0.0f;
    else {
        if ( StringToDouble(all_tokens[score_change_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[score_change_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[score_change_column]);
            return false;
        }

        temp_parameters.score_change = float(temp_double);
    }

    // pixel_size

    if ( pixel_size_column == -1 )
        temp_parameters.pixel_size = 0.0f;
    else {
        if ( StringToDouble(all_tokens[pixel_size_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[pixel_size_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[pixel_size_column]);
            return false;
        }

        temp_parameters.pixel_size = float(temp_double);
    }

    // voltage

    if ( microscope_voltage_kv_column == -1 )
        temp_parameters.microscope_voltage_kv = 0.0f;
    else {
        if ( StringToDouble(all_tokens[microscope_voltage_kv_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[microscope_voltage_kv_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[microscope_voltage_kv_column]);
            return false;
        }

        temp_parameters.microscope_voltage_kv = float(temp_double);
    }

    // Cs

    if ( microscope_spherical_aberration_mm_column == -1 )
        temp_parameters.microscope_spherical_aberration_mm = 2.7f;
    else {
        if ( StringToDouble(all_tokens[microscope_spherical_aberration_mm_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[microscope_spherical_aberration_mm_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[microscope_spherical_aberration_mm_column]);
            return false;
        }

        temp_parameters.microscope_spherical_aberration_mm = float(temp_double);
    }

    // amplitude_contrast

    if ( amplitude_contrast_column == -1 )
        temp_parameters.amplitude_contrast = 0.07f;
    else {
        if ( StringToDouble(all_tokens[amplitude_contrast_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[amplitude_contrast_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[amplitude_contrast_column]);
            return false;
        }

        temp_parameters.amplitude_contrast = float(temp_double);
    }

    // beam_tilt_x

    if ( beam_tilt_x_column == -1 )
        temp_parameters.beam_tilt_x = 0.0f;
    else {
        if ( StringToDouble(all_tokens[beam_tilt_x_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_x_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_x_column]);
            return false;
        }

        temp_parameters.beam_tilt_x = float(temp_double);
    }

    // beam_tilt_y

    if ( beam_tilt_y_column == -1 )
        temp_parameters.beam_tilt_y = 0.0f;
    else {
        if ( StringToDouble(all_tokens[beam_tilt_y_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_y_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_y_column]);
            return false;
        }

        temp_parameters.beam_tilt_y = float(temp_double);
    }

    // image_shift_x

    if ( image_shift_x_column == -1 )
        temp_parameters.image_shift_x = 0.0f;
    else {
        if ( StringToDouble(all_tokens[image_shift_x_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_shift_x_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[image_shift_x_column]);
            return false;
        }

        temp_parameters.image_shift_x = float(temp_double);
    }

    // image_shift_y

    if ( image_shift_y_column == -1 )
        temp_parameters.image_shift_y = 0.0f;
    else {
        if ( StringToDouble(all_tokens[image_shift_y_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_shift_y_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[image_shift_y_column]);
            return false;
        }

        temp_parameters.image_shift_y = float(temp_double);
    }

    // best 2D class

    if ( best_2d_class_column == -1 )
        temp_parameters.best_2d_class = 0;
    else {
        if ( StringToLong(all_tokens[best_2d_class_column], temp_long) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[best_2d_class_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[best_2d_class_column]);
            return false;
        }

        temp_parameters.best_2d_class = int(temp_long);
    }

    // beam tilt group

    if ( beam_tilt_group_column == -1 )
        temp_parameters.beam_tilt_group = 0;
    else {
        if ( StringToLong(all_tokens[beam_tilt_group_column], temp_long) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_group_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_group_column]);
            return false;
        }

        temp_parameters.beam_tilt_group = int(temp_long);
    }

    // particle group

    if ( particle_group_column == -1 )
        temp_parameters.particle_group = 0;
    else {
        if ( StringToLong(all_tokens[particle_group_column], temp_long) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[particle_group_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[particle_group_column]);
            return false;
        }

        temp_parameters.particle_group = int(temp_long);
    }

    // assigned subset (for half-dataset refinement, or half map FSCs, etc)

    if ( assigned_subset_column == -1 )
        temp_parameters.assigned_subset = 0;
    else {
        if ( StringToLong(all_tokens[assigned_subset_column], temp_long) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[assigned_subset_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[assigned_subset_column]);
            return false;
        }

        temp_parameters.assigned_subset = int(temp_long);
    }

    // pre exposure

    if ( pre_exposure_column == -1 )
        temp_parameters.pre_exposure = 0.0f;
    else {
        if ( StringToDouble(all_tokens[pre_exposure_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[pre_exposure_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[pre_exposure_column]);
            return false;
        }

        temp_parameters.pre_exposure = float(temp_double);
    }

    // total exposure

    if ( total_exposure_column == -1 )
        temp_parameters.total_exposure = 0.0f;
    else {
        if ( StringToDouble(all_tokens[total_exposure_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[total_exposure_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[total_exposure_column]);
            return false;
        }

        temp_parameters.total_exposure = float(temp_double);
    }

    // original x position

    if ( original_x_position_column == -1 )
        temp_parameters.original_x_position = 0.0f;
    else {
        if ( StringToDouble(all_tokens[original_x_position_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[original_x_position_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[original_x_position_column]);
            return false;
        }

        temp_parameters.original_x_position = float(temp_double);
    }

    // original y position

    if ( original_y_position_column == -1 )
        temp_parameters.original_y_position = 0.0f;
    else {
        if ( StringToDouble(all_tokens[original_y_position_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[original_y_position_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[original_y_position_column]);
            return false;
        }

        temp_parameters.original_y_position = float(temp_double);
    }

    // stack filename

    if ( stack_filename_column == -1 )
        temp_parameters.stack_filename = "";
    else {
        temp_parameters.stack_filename = Trim(all_tokens[stack_filename_column]);
        if ( StripEnclosingSingleQuotesFromString(temp_parameters.stack_filename) == false ) {
            MyPrintfRed("Error: stack file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_parameters.stack_filename);
            temp_parameters.stack_filename = "";
        }
    }

    // original_image_filename

    if ( original_image_filename_column == -1 )
        temp_parameters.original_image_filename = "";
    else {
        temp_parameters.original_image_filename = Trim(all_tokens[original_image_filename_column]);

        if ( StripEnclosingSingleQuotesFromString(temp_parameters.original_image_filename) == false ) {
            MyPrintfRed("Error: original image file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_parameters.original_image_filename);
            temp_parameters.original_image_filename = "";
        }
    }

    // reference_3d_filename

    if ( reference_3d_filename_column == -1 )
        temp_parameters.reference_3d_filename = "";
    else {
        temp_parameters.reference_3d_filename = Trim(all_tokens[reference_3d_filename_column]);

        if ( StripEnclosingSingleQuotesFromString(temp_parameters.reference_3d_filename) == false ) {
            MyPrintfRed("Error: reference 3d file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_parameters.reference_3d_filename);
            temp_parameters.reference_3d_filename = "";
        }
    }

    cached_parameters->push_back(temp_parameters);

    return true;
}

bool cisTEMStarFileReader::ReadTextFile(std::string wanted_filename, std::string* error_string, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer, bool exclude_negative_film_numbers) {
    Open(wanted_filename, alternate_cached_parameters_pointer);
    std::string current_line;

    MyDebugAssertTrue(input_file_is_opened, "File not open");

    bool found_valid_data_block = false;
    bool found_valid_loop_block = false;

    current_line_number = -1;
    // find a data block

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);
        if ( current_line.find("data_") != std::string::npos ) {
            found_valid_data_block = true;
            break;
        }
    }

    if ( found_valid_data_block == false ) {
        MyPrintWithDetails("Error: Couldn't find a valid data block in star file (%s)\n", wanted_filename);

        if ( error_string != NULL )
            *error_string = Format("Error: Couldn't find a valid data block in star file (%s)\n", wanted_filename);
        return false;
    }

    // find a loop block

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);

        if ( current_line.find("loop_") != std::string::npos ) {
            found_valid_loop_block = true;
            break;
        }
    }

    if ( found_valid_loop_block == false ) {
        MyPrintWithDetails("Error: Couldn't find a valid loop block in star file (%s)\n", wanted_filename);
        if ( error_string != NULL )
            *error_string = Format("Error: Couldn't find a valid loop block in star file (%s)\n", wanted_filename);
        return false;
    }

    // now we can get headers..

    ResetColumnPositions( );

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);

        if ( current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';' )
            continue;
        if ( current_line[0] != '_' )
            break;

        // otherwise it is a label, is it a label we want though?

        if ( StartsWith(current_line, "_cisTEMPositionInStack ") == true ) {
            if ( position_in_stack_column != -1 )
                Printf("Warning :: _cisTEMPositionInStack occurs more than once. I will take the last occurrence\n");
            position_in_stack_column                    = current_column;
            parameters_that_were_read.position_in_stack = true;
        }
        else if ( StartsWith(current_line, "_cisTEMAnglePsi ") == true ) {
            if ( psi_column != -1 )
                Printf("Warning :: _cisTEMAnglePsi occurs more than once. I will take the last occurrence\n");
            psi_column                    = current_column;
            parameters_that_were_read.psi = true;
        }
        else if ( StartsWith(current_line, "_cisTEMAngleTheta ") == true ) {
            if ( theta_column != -1 )
                Printf("Warning :: _cisTEMAngleTheta occurs more than once. I will take the last occurrence\n");
            theta_column                    = current_column;
            parameters_that_were_read.theta = true;
        }
        else if ( StartsWith(current_line, "_cisTEMAnglePhi ") == true ) {
            if ( phi_column != -1 )
                Printf("Warning :: _cisTEMAnglePhi occurs more than once. I will take the last occurrence\n");
            phi_column                    = current_column;
            parameters_that_were_read.phi = true;
        }
        else if ( StartsWith(current_line, "_cisTEMXShift ") == true ) {
            if ( x_shift_column != -1 )
                Printf("Warning :: _cisTEMXShift occurs more than once. I will take the last occurrence\n");
            x_shift_column                    = current_column;
            parameters_that_were_read.x_shift = true;
        }
        else if ( StartsWith(current_line, "_cisTEMYShift ") == true ) {
            if ( y_shift_column != -1 )
                Printf("Warning :: _cisTEMYShift occurs more than once. I will take the last occurrence\n");
            y_shift_column                    = current_column;
            parameters_that_were_read.y_shift = true;
        }
        else if ( StartsWith(current_line, "_cisTEMDefocus1 ") == true ) {
            if ( defocus_1_column != -1 )
                Printf("Warning :: _cisTEMDefocus1 occurs more than once. I will take the last occurrence\n");
            defocus_1_column                    = current_column;
            parameters_that_were_read.defocus_1 = true;
        }
        else if ( StartsWith(current_line, "_cisTEMDefocus2 ") == true ) {
            if ( defocus_2_column != -1 )
                Printf("Warning :: _cisTEMDefocus2 occurs more than once. I will take the last occurrence\n");
            defocus_2_column                    = current_column;
            parameters_that_were_read.defocus_2 = true;
        }
        else if ( StartsWith(current_line, "_cisTEMDefocusAngle ") == true ) {
            if ( defocus_angle_column != -1 )
                Printf("Warning :: _cisTEMDefocusAngle occurs more than once. I will take the last occurrence\n");
            defocus_angle_column                    = current_column;
            parameters_that_were_read.defocus_angle = true;
        }
        else if ( StartsWith(current_line, "_cisTEMPhaseShift ") == true ) {
            if ( phase_shift_column != -1 )
                Printf("Warning :: _cisTEMPhaseShift occurs more than once. I will take the last occurrence\n");
            phase_shift_column                    = current_column;
            parameters_that_were_read.phase_shift = true;
        }
        else if ( StartsWith(current_line, "_cisTEMImageActivity ") == true ) {
            if ( image_is_active_column != -1 )
                Printf("Warning :: _cisTEMImageActivity occurs more than once. I will take the last occurrence\n");
            image_is_active_column                    = current_column;
            parameters_that_were_read.image_is_active = true;
        }
        else if ( StartsWith(current_line, "_cisTEMOccupancy ") == true ) {
            if ( occupancy_column != -1 )
                Printf("Warning :: _cisTEMOccupancy occurs more than once. I will take the last occurrence\n");
            occupancy_column                    = current_column;
            parameters_that_were_read.occupancy = true;
        }
        else if ( StartsWith(current_line, "_cisTEMLogP ") == true ) {
            if ( logp_column != -1 )
                Printf("Warning :: _cisTEMLogP occurs more than once. I will take the last occurrence\n");
            logp_column                    = current_column;
            parameters_that_were_read.logp = true;
        }
        else if ( StartsWith(current_line, "_cisTEMSigma ") == true ) {
            if ( sigma_column != -1 )
                Printf("Warning :: _cisTEMSigma occurs more than once. I will take the last occurrence\n");
            sigma_column                    = current_column;
            parameters_that_were_read.sigma = true;
        }
        else if ( StartsWith(current_line, "_cisTEMScore ") == true ) {
            if ( score_column != -1 )
                Printf("Warning :: _cisTEMScore occurs more than once. I will take the last occurrence\n");
            score_column                    = current_column;
            parameters_that_were_read.score = true;
        }
        else if ( StartsWith(current_line, "_cisTEMScoreChange ") == true ) {
            if ( score_change_column != -1 )
                Printf("Warning :: _cisTEMScoreChange occurs more than once. I will take the last occurrence\n");
            score_change_column                    = current_column;
            parameters_that_were_read.score_change = true;
        }
        else if ( StartsWith(current_line, "_cisTEMPixelSize ") == true ) {
            if ( pixel_size_column != -1 )
                Printf("Warning :: _cisTEMPixelSize occurs more than once. I will take the last occurrence\n");
            pixel_size_column                    = current_column;
            parameters_that_were_read.pixel_size = true;
        }
        else if ( StartsWith(current_line, "_cisTEMMicroscopeVoltagekV ") == true ) {
            if ( microscope_voltage_kv_column != -1 )
                Printf("Warning :: _cisTEMMicroscopeVoltagekV occurs more than once. I will take the last occurrence\n");
            microscope_voltage_kv_column                    = current_column;
            parameters_that_were_read.microscope_voltage_kv = true;
        }
        else if ( StartsWith(current_line, "_cisTEMMicroscopeCsMM ") == true ) {
            if ( microscope_spherical_aberration_mm_column != -1 )
                Printf("Warning :: _cisTEMMicroscopeCsMM occurs more than once. I will take the last occurrence\n");
            microscope_spherical_aberration_mm_column                    = current_column;
            parameters_that_were_read.microscope_spherical_aberration_mm = true;
        }
        else if ( StartsWith(current_line, "_cisTEMAmplitudeContrast ") == true ) {
            if ( amplitude_contrast_column != -1 )
                Printf("Warning :: _cisTEMAmplitudeContrast occurs more than once. I will take the last occurrence\n");
            amplitude_contrast_column                    = current_column;
            parameters_that_were_read.amplitude_contrast = true;
        }
        else if ( StartsWith(current_line, "_cisTEMBeamTiltX ") == true ) {
            if ( beam_tilt_x_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltX occurs more than once. I will take the last occurrence\n");
            beam_tilt_x_column                    = current_column;
            parameters_that_were_read.beam_tilt_x = true;
        }
        else if ( StartsWith(current_line, "_cisTEMBeamTiltY ") == true ) {
            if ( beam_tilt_y_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltY occurs more than once. I will take the last occurrence\n");
            beam_tilt_y_column                    = current_column;
            parameters_that_were_read.beam_tilt_y = true;
        }
        else if ( StartsWith(current_line, "_cisTEMImageShiftX ") == true ) {
            if ( image_shift_x_column != -1 )
                Printf("Warning :: _cisTEMImageShiftX occurs more than once. I will take the last occurrence\n");
            image_shift_x_column                    = current_column;
            parameters_that_were_read.image_shift_x = true;
        }
        else if ( StartsWith(current_line, "_cisTEMImageShiftY ") == true ) {
            if ( image_shift_y_column != -1 )
                Printf("Warning :: _cisTEMImageShiftY occurs more than once. I will take the last occurrence\n");
            image_shift_y_column                    = current_column;
            parameters_that_were_read.image_shift_y = true;
        }
        else if ( StartsWith(current_line, "_cisTEMBest2DClass ") == true ) {
            if ( best_2d_class_column != -1 )
                Printf("Warning :: _cisTEMBest2DClass occurs more than once. I will take the last occurrence\n");
            best_2d_class_column                    = current_column;
            parameters_that_were_read.best_2d_class = true;
        }
        else if ( StartsWith(current_line, "_cisTEMBeamTiltGroup ") == true ) {
            if ( beam_tilt_group_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltGroup occurs more than once. I will take the last occurrence\n");
            beam_tilt_group_column                    = current_column;
            parameters_that_were_read.beam_tilt_group = true;
        }
        else if ( StartsWith(current_line, "_cisTEMParticleGroup ") == true ) {
            if ( particle_group_column != -1 )
                Printf("Warning :: _cisTEMParticleGroup occurs more than once. I will take the last occurrence\n");
            particle_group_column                    = current_column;
            parameters_that_were_read.particle_group = true;
        }
        else if ( StartsWith(current_line, "_cisTEMAssignedSubset ") == true ) {
            if ( assigned_subset_column != -1 )
                Printf("Warning :: _cisTEMAssignedSubset occurs more than once. I will take the last occurrence\n");
            assigned_subset_column                    = current_column;
            parameters_that_were_read.assigned_subset = true;
        }
        else if ( StartsWith(current_line, "_cisTEMPreExposure ") == true ) {
            if ( pre_exposure_column != -1 )
                Printf("Warning :: _cisTEMPreExposure occurs more than once. I will take the last occurrence\n");
            pre_exposure_column                    = current_column;
            parameters_that_were_read.pre_exposure = true;
        }
        else if ( StartsWith(current_line, "_cisTEMTotalExposure ") == true ) {
            if ( total_exposure_column != -1 )
                Printf("Warning :: _cisTEMTotalExposure occurs more than once. I will take the last occurrence\n");
            total_exposure_column                    = current_column;
            parameters_that_were_read.total_exposure = true;
        }
        else if ( StartsWith(current_line, "_cisTEMOriginalXPosition") == true ) {
            if ( original_x_position_column != -1 )
                Printf("Warning :: _cisTEMOriginalXPosition occurs more than once. I will take the last occurrence\n");
            original_x_position_column                    = current_column;
            parameters_that_were_read.original_x_position = true;
        }
        else if ( StartsWith(current_line, "_cisTEMOriginalYPosition") == true ) {
            if ( original_y_position_column != -1 )
                Printf("Warning :: _cisTEMOriginalYPosition occurs more than once. I will take the last occurrence\n");
            original_y_position_column                    = current_column;
            parameters_that_were_read.original_y_position = true;
        }
        else if ( StartsWith(current_line, "_cisTEMReference3DFilename ") == true ) {
            if ( reference_3d_filename_column != -1 )
                Printf("Warning :: _cisTEMReference3DFilename occurs more than once. I will take the last occurrence\n");
            reference_3d_filename_column                    = current_column;
            parameters_that_were_read.reference_3d_filename = true;
        }
        else if ( StartsWith(current_line, "_cisTEMOriginalImageFilename ") == true ) {
            if ( original_image_filename_column != -1 )
                Printf("Warning :: _cisTEMOriginalImageFilename occurs more than once. I will take the last occurrence\n");
            original_image_filename_column                    = current_column;
            parameters_that_were_read.original_image_filename = true;
        }
        else if ( StartsWith(current_line, "_cisTEMStackFilename ") == true ) {
            if ( stack_filename_column != -1 )
                Printf("Warning :: _cisTEMStackFilename occurs more than once. I will take the last occurrence\n");
            stack_filename_column                    = current_column;
            parameters_that_were_read.stack_filename = true;
        }

        current_column++;
    }

    // quick checks we have all the desired info.
    /*
	if (phi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		return false;
	}

	if (theta_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		return false;
	}

	if (psi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		return false;
	}

	if (xshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		return false;
	}

	if (yshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus1_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus2_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus_angle_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		return false;
	}

	if (phase_shift_column == -1)
	{
	//	MyPrintWithDetails("Warning: Couldn't find _rlnPhaseShift in star file (%s) - phase shift will be set to 0.0\n", wanted_filename);
	}

*/

    // we have the headers, the current line should be the first parameter to extract the info

    if ( ExtractParametersFromLine(current_line, error_string, exclude_negative_film_numbers) == false )
        return false;

    // loop over the data lines and fill in..

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);

        if ( current_line.empty() == true )
            break;
        if ( current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';' )
            continue;

        if ( ExtractParametersFromLine(current_line, error_string, exclude_negative_film_numbers) == false )
            return false;
    }

    return true;
}

bool cisTEMStarFileReader::ReadBinaryFile(std::string wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer, bool exclude_negative_film_numbers) {
    Open(wanted_filename, alternate_cached_parameters_pointer, true);
    MyDebugAssertTrue(binary_file_size > 2, "Input binary file is too small")

            int number_of_columns;
    int         number_of_lines;

    int current_column;
    int current_line;

    binary_buffer_position = 0;

    // get the number of columns and line

    if ( SafelyReadFromBinaryBufferIntoInteger(number_of_columns) == false )
        return false;
    if ( SafelyReadFromBinaryBufferIntoInteger(number_of_lines) == false )
        return false;

    if ( number_of_columns < 1 || number_of_lines < 1 ) {
        MyPrintWithDetails("Format Error %i columns and %i lines", number_of_columns, number_of_lines);
    }

    // Preallocate the memory for speed

    cached_parameters->reserve(number_of_lines);

    // get the order of columns..

    long    column_order_buffer[number_of_columns]; // so we can keep the order for reading later..
    uint8_t column_data_types[number_of_columns]; // in case we don't recognize a column - see cistem_paramters for more details // sizeof(c_ft::Enum) == sizeof(uint8_t)

    ResetColumnPositions( );

    for ( current_column = 0; current_column < number_of_columns; current_column++ ) {
        if ( SafelyReadFromBinaryBufferIntoLong(column_order_buffer[current_column]) == false )
            return false;
        if ( SafelyReadFromBinaryBufferIntoUnsignedChar(column_data_types[current_column]) == false )
            return false;

        if ( column_order_buffer[current_column] == POSITION_IN_STACK ) {
            if ( position_in_stack_column != -1 )
                Printf("Warning :: _cisTEMPositionInStack occurs more than once. I will take the last occurrence\n");
            position_in_stack_column                    = current_column;
            parameters_that_were_read.position_in_stack = true;
        }
        else if ( column_order_buffer[current_column] == PSI ) {
            if ( psi_column != -1 )
                Printf("Warning :: _cisTEMAnglePsi occurs more than once. I will take the last occurrence\n");
            psi_column                    = current_column;
            parameters_that_were_read.psi = true;
        }
        else if ( column_order_buffer[current_column] == THETA ) {
            if ( theta_column != -1 )
                Printf("Warning :: _cisTEMAngleTheta occurs more than once. I will take the last occurrence\n");
            theta_column                    = current_column;
            parameters_that_were_read.theta = true;
        }
        else if ( column_order_buffer[current_column] == PHI ) {
            if ( phi_column != -1 )
                Printf("Warning :: _cisTEMAnglePhi occurs more than once. I will take the last occurrence\n");
            phi_column                    = current_column;
            parameters_that_were_read.phi = true;
        }
        else if ( column_order_buffer[current_column] == X_SHIFT ) {
            if ( x_shift_column != -1 )
                Printf("Warning :: _cisTEMXShift occurs more than once. I will take the last occurrence\n");
            x_shift_column                    = current_column;
            parameters_that_were_read.x_shift = true;
        }
        else if ( column_order_buffer[current_column] == Y_SHIFT ) {
            if ( y_shift_column != -1 )
                Printf("Warning :: _cisTEMYShift occurs more than once. I will take the last occurrence\n");
            y_shift_column                    = current_column;
            parameters_that_were_read.y_shift = true;
        }
        else if ( column_order_buffer[current_column] == DEFOCUS_1 ) {
            if ( defocus_1_column != -1 )
                Printf("Warning :: _cisTEMDefocus1 occurs more than once. I will take the last occurrence\n");
            defocus_1_column                    = current_column;
            parameters_that_were_read.defocus_1 = true;
        }
        else if ( column_order_buffer[current_column] == DEFOCUS_2 ) {
            if ( defocus_2_column != -1 )
                Printf("Warning :: _cisTEMDefocus2 occurs more than once. I will take the last occurrence\n");
            defocus_2_column                    = current_column;
            parameters_that_were_read.defocus_2 = true;
        }
        else if ( column_order_buffer[current_column] == DEFOCUS_ANGLE ) {
            if ( defocus_angle_column != -1 )
                Printf("Warning :: _cisTEMDefocusAngle occurs more than once. I will take the last occurrence\n");
            defocus_angle_column                    = current_column;
            parameters_that_were_read.defocus_angle = true;
        }
        else if ( column_order_buffer[current_column] == PHASE_SHIFT ) {
            if ( phase_shift_column != -1 )
                Printf("Warning :: _cisTEMPhaseShift occurs more than once. I will take the last occurrence\n");
            phase_shift_column                    = current_column;
            parameters_that_were_read.phase_shift = true;
        }
        else if ( column_order_buffer[current_column] == IMAGE_IS_ACTIVE ) {
            if ( image_is_active_column != -1 )
                Printf("Warning :: _cisTEMImageActivity occurs more than once. I will take the last occurrence\n");
            image_is_active_column                    = current_column;
            parameters_that_were_read.image_is_active = true;
        }
        else if ( column_order_buffer[current_column] == OCCUPANCY ) {
            if ( occupancy_column != -1 )
                Printf("Warning :: _cisTEMOccupancy occurs more than once. I will take the last occurrence\n");
            occupancy_column                    = current_column;
            parameters_that_were_read.occupancy = true;
        }
        else if ( column_order_buffer[current_column] == LOGP ) {
            if ( logp_column != -1 )
                Printf("Warning :: _cisTEMLogP occurs more than once. I will take the last occurrence\n");
            logp_column                    = current_column;
            parameters_that_were_read.logp = true;
        }
        else if ( column_order_buffer[current_column] == SIGMA ) {
            if ( sigma_column != -1 )
                Printf("Warning :: _cisTEMSigma occurs more than once. I will take the last occurrence\n");
            sigma_column                    = current_column;
            parameters_that_were_read.sigma = true;
        }
        else if ( column_order_buffer[current_column] == SCORE ) {
            if ( score_column != -1 )
                Printf("Warning :: _cisTEMScore occurs more than once. I will take the last occurrence\n");
            score_column                    = current_column;
            parameters_that_were_read.score = true;
        }
        else if ( column_order_buffer[current_column] == SCORE_CHANGE ) {
            if ( score_change_column != -1 )
                Printf("Warning :: _cisTEMScoreChange occurs more than once. I will take the last occurrence\n");
            score_change_column                    = current_column;
            parameters_that_were_read.score_change = true;
        }
        else if ( column_order_buffer[current_column] == PIXEL_SIZE ) {
            if ( pixel_size_column != -1 )
                Printf("Warning :: _cisTEMPixelSize occurs more than once. I will take the last occurrence\n");
            pixel_size_column                    = current_column;
            parameters_that_were_read.pixel_size = true;
        }
        else if ( column_order_buffer[current_column] == MICROSCOPE_VOLTAGE ) {
            if ( microscope_voltage_kv_column != -1 )
                Printf("Warning :: _cisTEMMicroscopeVoltagekV occurs more than once. I will take the last occurrence\n");
            microscope_voltage_kv_column                    = current_column;
            parameters_that_were_read.microscope_voltage_kv = true;
        }
        else if ( column_order_buffer[current_column] == MICROSCOPE_CS ) {
            if ( microscope_spherical_aberration_mm_column != -1 )
                Printf("Warning :: _cisTEMMicroscopeCsMM occurs more than once. I will take the last occurrence\n");
            microscope_spherical_aberration_mm_column                    = current_column;
            parameters_that_were_read.microscope_spherical_aberration_mm = true;
        }
        else if ( column_order_buffer[current_column] == AMPLITUDE_CONTRAST ) {
            if ( amplitude_contrast_column != -1 )
                Printf("Warning :: _cisTEMAmplitudeContrast occurs more than once. I will take the last occurrence\n");
            amplitude_contrast_column                    = current_column;
            parameters_that_were_read.amplitude_contrast = true;
        }
        else if ( column_order_buffer[current_column] == BEAM_TILT_X ) {
            if ( beam_tilt_x_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltX occurs more than once. I will take the last occurrence\n");
            beam_tilt_x_column                    = current_column;
            parameters_that_were_read.beam_tilt_x = true;
        }
        else if ( column_order_buffer[current_column] == BEAM_TILT_Y ) {
            if ( beam_tilt_y_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltY occurs more than once. I will take the last occurrence\n");
            beam_tilt_y_column                    = current_column;
            parameters_that_were_read.beam_tilt_y = true;
        }
        else if ( column_order_buffer[current_column] == IMAGE_SHIFT_X ) {
            if ( image_shift_x_column != -1 )
                Printf("Warning :: _cisTEMImageShiftX occurs more than once. I will take the last occurrence\n");
            image_shift_x_column                    = current_column;
            parameters_that_were_read.image_shift_x = true;
        }
        else if ( column_order_buffer[current_column] == IMAGE_SHIFT_Y ) {
            if ( image_shift_y_column != -1 )
                Printf("Warning :: _cisTEMImageShiftY occurs more than once. I will take the last occurrence\n");
            image_shift_y_column                    = current_column;
            parameters_that_were_read.image_shift_y = true;
        }
        else if ( column_order_buffer[current_column] == BEST_2D_CLASS ) {
            if ( best_2d_class_column != -1 )
                Printf("Warning :: _cisTEMBest2DClass occurs more than once. I will take the last occurrence\n");
            best_2d_class_column                    = current_column;
            parameters_that_were_read.best_2d_class = true;
        }
        else if ( column_order_buffer[current_column] == BEAM_TILT_GROUP ) {
            if ( beam_tilt_group_column != -1 )
                Printf("Warning :: _cisTEMBeamTiltGroup occurs more than once. I will take the last occurrence\n");
            beam_tilt_group_column                    = current_column;
            parameters_that_were_read.beam_tilt_group = true;
        }
        else if ( column_order_buffer[current_column] == PARTICLE_GROUP ) {
            if ( particle_group_column != -1 )
                Printf("Warning :: _cisTEMParticleGroup occurs more than once. I will take the last occurrence\n");
            particle_group_column                    = current_column;
            parameters_that_were_read.particle_group = true;
        }
        else if ( column_order_buffer[current_column] == ASSIGNED_SUBSET ) {
            if ( assigned_subset_column != -1 )
                Printf("Warning :: _cisTEMAssignedSubset occurs more than once. I will take the last occurrence\n");
            assigned_subset_column                    = current_column;
            parameters_that_were_read.assigned_subset = true;
        }
        else if ( column_order_buffer[current_column] == PRE_EXPOSURE ) {
            if ( pre_exposure_column != -1 )
                Printf("Warning :: _cisTEMPreExposure occurs more than once. I will take the last occurrence\n");
            pre_exposure_column                    = current_column;
            parameters_that_were_read.pre_exposure = true;
        }
        else if ( column_order_buffer[current_column] == TOTAL_EXPOSURE ) {
            if ( total_exposure_column != -1 )
                Printf("Warning :: _cisTEMTotalExposure occurs more than once. I will take the last occurrence\n");
            total_exposure_column                    = current_column;
            parameters_that_were_read.total_exposure = true;
        }
        else if ( column_order_buffer[current_column] == ORIGINAL_X_POSITION ) {
            if ( original_x_position_column != -1 )
                Printf("Warning :: _cisTEMOriginalXPosition occurs more than once. I will take the last occurrence\n");
            original_x_position_column                    = current_column;
            parameters_that_were_read.original_x_position = true;
        }
        else if ( column_order_buffer[current_column] == ORIGINAL_Y_POSITION ) {
            if ( original_y_position_column != -1 )
                Printf("Warning :: _cisTEMOriginalYPosition occurs more than once. I will take the last occurrence\n");
            original_y_position_column                    = current_column;
            parameters_that_were_read.original_y_position = true;
        }
        else if ( column_order_buffer[current_column] == REFERENCE_3D_FILENAME ) {
            if ( reference_3d_filename_column != -1 )
                Printf("Warning :: _cisTEMReference3DFilename occurs more than once. I will take the last occurrence\n");
            reference_3d_filename_column                    = current_column;
            parameters_that_were_read.reference_3d_filename = true;
        }
        else if ( column_order_buffer[current_column] == ORIGINAL_IMAGE_FILENAME ) {
            if ( original_image_filename_column != -1 )
                Printf("Warning :: _cisTEMOriginalImageFilename occurs more than once. I will take the last occurrence\n");
            original_image_filename_column                    = current_column;
            parameters_that_were_read.original_image_filename = true;
        }
        else if ( column_order_buffer[current_column] == STACK_FILENAME ) {
            if ( stack_filename_column != -1 )
                Printf("Warning :: _cisTEMStackFilename occurs more than once. I will take the last occurrence\n");
            stack_filename_column                    = current_column;
            parameters_that_were_read.stack_filename = true;
        }
    }

    // quick checks we have all the desired info.
    /*
	if (phi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		return false;
	}

	if (theta_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		return false;
	}

	if (psi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		return false;
	}

	if (xshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		return false;
	}

	if (yshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus1_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus2_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus_angle_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = Format("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		return false;
	}

	if (phase_shift_column == -1)
	{
	//	MyPrintWithDetails("Warning: Couldn't find _rlnPhaseShift in star file (%s) - phase shift will be set to 0.0\n", wanted_filename);
	}

 	 */

    // extract the data..

    cisTEMParameterLine temp_parameters;

    for ( current_line = 0; current_line < number_of_lines; current_line++ ) {
        temp_parameters.SetAllToZero( );

        for ( current_column = 0; current_column < number_of_columns; current_column++ ) {
            if ( column_order_buffer[current_column] == POSITION_IN_STACK ) {
                if ( SafelyReadFromBinaryBufferIntoUnsignedInteger(temp_parameters.position_in_stack) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PSI ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.psi) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == THETA ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.theta) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PHI ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.phi) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == X_SHIFT ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.x_shift) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == Y_SHIFT ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.y_shift) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == DEFOCUS_1 ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.defocus_1) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == DEFOCUS_2 ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.defocus_2) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == DEFOCUS_ANGLE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.defocus_angle) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PHASE_SHIFT ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.phase_shift) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == IMAGE_IS_ACTIVE ) {
                if ( SafelyReadFromBinaryBufferIntoInteger(temp_parameters.image_is_active) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == OCCUPANCY ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.occupancy) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == LOGP ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.logp) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == SIGMA ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.sigma) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == SCORE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.score) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == SCORE_CHANGE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.score_change) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PIXEL_SIZE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.pixel_size) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == MICROSCOPE_VOLTAGE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.microscope_voltage_kv) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == MICROSCOPE_CS ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.microscope_spherical_aberration_mm) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == AMPLITUDE_CONTRAST ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.amplitude_contrast) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == BEAM_TILT_X ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.beam_tilt_x) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == BEAM_TILT_Y ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.beam_tilt_y) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == IMAGE_SHIFT_X ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.image_shift_x) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == IMAGE_SHIFT_Y ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.image_shift_y) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == BEST_2D_CLASS ) {
                if ( SafelyReadFromBinaryBufferIntoInteger(temp_parameters.best_2d_class) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == BEAM_TILT_GROUP ) {
                if ( SafelyReadFromBinaryBufferIntoInteger(temp_parameters.beam_tilt_group) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == STACK_FILENAME ) {
                if ( SafelyReadFromBinaryBufferIntoString(temp_parameters.stack_filename) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == ORIGINAL_IMAGE_FILENAME ) {
                if ( SafelyReadFromBinaryBufferIntoString(temp_parameters.original_image_filename) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == REFERENCE_3D_FILENAME ) {
                if ( SafelyReadFromBinaryBufferIntoString(temp_parameters.reference_3d_filename) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PARTICLE_GROUP ) {
                if ( SafelyReadFromBinaryBufferIntoInteger(temp_parameters.particle_group) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == ASSIGNED_SUBSET ) {
                if ( SafelyReadFromBinaryBufferIntoInteger(temp_parameters.assigned_subset) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == PRE_EXPOSURE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.pre_exposure) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == TOTAL_EXPOSURE ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.total_exposure) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == ORIGINAL_X_POSITION ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.original_x_position) == false )
                    return false;
            }
            else if ( column_order_buffer[current_column] == ORIGINAL_Y_POSITION ) {
                if ( SafelyReadFromBinaryBufferIntoFloat(temp_parameters.original_y_position) == false )
                    return false;
            }
            else // We do not recongnize this column type
            {
                if ( current_line == 0 )
                    Printf("Unknown Column Type in Binary File (%li) - it will be ignored.\n");

                if ( column_data_types[current_column] == c_ft::integer_t ) {
                    int buffer_int;
                    if ( SafelyReadFromBinaryBufferIntoInteger(buffer_int) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::integer_unsigned_t ) {
                    unsigned int buffer_int;
                    if ( SafelyReadFromBinaryBufferIntoUnsignedInteger(buffer_int) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::float_t ) {
                    float buffer_float;
                    if ( SafelyReadFromBinaryBufferIntoFloat(buffer_float) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::long_t ) {
                    long buffer_long;
                    if ( SafelyReadFromBinaryBufferIntoLong(buffer_long) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::double_t ) {
                    double buffer_double;
                    if ( SafelyReadFromBinaryBufferIntoDouble(buffer_double) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::char_t ) {
                    char buffer_char;
                    if ( SafelyReadFromBinaryBufferIntoChar(buffer_char) == false )
                        return false;
                }
                else if ( column_data_types[current_column] == c_ft::variable_length_t ) {
                    std::string buffer_string;
                    if ( SafelyReadFromBinaryBufferIntoString(buffer_string) == false )
                        return false;
                }
            }
        }

        if ( temp_parameters.image_is_active >= 0 || exclude_negative_film_numbers == false )
            cached_parameters->push_back(temp_parameters);
    }

    return true;
}
