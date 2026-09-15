#include "core_headers.h"

StarFileParameters::StarFileParameters( ) {
    position_in_stack = -1;
    phi               = 0;
    theta             = 0;
    psi               = 0;
    x_coordinate      = 0;
    y_coordinate      = 0;
    x_shift           = 0;
    y_shift           = 0;
    defocus1          = 0;
    defocus2          = 0;
    defocus_angle     = 0;
    phase_shift       = 0;
    micrograph_name   = "";
    random_subset     = -1;
}

BasicStarFileReader::BasicStarFileReader( ) {
    filename = "";

    input_file_is_opened = false;
    current_line_number  = -1;

    current_position_in_stack = 0;
    current_column            = 0;

    phi_column             = -1;
    theta_column           = -1;
    psi_column             = -1;
    xcoordinate_column     = -1;
    ycoordinate_column     = -1;
    xshift_column          = -1;
    yshift_column          = -1;
    defocus1_column        = -1;
    defocus2_column        = -1;
    defocus_angle_column   = -1;
    phase_shift_column     = -1;
    micrograph_name_column = -1;
    image_name_column      = -1;
    random_subset_column   = -1;
}

BasicStarFileReader::BasicStarFileReader(std::string wanted_filename) {
    ReadFile(wanted_filename);
}

BasicStarFileReader::~BasicStarFileReader( ) {
    Close( );
}

void BasicStarFileReader::Open(std::string wanted_filename) {
    Close( );
    cached_parameters.clear( );

    filename = wanted_filename;

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

void BasicStarFileReader::Close( ) {
    cached_parameters.clear( );

    input_file_lines.clear( );
    input_file_is_opened = false;
    current_line_number  = -1;
}

bool BasicStarFileReader::ExtractParametersFromLine(std::string& wanted_line, std::string* error_string) {
    // extract info.

    std::vector<std::string> all_tokens = SplitString(wanted_line);
    StarFileParameters       temp_parameters;
    double                   temp_double;

    current_position_in_stack++;
    temp_parameters.position_in_stack = current_position_in_stack;

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

    // xcoordinate

    if ( xcoordinate_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[xcoordinate_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[xcoordinate_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[xcoordinate_column]);
        return false;
    }

    temp_parameters.x_coordinate = float(temp_double);

    // ycoordinate

    if ( ycoordinate_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[ycoordinate_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[ycoordinate_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[ycoordinate_column]);
        return false;
    }

    temp_parameters.y_coordinate = float(temp_double);

    // xshift

    if ( xshift_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[xshift_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[xshift_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[xshift_column]);
        return false;
    }

    temp_parameters.x_shift = float(temp_double);

    // yshift

    if ( yshift_column == -1 )
        temp_double = 0.0;
    else if ( StringToDouble(all_tokens[yshift_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[yshift_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[yshift_column]);
        return false;
    }

    temp_parameters.y_shift = float(temp_double);

    // defocus1

    if ( StringToDouble(all_tokens[defocus1_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus1_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[defocus1_column]);
        return false;
    }

    temp_parameters.defocus1 = float(temp_double);

    // defocus2

    if ( StringToDouble(all_tokens[defocus2_column], temp_double) == false ) {
        MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus2_column]);
        if ( error_string != NULL )
            *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[defocus2_column]);
        return false;
    }

    temp_parameters.defocus2 = float(temp_double);

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

    // random_subset

    if ( random_subset_column == -1 )
        temp_parameters.random_subset = -1;
    else {
        if ( StringToDouble(all_tokens[random_subset_column], temp_double) == false ) {
            MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[random_subset_column]);
            if ( error_string != NULL )
                *error_string = Format("Error: Converting to a number (%s)\n", all_tokens[random_subset_column]);
            return false;
        }

        temp_parameters.random_subset = int(temp_double);
    }

    // Micrograph Name

    if ( micrograph_name_column == -1 )
        temp_parameters.micrograph_name = "";
    else
        temp_parameters.micrograph_name = all_tokens[micrograph_name_column];

    // Image Name

    if ( image_name_column == -1 )
        temp_parameters.image_name = "";
    else
        temp_parameters.image_name = all_tokens[image_name_column];

    cached_parameters.push_back(temp_parameters);

    return true;
}

bool BasicStarFileReader::ReadFile(std::string wanted_filename, std::string* error_string) {
    Open(wanted_filename);
    std::string current_line;

    MyDebugAssertTrue(input_file_is_opened, "File not open");

    bool found_valid_data_block = false;
    bool found_valid_loop_block = false;

    x_shifts_are_in_angst = false;
    y_shifts_are_in_angst = false;

    current_line_number = -1;
    // find a data block

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);
        if ( current_line.find("data_") != std::string::npos ) {
            if ( Contains(current_line, "data_optics") == true )
                continue;
            else {
                found_valid_data_block = true;
                break;
            }
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

    while ( AtEndOfFile( ) == false ) {
        current_line = ReturnNextLine( );
        TrimRight(current_line);
        TrimLeft(current_line);

        if ( current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';' )
            continue;
        if ( current_line[0] != '_' )
            break;

        // otherwise it is a label, is it a label we want though?

        if ( StartsWith(current_line, "_rlnAngleRot") == true )
            phi_column = current_column;
        else if ( StartsWith(current_line, "_rlnAngleTilt") == true )
            theta_column = current_column;
        else if ( StartsWith(current_line, "_rlnAnglePsi") == true )
            psi_column = current_column;
        else if ( StartsWith(current_line, "_rlnCoordinateX") == true )
            xcoordinate_column = current_column;
        if ( StartsWith(current_line, "_rlnCoordinateY") == true )
            ycoordinate_column = current_column;
        if ( StartsWith(current_line, "_rlnOriginX") == true ) {
            xshift_column = current_column;
            if ( StartsWith(current_line, "_rlnOriginXAngst") == true )
                x_shifts_are_in_angst = true;
            else
                ;
        }
        else if ( StartsWith(current_line, "_rlnOriginY") == true ) {
            yshift_column = current_column;
            if ( StartsWith(current_line, "_rlnOriginYAngst") == true )
                y_shifts_are_in_angst = true;
            else
                ;
        }
        else if ( StartsWith(current_line, "_rlnDefocusU") == true )
            defocus1_column = current_column;
        else if ( StartsWith(current_line, "_rlnDefocusV") == true )
            defocus2_column = current_column;
        else if ( StartsWith(current_line, "_rlnDefocusAngle") == true )
            defocus_angle_column = current_column;
        else if ( StartsWith(current_line, "_rlnPhaseShift") == true )
            phase_shift_column = current_column;
        else if ( StartsWith(current_line, "_rlnMicrographName") == true )
            micrograph_name_column = current_column;
        else if ( StartsWith(current_line, "_rlnRandomSubset") == true )
            random_subset_column = current_column;
        else if ( StartsWith(current_line, "_rlnImageName") == true )
            image_name_column = current_column;

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
*/
    if ( defocus1_column == -1 ) {
        MyPrintWithDetails("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
        if ( error_string != NULL )
            *error_string = Format("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
        return false;
    }

    if ( defocus2_column == -1 ) {
        MyPrintWithDetails("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
        if ( error_string != NULL )
            *error_string = Format("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
        return false;
    }

    if ( defocus_angle_column == -1 ) {
        MyPrintWithDetails("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
        if ( error_string != NULL )
            *error_string = Format("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
        return false;
    }

    if ( phase_shift_column == -1 ) {
        //	MyPrintWithDetails("Warning: Couldn't find _rlnPhaseShift in star file (%s) - phase shift will be set to 0.0\n", wanted_filename);
    }

    // we have the headers, the current line should be the first parameter to extract the info

    if ( ExtractParametersFromLine(current_line, error_string) == false )
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

        if ( ExtractParametersFromLine(current_line, error_string) == false )
            return false;
    }

    return true;
}
