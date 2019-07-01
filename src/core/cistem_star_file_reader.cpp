#include "core_headers.h"

cisTEMStarFileReader::cisTEMStarFileReader()
{
	filename = "";
	input_file = NULL;

	current_position_in_stack = 0;
	current_column = 0;

	cached_parameters = NULL;
	using_external_array = false;

	position_in_stack_column = -1;
	image_is_active_column = -1;
	psi_column = -1;
	theta_column = -1;
	phi_column = -1;
	x_shift_column = -1;
	y_shift_column = -1;
	defocus_1_column = -1;
	defocus_2_column = -1;
	defocus_angle_column = -1;
	phase_shift_column = -1;
	occupancy_column = -1;
	logp_column = -1;
	sigma_column = -1;
	score_column = -1;
	score_change_column = -1;
	pixel_size_column = -1;
	microscope_voltage_kv_column = -1;
	microscope_spherical_aberration_mm_column = -1;
	amplitude_contrast_column = -1;
	beam_tilt_x_column = -1;
	beam_tilt_y_column = -1;
	image_shift_x_column = -1;
	image_shift_y_column = -1;
}

cisTEMStarFileReader::cisTEMStarFileReader(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer, bool exclude_negative_film_numbers)
{
	filename = "";
	input_file = NULL;

	current_position_in_stack = 0;
	current_column = 0;

	cached_parameters = NULL;

	position_in_stack_column = -1;
	image_is_active_column = -1;
	psi_column = -1;
	theta_column = -1;
	phi_column = -1;
	x_shift_column = -1;
	y_shift_column = -1;
	defocus_1_column = -1;
	defocus_2_column = -1;
	defocus_angle_column = -1;
	phase_shift_column = -1;
	occupancy_column = -1;
	logp_column = -1;
	sigma_column = -1;
	score_column = -1;
	score_change_column = -1;
	pixel_size_column = -1;
	microscope_voltage_kv_column = -1;
	microscope_spherical_aberration_mm_column = -1;
	amplitude_contrast_column = -1;
	beam_tilt_x_column = -1;
	beam_tilt_y_column = -1;
	image_shift_x_column = -1;
	image_shift_y_column = -1;

	if (alternate_cached_parameters_pointer == NULL)
	{
		cached_parameters = new ArrayOfcisTEMParameterLines;
		using_external_array = false;
	}
	else
	{
		cached_parameters = alternate_cached_parameters_pointer;
		using_external_array = true;
	}

	ReadFile(wanted_filename, NULL, alternate_cached_parameters_pointer, exclude_negative_film_numbers);
}

cisTEMStarFileReader::~cisTEMStarFileReader()
{
	Close();
}

void cisTEMStarFileReader::Open(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer)
{
	Close();

	if (cached_parameters != NULL && using_external_array == false)
	{
		delete cached_parameters;
		cached_parameters = NULL;
	}
	filename = wanted_filename;

	input_file = new wxTextFile(wanted_filename);
	input_file->Open();

	//if (input_file_stream->IsOk() == false)
	if (input_file->IsOpened() == false)
	{
		MyPrintWithDetails("Error: Cannot open star file (%s) for read\n", wanted_filename);
		DEBUG_ABORT;
	}

	if (alternate_cached_parameters_pointer == NULL)
	{
		cached_parameters = new ArrayOfcisTEMParameterLines;
		using_external_array = false;
	}
	else
	{
		cached_parameters = alternate_cached_parameters_pointer;
		using_external_array = true;
	}
}

void cisTEMStarFileReader::Close()
{
	if (cached_parameters != NULL && using_external_array == false)
	{
		delete cached_parameters;
		cached_parameters = NULL;
	}
	if (input_file != NULL) delete input_file;

}

bool cisTEMStarFileReader::ExtractParametersFromLine(wxString &wanted_line, wxString *error_string, bool exclude_negative_film_numbers)
{
	// extract info.

	wxArrayString all_tokens;
	wxStringTokenizer tokens(wanted_line);
	cisTEMParameterLine temp_parameters;

	double temp_double;
	long temp_long;

	while (tokens.HasMoreTokens() == true)
	{
		all_tokens.Add(tokens.GetNextToken());
	}

	// start with image is active, because sometimes we can just stop..

	// image is active

	if (image_is_active_column == -1) temp_parameters.image_is_active = 1.0;
	else
	{
		if (all_tokens[image_is_active_column].ToLong(&temp_long) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_is_active_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[image_is_active_column]);
			return false;
		}

		temp_parameters.image_is_active = int(temp_long);
	}

	if (temp_parameters.image_is_active < 0 && exclude_negative_film_numbers == true) return true;

	// position in stack

	if (position_in_stack_column == -1) temp_double = -1;
	else
	if (all_tokens[position_in_stack_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[position_in_stack_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[position_in_stack_column]);
		return false;
	}

	temp_parameters.position_in_stack = int(temp_double);


	// phi

	if (phi_column == -1) temp_double = 0.0;
	else
	if (all_tokens[phi_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[phi_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[phi_column]);
		return false;
	}

	temp_parameters.phi = float(temp_double);

	// theta

	if (theta_column == -1) temp_double = 0.0;
	else
	if (all_tokens[theta_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[theta_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[theta_column]);
		return false;
	}

	temp_parameters.theta = float(temp_double);

	// psi

	if (psi_column == -1) temp_double = 0.0;
	else
	if (all_tokens[psi_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[psi_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[psi_column]);
		return false;
	}

	temp_parameters.psi = float(temp_double);

	// xshift

	if (x_shift_column == -1) temp_double = 0.0;
	else
	if (all_tokens[x_shift_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[x_shift_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[x_shift_column]);
		return false;
	}

	temp_parameters.x_shift = float(temp_double);

	// yshift

	if (y_shift_column == -1) temp_double = 0.0;
	else
	if (all_tokens[y_shift_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[y_shift_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[y_shift_column]);
		return false;
	}

	temp_parameters.y_shift = float(temp_double);

	// defocus1

	if (all_tokens[defocus_1_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_1_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[defocus_1_column]);
		return false;
	}

	temp_parameters.defocus_1 = float(temp_double);

	// defocus2

	if (all_tokens[defocus_2_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_2_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[defocus_2_column]);
		return false;
	}

	temp_parameters.defocus_2 = float(temp_double);

	// defocus_angle

	if (all_tokens[defocus_angle_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus_angle_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[defocus_angle_column]);
		return false;
	}

	temp_parameters.defocus_angle = float(temp_double);

	// phase_shift

	if (phase_shift_column == -1) temp_parameters.phase_shift = 0.0;
	else
	{
		if (all_tokens[phase_shift_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[phase_shift_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[phase_shift_column]);
			return false;
		}

		temp_parameters.phase_shift = deg_2_rad(float(temp_double));
	}

	// occupancy

	if (occupancy_column == -1) temp_parameters.occupancy = 100.0f;
	else
	{
		if (all_tokens[occupancy_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[occupancy_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[occupancy_column]);
			return false;
		}

		temp_parameters.occupancy = float(temp_double);
	}

	// logp

	if (logp_column == -1) temp_parameters.logp = 100.0f;
	else
	{
		if (all_tokens[logp_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[logp_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[logp_column]);
			return false;
		}

		temp_parameters.logp = float(temp_double);
	}

	// sigma

	if (sigma_column == -1) temp_parameters.sigma = 10.0f;
	else
	{
		if (all_tokens[sigma_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[sigma_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[sigma_column]);
			return false;
		}

		temp_parameters.sigma = float(temp_double);
	}

	// score

	if (score_column == -1) temp_parameters.score = 0.0f;
	else
	{
		if (all_tokens[score_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[score_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[score_column]);
			return false;
		}

		temp_parameters.score = float(temp_double);
		//wxPrintf("Score = %f, image_is_active = %i, exclude negative = %s\n", temp_parameters.score, temp_parameters.image_is_active, BoolToYesNo(exclude_negative_film_numbers));
	}

	// score_change

	if (score_change_column == -1) temp_parameters.score_change = 0.0f;
	else
	{
		if (all_tokens[score_change_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[score_change_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[score_change_column]);
			return false;
		}

		temp_parameters.score_change = float(temp_double);
	}

	// pixel_size

	if (pixel_size_column == -1) temp_parameters.pixel_size = 0.0f;
	else
	{
		if (all_tokens[pixel_size_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[pixel_size_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[pixel_size_column]);
			return false;
		}

		temp_parameters.pixel_size = float(temp_double);
	}

	// voltage

	if (microscope_voltage_kv_column == -1) temp_parameters.microscope_voltage_kv = 0.0f;
	else
	{
		if (all_tokens[microscope_voltage_kv_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[microscope_voltage_kv_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[microscope_voltage_kv_column]);
			return false;
		}

		temp_parameters.microscope_voltage_kv = float(temp_double);
	}

	// Cs

	if ( microscope_spherical_aberration_mm_column == -1) temp_parameters.microscope_spherical_aberration_mm = 2.7f;
	else
	{
		if (all_tokens[microscope_spherical_aberration_mm_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[microscope_spherical_aberration_mm_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[microscope_spherical_aberration_mm_column]);
			return false;
		}

		temp_parameters.microscope_spherical_aberration_mm = float(temp_double);
	}

	// amplitude_contrast

	if ( amplitude_contrast_column == -1) temp_parameters.amplitude_contrast = 0.07f;
	else
	{
		if (all_tokens[amplitude_contrast_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[amplitude_contrast_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[amplitude_contrast_column]);
			return false;
		}

		temp_parameters.amplitude_contrast = float(temp_double);
	}

	// beam_tilt_x

	if ( beam_tilt_x_column == -1) temp_parameters.beam_tilt_x = 0.0f;
	else
	{
		if (all_tokens[beam_tilt_x_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_x_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_x_column]);
			return false;
		}

		temp_parameters.beam_tilt_x = float(temp_double);
	}

	// beam_tilt_y

	if ( beam_tilt_y_column == -1) temp_parameters.beam_tilt_y = 0.0f;
	else
	{
		if (all_tokens[beam_tilt_y_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_y_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[beam_tilt_y_column]);
			return false;
		}

		temp_parameters.beam_tilt_y = float(temp_double);
	}

	// image_shift_x

	if ( image_shift_x_column == -1) temp_parameters.image_shift_x = 0.0f;
	else
	{
		if (all_tokens[image_shift_x_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_shift_x_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[image_shift_x_column]);
			return false;
		}

		temp_parameters.image_shift_x = float(temp_double);
	}

	// image_shift_y

	if ( image_shift_y_column == -1) temp_parameters.image_shift_y = 0.0f;
	else
	{
		if (all_tokens[image_shift_y_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[image_shift_y_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[image_shift_y_column]);
			return false;
		}

		temp_parameters.image_shift_y = float(temp_double);
	}

	cached_parameters->Add(temp_parameters);

	return true;

}



bool cisTEMStarFileReader::ReadFile(wxString wanted_filename, wxString *error_string, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer, bool exclude_negative_film_numbers)
{
	Open(wanted_filename, alternate_cached_parameters_pointer);
	wxString current_line;

	//MyDebugAssertTrue(input_file_stream != NULL, "FileStream is NULL!");
	MyDebugAssertTrue(input_file->IsOpened(), "File not open");

	bool found_valid_data_block = false;
	bool found_valid_loop_block = false;

	input_file->GoToLine(-1);
	// find a data block

	//while (input_file_stream->Eof() == false)
	while (input_file->Eof() == false)
	{
		//current_line = input_text_stream->ReadLine();
		current_line = input_file->GetNextLine();
		current_line = current_line.Trim(true);
		current_line = current_line.Trim(false);
		if (current_line.Find("data_") != wxNOT_FOUND)
		{
			found_valid_data_block = true;
			break;
		}
	}

	if (found_valid_data_block == false)
	{
		MyPrintWithDetails("Error: Couldn't find a valid data block in star file (%s)\n", wanted_filename);

		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find a valid data block in star file (%s)\n", wanted_filename);
		return false;
	}

	// find a loop block

	//while (input_file_stream->Eof() == false)
	while (input_file->Eof() == false)
	{
		//current_line = input_text_stream->ReadLine();
		current_line = input_file->GetNextLine();
		current_line = current_line.Trim(true);
		current_line = current_line.Trim(false);

		if (current_line.Find("loop_") != wxNOT_FOUND)
		{
			found_valid_loop_block = true;
			break;
		}
	}

	if (found_valid_loop_block == false)
	{
		MyPrintWithDetails("Error: Couldn't find a valid loop block in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find a valid loop block in star file (%s)\n", wanted_filename);
		return false;
	}

	// now we can get headers..

	//while (input_file_stream->Eof() == false)
	while (input_file->Eof() == false)
	{
		//current_line = input_text_stream->ReadLine();
		current_line = input_file->GetNextLine();
		current_line = current_line.Trim(true);
		current_line = current_line.Trim(false);

	    if (current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';') continue;
	    if (current_line[0] != '_') break;

	    // otherwise it is a label, is it a label we want though?

	    if (current_line.StartsWith("_cisTEMPositionInStack ") == true) position_in_stack_column = current_column;
	    else
	    if (current_line.StartsWith("_cisTEMAnglePsi ") == true) psi_column = current_column;
	    else
		if (current_line.StartsWith("_cisTEMAngleTheta ") == true) theta_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMAnglePhi ") == true) phi_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMXShift ") == true) x_shift_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMYShift ") == true) y_shift_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMDefocus1 ") == true) defocus_1_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMDefocus2 ") == true) defocus_2_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMDefocusAngle ") == true) defocus_angle_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMPhaseShift ") == true) phase_shift_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMImageActivity ") == true) image_is_active_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMOccupancy ") == true) occupancy_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMLogP ") == true) logp_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMSigma ") == true) sigma_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMScore ") == true) score_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMScoreChange ") == true) score_change_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMPixelSize ") == true) pixel_size_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMMicroscopeVoltagekV ") == true) microscope_voltage_kv_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMMicroscopeCsMM ") == true) microscope_spherical_aberration_mm_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMAmplitudeContrast ") == true) amplitude_contrast_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMBeamTiltX ") == true) beam_tilt_x_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMBeamTiltY ") == true) beam_tilt_y_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMImageShiftX ") == true) image_shift_x_column = current_column;
		else
		if (current_line.StartsWith("_cisTEMImageShiftY ") == true) image_shift_y_column = current_column;

	    current_column++;
	}

	// quick checks we have all the desired info.
/*
	if (phi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnAngleRot in star file (%s)\n", wanted_filename);
		return false;
	}

	if (theta_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnAngleTilt in star file (%s)\n", wanted_filename);
		return false;
	}

	if (psi_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnAnglePsi in star file (%s)\n", wanted_filename);
		return false;
	}

	if (xshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnOriginX in star file (%s)\n", wanted_filename);
		return false;
	}

	if (yshift_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnOriginY in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus1_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnDefocusU in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus2_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnDefocusV in star file (%s)\n", wanted_filename);
		return false;
	}

	if (defocus_angle_column == -1)
	{
		MyPrintWithDetails("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		if (error_string != NULL) *error_string = wxString::Format("Error: Couldn't find _rlnDefocusAngle in star file (%s)\n", wanted_filename);
		return false;
	}

	if (phase_shift_column == -1)
	{
	//	MyPrintWithDetails("Warning: Couldn't find _rlnPhaseShift in star file (%s) - phase shift will be set to 0.0\n", wanted_filename);
	}

*/

	// we have the headers, the current line should be the first parameter to extract the info

	if (ExtractParametersFromLine(current_line, error_string, exclude_negative_film_numbers) == false) return false;

	// loop over the data lines and fill in..

	//while (input_file_stream->Eof() == false)
	while (input_file->Eof() == false)
	{
		//current_line = input_text_stream->ReadLine();
		current_line = input_file->GetNextLine();
		current_line = current_line.Trim(true);
		current_line = current_line.Trim(false);

		if (current_line.IsEmpty() == true) break;
		if (current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';') continue;

		if (ExtractParametersFromLine(current_line, error_string, exclude_negative_film_numbers) == false) return false;
	}

	return true;
}

