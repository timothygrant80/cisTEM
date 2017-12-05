#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofStarFileParameters);

StarFileParameters::StarFileParameters()
{
	position_in_stack = -1;
	phi = 0;
	theta = 0;
	psi = 0;
	x_shift = 0;
	y_shift = 0;
	defocus1 = 0;
	defocus2 = 0;
	defocus_angle = 0;
	phase_shift = 0;
}

BasicStarFileReader::BasicStarFileReader()
{
	filename = "";
//	input_file_stream = NULL;
//	input_text_stream = NULL;

	input_file = NULL;

	current_position_in_stack = 0;
	current_column = 0;

	phi_column = -1;
	theta_column = -1;
	psi_column = -1;
	xshift_column = -1;
	yshift_column = -1;
	defocus1_column = -1;
	defocus2_column = -1;
	defocus_angle_column = -1;
	phase_shift_column = -1;

}

BasicStarFileReader::BasicStarFileReader(wxString wanted_filename)
{
	ReadFile(wanted_filename);
}

BasicStarFileReader::~BasicStarFileReader()
{
	Close();
}

void BasicStarFileReader::Open(wxString wanted_filename)
{
	Close();
	cached_parameters.Clear();

	filename = wanted_filename;

//	input_file_stream = new wxFileInputStream(wanted_filename);
//	input_text_stream = new wxTextInputStream(*input_file_stream);

	input_file = new wxTextFile(wanted_filename);
	input_file->Open();

	//if (input_file_stream->IsOk() == false)
	if (input_file->IsOpened() == false)
	{
		MyPrintWithDetails("Error: Cannot open star file (%s) for read\n", wanted_filename);
		DEBUG_ABORT;
	}
}

void BasicStarFileReader::Close()
{
	cached_parameters.Clear();

/*	if (input_text_stream != NULL) delete input_text_stream;
	if (input_file_stream != NULL)
	{
		if (input_file_stream->GetFile()->IsOpened() == true) input_file_stream->GetFile()->Close();
		delete input_file_stream;
	}

	input_file_stream = NULL;
	input_text_stream = NULL;*/

	if (input_file != NULL) delete input_file;

}

bool BasicStarFileReader::ExtractParametersFromLine(wxString &wanted_line, wxString *error_string)
{
	// extract info.

	wxArrayString all_tokens;
	wxStringTokenizer tokens(wanted_line);
	StarFileParameters temp_parameters;
	double temp_double;

	while (tokens.HasMoreTokens() == true)
	{
		all_tokens.Add(tokens.GetNextToken());
	}

	current_position_in_stack++;
	temp_parameters.position_in_stack = current_position_in_stack;

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

	if (xshift_column == -1) temp_double = 0.0;
	else
	if (all_tokens[xshift_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[xshift_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[xshift_column]);
		return false;
	}

	temp_parameters.x_shift = float(temp_double);

	// yshift

	if (yshift_column == -1) temp_double = 0.0;
	else
	if (all_tokens[yshift_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[yshift_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[yshift_column]);
		return false;
	}

	temp_parameters.y_shift = float(temp_double);

	// defocus1

	if (all_tokens[defocus1_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus1_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[defocus1_column]);
		return false;
	}

	temp_parameters.defocus1 = float(temp_double);

	// defocus2

	if (all_tokens[defocus2_column].ToDouble(&temp_double) == false)
	{
		MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[defocus2_column]);
		if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[defocus2_column]);
		return false;
	}

	temp_parameters.defocus2 = float(temp_double);

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

	cached_parameters.Add(temp_parameters);

	return true;

}



bool BasicStarFileReader::ReadFile(wxString wanted_filename, wxString *error_string)
{
	Open(wanted_filename);
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

	    if (current_line.StartsWith("_rlnAngleRot") == true) phi_column = current_column;
	    else
		if (current_line.StartsWith("_rlnAngleTilt") == true) theta_column = current_column;
		else
		if (current_line.StartsWith("_rlnAnglePsi") == true) psi_column = current_column;
		else
		if (current_line.StartsWith("_rlnOriginX") == true) xshift_column = current_column;
		else
		if (current_line.StartsWith("_rlnOriginY") == true) yshift_column = current_column;
		else
		if (current_line.StartsWith("_rlnDefocusU") == true) defocus1_column = current_column;
		else
		if (current_line.StartsWith("_rlnDefocusV") == true) defocus2_column = current_column;
		else
		if (current_line.StartsWith("_rlnDefocusAngle") == true) defocus_angle_column = current_column;
		else
		if (current_line.StartsWith("_rlnPhaseShift") == true) phase_shift_column = current_column;

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
*/
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

	// we have the headers, the current line should be the first parameter to extract the info

	if (ExtractParametersFromLine(current_line, error_string) == false) return false;

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

		if (ExtractParametersFromLine(current_line, error_string) == false) return false;
	}

	return true;
}

