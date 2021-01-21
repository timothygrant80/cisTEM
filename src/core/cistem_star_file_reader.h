// **************************************
// ADDING NEW COLUMNS TO THE DATA FILES..
// **************************************
/*

cistem_parameters.h
-------------------

1. Add a bitwise definition for your data type to the top of cistem_parameters.h
2. Add it as a new member variable to  cisTEMParameterLine in cistem_parameters.h
3. Add it as a new member variable to  cisTEMParameterMask in cistem_parameters.h
4. Add a new method to return the parameter from a given line e.g. ReturnPositionInStack

cistem_parameters.cpp
---------------------

1. Add it to void cisTEMParameterMask::SetAllToTrue(), void cisTEMParameterMask::SetAllToFalse() and cisTEMParameterMask::SetActiveParameters()
2. Add it to cisTEMParameterLine::SetAllToZero() and  cisTEMParameterLine::ReplaceNanAndInfWithOther if it is a number
3. If it makes sense to add this parameter etc, add it to  cisTEMParameterLine::Add, cisTEMParameterLine::Subtract,  cisTEMParameterLine::AddSquare
4. Add it to cisTEMParameters:: ReturnNumberOfParametersToWrite
5. Add it to cisTEMParameters::WriteTocisTEMBinaryFile - you will need to add it 2 places, the first is in the block that currently ends at line ~940, and looks like this:-

	if (parameters_to_write.total_exposure == true)
	{
		bitmask_identifier = TOTAL_EXPOSURE;
		data_type = VARIABLE_LENGTH;
		fwrite ( &bitmask_identifier, sizeof(long), 1, cisTEM_bin_file );
		fwrite ( &data_type, sizeof(char), 1, cisTEM_bin_file );
	}

You will need to change parameters_to_write.total_exposure to your variable, bitmask_identifier = to your identifier and data_type to your
data type (it can be INTEGER, UNSIGNED_INTEGER, FLOAT, LONG, BYTE, DOUBLE or VARIABLE)

You will then have to add it to loop where the data values are written out - this currently ends at line ~996, and looks like this :-

	if (parameters_to_write.total_exposure == true) fwrite ( &all_parameters[particle_counter].total_exposure, sizeof(float), 1, cisTEM_bin_file );

you need to change parameters_to_write.total_exposure and all_parameters[particle_counter].total_exposure to your variable.  Then make sure you change float to your data type.

6. Add it to cisTEMParameters::WriteTocisTEMStarFile - in a few places.

Firstly, the block that currently ends at ~line 1240 and looks like :-

	if (parameters_to_write.total_exposure == true)
	{
		fprintf(cisTEM_star_file, "_cisTEMTotalExposure #%i\n", column_counter);
		column_counter++;
	}

Changeparameters_to_write.total_exposure to your variable, and _cisTEMTotalExposure to whatever you want the star file descriptor to be for your variable.

Add a header for your variable to the block below which looks like :-

	if (parameters_to_write.total_exposure == true) 					data_line += " TOTEXP ";

Finally, add it to the loop that writes the actual data, which currently ends at line ~1325 and looks like :-

	if (parameters_to_write.total_exposure == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].total_exposure);

Change parameters_to_write.total_exposure and all_parameters[particle_counter].total_exposure to your variable, and change "%7.2f " to whatever formatting is
suitable for your variable.

cistem_star_file_reader.h
-------------------------

1. Add an int member variable to cisTEMStarFileReader to hold the found column for your variable (e.g. int total_exposure_column)
2. Also add an inline method to return your variable (e.g. 	inline int ReturnTotalExpsosure(int line_number) {return cached_parameters->Item(line_number).total_exposure;})

cistem_star_file_reader.cpp
---------------------------

1. Add your variable to cisTEMStarFileReader::ResetColumnPositions
2. Add your variable to cisTEMStarFileReader::ExtractParametersFromLine.

Add it to the block which currently ends ~line 675 and looks like this :-

	if ( total_exposure_column == -1) temp_parameters.total_exposure = 0.0f;
	else
	{
		if (all_tokens[total_exposure_column].ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[total_exposure_column]);
			if (error_string != NULL) *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[total_exposure_column]);
			return false;
		}

		temp_parameters.total_exposure = float(temp_double);
	}

change the comment to be your variable, change  total_exposure_column to be your column variable (thisis on 3 lines), and temp_parameters.total_exposure (2 different places)to be your variable, change the = set the variable
to the default value that should be taken if the column does not exist.

3.Change cisTEMStarFileReader::ReadFile.  Add to the section which currently ends at line ~1085 and looks like :-

		if (current_line.StartsWith("_cisTEMStackFilename ") == true)
		{
	    	if (stack_filename_column != -1) wxPrintf("Warning :: _cisTEMStackFilename occurs more than once. I will take the last occurrence\n");
		   	stack_filename_column = current_column;
			parameters_that_were_read.stack_filename = true;
		}

Change _cisTEMStackFilename (2 places) to be the header name which you chose earlier and added to WriteCisTEMStarFile in cistem_parameters.cpp.
Change stack_filename_column to your columns variable and parameters_that_were_read.stack_filename to your variable.

4. Change cisTEMStarFileReader::ReadBinaryFile - You will have to change

Change the section that currently ends at line ~1450 and looks like :-

		if (column_order_buffer[current_column] == STACK_FILENAME)
		{
	    	if (stack_filename_column != -1) wxPrintf("Warning :: _cisTEMStackFilename occurs more than once. I will take the last occurrence\n");
		   	stack_filename_column = current_column;
		   	parameters_that_were_read.stack_filename = true;
		}

Change STACK_FILENAME to be whatever bitwise define you chose at the top of cistem_parameters.h for your variable. Change stack_filename_column (2 places) to be your variable and
_cisTEMStackFilename to be your header name. change parameters_that_were_read.stack_filename to your variable.

Change the section that currently ends at line ~1690 and add your variable.  E.g. for total exposure it looks like :-

		if (column_order_buffer[current_column] == TOTAL_EXPOSURE)
		{
			if (SafelyReadFromBinaryBufferIntoFloat(temp_parameters.total_exposure) == false) return false;
		}

Change TOTAL_EXPOSURE to your bitwise type and temp_parameters.total_exposure to your variable.  You will have to change the SafelyReadFromBinaryBufferInto function to be the correct data type.

console_test.cpp
----------------

Add your variable to TestStarToBinaryFileConversion, setting it to some random variable.

*/

class  cisTEMStarFileReader {

private:

	int current_position_in_stack;
	int current_column;

	int 	position_in_stack_column;
	int 	image_is_active_column;
	int	 	psi_column;
	int		theta_column;
	int		phi_column;
	int		x_shift_column;
	int		y_shift_column;
	int		defocus_1_column;
	int		defocus_2_column;
	int		defocus_angle_column;
	int		phase_shift_column;
	int		occupancy_column;
	int		logp_column;
	int		sigma_column;
	int		score_column;
	int		score_change_column;
	int		pixel_size_column;
	int		microscope_voltage_kv_column;
	int		microscope_spherical_aberration_mm_column;
	int		amplitude_contrast_column;
	int		beam_tilt_x_column;
	int		beam_tilt_y_column;
	int		image_shift_x_column;
	int		image_shift_y_column;
	int		stack_filename_column;
	int     original_image_filename_column;
	int     reference_3d_filename_column;
	int     best_2d_class_column;
	int     beam_tilt_group_column;
	int		particle_group_column;
	int 	assigned_subset_column;
	int		pre_exposure_column;
	int		total_exposure_column;

	long binary_buffer_position;

	// The following "Safely" functions are to read data from the buffer with error checking to make sure there is no segfault

	inline bool SafelyReadFromBinaryBufferIntoInteger(int &integer_to_read_into)
	{
		if (binary_buffer_position + sizeof(int) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short");
			return false;
		}

		int *temp_int_pointer = reinterpret_cast <int *> (&binary_file_read_buffer[binary_buffer_position]);
		integer_to_read_into = *temp_int_pointer;
		binary_buffer_position += sizeof(int);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntoUnsignedInteger(unsigned int &unsigned_integer_to_read_into)
	{
		if (binary_buffer_position + sizeof(unsigned int) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short");
			return false;
		}

		unsigned int *temp_int_pointer = reinterpret_cast <unsigned int *> (&binary_file_read_buffer[binary_buffer_position]);
		unsigned_integer_to_read_into = *temp_int_pointer;
		binary_buffer_position += sizeof(unsigned int);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntoFloat(float &float_to_read_into)
	{
		if (binary_buffer_position + sizeof(float) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short\n");
			return false;
		}

		float *temp_float_pointer = reinterpret_cast <float *> (&binary_file_read_buffer[binary_buffer_position]);
		float_to_read_into = *temp_float_pointer;
		binary_buffer_position += sizeof(float);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntoLong(long &long_to_read_into)
	{
		if (binary_buffer_position + sizeof(long) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short\n");
			return false;
		}

		long *temp_long_pointer = reinterpret_cast <long *> (&binary_file_read_buffer[binary_buffer_position]);
		long_to_read_into = *temp_long_pointer;
		binary_buffer_position += sizeof(long);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntoChar(char &char_to_read_into)
	{
		if (binary_buffer_position + sizeof(char) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short\n");
			return false;
		}

		char *temp_long_pointer = &binary_file_read_buffer[binary_buffer_position];
		char_to_read_into = *temp_long_pointer;
		binary_buffer_position += sizeof(char);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntoDouble(double &double_to_read_into)
	{
		if (binary_buffer_position + sizeof(double) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short\n");
			return false;
		}

		double *temp_double_pointer = reinterpret_cast <double *> (&binary_file_read_buffer[binary_buffer_position]);
		double_to_read_into = *temp_double_pointer;
		binary_buffer_position += sizeof(double);
		return true;
	}

	inline bool SafelyReadFromBinaryBufferIntowxString(wxString &wxstring_to_read_into)
	{
		int length_of_string;
		if (SafelyReadFromBinaryBufferIntoInteger(length_of_string) == false) return false;

		if (length_of_string < 0)
		{
			MyPrintWithDetails("Error Reading string, length is %i", length_of_string);
		}

		if (binary_buffer_position + length_of_string * sizeof(char) - 1 >= binary_file_size)
		{
			MyPrintWithDetails("Error: Binary file is too short\n");
			return false;
		}

		char string_buffer[length_of_string + 1];

		for (int array_counter = 0; array_counter < length_of_string; array_counter++)
		{
			string_buffer[array_counter] = binary_file_read_buffer[binary_buffer_position + array_counter];
		}

		string_buffer[length_of_string] = 0;
		wxstring_to_read_into = string_buffer;

		binary_buffer_position += sizeof(char) * length_of_string;
		return true;
	}

public:

	wxString    filename;
	wxTextFile *input_text_file;

	char *binary_file_read_buffer;
	long binary_file_size;

	bool using_external_array;

	ArrayOfcisTEMParameterLines *cached_parameters;
	cisTEMParameterMask 		parameters_that_were_read;

	cisTEMStarFileReader();
	~cisTEMStarFileReader();

	cisTEMStarFileReader(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);

	void Open(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool read_as_binary = false);
	void Close();
	bool ReadFile(wxString wanted_filename, wxString *error_string = NULL, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);
	bool ReadBinaryFile(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);


	bool ExtractParametersFromLine(wxString &wanted_line, wxString *error_string = NULL, bool exclude_negative_film_numbers = false);
	void Reset();
	void ResetColumnPositions();

	inline int   ReturnPositionInStack(int line_number) { return cached_parameters->Item(line_number).position_in_stack;}
	inline int   ReturnImageIsActive(int line_number) { return cached_parameters->Item(line_number).image_is_active;}
	inline float ReturnPhi(int line_number) { return cached_parameters->Item(line_number).phi;}
	inline float ReturnTheta(int line_number) { return cached_parameters->Item(line_number).theta;}
	inline float ReturnPsi(int line_number) { return cached_parameters->Item(line_number).psi;}
	inline float ReturnXShift(int line_number) { return cached_parameters->Item(line_number).x_shift;}
	inline float ReturnYShift(int line_number) { return cached_parameters->Item(line_number).y_shift;}
	inline float ReturnDefocus1(int line_number) { return cached_parameters->Item(line_number).defocus_1;}
	inline float ReturnDefocus2(int line_number) { return cached_parameters->Item(line_number).defocus_2;}
	inline float ReturnDefocusAngle(int line_number) { return cached_parameters->Item(line_number).defocus_angle;}
	inline float ReturnPhaseShift(int line_number) { return cached_parameters->Item(line_number).phase_shift;}
	inline int   ReturnLogP(int line_number) { return cached_parameters->Item(line_number).logp;}
	inline float ReturnSigma(int line_number) { return cached_parameters->Item(line_number).sigma;}
	inline float ReturnScore(int line_number) { return cached_parameters->Item(line_number).score;}
	inline float ReturnScoreChange(int line_number) { return cached_parameters->Item(line_number).score_change;}
	inline float ReturnPixelSize(int line_number) { return cached_parameters->Item(line_number).pixel_size;}
	inline float ReturnMicroscopekV(int line_number) { return cached_parameters->Item(line_number).microscope_voltage_kv;}
	inline float ReturnMicroscopeCs(int line_number) { return cached_parameters->Item(line_number).microscope_spherical_aberration_mm;}
	inline float ReturnAmplitudeContrast(int line_number) { return cached_parameters->Item(line_number).amplitude_contrast;}
	inline float ReturnBeamTiltX(int line_number) { return cached_parameters->Item(line_number).beam_tilt_x;}
	inline float ReturnBeamTiltY(int line_number) { return cached_parameters->Item(line_number).beam_tilt_y;}
	inline float ReturnImageShiftX(int line_number) { return cached_parameters->Item(line_number).image_shift_x;}
	inline float ReturnImageShiftY(int line_number) { return cached_parameters->Item(line_number).image_shift_y;}
	inline wxString	ReturnStackFilename(int line_number) {return cached_parameters->Item(line_number).stack_filename;}
	inline wxString ReturnOriginalImageFilename(int line_number) {return cached_parameters->Item(line_number).original_image_filename;}
	inline wxString ReturnReference3DFilename(int line_number) {return cached_parameters->Item(line_number).reference_3d_filename;}
	inline int ReturnBest2DClass(int line_number) {return cached_parameters->Item(line_number).best_2d_class;}
	inline int ReturnBeamTiltGroup(int line_number) {return cached_parameters->Item(line_number).beam_tilt_group;}
	inline int ReturnParticleGroup(int line_number) {return cached_parameters->Item(line_number).particle_group;}
	inline int ReturnAssignedSubset(int line_number) {return cached_parameters->Item(line_number).assigned_subset;}
	inline int ReturnPreExposure(int line_number) {return cached_parameters->Item(line_number).pre_exposure;}
	inline int ReturnTotalExpsosure(int line_number) {return cached_parameters->Item(line_number).total_exposure;}


};
