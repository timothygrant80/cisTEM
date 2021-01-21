/*  \brief  class for cisTEM parameters */

#define POSITION_IN_STACK 			1
#define IMAGE_IS_ACTIVE 			2
#define PSI				 			4
#define X_SHIFT						8
#define Y_SHIFT						16
#define DEFOCUS_1					32
#define DEFOCUS_2					64
#define DEFOCUS_ANGLE				128
#define PHASE_SHIFT					256
#define OCCUPANCY					512
#define LOGP						1024
#define SIGMA						2048
#define SCORE						4096
#define SCORE_CHANGE				8192
#define PIXEL_SIZE					16384
#define MICROSCOPE_VOLTAGE			32768
#define	MICROSCOPE_CS				65536
#define AMPLITUDE_CONTRAST			131072
#define BEAM_TILT_X					262144
#define BEAM_TILT_Y					524288
#define IMAGE_SHIFT_X				1048576
#define IMAGE_SHIFT_Y				2097152
#define THETA						4194304
#define PHI							8388608
#define STACK_FILENAME				16777216
#define ORIGINAL_IMAGE_FILENAME 	33554432
#define REFERENCE_3D_FILENAME		67108864
#define BEST_2D_CLASS				134217728
#define BEAM_TILT_GROUP				268435456
#define PARTICLE_GROUP				536870912
#define PRE_EXPOSURE				1073741824
#define TOTAL_EXPOSURE				2147483648
#define ASSIGNED_SUBSET				4294967296

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

class cisTEMParameterLine {

public:

	unsigned int 	position_in_stack;
	int 			image_is_active;
	float		 	psi;
	float			theta;
	float			phi;
	float			x_shift;
	float			y_shift;
	float			defocus_1;
	float			defocus_2;
	float			defocus_angle;
	float			phase_shift;
	float			occupancy;
	float			logp;
	float			sigma;
	float 			score;
	float			score_change;
	float			pixel_size;
	float			microscope_voltage_kv;
	float			microscope_spherical_aberration_mm;
	float			amplitude_contrast;
	float			beam_tilt_x;
	float			beam_tilt_y;
	float			image_shift_x;
	float			image_shift_y;
	wxString		stack_filename;
	wxString		original_image_filename;
	wxString		reference_3d_filename;
	int				best_2d_class;
	int 			beam_tilt_group; // identify particles expected to have the same beam tilt parameters
	int				particle_group;  // identify images of the same particle (all images of a given particle should have the same PARTICLE_GROUP number. E.g. across a tilt-series or movie, i.e. a frame-series))
	int				assigned_subset; // used for example to assign particles to half-datasets, half-maps for the purposes of FSCs
	float			pre_exposure;
	float			total_exposure;

	//void SwapPsiAndPhi(); shouldn't need
	void Add(cisTEMParameterLine &line_to_add);
	void Subtract(cisTEMParameterLine &line_to_subtract);
	void AddSquare(cisTEMParameterLine &line_to_add);

	void SetAllToZero();
	void SetAllToDefault();
	void ReplaceNanAndInfWithOther(cisTEMParameterLine &other_params);




	cisTEMParameterLine();
	~cisTEMParameterLine();
};

class cisTEMParameterMask {

public:

	bool position_in_stack;
	bool image_is_active;
	bool psi;
	bool theta;
	bool phi;
	bool x_shift;
	bool y_shift;
	bool defocus_1;
	bool defocus_2;
	bool defocus_angle;
	bool phase_shift;
	bool occupancy;
	bool logp;
	bool sigma;
	bool score;
	bool score_change;
	bool pixel_size;
	bool microscope_voltage_kv;
	bool microscope_spherical_aberration_mm;
	bool amplitude_contrast;
	bool beam_tilt_x;
	bool beam_tilt_y;
	bool image_shift_x;
	bool image_shift_y;
	bool stack_filename;
	bool original_image_filename;
	bool reference_3d_filename;
	bool best_2d_class;
	bool beam_tilt_group;
	bool particle_group;
	bool assigned_subset;
	bool pre_exposure;
	bool total_exposure;

	void SetAllToTrue();
	void SetAllToFalse();
	void SetActiveParameters(long parameters_to_set); // uses takes the defines above bitwise

	cisTEMParameterMask();

};

WX_DECLARE_OBJARRAY(cisTEMParameterLine, ArrayOfcisTEMParameterLines);

class cisTEMParameters {


public :

	wxArrayString			    	header_comments;
	ArrayOfcisTEMParameterLines 	all_parameters;

	cisTEMParameterMask 			parameters_to_write;
	cisTEMParameterMask 			parameters_that_were_read;

	// for defocus dependance



	float		average_defocus;
	float		defocus_coeff_a;
	float		defocus_coeff_b;

	cisTEMParameters();
	~cisTEMParameters();

	void ReadFromcisTEMStarFile(wxString wanted_filename, bool exclude_negative_film_numbers = false );
	void ReadFromcisTEMBinaryFile(wxString wanted_filename, bool exclude_negative_film_numbers = false);

	void ReadFromFrealignParFile(wxString wanted_filename,
								float wanted_pixel_size = 0.0f,
								float wanted_microscope_voltage = 0.0f,
								float wanted_microscope_cs = 0.0f,
								float wanted_amplitude_contrast = 0.0f,
								float wanted_beam_tilt_x = 0.0f,
								float wanted_beam_tilt_y = 0.0f,
								float wanted_image_shift_x = 0.0f,
								float wanted_image_shift_y = 0.0f,
								int	  wanted_particle_group = 1,
								float wanted_pre_exposure = 0.0f,
								float wanted_total_exposure = 0.1f);

	int ReturnNumberOfParametersToWrite();
	int ReturnNumberOfLinesToWrite(int first_image_to_write, int last_image_to_write);
	void WriteTocisTEMBinaryFile(wxString wanted_filename, int first_image_to_write = -1, int last_image_to_write = -1);

	void AddCommentToHeader(wxString comment_to_add);
	void WriteTocisTEMStarFile(wxString wanted_filename, int first_line_to_write = -1, int last_line_to_write = -1, int first_image_to_write = -1, int last_image_to_write = -1);

	void ClearAll();

	void PreallocateMemoryAndBlank(int number_to_allocate);

	inline long   ReturnNumberofLines() { return all_parameters.GetCount();}
	inline cisTEMParameterLine ReturnLine(int line_number) { return all_parameters.Item(line_number);}

	inline int   ReturnPositionInStack(int line_number) { return all_parameters.Item(line_number).position_in_stack;}
	inline int   ReturnImageIsActive(int line_number) { return all_parameters.Item(line_number).image_is_active;}
	inline float ReturnPhi(int line_number) { return all_parameters.Item(line_number).phi;}
	inline float ReturnTheta(int line_number) { return all_parameters.Item(line_number).theta;}
	inline float ReturnPsi(int line_number) { return all_parameters.Item(line_number).psi;}
	inline float ReturnXShift(int line_number) { return all_parameters.Item(line_number).x_shift;}
	inline float ReturnYShift(int line_number) { return all_parameters.Item(line_number).y_shift;}
	inline float ReturnDefocus1(int line_number) { return all_parameters.Item(line_number).defocus_1;}
	inline float ReturnDefocus2(int line_number) { return all_parameters.Item(line_number).defocus_2;}
	inline float ReturnDefocusAngle(int line_number) { return all_parameters.Item(line_number).defocus_angle;}
	inline float ReturnPhaseShift(int line_number) { return all_parameters.Item(line_number).phase_shift;}
	inline float ReturnOccupancy(int line_number) { return all_parameters.Item(line_number).occupancy;}
	inline float ReturnLogP(int line_number) { return all_parameters.Item(line_number).logp;}
	inline float ReturnSigma(int line_number) { return all_parameters.Item(line_number).sigma;}
	inline float ReturnScore(int line_number) { return all_parameters.Item(line_number).score;}
	inline float ReturnScoreChange(int line_number) { return all_parameters.Item(line_number).score_change;}
	inline float ReturnPixelSize(int line_number) { return all_parameters.Item(line_number).pixel_size;}
	inline float ReturnMicroscopekV(int line_number) { return all_parameters.Item(line_number).microscope_voltage_kv;}
	inline float ReturnMicroscopeCs(int line_number) { return all_parameters.Item(line_number).microscope_spherical_aberration_mm;}
	inline float ReturnAmplitudeContrast(int line_number) { return all_parameters.Item(line_number).amplitude_contrast;}
	inline float ReturnBeamTiltX(int line_number) { return all_parameters.Item(line_number).beam_tilt_x;}
	inline float ReturnBeamTiltY(int line_number) { return all_parameters.Item(line_number).beam_tilt_y;}
	inline float ReturnImageShiftX(int line_number) { return all_parameters.Item(line_number).image_shift_x;}
	inline float ReturnImageShiftY(int line_number) { return all_parameters.Item(line_number).image_shift_y;}
	inline wxString	ReturnStackFilename(int line_number) {return all_parameters.Item(line_number).stack_filename;}
	inline wxString ReturnOriginalImageFilename(int line_number) {return all_parameters.Item(line_number).original_image_filename;}
	inline wxString ReturnReference3DFilename(int line_number) {return all_parameters.Item(line_number).reference_3d_filename;}
	inline int ReturnBest2DClass(int line_number) {return all_parameters.Item(line_number).best_2d_class;}
	inline int ReturnBeamTiltGroup(int line_number) {return all_parameters.Item(line_number).beam_tilt_group;}
	inline int ReturnParticleGroup(int line_number) {return all_parameters.Item(line_number).particle_group;}
	inline int ReturnAssignedSubset(int line_number) {return all_parameters.Item(line_number).assigned_subset;}
	inline float ReturnPreExposure(int line_number) {return all_parameters.Item(line_number).pre_exposure;}
	inline float ReturnTotalExposure(int line_number) {return all_parameters.Item(line_number).total_exposure;}

	float ReturnAverageSigma(bool exclude_negative_film_numbers = false);
	float ReturnAverageOccupancy(bool exclude_negative_film_numbers = false);
	float ReturnAverageScore(bool exclude_negative_film_numbers = false);

	bool  ContainsMultipleParticleGroups();

	void RemoveSigmaOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers = false, bool reciprocal_square = false);
	void RemoveScoreOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers = false, bool reciprocal_square = false);

	void CalculateDefocusDependence(bool exclude_negative_film_numbers = false);
	void AdjustScores(bool exclude_negative_film_numbers = false);
	float ReturnScoreAdjustment(float defocus);
	float ReturnScoreThreshold(float wanted_percentage, bool exclude_negative_film_numbers = false);

	float ReturnMinScore(bool exclude_negative_film_numbers = false);
	float ReturnMaxScore(bool exclude_negative_film_numbers = false);
	int ReturnMinPositionInStack(bool exclude_negative_film_numbers = false);
	int ReturnMaxPositionInStack(bool exclude_negative_film_numbers = false);

	void SetAllReference3DFilename(wxString wanted_filename);
	void SortByReference3DFilename();


	cisTEMParameterLine ReturnParameterAverages(bool only_average_active = true);
	cisTEMParameterLine ReturnParameterVariances(bool only_average_active = true);

};
