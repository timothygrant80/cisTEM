/*  \brief  class for cisTEM parameters */

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


	//void SwapPsiAndPhi(); shouldn't need
	void Add(cisTEMParameterLine &line_to_add);
	void Subtract(cisTEMParameterLine &line_to_subtract);
	void AddSquare(cisTEMParameterLine &line_to_add);

	void SetAllToZero();
	void ReplaceNanAndInfWithOther(cisTEMParameterLine &other_params);




	cisTEMParameterLine();
	~cisTEMParameterLine();
};

WX_DECLARE_OBJARRAY(cisTEMParameterLine, ArrayOfcisTEMParameterLines);

class cisTEMParameters {


public :

	wxArrayString			     header_comments;
	ArrayOfcisTEMParameterLines  all_parameters;

	// for defocus dependance

	float		average_defocus;
	float		defocus_coeff_a;
	float		defocus_coeff_b;

	cisTEMParameters();
	~cisTEMParameters();

	void ReadFromcisTEMStarFile(wxString wanted_filename, bool exclude_negative_film_numbers = false );
	void ReadFromFrealignParFile(wxString wanted_filename, float wanted_pixel_size = 0.0f, float wanted_microscope_voltage = 0.0f, float wanted_microscope_cs = 0.0f, float wanted_amplitude_contrast = 0.0f, float wanted_beam_tilt_x = 0.0f, float wanted_beam_tilt_y = 0.0f, float wanted_image_shift_x = 0.0f, float wanted_image_shift_y = 0.0f);

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

	float ReturnAverageSigma(bool exclude_negative_film_numbers = false);
	float ReturnAverageOccupancy(bool exclude_negative_film_numbers = false);
	float ReturnAverageScore(bool exclude_negative_film_numbers = false);

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

	cisTEMParameterLine ReturnParameterAverages(bool only_average_active = true);
	cisTEMParameterLine ReturnParameterVariances(bool only_average_active = true);

};
