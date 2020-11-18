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

public:

	wxString    filename;
	wxTextFile *input_file;

	bool using_external_array;

	ArrayOfcisTEMParameterLines *cached_parameters;
	cisTEMParameterMask 		parameters_that_were_read;

	cisTEMStarFileReader();
	~cisTEMStarFileReader();

	cisTEMStarFileReader(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);

	void Open(wxString wanted_filename, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL);
	void Close();
	bool ReadFile(wxString wanted_filename, wxString *error_string = NULL, ArrayOfcisTEMParameterLines *alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);

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
