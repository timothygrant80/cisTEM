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
	int				logp;
	float			sigma;
	float 			score;
	float			score_change;
	float			pixel_size;
	float			microscope_voltage_kv;
	float			microscope_spherical_aberration_mm;
	float			beam_tilt_x;
	float			beam_tilt_y;



	cisTEMParameterLine();
	~cisTEMParameterLine();
};

WX_DECLARE_OBJARRAY(cisTEMParameterLine, ArrayOfcisTEMParameterLines);

class cisTEMParameters {


public :

	wxArrayString			     header_comments;
	ArrayOfcisTEMParameterLines  all_parameters;

	cisTEMParameters();
	~cisTEMParameters();

	void ReadFromcisTEMStarFile(wxString wanted_filename);
	void ReadFromFrealignParFile(wxString wanted_filename, float wanted_pixel_size = 0.0f, float wanted_microscope_voltage = 0.0f, float wanted_microscope_cs = 0.0f, float wanted_beam_tilt_x = 0.0f, float wanted_beam_tilt_y = 0.0f);

	void AddCommentToHeader(wxString comment_to_add);
	void WriteTocisTEMStarFile(wxString wanted_filename);

	void ClearAll();

	void PreallocateMemoryAndBlank(int number_to_allocate);

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
	inline int   ReturnLogP(int line_number) { return all_parameters.Item(line_number).logp;}
	inline float ReturnSigma(int line_number) { return all_parameters.Item(line_number).sigma;}
	inline float ReturnScore(int line_number) { return all_parameters.Item(line_number).score;}
	inline float ReturnScoreChange(int line_number) { return all_parameters.Item(line_number).score_change;}
	inline float ReturnPixelSize(int line_number) { return all_parameters.Item(line_number).pixel_size;}
	inline float ReturnMicroscopekV(int line_number) { return all_parameters.Item(line_number).microscope_voltage_kv;}
	inline float ReturnMicroscopeCs(int line_number) { return all_parameters.Item(line_number).microscope_spherical_aberration_mm;}
	inline float ReturnBeamTiltX(int line_number) { return all_parameters.Item(line_number).beam_tilt_x;}
	inline float ReturnBeamTiltY(int line_number) { return all_parameters.Item(line_number).beam_tilt_y;}

};
