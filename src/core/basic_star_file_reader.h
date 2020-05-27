class StarFileParameters {

public :

	int position_in_stack;
	float phi;
	float theta;
	float psi;
	float x_shift;
	float y_shift;
	float defocus1;
	float defocus2;
	float defocus_angle;
	float phase_shift;
	wxString micrograph_name;

	StarFileParameters();
};

WX_DECLARE_OBJARRAY(StarFileParameters, ArrayofStarFileParameters);

class BasicStarFileReader {

	int current_position_in_stack;
	int current_column;

	int phi_column;
	int theta_column;
	int psi_column;
	int xshift_column;
	int yshift_column;
	int defocus1_column;
	int defocus2_column;
	int defocus_angle_column;
	int phase_shift_column;
	int micrograph_name_column;

public:

	wxString    filename;
  //  wxFileInputStream *input_file_stream;
  //  wxTextInputStream *input_text_stream;

	wxTextFile *input_file;

	ArrayofStarFileParameters cached_parameters;

	BasicStarFileReader();
	~BasicStarFileReader();

	BasicStarFileReader(wxString wanted_filename);
	void Open(wxString wanted_filename);
	void Close();
	bool ReadFile(wxString wanted_filename, wxString *error_string = NULL);

	bool ExtractParametersFromLine(wxString &wanted_line, wxString *error_string = NULL);

	bool XShiftsAreInAngst;
	bool YShiftsAreInAngst;

	inline int   ReturnPositionInStack(int line_number) { return cached_parameters[line_number].position_in_stack;}
	inline float ReturnPhi(int line_number) { return cached_parameters[line_number].phi;}
	inline float ReturnTheta(int line_number) { return cached_parameters[line_number].theta;}
	inline float ReturnPsi(int line_number) { return cached_parameters[line_number].psi;}
	inline float ReturnXShift(int line_number) { return cached_parameters[line_number].x_shift;}
	inline float ReturnYShift(int line_number) { return cached_parameters[line_number].y_shift;}
	inline float ReturnDefocus1(int line_number) { return cached_parameters[line_number].defocus1;}
	inline float ReturnDefocus2(int line_number) { return cached_parameters[line_number].defocus2;}
	inline float ReturnDefocusAngle(int line_number) { return cached_parameters[line_number].defocus_angle;}
	inline float ReturnPhaseShift(int line_number) { return cached_parameters[line_number].phase_shift;}
	inline wxString ReturnMicrographName(int line_number) { return cached_parameters[line_number].micrograph_name;}

};
