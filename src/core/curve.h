class Curve {

private:

public:

	bool have_polynomial;
	bool have_savitzky_golay;
	int number_of_points;
	int allocated_space_for_points;

	float *data_x;
	float *data_y;

	float *polynomial_fit;
	float *savitzky_golay_fit;

	int savitzky_golay_window_size;
	int savitzky_golay_polynomial_order;
	float **savitzky_golay_coefficients;

	int polynomial_order;
	float *polynomial_coefficients;


	// Constructors, destructors
	Curve();
	~Curve();
	Curve( const Curve &other_curve);

	Curve & operator = (const Curve &other_curve);
	Curve & operator = (const Curve *other_curve);

	void ResampleCurve(Curve *input_curve, int wanted_number_of_points);
	float ReturnLinearInterpolationFromI(float wanted_i);
	float ReturnLinearInterpolationFromX(float wanted_x_value);
	void PrintToStandardOut();
	void WriteToFile(wxString output_file);
	void CopyFrom(Curve *other_curve);
	void ClearData();
	void AddPoint(float x_value, float y_value);
	void FitPolynomialToData(int wanted_polynomial_order = 6);
	void FitSavitzkyGolayToData(int wanted_window_size, int wanted_polynomial_order);
	float ReturnSavitzkyGolayInterpolationFromX( float wanted_x );
	int ReturnIndexOfNearestPointFromX( float wanted_x );
	void DeleteSavitzkyGolayCoefficients();
	void AllocateSavitzkyGolayCoefficients();
	void CheckMemory();
};
