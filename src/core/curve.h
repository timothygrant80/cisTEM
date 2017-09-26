
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
	float ReturnLinearInterpolationFromX(float wanted_x);
	void AddValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x);
	void AddValueAtXUsingNearestNeighborInterpolation(float wanted_x, float value_to_add);
	int ReturnIndexOfNearestPreviousBin(float wanted_x);
	void PrintToStandardOut();
	void WriteToFile(wxString output_file);
	void CopyFrom(Curve *other_curve);
	void ClearData();
	void MultiplyByConstant(float constant_to_multiply_by);
	void MultiplyXByConstant(float constant_to_multiply_by);
	void AddPoint(float x_value, float y_value);
	void FitPolynomialToData(int wanted_polynomial_order = 6);
	void FitSavitzkyGolayToData(int wanted_window_size, int wanted_polynomial_order);
	float ReturnSavitzkyGolayInterpolationFromX( float wanted_x );
	int ReturnIndexOfNearestPointFromX( float wanted_x );
	void DeleteSavitzkyGolayCoefficients();
	void AllocateSavitzkyGolayCoefficients();
	void CheckMemory();
	void AddWith(Curve *other_curve);
	void SetupXAxis(const float lower_bound, const float upper_bound, const int wanted_number_of_points);
	float ReturnMaximumValue();
	float ReturnMode();
	void ComputeMaximumValueAndMode(float &maximum_value, float &mode);
	float ReturnFullWidthAtGivenValue(const float &wanted_value);
	void NormalizeMaximumValue();
	void ZeroYData();
	void ApplyCTF(CTF ctf_to_apply, float azimuth_in_radians =  0.0);
	void SquareRoot();
	void Reciprocal();
	void ZeroAfterIndex(int index);
	float ReturnAverageValue();

	void GetXMinMax(float &min_value, float &max_value);
	void GetYMinMax(float &min_value, float &max_value);
};

WX_DECLARE_OBJARRAY(Curve, ArrayofCurves);
