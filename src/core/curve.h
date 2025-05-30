
class Curve {

  private:
    int              savitzky_golay_window_size      = 0;
    int              savitzky_golay_polynomial_order = 0;
    int              polynomial_order                = 0;
    std::vector<int> index_of_last_point_used;

    inline int GetIndexOfLastPointUsed( ) const {
        return index_of_last_point_used[ReturnThreadNumberOfCurrentThread( )];
    };

    inline void SetIndexOfLastPointUsed(int wanted_index) {
        index_of_last_point_used[ReturnThreadNumberOfCurrentThread( )] = wanted_index;
    };

    void DebugCheckForNegativeValues( ) {
#ifdef DEBUG
        for ( auto& value : data_y ) {
            if ( value < 0.0 )
                MyDebugAssertTrue(false, "This routine assumes all Y values are positive, but value is %f\n", value);
        }
#endif
    }

    inline void DebugCheckValidX(float wanted_x) {
        MyDebugAssertFalse(data_x.empty( ), "X data is empty\n");
        MyDebugAssertTrue(wanted_x >= data_x.front( ) - (data_x.back( ) - data_x.front( )) * 0.01 && wanted_x <= data_x.back( ) + (data_x.back( ) - data_x.front( )) * 0.01,
                          "Wanted X (%f) falls outside of range (%f to %f)\n", wanted_x, data_x.front( ), data_x.back( ));
    }

    inline void DebugCheckEmpty( ) {
        MyDebugAssertTrue(data_x.size( ) > 0, "X data is empty\n");
        MyDebugAssertTrue(data_y.size( ) > 0, "Y data is empty\n");
    }

    CurvePoint ReturnValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x);
    int        ReturnIndexOfNearestPreviousBin(float wanted_x);

  public:
    std::vector<float> data_x;
    std::vector<float> data_y;

    std::vector<float> polynomial_fit;

    std::vector<float> gaussian_fit;
    std::vector<float> savitzky_golay_fit;

    std::vector<std::vector<float>> savitzky_golay_coefficients;

    std::vector<float> polynomial_coefficients;
    std::vector<float> gaussian_coefficients;

    // Constructors, destructors
    Curve( );
    ~Curve( );
    Curve(const Curve& other_curve);

    Curve& operator=(const Curve& other_curve);
    Curve& operator=(const Curve* other_curve);

    inline int NumberOfPoints( ) const {
        MyDebugAssertTrue(data_x.size( ) == data_y.size( ), "X and Y data sizes do not match\n");
        return data_x.size( );
    }

    void MakeThreadSafeForNThreads(int wanted_number_of_threads);

    void  ResampleCurve(Curve* input_curve, int wanted_number_of_points);
    float ReturnLinearInterpolationFromI(float wanted_i);
    float ReturnLinearInterpolationFromX(float wanted_x);

    void AddValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x);
    void AddValueAtXUsingNearestNeighborInterpolation(float wanted_x, float value_to_add);

    void  PrintToStandardOut( );
    void  WriteToFile(wxString output_filename);
    void  WriteToFile(wxString output_filename, wxString header_line);
    void  CopyFrom(Curve* other_curve);
    void  CopyDataFromArrays(double* x_series, double* y_series, const int wanted_number_of_points);
    void  CopyYValuesFromArray(double* y_series, const int wanted_number_of_points);
    void  ClearData( );
    void  MultiplyByConstant(float constant_to_multiply_by);
    void  MultiplyXByConstant(float constant_to_multiply_by);
    void  AddPoint(float x_value, float y_value);
    void  FitPolynomialToData(int wanted_polynomial_order = 6);
    void  FitGaussianToData(float lower_bound_x = -FLT_MAX, float upper_bound_x = FLT_MAX, bool apply_x_weighting = false);
    void  FitSavitzkyGolayToData(int wanted_window_size, int wanted_polynomial_order);
    float ReturnSavitzkyGolayInterpolationFromX(float wanted_x);
    int   ReturnIndexOfNearestPointFromX(float wanted_x);
    void  DeleteSavitzkyGolayCoefficients( );
    void  AllocateSavitzkyGolayCoefficients( );
    void  CheckMemory( );
    void  AllocateMemory(int wanted_number_of_points);
    void  AddWith(Curve* other_curve);
    void  DivideBy(Curve* other_curve);
    void  SetupXAxis(const float lower_bound, const float upper_bound, const int wanted_number_of_points);
    void  SetupXAxisForFourierSpace(const int logical_x_dimension, const float dimension);
    float ReturnMaximumValue( );
    float ReturnMode( );
    void  ComputeMaximumValueAndMode(float& maximum_value, float& mode);
    float ReturnFullWidthAtGivenValue(const float& wanted_value);
    void  NormalizeMaximumValue( );
    void  Logarithm( );
    void  Ln( );
    void  ZeroYData( );
    void  ApplyCTF(CTF ctf_to_apply, float azimuth_in_radians = 0.0);
    void  ApplyPowerspectrumWithThickness(CTF ctf_to_apply, float azimuth_in_radians = 0.0);
    void  SquareRoot( );
    void  Reciprocal( );
    void  DivideBy(Curve& other_curve);
    void  MultiplyBy(Curve& other_curve);
    void  ZeroAfterIndex(int index);
    void  FlattenBeforeIndex(int index);
    float ReturnAverageValue( );
    void  ApplyCosineMask(float wanted_x_of_cosine_start, float wanted_cosine_width_in_x, bool undo = false);
    void  ApplyGaussianLowPassFilter(float sigma); // Assumption is that X is recipricoal pixels
    void  Absolute( );
    bool  YIsAlmostEqual(Curve& other_curve);
    void  AddConstant(float constant_to_add);

    void GetXMinMax(float& min_value, float& max_value);
    void GetYMinMax(float& min_value, float& max_value);

    void SetYToConstant(float wanted_constant);
};

WX_DECLARE_OBJARRAY(Curve, ArrayofCurves);
