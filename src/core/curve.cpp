#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofCurves);

void LS_POLY(float* x_data, float* y_data, int n_points, int order_of_polynomial, float* output_smoothed_curve, float* output_coefficients);

Curve::Curve( ) {
    polynomial_order                = 0;
    savitzky_golay_polynomial_order = 0;
    savitzky_golay_window_size      = 0;
    index_of_last_point_used        = 0;
}

// copy constructor
Curve::Curve(const Curve& other_curve) {

    DeleteSavitzkyGolayCoefficients( );
    savitzky_golay_polynomial_order = 0;
    savitzky_golay_window_size      = 0;

    polynomial_coefficients.clear( );
    polynomial_order = 0;

    index_of_last_point_used = 0;

    *this = other_curve;
}

Curve::~Curve( ) {
}

void Curve::DeleteSavitzkyGolayCoefficients( ) {
    for ( auto& coefficient : savitzky_golay_coefficients ) {
        coefficient.clear( );
    }
    savitzky_golay_coefficients.clear( );
}

void Curve::AllocateSavitzkyGolayCoefficients( ) {
    MyDebugAssertTrue(savitzky_golay_polynomial_order > 0, "Oops, looks like the SG polynomial order was not set properly\n");
    savitzky_golay_coefficients.clear( );
    savitzky_golay_coefficients.reserve(NumberOfPoints( ));

    std::vector<float> temp;
    temp.assign(savitzky_golay_polynomial_order + 1, 0.0f);

    for ( int i = 0; i < NumberOfPoints( ); i++ ) {
        savitzky_golay_coefficients.push_back(temp);
    }
}

Curve& Curve::operator=(const Curve* other_curve) {
    // Check for self assignment
    if ( this != other_curve ) {
        int counter;

        if ( NumberOfPoints( ) != other_curve->NumberOfPoints( ) ) {
            data_x.clear( );
            data_y.clear( );

            data_x.reserve(other_curve->NumberOfPoints( ));
            data_y.reserve(other_curve->NumberOfPoints( ));

            for ( auto& x : other_curve->data_x ) {
                data_x.push_back(x);
            }
            for ( auto& y : other_curve->data_y ) {
                data_y.push_back(y);
            }

            polynomial_order                = other_curve->polynomial_order;
            savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;

            polynomial_fit.clear( );
            polynomial_coefficients.clear( );

            savitzky_golay_fit.clear( );
            DeleteSavitzkyGolayCoefficients( );

            if ( ! other_curve->polynomial_fit.empty( ) ) {
                polynomial_fit.reserve(other_curve->NumberOfPoints( ));
                polynomial_coefficients.reserve(polynomial_order);
            }

            if ( ! other_curve->savitzky_golay_fit.empty( ) ) {
                savitzky_golay_fit.reserve(other_curve->NumberOfPoints( ));
                AllocateSavitzkyGolayCoefficients( );
            }
        }
        else {
            polynomial_order                = other_curve->polynomial_order;
            savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;

            if ( polynomial_fit.empty( ) != other_curve->polynomial_fit.empty( ) ) {
                if ( ! polynomial_fit.empty( ) ) {
                    polynomial_coefficients.clear( );
                    polynomial_fit.clear( );
                }
                else {
                    polynomial_fit.reserve(NumberOfPoints( ));
                    polynomial_coefficients.reserve(polynomial_order);
                }
            }

            if ( savitzky_golay_fit.empty( ) != other_curve->savitzky_golay_fit.empty( ) ) {
                if ( ! savitzky_golay_fit.empty( ) ) {
                    savitzky_golay_fit.clear( );
                    DeleteSavitzkyGolayCoefficients( );
                }
                else {
                    savitzky_golay_fit.reserve(NumberOfPoints( ));
                    AllocateSavitzkyGolayCoefficients( );
                }
            }
        }

        savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;
        savitzky_golay_window_size      = other_curve->savitzky_golay_window_size;

        index_of_last_point_used = other_curve->index_of_last_point_used;

        if ( ! other_curve->polynomial_fit.empty( ) ) {
            for ( auto& fit : other_curve->polynomial_fit ) {
                polynomial_fit.push_back(fit);
            }
            for ( auto& coefficient : other_curve->polynomial_coefficients ) {
                polynomial_coefficients.push_back(coefficient);
            }
        }

        if ( ! other_curve->savitzky_golay_fit.empty( ) ) {
            savitzky_golay_fit.clear( );
            for ( auto& fit : other_curve->savitzky_golay_fit ) {
                savitzky_golay_fit.push_back(fit);
            }
            savitzky_golay_coefficients.clear( );
            for ( auto& coefficient : other_curve->savitzky_golay_coefficients ) {
                savitzky_golay_coefficients.push_back(coefficient);
            }
        }
    }

    return *this;
}

Curve& Curve::operator=(const Curve& other_curve) {
    *this = &other_curve;
    return *this;
}

void Curve::SetupXAxis(const float lower_bound, const float upper_bound, const int wanted_number_of_points) {
    MyDebugAssertFalse(isnan(lower_bound), "Lower bound is NaN");
    MyDebugAssertFalse(isnan(upper_bound), "Lower bound is NaN");
    MyDebugAssertTrue(wanted_number_of_points > 1, "Bad number of points");

    ClearData( );

    for ( int counter = 0; counter < wanted_number_of_points; counter++ ) {
        AddPoint(counter * (upper_bound - lower_bound) / float(wanted_number_of_points - 1) + lower_bound, 0.0);
    }
}

/**
 * @brief Sets all data_y to 0 (calls SetYToConstant(0.f))
 * 
 * @param wanted_constant 
 */
void Curve::ZeroYData( ) {
    SetYToConstant(0.0f);
}

/**
 * @brief Sets all data_y to a constant float value
 * 
 * @param wanted_constant 
 */
void Curve::SetYToConstant(float wanted_constant) {
    DebugCheckEmpty( );
    data_y.assign(NumberOfPoints( ), wanted_constant);
}

/**
 * @brief Adds the data_y of another curve to the data_y of this curve. The number of points in both curves must be the same.
 * 
 * @param other_curve 
 */
void Curve::AddWith(Curve* other_curve) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(NumberOfPoints( ) == other_curve->data_y.size( ), "Different number of points");

    std::transform(data_y.begin( ), data_y.end( ), other_curve->data_y.begin( ), data_y.begin( ), std::plus<float>( ));
}

/**
 * @brief Divides the data_y of this curve by the data_y of another curve. The number of points in both curves must be the same. Division by zero returns zero.
 * 
 * @param other_curve 
 */
void Curve::DivideBy(Curve* other_curve) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(NumberOfPoints( ) == other_curve->data_y.size( ), "Different number of points");

    std::transform(data_y.begin( ), data_y.end( ), other_curve->data_y.begin( ), data_y.begin( ), [](const float my_val, const float other_val) {
        if ( other_val == 0.0f )
            return 0.0f;
        else
            return my_val / other_val;
    });
}

/**
 * @brief Returns the arithmetic average of the data_y of the curve
 * 
 * @return float 
 */
float Curve::ReturnAverageValue( ) {
    DebugCheckEmpty( );

    float sum = std::accumulate(data_y.begin( ), data_y.end( ), 0.0f);

    return sum / float(NumberOfPoints( ));
}

/**
 * @brief All values after index are set to 0.f
 * 
 * @param index 
 */
void Curve::ZeroAfterIndex(int index) {
    DebugCheckEmpty( );

    for ( int counter = index + 1; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] = 0.0;
    }
}

/**
 * @brief All values before index are set to the value at index
 * 
 * @param index 
 */
void Curve::FlattenBeforeIndex(int index) {
    DebugCheckEmpty( );

    if ( index > NumberOfPoints( ) )
        index = NumberOfPoints( ) - 1;
    for ( int counter = 0; counter < index; counter++ ) {
        data_y[counter] = data_y[index];
    }
}

/**
 * @brief Up or down sample a curve using linear interpolation. The ends are clamped to the value of the input.
 * 
 * @param wanted_x 
 * @return int 
 */
void Curve::ResampleCurve(Curve* input_curve, int wanted_number_of_points) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(wanted_number_of_points > 1, "wanted_number_of_points is smaller than 2");

    Curve temp_curve;

    float i_x, x;

    for ( int i = 0; i < wanted_number_of_points; i++ ) {
        // if wanted_number_of_points > NumberOfPoints( ) then i_x will be the scaled coordinate clamped at 0 and NumberOfPoints( )
        i_x = float(i * (input_curve->NumberOfPoints( ) - 1)) / float(wanted_number_of_points - 1);
        x   = input_curve->data_x.front( ) + i_x / (input_curve->NumberOfPoints( ) - 1) * (input_curve->data_x[input_curve->NumberOfPoints( ) - 1] - input_curve->data_x.front( ));
        temp_curve.AddPoint(x, input_curve->ReturnLinearInterpolationFromI(i_x));
    }
    CopyFrom(&temp_curve);
}

/**
 * @brief Returns the index of the nearest point to the wanted_x
 * 
 * @param wanted_x 
 * @return int 
 */
float Curve::ReturnLinearInterpolationFromI(float wanted_i) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(wanted_i <= NumberOfPoints( ) - 1, "Index too high");
    MyDebugAssertTrue(wanted_i >= 0, "Index too low");

    int i = int(wanted_i);

    float distance_below = wanted_i - i;
    float distance_above = 1.0 - distance_below;

    if ( distance_below == 0.0 )
        return data_y.at(i);
    if ( distance_above == 0.0 )
        return data_y.at(i + 1);

    return (1.0 - distance_above) * data_y.at(i + 1) + (1.0 - distance_below) * data_y.at(i);
}

/**
 * @brief Returns the index of the nearest point to the wanted_x
 * 
 * @param wanted_x 
 * @return int 
 */
float Curve::ReturnLinearInterpolationFromX(float wanted_x) {
    DebugCheckEmpty( );
    DebugCheckValidX(wanted_x);

    float value_to_return = 0.0f;

    const int index_of_previous_bin = ReturnIndexOfNearestPreviousBin(wanted_x);

    if ( index_of_previous_bin == NumberOfPoints( ) - 1 ) {
        value_to_return = data_y[NumberOfPoints( ) - 1];
    }
    else {
        float distance = (wanted_x - data_x.at(index_of_previous_bin)) / (data_x.at(index_of_previous_bin + 1) - data_x.at(index_of_previous_bin));
        value_to_return += data_y.at(index_of_previous_bin) * (1.0f - distance);
        value_to_return += data_y.at(index_of_previous_bin + 1) * distance;
    }
    return value_to_return;
}

/**
 * @brief 
 * 
 * @param maximum_value 
 * @param mode 
 */
void Curve::ComputeMaximumValueAndMode(float& maximum_value, float& mode) {
    DebugCheckEmpty( );
    maximum_value = -std::numeric_limits<float>::max( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        if ( data_y[counter] > maximum_value ) {
            maximum_value = data_y.at(counter);
            mode          = data_x.at(counter);
        }
    }
}

/**
 * @brief Returns the maximum value and also the mode, but the mode is ONLY for a curve that is storing a histogram, or any other container where the y_data represent COUNTS>
 *  This is meant to be used to measure, e.g., the full-width at half-maximum of a histogram stored as a curve
 * @return float 
 */

float Curve::ReturnFullWidthAtGivenValue(const float& wanted_value) {
    DebugCheckEmpty( );

    int first_bin_above_value = -1;
    int last_bin_above_value  = -1;

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        if ( first_bin_above_value == -1 && data_y[counter] > wanted_value )
            first_bin_above_value = counter;
        if ( last_bin_above_value == -1 && first_bin_above_value != -1 && data_y[counter] < wanted_value )
            last_bin_above_value = counter - 1;
    }

    MyDebugAssertTrue(first_bin_above_value != -1, "Could not find first bin above value");
    MyDebugAssertTrue(last_bin_above_value != -1, "Could not find last bin above value");

    return data_x.at(last_bin_above_value + 1) - data_x.at(first_bin_above_value);
}

/**
 * @brief Returns the maximum value of the data_y of a curve object. It does not matter if the data represents a histogram or not.
 * @return float 
 */

float Curve::ReturnMaximumValue( ) {
    float maximum_value, mode;
    ComputeMaximumValueAndMode(maximum_value, mode);
    return maximum_value;
}

/**
 * @brief Returns the mode ONLY for a curve that is storing a histogram, or any other container where the y_data represent COUNTS>
 * 
 * @return float 
 */
float Curve::ReturnMode( ) {
    float maximum_value, mode;
    ComputeMaximumValueAndMode(maximum_value, mode);
    return mode;
}

/**
 * @brief Scale the Y values so that the peak is at 1.0 (assumes all Y values >=0)
 * 
 */
void Curve::NormalizeMaximumValue( ) {
#ifdef DEBUG
    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        if ( data_y[counter] < 0.0 )
            MyDebugAssertTrue(false, "This routine assumes all Y values are positive, but value %i is %f\n", counter, data_y[counter]);
    }
#endif

    const float maximum_value = ReturnMaximumValue( );

    if ( maximum_value > 0.0f ) {
        float factor = 1.0f / maximum_value;
        std::transform(data_y.begin( ), data_y.end( ), data_y.begin( ), [factor](const float& value) { return value * factor; });
    }
}

/**
 * @brief Replace Y values with their log, base 10
 * 
 */
void Curve::Logarithm( ) {
    DebugCheckForNegativeValues( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] = std::log10(data_y[counter]);
    }
}

/**
 * @brief Replace Y values with their natural logarthm base e
 * 
 */
void Curve::Ln( ) {
    DebugCheckForNegativeValues( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] = std::log(data_y[counter]);
    }
}

/**
 * @brief Element-wise square root of the Y values
 * 
 */
void Curve::SquareRoot( ) {
    DebugCheckForNegativeValues( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] = sqrtf(data_y[counter]);
    }
}

CurvePoint Curve::ReturnValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x) {
    MyDebugAssertFalse(NumberOfPoints( ) < 2, "No points in curve");
    DebugCheckValidX(wanted_x);

    int        index_of_previous_bin;
    CurvePoint return_value;
    if ( assume_linear_x ) {
        if ( data_x[0] == data_x[1] ) {
            // Special case where x is constant (it's not really a curve in that case, but we still have to deal with this)
            index_of_previous_bin = NumberOfPoints( ) - 2;
        }
        else {
            index_of_previous_bin = int((wanted_x - data_x[0]) / (data_x[1] - data_x[0]));
        }
    }
    else {
        index_of_previous_bin = ReturnIndexOfNearestPreviousBin(wanted_x);
    }

    MyDebugAssertTrue(index_of_previous_bin >= 0 && index_of_previous_bin < NumberOfPoints( ), "Oops. Bad index_of_previous_bin: %i\n", index_of_previous_bin);

    if ( index_of_previous_bin == NumberOfPoints( ) - 1 ) {
        return_value.index_m = index_of_previous_bin;
        return_value.index_n = index_of_previous_bin;
        return_value.value_m = value_to_add;
        return_value.value_n = 0.0;
    }
    else {
        float distance       = (wanted_x - data_x.at(index_of_previous_bin)) / (data_x.at(index_of_previous_bin + 1) - data_x.at(index_of_previous_bin));
        return_value.index_m = index_of_previous_bin;
        return_value.index_n = index_of_previous_bin + 1;
        return_value.value_m = value_to_add * (1.0 - distance);
        return_value.value_n = value_to_add * distance;
    }

    return return_value;
}

/**
 * @brief If the data_x values form a linear axis (i.e. they are regularly spaced), give assume_linear_x true.
 * 
 * @param wanted_x 
 * @param value_to_add 
 * @param assume_linear_x 
 */
void Curve::AddValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x) {
    DebugCheckEmpty( );
    DebugCheckValidX(wanted_x);

    CurvePoint return_value;

    return_value = ReturnValueAtXUsingLinearInterpolation(wanted_x, value_to_add, assume_linear_x);
    data_y[return_value.index_m] += return_value.value_m;
    data_y[return_value.index_n] += return_value.value_n;
}

/**
 * @brief Simple nearest neighbor interpolation
 * 
 * @param wanted_x 
 * @param value_to_add 
 */
void Curve::AddValueAtXUsingNearestNeighborInterpolation(float wanted_x, float value_to_add) {
    DebugCheckEmpty( );
    DebugCheckValidX(wanted_x);

    data_y[ReturnIndexOfNearestPointFromX(wanted_x)] += value_to_add;
}

/**
 * @brief Print the x/y data of the Curve object to stdout
 * 
 */
void Curve::PrintToStandardOut( ) {
    for ( int i = 0; i < NumberOfPoints( ); i++ ) {
        wxPrintf("%f %f\n", data_x[i], data_y[i]);
    }
}

/**
 * @brief Write the curve out to a numeric text file
 * 
 * @param output_file 
 * @param header_line 
 */
void Curve::WriteToFile(wxString output_file, wxString header_line) {
    DebugCheckEmpty( );

    std::array<float, 2> temp_float;

    NumericTextFile output_curve_file(output_file, OPEN_TO_WRITE, 2);
    output_curve_file.WriteCommentLine(header_line);
    for ( int i = 0; i < NumberOfPoints( ); i++ ) {
        temp_float[0] = data_x[i];
        temp_float[1] = data_y[i];

        output_curve_file.WriteLine(temp_float.data( ));
    }
}

// FIXME: Why is this called externally?
/**
 * @brief Set the header for the curve object being written out to a numeric text file
 * 
 * @param output_file 
 */
void Curve::WriteToFile(wxString output_file) {
    WriteToFile(output_file, "C            X              Y");
}

/**
 * @brief Delete existing data and copy from the arrays provided. 
 * 
 * @param x_series 
 * @param y_series 
 * @param wanted_number_of_points 
 */
void Curve::CopyDataFromArrays(double* x_series, double* y_series, int wanted_number_of_points) {
    ClearData( );
    for ( int counter = 0; counter < wanted_number_of_points; counter++ ) {
        AddPoint(float(x_series[counter]), float(y_series[counter]));
    }
}

/**
 * @brief Delete existing data and copy from the array provided. The X values are assumed to be 0, 1, 2, 3, ...
 * 
 * @param y_series 
 * @param wanted_number_of_points 
 */
void Curve::CopyYValuesFromArray(double* y_series, int wanted_number_of_points) {
    ClearData( );
    for ( int counter = 0; counter < wanted_number_of_points; counter++ ) {
        AddPoint(float(counter), float(y_series[counter]));
    }
}

/**
 * @brief Call the assignment op for the curve object
 * 
 * @param other_curve 
 */
void Curve::CopyFrom(Curve* other_curve) {
    *this = other_curve;
}

/**
 * @brief Return the min and max X value by reference
 * 
 * @param min_value 
 * @param max_value 
 */
void Curve::GetXMinMax(float& min_value, float& max_value) {
    min_value = std::numeric_limits<float>::max( );
    max_value = -std::numeric_limits<float>::max( );

    for ( auto& x : data_x ) {
        min_value = std::min(min_value, x);
        max_value = std::max(max_value, x);
    }
}

/**
 * @brief Return the min and max Y value by reference
 * 
 * @param min_value 
 * @param max_value 
 */
void Curve::GetYMinMax(float& min_value, float& max_value) {
    min_value = std::numeric_limits<float>::max( );
    max_value = -std::numeric_limits<float>::max( );

    for ( auto& y : data_y ) {
        min_value = std::min(min_value, y);
        max_value = std::max(max_value, y);
    }
}

/**
 * @brief push back a new x/y pair to the stored data vectors.
 * 
 * @param x_value 
 * @param y_value 
 */
void Curve::AddPoint(float x_value, float y_value) {
    data_x.push_back(x_value);
    data_y.push_back(y_value);
}

/**
 * @brief Reset data, fit and coefficient vectors.
 * 
 */
void Curve::ClearData( ) {
    data_x.clear( );
    data_y.clear( );

    polynomial_fit.clear( );
    polynomial_coefficients.clear( );

    savitzky_golay_fit.clear( );
    DeleteSavitzkyGolayCoefficients( );
}

/**
 * @brief Multiply all Y values by a constant
 * 
 * @param constant_to_multiply_by 
 */
void Curve::MultiplyByConstant(float constant_to_multiply_by) {
    DebugCheckEmpty( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] *= constant_to_multiply_by;
    }
}

/**
 * @brief It is assumed that the X axis has spatial frequencies in reciprocal pixels (0.5 is Nyquist)
 * 
 * @param ctf_to_apply 
 * @param azimuth_in_radians 
 */
void Curve::ApplyCTF(CTF ctf_to_apply, float azimuth_in_radians) {
    DebugCheckEmpty( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] *= ctf_to_apply.Evaluate(powf(data_x[counter], 2), azimuth_in_radians);
    }
}

/**
 * @brief It is assumed that the X axis has spatial frequencies in reciprocal pixels (0.5 is Nyquist)
 * 
 * @param ctf_to_apply 
 * @param azimuth_in_radians 
 */
void Curve::ApplyPowerspectrumWithThickness(CTF ctf_to_apply, float azimuth_in_radians) {
    DebugCheckEmpty( );

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] *= ctf_to_apply.EvaluatePowerspectrumWithThickness(powf(data_x[counter], 2), azimuth_in_radians);
    }
}

/**
 * @brief  Assumption is that X is recipricoal pixels
 * 
 * @param sigma 
 */
void Curve::ApplyGaussianLowPassFilter(float sigma) {
    float frequency_squared;
    float one_over_two_sigma_squared = 0.5 / powf(sigma, 2);

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        frequency_squared = powf(data_x[counter], 2);
        data_y[counter] *= expf(-frequency_squared * one_over_two_sigma_squared);
    }
}

/**
 * @brief mask (FIXME what is undo)
 * 
 * @param wanted_x_of_cosine_start 
 * @param wanted_cosine_width_in_x 
 * @param undo 
 */
void Curve::ApplyCosineMask(float wanted_x_of_cosine_start, float wanted_cosine_width_in_x, bool undo) {
    DebugCheckEmpty( );

    float current_x;
    float edge;

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {

        current_x = data_x[counter];
        if ( current_x >= wanted_x_of_cosine_start && current_x <= wanted_x_of_cosine_start + wanted_cosine_width_in_x ) {
            edge = (1.0 + cosf(PI * (current_x - wanted_x_of_cosine_start) / wanted_cosine_width_in_x)) / 2.0;
            if ( undo ) {
                //MyDebugAssertFalse(edge == 0.0,"Raised cosine should not be 0.0");
                if ( edge > 0.0 )
                    data_y[counter] /= edge;
            }
            else {
                data_y[counter] *= edge;
            }
        }
        else if ( current_x > wanted_x_of_cosine_start + wanted_cosine_width_in_x ) {
            data_y[counter] = 0.0;
        }
    }
}

/**
 * @brief Return the interpolation of the Savitzky-Golay polynomial at the wanted_x
 * 
 * @param wanted_x 
 * @return float 
 */
float Curve::ReturnSavitzkyGolayInterpolationFromX(float wanted_x) {

    // Find the nearest data point to the wanted_x
    int index_of_nearest_point = ReturnIndexOfNearestPointFromX(wanted_x);

    // Savitzky-Golay coefficients are zero for last index
    if ( index_of_nearest_point == NumberOfPoints( ) - 1 )
        index_of_nearest_point--;

    // Evaluate the polynomial defined at the nearest point.
    // TODO: use a better algorithm to evaluate the poynomial, e.g. Horner, see Numerical Recipes
    double y = savitzky_golay_coefficients[index_of_nearest_point][0];
    for ( int order = 1; order <= savitzky_golay_polynomial_order; order++ ) {
        y += pow(wanted_x, order) * savitzky_golay_coefficients[index_of_nearest_point][order];
    }

    return float(y);
}

/**
 * @brief Return the interpolation of the Savitzky-Golay polynomial at the wanted_x
 * 
 * @param wanted_x 
 * @return int 
 */
int Curve::ReturnIndexOfNearestPointFromX(float wanted_x) {

    int   index_of_nearest_point    = 0;
    int   counter                   = 0;
    float distance_to_current_point = wanted_x - data_x[counter];
    float distance_to_nearest_point = distance_to_current_point;
    for ( counter = 1; counter < NumberOfPoints( ); counter++ ) {
        distance_to_current_point = wanted_x - data_x[counter];
        if ( fabs(distance_to_current_point) <= fabs(distance_to_nearest_point) ) {
            distance_to_nearest_point = distance_to_current_point;
            index_of_nearest_point    = counter;
        }
        else {
            break;
        }
    }
    return index_of_nearest_point;
}

/**
 * @brief TODO: write a faster verions of this
 * 
 * @param wanted_x 
 * @return
 */
int Curve::ReturnIndexOfNearestPreviousBin(float wanted_x) {
    DebugCheckEmpty( );
    DebugCheckValidX(wanted_x);

    for ( int counter = index_of_last_point_used; counter < NumberOfPoints( ) - 1; counter++ ) {
        if ( wanted_x >= data_x[counter] && wanted_x < data_x[counter + 1] ) {
            index_of_last_point_used = counter;
            return counter;
        }
    }
    for ( int counter = index_of_last_point_used - 1; counter >= 0; counter-- ) {
        if ( wanted_x >= data_x[counter] && wanted_x < data_x[counter + 1] ) {
            index_of_last_point_used = counter;
            return counter;
        }
    }

    if ( wanted_x < data_x[0] ) {
        index_of_last_point_used = 0;
        return 0;
    }
    else if ( wanted_x >= data_x.back( ) ) {
        index_of_last_point_used = NumberOfPoints( ) - 1;
        return NumberOfPoints( ) - 1;
    }

    // Should never get here
    MyDebugAssertTrue(false, "Oops, programming error\n");
    return 0;
}

/**
 * @brief Get the savitzky-golay polynomial order
 * 
 * @param wanted_window_size 
 * @param wanted_polynomial_order 
 */
void Curve::FitSavitzkyGolayToData(int wanted_window_size, int wanted_polynomial_order) {
    // make sure the window size is odd

    MyDebugAssertTrue(IsOdd(wanted_window_size) == true, "Window must be odd!");
    MyDebugAssertTrue(wanted_window_size < NumberOfPoints( ), "Window size is larger than the number of points!");
    MyDebugAssertTrue(wanted_polynomial_order < wanted_window_size, "polynomial order is larger than the window size!");

    int pixel_counter;
    int polynomial_counter;

    int end_start;

    int half_pixel = wanted_window_size / 2;

    float* fit_array_x      = new float[wanted_window_size];
    float* fit_array_y      = new float[wanted_window_size];
    float* output_fit_array = new float[wanted_window_size];
    //float *coefficients = new float[wanted_polynomial_order+1];

    // Remember the polymomal order and the window size
    savitzky_golay_polynomial_order = wanted_polynomial_order;
    savitzky_golay_window_size      = wanted_window_size;

    // Allocate array of coefficient arrays, to be kept in memory for later use (e.g. for interpolation)
    DeleteSavitzkyGolayCoefficients( );
    // Allocate memory for smooth y values
    savitzky_golay_fit.assign(NumberOfPoints( ), 0.0f);

    AllocateSavitzkyGolayCoefficients( );

    // loop over all the points..

    for ( pixel_counter = 0; pixel_counter < NumberOfPoints( ) - 2 * half_pixel; pixel_counter++ ) {
        // for this pixel, extract the window, fit the polynomial, and copy the average into the output array
        for ( polynomial_counter = 0; polynomial_counter < wanted_window_size; polynomial_counter++ ) {
            fit_array_x[polynomial_counter] = data_x.at(pixel_counter + polynomial_counter);
            fit_array_y[polynomial_counter] = data_y.at(pixel_counter + polynomial_counter);
        }

        // fit a polynomial to this data..
        LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients.at(half_pixel + pixel_counter).data( ));
        // take the middle pixel, and put it into the output array..

        savitzky_golay_fit.at(half_pixel + pixel_counter) = output_fit_array[half_pixel];
    }

    // now we need to take care of the ends - first the start..
    // DNM: Need to take actual points beyond the end of the fitted points in the middle
    for ( polynomial_counter = 0; polynomial_counter < wanted_window_size; polynomial_counter++ ) {
        fit_array_x[polynomial_counter] = data_x.at(polynomial_counter);

        if ( polynomial_counter < half_pixel || polynomial_counter >= NumberOfPoints( ) - half_pixel )
            fit_array_y[polynomial_counter] = data_y.at(polynomial_counter);
        else
            fit_array_y[polynomial_counter] = savitzky_golay_fit.at(polynomial_counter);
    }

    // fit a polynomial to this data..

    LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients.at(half_pixel - 1).data( ));

    // copy the required data back..
    for ( pixel_counter = 0; pixel_counter < half_pixel - 1; pixel_counter++ ) {
        for ( polynomial_counter = 0; polynomial_counter <= savitzky_golay_polynomial_order; polynomial_counter++ ) {
            savitzky_golay_coefficients.at(pixel_counter).at(polynomial_counter) = savitzky_golay_coefficients.at(half_pixel - 1).at(polynomial_counter);
        }
    }
    for ( polynomial_counter = 0; polynomial_counter < half_pixel; polynomial_counter++ ) {
        //savitzky_golay_coefficients[polynomial_counter] = savitzky_golay_coefficients[half_pixel - 1];
        savitzky_golay_fit.at(polynomial_counter) = output_fit_array[polynomial_counter];
    }

    // now the end..
    end_start     = NumberOfPoints( ) - wanted_window_size;
    pixel_counter = 0;
    for ( polynomial_counter = end_start; polynomial_counter < NumberOfPoints( ); polynomial_counter++ ) {
        fit_array_x[pixel_counter] = data_x.at(polynomial_counter);

        if ( pixel_counter > half_pixel )
            fit_array_y[pixel_counter] = data_y.at(polynomial_counter);
        else
            fit_array_y[pixel_counter] = savitzky_golay_fit.at(polynomial_counter);

        pixel_counter++;
    }

    // fit a polynomial to this data..

    LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients.at(NumberOfPoints( ) - half_pixel).data( ));

    // copy the required data back..

    for ( pixel_counter = NumberOfPoints( ) - half_pixel + 1; pixel_counter < NumberOfPoints( ) - 1; pixel_counter++ ) {
        for ( polynomial_counter = 0; polynomial_counter <= savitzky_golay_polynomial_order; polynomial_counter++ ) {
            savitzky_golay_coefficients.at(pixel_counter).at(polynomial_counter) = savitzky_golay_coefficients.at(NumberOfPoints( ) - half_pixel).at(polynomial_counter);
        }
    }

    pixel_counter = half_pixel + 1;
    for ( polynomial_counter = NumberOfPoints( ) - half_pixel; polynomial_counter < NumberOfPoints( ); polynomial_counter++ ) {
        //savitzky_golay_coefficients[polynomial_counter] = savitzky_golay_coefficients[NumberOfPoints( ) - half_pixel];
        savitzky_golay_fit.at(polynomial_counter) = output_fit_array[pixel_counter];
        pixel_counter++;
    }

    delete[] fit_array_x;
    delete[] fit_array_y;
    delete[] output_fit_array;
}

/**
 * @brief fit a basic polynomial to the data
 * 
 * @param wanted_polynomial_order 
 */
void Curve::FitPolynomialToData(int wanted_polynomial_order) {
    polynomial_fit.clear( );

    polynomial_fit.assign(NumberOfPoints( ), 0.0f);
    polynomial_order = wanted_polynomial_order;
    polynomial_coefficients.assign(polynomial_order + 1, 0.0f);

    LS_POLY(data_x.data( ), data_y.data( ), NumberOfPoints( ), polynomial_order, polynomial_fit.data( ), polynomial_coefficients.data( )); // weird old code to do the fit
}

/**
 * @brief Fit a 1d gaussian to the data
 * 
 * @param lower_bound_x 
 * @param upper_bound_x 
 * @param apply_x_weighting 
 */
void Curve::FitGaussianToData(float lower_bound_x, float upper_bound_x, bool apply_x_weighting) {
    //	MyDebugAssertTrue(first_point >= 0 && first_point < NumberOfPoints( ) - 1, "first_point out of range");
    //	MyDebugAssertTrue(last_point >= 0 && first_point < NumberOfPoints( ), "last_point out of range");

    gaussian_coefficients.clear( );
    gaussian_fit.clear( );

    gaussian_fit.assign(NumberOfPoints( ), 0.f);
    gaussian_coefficients.reserve(2);

    float s, sx, sxx, sxy, sy;
    float delta, sigma2;
    int   i;

    s      = 0.0f;
    sx     = 0.0f;
    sxx    = 0.0f;
    sxy    = 0.0f;
    sy     = 0.0f;
    sigma2 = 1.0f;

    for ( i = 0; i < NumberOfPoints( ); i++ ) {
        if ( data_x[i] >= lower_bound_x && data_x[i] <= upper_bound_x ) {
            if ( apply_x_weighting )
                sigma2 = powf(data_x[i], 2);
            if ( data_y[i] > 0.0f ) {
                s += sigma2;
                sx += powf(data_x[i], 2) * sigma2;
                sxx += powf(data_x[i], 4) * sigma2;
                sxy += powf(data_x[i], 2) * log(data_y[i]) * sigma2;
                sy += log(data_y[i]) * sigma2;
            }
        }
    }

    delta                    = s * sxx - sx * sx;
    gaussian_coefficients[0] = expf((sxx * sy - sx * sxy) / delta);
    gaussian_coefficients[1] = (s * sxy - sx * sy) / delta;

    for ( i = 0; i < NumberOfPoints( ); i++ ) {
        gaussian_fit[i] = gaussian_coefficients[0] * expf(gaussian_coefficients[1] * powf(data_x[i], 2));
    }
}

/**
 * @brief in-place reciprocal of the curve, zero values are not modified.
 * 
 */
void Curve::Reciprocal( ) {
    DebugCheckForNegativeValues( );

    for ( auto& y : data_y ) {
        if ( y != 0.0 )
            y = 1.0 / y;
    }
}

/**
 * @brief in-place absolute value of the curve
 * 
 */
void Curve::Absolute( ) {
    for ( auto& y : data_y ) {
        y = fabs(y);
    }
}

/**
 * @brief Check that the values of both curves are nearly the same, as defined in functions::FloatsAreAlmostTheSame
 * 
 * @param other_curve 
 * @return true 
 * @return false 
 */
bool Curve::YIsAlmostEqual(Curve& other_curve) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(NumberOfPoints( ) == other_curve.NumberOfPoints( ), "Number of points in curves not equal");
    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        if ( ! FloatsAreAlmostTheSame(data_y[counter], other_curve.data_y[counter]) )
            return false;
    }
    return true;
}

/**
 * @brief Add a constant float to each Y value
 * 
 * @param constant_to_add 
 */
void Curve::AddConstant(float constant_to_add) {
    for ( auto& y : data_y ) {
        y += constant_to_add;
    }
}

/**
 * @brief Divide one cuve by another element wise, zero values are not modified.
 * 
 * @param other_curve 
 */
void Curve::DivideBy(Curve& other_curve) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(NumberOfPoints( ) == other_curve.NumberOfPoints( ), "Number of points in curves not equal");

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        if ( other_curve.data_y[counter] != 0.0 )
            data_y[counter] = data_y[counter] / other_curve.data_y[counter];
    }
}

/**
 * @brief Multiply one curve by another element wise
 * 
 * @param other_curve 
 */
void Curve::MultiplyBy(Curve& other_curve) {
    DebugCheckEmpty( );
    MyDebugAssertTrue(NumberOfPoints( ) == other_curve.NumberOfPoints( ), "Number of points in curves not equal");

    for ( int counter = 0; counter < NumberOfPoints( ); counter++ ) {
        data_y[counter] = data_y[counter] * other_curve.data_y[counter];
    }
}

/**
 * @brief Multiply the X values by a constant
 * 
 * @param constant_to_multiply_by 
 */
void Curve::MultiplyXByConstant(float constant_to_multiply_by) {
    DebugCheckEmpty( );

    for ( auto& x : data_x ) {
        x *= constant_to_multiply_by;
    }
}

/***************************************************
*      Program to demonstrate least squares        *
*         polynomial fitting subroutine            *
* ------------------------------------------------ *
* Reference: BASIC Scientific Subroutines, Vol. II *
* By F.R. Ruckdeschel, BYTE/McGRAWW-HILL, 1981 [1].*
*                                                  *
*                C++ version by J-P Moreau, Paris  *
*                       (www.jpmoreau.fr)          *
* ------------------------------------------------ *

typedef double TAB[SIZE+1];

int    i,l,m,n;
double dd,e1,vv;

TAB    x,y,v,a,b,c,d,c2,e,f;



*****************************************************************
*         LEAST SQUARES POLYNOMIAL FITTING SUBROUTINE           *
* ------------------------------------------------------------- *
* This program least squares fits a polynomial to input data.   *
* forsythe orthogonal polynomials are used in the fitting.      *
* The number of data points is n.                               *
* The data is input to the subroutine in x[i], y[i] pairs.      *
* The coefficients are returned in c[i],                        *
* the smoothed data is returned in v[i],                        *
* the order of the fit is specified by m.                       *
* The standard deviation of the fit is returned in d.           *
* There are two options available by use of the parameter e:    *
*  1. if e = 0, the fit is to order m,                          *
*  2. if e > 0, the order of fit increases towards m, but will  *
*     stop if the relative standard deviation does not decrease *
*     by more than e between successive fits.                   *
* The order of the fit then obtained is l.                      *
****************************************************************/

void LS_POLY(float* x_data, float* y_data, int n_points, int order_of_polynomial, float* output_smoothed_curve, float* output_coefficients) {

    double a[order_of_polynomial + 2];
    double b[order_of_polynomial + 2];
    double c[order_of_polynomial + 3];
    double c2[order_of_polynomial + 2];
    double f[order_of_polynomial + 2];

    double v[n_points + 1];
    double d[n_points + 1];
    double e[n_points + 1];
    double x[n_points + 1];
    double y[n_points + 1];

    int l; //
    int n = n_points;
    int m = order_of_polynomial;

    double e1 = 0.0; //
    double dd;
    double vv;

    //Labels: e10,e15,e20,e30,e50,fin;
    int i;
    int l2;
    int n1;

    double a1;
    double a2;
    double b1;
    double b2;
    double c1;
    double d1;
    double f1;
    double f2;
    double v1;
    double v2;
    double w;

    l  = 0;
    n1 = m + 1;
    v1 = 1e7;

    for ( i = 0; i < n_points; i++ ) {
        x[i + 1] = x_data[i];
        y[i + 1] = y_data[i];

        //wxPrintf("Before %i = %f\n", i, y_data[i]);
    }

    // Initialize the arrays
    for ( i = 1; i < n1 + 1; i++ ) {
        a[i] = 0;
        b[i] = 0;
        c[i] = 0;
        f[i] = 0;
    };
    c[n1 + 1] = 0;

    for ( i = 1; i < n + 1; i++ ) {
        v[i] = 0;
        d[i] = 0;
    }
    d1 = sqrt(n);
    w  = d1;
    for ( i = 1; i < n + 1; i++ ) {
        e[i] = 1 / w;
    }
    f1 = d1;
    a1 = 0;
    for ( i = 1; i < n + 1; i++ ) {
        a1 = a1 + x[i] * e[i] * e[i];
    }
    c1 = 0;
    for ( i = 1; i < n + 1; i++ ) {
        c1 = c1 + y[i] * e[i];
    }
    b[1] = 1 / f1;
    f[1] = b[1] * c1;
    for ( i = 1; i < n + 1; i++ ) {
        v[i] = v[i] + e[i] * c1;
    }
    m = 1;
e10: // Save latest results
    for ( i = 1; i < l + 1; i++ )
        c2[i] = c[i];
    l2 = l;
    v2 = v1;
    f2 = f1;
    a2 = a1;
    f1 = 0;
    for ( i = 1; i < n + 1; i++ ) {
        b1   = e[i];
        e[i] = (x[i] - a2) * e[i] - f2 * d[i];
        d[i] = b1;
        f1   = f1 + e[i] * e[i];
    }
    f1 = sqrt(f1);
    for ( i = 1; i < n + 1; i++ )
        e[i] = e[i] / f1;
    a1 = 0;
    for ( i = 1; i < n + 1; i++ )
        a1 = a1 + x[i] * e[i] * e[i];
    c1 = 0;
    for ( i = 1; i < n + 1; i++ )
        c1 = c1 + e[i] * y[i];
    m = m + 1;
    i = 0;
e15:
    l  = m - i;
    b2 = b[l];
    d1 = 0;
    if ( l > 1 )
        d1 = b[l - 1];
    d1   = d1 - a2 * b[l] - f2 * a[l];
    b[l] = d1 / f1;
    a[l] = b2;
    i    = i + 1;
    if ( i != m )
        goto e15;
    for ( i = 1; i < n + 1; i++ )
        v[i] = v[i] + e[i] * c1;
    for ( i = 1; i < n1 + 1; i++ ) {
        f[i] = f[i] + b[i] * c1;
        c[i] = f[i];
    }
    vv = 0;
    for ( i = 1; i < n + 1; i++ )
        vv = vv + (v[i] - y[i]) * (v[i] - y[i]);
    //Note the division is by the number of degrees of freedom
    vv = sqrt(vv / (n - l - 1));
    l  = m;
    if ( e1 == 0 )
        goto e20;
    //Test for minimal improvement
    if ( fabs(v1 - vv) / vv < e1 )
        goto e50;
    //if error is larger, quit
    if ( e1 * vv > e1 * v1 )
        goto e50;
    v1 = vv;
e20:
    if ( m == n1 )
        goto e30;
    goto e10;
e30: //Shift the c[i] down, so c(0) is the constant term
    for ( i = 1; i < l + 1; i++ )
        c[i - 1] = c[i];
    c[l] = 0;
    //l is the order of the polynomial fitted
    l  = l - 1;
    dd = vv;
    goto fin;
e50: // Aborted sequence, recover last values
    l  = l2;
    vv = v2;
    for ( i = 1; i < l + 1; i++ )
        c[i] = c2[i];
    goto e30;
fin:;

    for ( i = 0; i < n_points; i++ ) {
        output_smoothed_curve[i] = v[i + 1];
        //	output_smoothed_curve[i] = y[i + 1];
        //wxPrintf("After %i = %f\n", i, output_smoothed_curve[i]);
    }

    // coefficient 0: constant
    // coefficient 1: linear
    // coefficient 2: square
    // coefficient 3: cube
    // ...
    for ( i = 0; i <= order_of_polynomial; i++ ) {
        output_coefficients[i] = c[i];
    }
}
