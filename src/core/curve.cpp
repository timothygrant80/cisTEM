#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofCurves);

void LS_POLY(float *x_data, float *y_data, int number_of_points, int order_of_polynomial, float *output_smoothed_curve, float *output_coefficients);

Curve::Curve()
{
	have_polynomial = false;
	have_savitzky_golay = false;

	number_of_points = 0;
	allocated_space_for_points = 100;

	data_x = new float[100];
	data_y = new float[100];

	polynomial_fit = NULL; // allocate on fit..
	savitzky_golay_fit = NULL;
	savitzky_golay_coefficients = NULL;

	polynomial_order = 0;
	polynomial_coefficients = NULL;

	savitzky_golay_polynomial_order = 0;
	savitzky_golay_window_size = 0;

	index_of_last_point_used = 0;
}

Curve::Curve( const Curve &other_curve) // copy constructor
{
	MyDebugPrint("Warning: copying a curve object");

	have_polynomial = false;
	have_savitzky_golay = false;

	number_of_points = 0;
	allocated_space_for_points = 100;

	data_x = new float[100];
	data_y = new float[100];

	polynomial_fit = NULL; // allocate on fit..
	savitzky_golay_fit = NULL;
	savitzky_golay_coefficients = NULL;

	polynomial_order = 0;
	polynomial_coefficients = NULL;

	savitzky_golay_polynomial_order = 0;
	savitzky_golay_window_size = 0;

	index_of_last_point_used = 0;

	*this = other_curve;
	 //DEBUG_ABORT;
}


Curve::~Curve()
{
	delete [] data_x;
	delete [] data_y;

	if (have_polynomial == true)
	{
		delete [] polynomial_fit;
		delete [] polynomial_coefficients;

		polynomial_fit = NULL;
		have_polynomial = false;
	}

	if (have_savitzky_golay == true)
	{
		delete [] savitzky_golay_fit;
		savitzky_golay_fit = NULL;
		DeleteSavitzkyGolayCoefficients();
		have_savitzky_golay = false;
	}
}

void Curve::DeleteSavitzkyGolayCoefficients()
{
	MyDebugAssertTrue(savitzky_golay_coefficients != NULL, "Oops trying to deallocate coefficients when they are not allocated\n");
	for (int pixel_counter = 0; pixel_counter < number_of_points; pixel_counter ++ )
	{
		if (savitzky_golay_coefficients[pixel_counter] != NULL)
		{
			delete  [] savitzky_golay_coefficients[pixel_counter]; 
		}
	}
	delete [] savitzky_golay_coefficients;
	savitzky_golay_coefficients = NULL;
}

void Curve::AllocateSavitzkyGolayCoefficients()
{
	MyDebugAssertTrue(savitzky_golay_coefficients == NULL,"Oops, trying to allocate coffecients when they are already allocated\n");
	MyDebugAssertTrue(savitzky_golay_polynomial_order > 0, "Oops, looks like the SG polynomial order was not set properly\n");
	savitzky_golay_coefficients = new float*[number_of_points];
	int counter;
	for (int pixel_counter = 0; pixel_counter < number_of_points; pixel_counter ++ )
	{
		savitzky_golay_coefficients[pixel_counter] = new float[savitzky_golay_polynomial_order + 1];
		// Initialise
		for (counter = 0; counter < savitzky_golay_polynomial_order + 1; counter ++)
		{
			savitzky_golay_coefficients[pixel_counter][counter] = 0.0;
		}
	}
}

Curve & Curve::operator = (const Curve *other_curve)
{
	// Check for self assignment
	if(this != other_curve)
	{
		int counter;

		if (number_of_points != other_curve->number_of_points)
		{
			delete [] data_x;
			delete [] data_y;

			allocated_space_for_points = other_curve->allocated_space_for_points;

			data_x = new float[allocated_space_for_points];
			data_y = new float[allocated_space_for_points];

			polynomial_order = other_curve->polynomial_order;
			savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;

			if (have_polynomial == true)
			{
				delete [] polynomial_fit;
				delete [] polynomial_coefficients;
			}

			if (have_savitzky_golay == true)
			{
				delete [] savitzky_golay_fit;
				DeleteSavitzkyGolayCoefficients();
			}

			if (other_curve->have_polynomial == true)
			{
				polynomial_fit = new float[number_of_points];
				polynomial_coefficients = new float[polynomial_order];
			}

			if (other_curve->have_savitzky_golay == true)
			{
				savitzky_golay_fit = new float[number_of_points];
				AllocateSavitzkyGolayCoefficients();
			}


		}
		else
		{
			polynomial_order = other_curve->polynomial_order;
			savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;

			if (have_polynomial != other_curve->have_polynomial)
			{
				if (have_polynomial == true)
				{
					delete [] polynomial_coefficients;
					delete [] polynomial_fit;
				}
				else
				{
					polynomial_fit = new float[number_of_points];
					polynomial_coefficients = new float[polynomial_order];
				}

			}

			if (have_savitzky_golay != other_curve->have_savitzky_golay)
			{
				if (have_savitzky_golay == true)
				{
					delete [] savitzky_golay_fit;
					DeleteSavitzkyGolayCoefficients();
				}
				else
				{
					savitzky_golay_fit = new float[number_of_points];
					AllocateSavitzkyGolayCoefficients();
				}
			}
		}


		number_of_points = other_curve->number_of_points;
		have_polynomial = other_curve->have_polynomial;
		have_savitzky_golay = other_curve->have_savitzky_golay;
		savitzky_golay_polynomial_order = other_curve->savitzky_golay_polynomial_order;
		savitzky_golay_window_size = other_curve->savitzky_golay_window_size;

		index_of_last_point_used = other_curve->index_of_last_point_used;

		for (counter = 0; counter < number_of_points; counter++)
		{
			data_x[counter] = other_curve->data_x[counter];
			data_y[counter] = other_curve->data_y[counter];
		}

		if (have_polynomial == true)
		{
			for (counter = 0; counter < number_of_points; counter++)
			{
				polynomial_fit[counter] = other_curve->polynomial_fit[counter];
			}

			for (counter = 0; counter < polynomial_order; counter++)
			{
				polynomial_coefficients[counter] = other_curve->polynomial_coefficients[counter];
			}

		}

		if (have_savitzky_golay == true)
		{
			for (counter = 0; counter < number_of_points; counter++)
			{
				savitzky_golay_fit[counter] = other_curve->savitzky_golay_fit[counter];
				for (int degree = 0; degree <= savitzky_golay_polynomial_order; degree++)
				{
					savitzky_golay_coefficients[counter][degree] = other_curve->savitzky_golay_coefficients[counter][degree];
				}
			}
		}
	}

	return *this;
}

Curve & Curve::operator = (const Curve &other_curve)
{
	*this = &other_curve;
	return *this;
}

void Curve::SetupXAxis(const float lower_bound, const float upper_bound, const int wanted_number_of_points)
{
	MyDebugAssertFalse(isnan(lower_bound),"Lower bound is NaN");
	MyDebugAssertFalse(isnan(upper_bound),"Lower bound is NaN");
	MyDebugAssertTrue(wanted_number_of_points > 1,"Bad number of points");
	ClearData();
	AllocateMemory(wanted_number_of_points);

	for ( int counter = 0; counter < wanted_number_of_points; counter++ )
	{
		AddPoint(counter * (upper_bound - lower_bound)/float(wanted_number_of_points-1) + lower_bound, 0.0);
	}
}

void Curve::ZeroYData()
{
	for (int counter = 0; counter < number_of_points; counter++)
	{
		data_y[counter] = 0;
	}
}

void Curve::SetYToConstant(float wanted_constant)
{
	MyDebugAssertTrue(number_of_points > 0, "No points to set");
	for (int counter = 0; counter < number_of_points; counter++)
	{
		data_y[counter] = wanted_constant;
	}
}

void Curve::AddWith(Curve *other_curve)
{
	MyDebugAssertTrue(number_of_points > 0, "No points to interpolate");
	MyDebugAssertTrue(number_of_points == other_curve->number_of_points, "Different number of points");

	for (int counter = 0; counter < number_of_points; counter++)
	{
		data_y[counter] += other_curve->data_y[counter];
	}
}

void Curve::DivideBy(Curve *other_curve)
{
	MyDebugAssertTrue(number_of_points > 0, "No points to interpolate");
	MyDebugAssertTrue(number_of_points == other_curve->number_of_points, "Different number of points");

	for (int counter = 0; counter < number_of_points; counter++)
	{
		if (other_curve->data_y[counter] != 0.0)
		data_y[counter] /= other_curve->data_y[counter];
	}
}

float Curve::ReturnAverageValue()
{
	MyDebugAssertTrue(number_of_points > 0, "No points to average");

	float sum = 0.0f;

	for (int counter = 0; counter < number_of_points; counter++)
	{
		sum += data_y[counter];
	}

	return sum / float(number_of_points);
}

void Curve::ZeroAfterIndex(int index)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	if (index + 1 <= number_of_points)
	{
		for (int counter = index + 1; counter < number_of_points; counter++)
		{
			data_y[counter] = 0.0;
		}
	}
}

void Curve::FlattenBeforeIndex(int index)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	if (index > number_of_points) index = number_of_points;
	for (int counter = 0; counter < index; counter++)
	{
		data_y[counter] = data_y[index];
	}
}

void Curve::ResampleCurve(Curve *input_curve, int wanted_number_of_points)
{
	MyDebugAssertTrue(input_curve->number_of_points > 0, "Input curve is empty");
	MyDebugAssertTrue(wanted_number_of_points > 1, "wanted_number_of_points is smaller than 2");

	Curve temp_curve;

	float i_x;

	for (int i = 0; i < wanted_number_of_points; i++)
	{
		i_x = float(i * input_curve->number_of_points) / float(wanted_number_of_points) * (1.0 - 1.0 / float(wanted_number_of_points - 1));
		temp_curve.AddPoint(i_x, input_curve->ReturnLinearInterpolationFromI(i_x));
	}

	CopyFrom(&temp_curve);
}

float Curve::ReturnLinearInterpolationFromI(float wanted_i)
{
	MyDebugAssertTrue(number_of_points > 0, "No points to interpolate");
	MyDebugAssertTrue(wanted_i <= number_of_points - 1, "Index too high");
	MyDebugAssertTrue(wanted_i >= 0, "Index too low");

	int i = int(wanted_i);

	float distance_below = wanted_i - i;
	float distance_above = 1.0 - distance_below;
	float distance;

	if (distance_below == 0.0) return data_y[i];
	if (distance_above == 0.0) return data_y[i + 1];

	return (1.0 - distance_above) * data_y[i + 1] + (1.0 - distance_below) * data_y[i];
}

float Curve::ReturnLinearInterpolationFromX(float wanted_x)
{
	MyDebugAssertTrue(number_of_points > 0, "No points to interpolate");
	MyDebugAssertTrue(wanted_x >= data_x[0]  - (data_x[number_of_points-1]-data_x[0])*0.01 && wanted_x <= data_x[number_of_points-1]  + (data_x[number_of_points-1]-data_x[0])*0.01, "Wanted X (%f) falls outside of range (%f to %f)\n",wanted_x, data_x[0],data_x[number_of_points-1]);

	float value_to_return = 0.0;

	const int index_of_previous_bin = ReturnIndexOfNearestPreviousBin(wanted_x);

	if (index_of_previous_bin == number_of_points-1)
	{
		value_to_return =  data_y[number_of_points-1];
	}
	else
	{
		float distance = (wanted_x - data_x[index_of_previous_bin])/(data_x[index_of_previous_bin+1] - data_x[index_of_previous_bin]);
		value_to_return += data_y[index_of_previous_bin]   * (1.0 - distance);
		value_to_return += data_y[index_of_previous_bin+1] * distance;
	}
	return value_to_return;
}

void Curve::ComputeMaximumValueAndMode(float &maximum_value, float &mode)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	maximum_value = - FLT_MAX;

	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] > maximum_value)
		{
			maximum_value = data_y[counter];
			mode = data_x[counter];
		}
	}
}

// This is meant to be used to measure, e.g., the full-width at half-maximum of a histogram
// stored as a curve
float Curve::ReturnFullWidthAtGivenValue(const float &wanted_value)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	int first_bin_above_value = -1;
	int last_bin_above_value = -1;

	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (first_bin_above_value == -1 && data_y[counter] > wanted_value) first_bin_above_value = counter;
		if (last_bin_above_value  == -1 && first_bin_above_value != -1 && data_y[counter] < wanted_value) last_bin_above_value  = counter - 1;
	}

	MyDebugAssertTrue(first_bin_above_value != -1,"Could not find first bin above value");
	MyDebugAssertTrue(last_bin_above_value != -1,"Could not find last bin above value");

	return data_x[last_bin_above_value + 1] - data_x[first_bin_above_value];
}

float Curve::ReturnMaximumValue()
{
	float maximum_value, mode;
	ComputeMaximumValueAndMode(maximum_value,mode);
	return maximum_value;
}

float Curve::ReturnMode()
{
	float maximum_value, mode;
	ComputeMaximumValueAndMode(maximum_value,mode);
	return mode;
}

// Scale the Y values so that the peak is at 1.0 (assumes all Y values >=0)
void Curve::NormalizeMaximumValue()
{
#ifdef DEBUG
	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] < 0.0) MyDebugAssertTrue(false,"This routine assumes all Y values are positive, but value %i is %f\n",counter,data_y[counter]);
	}
#endif

	const float maximum_value = ReturnMaximumValue();

	if (maximum_value > 0.0)
	{
		float factor = 1.0 / maximum_value;
		for ( int counter = 0; counter < number_of_points; counter ++ )
		{
			data_y[counter] *= factor;
		}
	}
}

// Replace Y values with their log, base 10
void Curve::Logarithm()
{
#ifdef DEBUG
	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] < 0.0) MyDebugAssertTrue(false,"This routine assumes all Y values are positive, but value %i is %f\n",counter,data_y[counter]);
	}
#endif


	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		data_y[counter] = log10(data_y[counter]);
	}
}

void Curve::SquareRoot()
{
#ifdef DEBUG
	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] < 0.0) MyDebugAssertTrue(false,"This routine assumes all Y values are positive, but value %i is %f\n",counter,data_y[counter]);
	}
#endif

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		data_y[counter] = sqrtf(data_y[counter]);
	}
}


CurvePoint Curve::ReturnValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(wanted_x >= data_x[0] - (data_x[number_of_points-1]-data_x[0])*0.01 && wanted_x <= data_x[number_of_points-1] + (data_x[number_of_points-1]-data_x[0])*0.01, "Wanted X (%f) falls outside of range (%f to %f)\n",wanted_x, data_x[0],data_x[number_of_points-1]);

	int index_of_previous_bin;
	CurvePoint return_value;
	if (assume_linear_x)
	{
		index_of_previous_bin = int((wanted_x - data_x[0]) / (data_x[1] - data_x[0]));
	}
	else
	{
		index_of_previous_bin = ReturnIndexOfNearestPreviousBin(wanted_x);
	}

	MyDebugAssertTrue(index_of_previous_bin >= 0 && index_of_previous_bin < number_of_points,"Oops. Bad index_of_previous_bin: %i\n",index_of_previous_bin);

	if (index_of_previous_bin == number_of_points - 1)
	{
		return_value.index_m = index_of_previous_bin;
		return_value.index_n = index_of_previous_bin;
		return_value.value_m = value_to_add;
		return_value.value_n = 0.0;
	}
	else
	{
		float distance = (wanted_x - data_x[index_of_previous_bin]) / (data_x[index_of_previous_bin + 1] - data_x[index_of_previous_bin]);
		return_value.index_m = index_of_previous_bin;
		return_value.index_n = index_of_previous_bin + 1;
		return_value.value_m = value_to_add * (1.0 - distance);
		return_value.value_n = value_to_add * distance;
	}

	return return_value;
}

// If it the data_x values form a linear axies (i.e. they are regularly spaced), give assume_linear_x true.
void Curve::AddValueAtXUsingLinearInterpolation(float wanted_x, float value_to_add, bool assume_linear_x)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(wanted_x >= data_x[0] - (data_x[number_of_points-1]-data_x[0])*0.01 && wanted_x <= data_x[number_of_points-1] + (data_x[number_of_points-1]-data_x[0])*0.01, "Wanted X (%f) falls outside of range (%f to %f)\n",wanted_x, data_x[0],data_x[number_of_points-1]);

	CurvePoint return_value;

	return_value = ReturnValueAtXUsingLinearInterpolation(wanted_x, value_to_add, assume_linear_x);
	data_y[return_value.index_m] += return_value.value_m;
	data_y[return_value.index_n] += return_value.value_n;
}

void Curve::AddValueAtXUsingNearestNeighborInterpolation(float wanted_x, float value_to_add)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(wanted_x >= data_x[0]  - (data_x[number_of_points-1]-data_x[0])*0.01 && wanted_x <= data_x[number_of_points-1]  + (data_x[number_of_points-1]-data_x[0])*0.01, "Wanted X (%f) falls outside of range (%f to %f)\n",wanted_x, data_x[0],data_x[number_of_points-1]);

	data_y[ReturnIndexOfNearestPointFromX(wanted_x)] += value_to_add;

}


void Curve::PrintToStandardOut()
{
	for (int i = 0; i < number_of_points; i++)
	{
		wxPrintf("%f %f\n",data_x[i],data_y[i]);
	}
}

void Curve::WriteToFile(wxString output_file, wxString header_line)
{
	MyDebugAssertTrue(number_of_points > 0, "Curve is empty");

	float temp_float[2];

	NumericTextFile output_curve_file(output_file, OPEN_TO_WRITE, 2);
	output_curve_file.WriteCommentLine(header_line);
	for (int i = 0; i < number_of_points; i++)
	{
		temp_float[0] = data_x[i];
		temp_float[1] = data_y[i];

		output_curve_file.WriteLine(temp_float);
	}
}

void Curve::WriteToFile(wxString output_file)
{
	WriteToFile(output_file,"C            X              Y");
}

void Curve::CopyDataFromArrays(double *x_series, double *y_series, int wanted_number_of_points)
{
	ClearData();
	for (int counter=0; counter < wanted_number_of_points; counter++)
	{
		AddPoint(float(x_series[counter]),float(y_series[counter]));
	}
}

void Curve::CopyYValuesFromArray(double *y_series, int wanted_number_of_points)
{
	ClearData();
	for (int counter=0; counter < wanted_number_of_points; counter++)
	{
		AddPoint(float(counter),float(y_series[counter]));
	}
}

void Curve::CopyFrom(Curve *other_curve)
{
	*this = other_curve;
}


void Curve::GetXMinMax(float &min_value, float &max_value)
{
	min_value = FLT_MAX;
	max_value = -FLT_MAX;

	for (int point_counter = 0; point_counter < number_of_points; point_counter++)
	{
		min_value = std::min(min_value, data_x[point_counter]);
		max_value = std::max(max_value, data_x[point_counter]);
	}

}

void Curve::GetYMinMax(float &min_value, float &max_value)
{
	min_value = FLT_MAX;
	max_value = -FLT_MAX;

	for (int point_counter = 0; point_counter < number_of_points; point_counter++)
	{
		min_value = std::min(min_value, data_y[point_counter]);
		max_value = std::max(max_value, data_y[point_counter]);
	}
}

void Curve::CheckMemory()
{
	if (number_of_points >= allocated_space_for_points)
	{
		// reallocate..

		if (allocated_space_for_points < 10000) allocated_space_for_points *= 2;
		else allocated_space_for_points += 10000;

		float *x_buffer = new float[allocated_space_for_points];
		float *y_buffer = new float[allocated_space_for_points];

		for (long counter = 0; counter < number_of_points; counter++)
		{
			x_buffer[counter] = data_x[counter];
			y_buffer[counter] = data_y[counter];
		}

		delete [] data_x;
		delete [] data_y;

		data_x = x_buffer;
		data_y = y_buffer;
	}
}

void Curve::AllocateMemory(int wanted_number_of_points)
{
	delete [] data_x;
	delete [] data_y;
	allocated_space_for_points = wanted_number_of_points;
	data_x = new float[allocated_space_for_points];
	data_y = new float[allocated_space_for_points];
	number_of_points = 0;
}

void Curve::AddPoint(float x_value, float y_value)
{
	// check memory

	CheckMemory();

	// add the point

	data_x[number_of_points] = x_value;
	data_y[number_of_points] = y_value;

	number_of_points++;
}

void Curve::ClearData()
{
	number_of_points = 0;

	if (have_polynomial == true)
	{
		delete [] polynomial_fit;
		delete [] polynomial_coefficients;

		have_polynomial = false;
	}

	if (have_savitzky_golay == true)
	{
		delete [] savitzky_golay_fit;

		have_savitzky_golay = false;
	}
}

void Curve::MultiplyByConstant(float constant_to_multiply_by)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		data_y[counter] *= constant_to_multiply_by;
	}
}

// It is assumed that the X axis has spatial frequencies in reciprocal pixels (0.5 is Nyquist)
void Curve::ApplyCTF(CTF ctf_to_apply, float azimuth_in_radians)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		data_y[counter] *= ctf_to_apply.Evaluate(powf(data_x[counter],2),azimuth_in_radians);
	}
}

void Curve::ApplyCosineMask(float wanted_x_of_cosine_start, float wanted_cosine_width_in_x, bool undo)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	float current_x;
	float edge;

	for (int counter = 0; counter < number_of_points; counter ++ )
	{

		current_x = data_x[counter];
		if (current_x >= wanted_x_of_cosine_start && current_x <= wanted_x_of_cosine_start + wanted_cosine_width_in_x)
		{
			edge = (1.0 + cosf(PI * (current_x - wanted_x_of_cosine_start) / wanted_cosine_width_in_x)) / 2.0;
			if (undo)
			{
				//MyDebugAssertFalse(edge == 0.0,"Raised cosine should not be 0.0");
				if (edge > 0.0) data_y[counter] /= edge;
			}
			else
			{
				data_y[counter] *= edge;
			}
		}
		else if (current_x > wanted_x_of_cosine_start + wanted_cosine_width_in_x)
		{
			data_y[counter] = 0.0;
		}

	}
}


float Curve::ReturnSavitzkyGolayInterpolationFromX( float wanted_x )
{
	//MyDebugAssertTrue(wanted_x >= data_x[0] && wanted_x <= data_x[number_of_points],"Wanted x (%f) outside of range (%f to %f)\n",wanted_x,data_x[0],data_x[number_of_points]);

	// Find the nearest data point to the wanted_x
	int index_of_nearest_point = ReturnIndexOfNearestPointFromX( wanted_x );

	// Savitzky-Golay coefficients are zero for last index
	if (index_of_nearest_point == number_of_points - 1) index_of_nearest_point--;

	// Evaluate the polynomial defined at the nearest point.
	// TODO: use a better algorithm to evaluate the poynomial, e.g. Horner, see Numerical Recipes
	double y = savitzky_golay_coefficients[index_of_nearest_point][0];
	for (int order = 1; order <= savitzky_golay_polynomial_order; order++)
	{
		y += pow(wanted_x,order) * savitzky_golay_coefficients[index_of_nearest_point][order];
	}

	return float(y);
}

int Curve::ReturnIndexOfNearestPointFromX( float wanted_x )
{
	//MyDebugAssertTrue(wanted_x >= data_x[0] && wanted_x <= data_x[number_of_points],"Wanted x (%f) outside of range (%f to %f)\n",wanted_x,data_x[0],data_x[number_of_points]);

	int index_of_nearest_point = 0;
	int counter = 0;
	float distance_to_current_point = wanted_x - data_x[counter];
	float distance_to_nearest_point = distance_to_current_point;
	for (counter = 1; counter < number_of_points; counter ++)
	{
		distance_to_current_point = wanted_x - data_x[counter];
		if (fabs(distance_to_current_point) <= fabs(distance_to_nearest_point))
		{
			distance_to_nearest_point = distance_to_current_point;
			index_of_nearest_point = counter;
		}
		else
		{
			break;
		}
	}
	return index_of_nearest_point;
}

//TODO: write a quicker version of this
int Curve::ReturnIndexOfNearestPreviousBin(float wanted_x)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(wanted_x >= data_x[0]  - (data_x[number_of_points-1]-data_x[0])*0.01 && wanted_x <= data_x[number_of_points-1]  + (data_x[number_of_points-1]-data_x[0])*0.01, "Wanted X (%f) falls outside of range (%f to %f)\n",wanted_x, data_x[0],data_x[number_of_points-1]);


	for (int counter = index_of_last_point_used; counter < number_of_points; counter++)
	{
		if (wanted_x >=data_x[counter] && wanted_x < data_x[counter+1])
		{
			index_of_last_point_used = counter;
			return counter;
		}
	}
	for (int counter = index_of_last_point_used-1; counter >=0; counter--)
	{
		if (wanted_x >=data_x[counter] && wanted_x < data_x[counter+1])
		{
			index_of_last_point_used = counter;
			return counter;
		}
	}

	if (wanted_x < data_x[0])
	{
		index_of_last_point_used = 0;
		return 0;
	}
	else if (wanted_x >= data_x[number_of_points-1])
	{
		index_of_last_point_used = number_of_points - 1;
		return number_of_points - 1;
	}


	// Should never get here
	MyDebugAssertTrue(false,"Oops, programming error\n");
	return 0;
}

void Curve::FitSavitzkyGolayToData(int wanted_window_size, int wanted_polynomial_order)
{
	// make sure the window size is odd

	MyDebugAssertTrue(IsOdd(wanted_window_size) == true, "Window must be odd!")
	MyDebugAssertTrue(wanted_window_size < number_of_points, "Window size is larger than the number of points!");
	MyDebugAssertTrue(wanted_polynomial_order < wanted_window_size, "polynomial order is larger than the window size!");

	int pixel_counter;
	int polynomial_counter;

	int end_start;

	int half_pixel = wanted_window_size / 2;

	float *fit_array_x = new float[wanted_window_size];
	float *fit_array_y = new float[wanted_window_size];
	float *output_fit_array = new float[wanted_window_size];
	//float *coefficients = new float[wanted_polynomial_order+1];

	// Remember the polymomal order and the window size
	savitzky_golay_polynomial_order = wanted_polynomial_order;
	savitzky_golay_window_size = wanted_window_size;


	// Allocate array of coefficient arrays, to be kept in memory for later use (e.g. for interpolation)
	if (savitzky_golay_coefficients != NULL) {
		DeleteSavitzkyGolayCoefficients();
	}
	// Allocate memory for smooth y values
	if (have_savitzky_golay == true)
	{
		delete [] savitzky_golay_fit;
	}

	AllocateSavitzkyGolayCoefficients();
	savitzky_golay_fit = new float[number_of_points];
	have_savitzky_golay = true;



	// loop over all the points..

	for (pixel_counter = 0; pixel_counter < number_of_points - 2 * half_pixel; pixel_counter++ )
	{
		// for this pixel, extract the window, fit the polynomial, and copy the average into the output array

		for (polynomial_counter = 0; polynomial_counter < wanted_window_size; polynomial_counter++)
		{
			fit_array_x[polynomial_counter] = data_x[pixel_counter + polynomial_counter];
			fit_array_y[polynomial_counter] = data_y[pixel_counter + polynomial_counter];
		}

		// fit a polynomial to this data..

		LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients[half_pixel + pixel_counter]);


		// take the middle pixel, and put it into the output array..

		savitzky_golay_fit[half_pixel + pixel_counter] = output_fit_array[half_pixel];
	}

	// now we need to take care of the ends - first the start..
	// DNM: Need to take actual points beyond the end of the fitted points in the middle
	for (polynomial_counter = 0; polynomial_counter < wanted_window_size; polynomial_counter++)
	{
		fit_array_x[polynomial_counter] = data_x[polynomial_counter];

		if (polynomial_counter < half_pixel || polynomial_counter >= number_of_points - half_pixel) fit_array_y[polynomial_counter] = data_y[polynomial_counter];
		else fit_array_y[polynomial_counter] = savitzky_golay_fit[polynomial_counter];
	}

	// fit a polynomial to this data..

	LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients[half_pixel - 1]);

	// copy the required data back..
	for (pixel_counter = 0; pixel_counter < half_pixel - 1; pixel_counter++)
	{
		for (polynomial_counter = 0; polynomial_counter <= savitzky_golay_polynomial_order; polynomial_counter ++)
		{
			savitzky_golay_coefficients[pixel_counter][polynomial_counter] = savitzky_golay_coefficients[half_pixel - 1][polynomial_counter];
		}
	}
	for (polynomial_counter = 0; polynomial_counter < half_pixel; polynomial_counter++)
	{
		//savitzky_golay_coefficients[polynomial_counter] = savitzky_golay_coefficients[half_pixel - 1];
		savitzky_golay_fit[polynomial_counter] = output_fit_array[polynomial_counter];
	}


	// now the end..

	end_start = number_of_points - wanted_window_size;
	pixel_counter = 0;
	for (polynomial_counter = end_start; polynomial_counter < number_of_points; polynomial_counter++)
	{
		fit_array_x[pixel_counter] = data_x[polynomial_counter];

		if (pixel_counter > half_pixel) fit_array_y[pixel_counter] = data_y[polynomial_counter];
		else fit_array_y[pixel_counter] = savitzky_golay_fit[polynomial_counter];

		pixel_counter++;
	}

	// fit a polynomial to this data..

	LS_POLY(fit_array_x, fit_array_y, wanted_window_size, wanted_polynomial_order, output_fit_array, savitzky_golay_coefficients[number_of_points - half_pixel]);

	// copy the required data back..

	for (pixel_counter = number_of_points - half_pixel + 1; pixel_counter < number_of_points - 1; pixel_counter++)
	{
		for (polynomial_counter = 0; polynomial_counter <= savitzky_golay_polynomial_order; polynomial_counter ++)
		{
			savitzky_golay_coefficients[pixel_counter][polynomial_counter] = savitzky_golay_coefficients[number_of_points - half_pixel][polynomial_counter];
		}
	}

	pixel_counter = half_pixel + 1;
	for (polynomial_counter = number_of_points - half_pixel; polynomial_counter < number_of_points; polynomial_counter++)
	{
		//savitzky_golay_coefficients[polynomial_counter] = savitzky_golay_coefficients[number_of_points - half_pixel];
		savitzky_golay_fit[polynomial_counter] = output_fit_array[pixel_counter];
		pixel_counter++;
	}


	delete [] fit_array_x;
	delete [] fit_array_y;
	delete [] output_fit_array;
}

void Curve::FitPolynomialToData(int wanted_polynomial_order)
{
	if (have_polynomial == true)
	{
		delete [] polynomial_coefficients;
		delete [] polynomial_fit;
	}

	polynomial_fit = new float[number_of_points];
	polynomial_order = wanted_polynomial_order;
	polynomial_coefficients = new float[polynomial_order + 1];
	have_polynomial = true;

	LS_POLY(data_x, data_y, number_of_points, polynomial_order, polynomial_fit, polynomial_coefficients); // weird old code to do the fit
}

void Curve::Reciprocal()
{
#ifdef DEBUG
	for ( int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] < 0.0) MyDebugAssertTrue(false,"This routine assumes all Y values are positive, but value %i is %f\n",counter,data_y[counter]);
	}
#endif

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		if (data_y[counter] != 0.0) data_y[counter] = 1.0 / data_y[counter];
	}
}

void Curve::DivideBy(Curve &other_curve)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(number_of_points == other_curve.number_of_points, "Number of points in curves not equal");

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		if (other_curve.data_y[counter] != 0.0) data_y[counter] = data_y[counter] / other_curve.data_y[counter];
	}
}

void Curve::MultiplyBy(Curve &other_curve)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");
	MyDebugAssertTrue(number_of_points == other_curve.number_of_points, "Number of points in curves not equal");

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		data_y[counter] = data_y[counter] * other_curve.data_y[counter];
	}
}

void Curve::MultiplyXByConstant(float constant_to_multiply_by)
{
	MyDebugAssertTrue(number_of_points > 0, "No points in curve");

	for (int counter = 0; counter < number_of_points; counter ++ )
	{
		data_x[counter] *= constant_to_multiply_by;
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



/****************************************************************
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
/*
void Curve::LS_POLY()
{

	double a[polynomial_order + 2];//
	double b[polynomial_order + 2];//
	double c[polynomial_order + 3];
	double c2[polynomial_order + 2];
	double f[polynomial_order + 2];//

	double v[number_of_points + 1];
	double d[number_of_points + 1];
	double e[number_of_points + 1];//
	double x[number_of_points + 1];
	double y[number_of_points + 1];

	int l;//
	int n = number_of_points;//
	int m = polynomial_order;//

	double e1 = 0.0;//
	double dd;
	double vv;

  //Labels: e10,e15,e20,e30,e50,fin;
  int i;
  int l2;
  int n1; //

  double a1;//
  double a2;
  double b1;
  double b2;
  double c1;//
  double d1;//
  double f1;//
  double f2;
  double v1; //
  double v2;
  double w;//

  l = 0;
  n1 = m + 1;
  v1 = 1e7;

  for (i = 0; i < number_of_points; i++)
  {
	  x[i + 1] = data_x[i];

	  wxPrintf("Before %i = %f\n", i, data_y[i]);
	  y[i + 1] = data_y[i];
  }

  // Initialize the arrays
  for (i = 1; i < n1+1; i++) {
    a[i] = 0; b[i] = 0; f[i] = 0;
  };
  for (i = 1; i < n+1; i++) {
    v[i] = 0; d[i] = 0;
  }
  d1 = sqrt(n); w = d1;
  for (i = 1; i < n+1; i++) {
    e[i] = 1 / w;
  }
  f1 = d1; a1 = 0;
  for (i = 1; i < n+1; i++) {
    a1 = a1 + x[i] * e[i] * e[i];
  }
  c1 = 0;
  for (i = 1; i < n+1; i++) {
    c1 = c1 + y[i] * e[i];
  }
  b[1] = 1 / f1; f[1] = b[1] * c1;
  for (i = 1; i < n+1; i++) {
    v[i] = v[i] + e[i] * c1;
  }
  m = 1;
e10: // Save latest results
  for (i = 1; i < l+1; i++)  c2[i] = c[i];
  l2 = l; v2 = v1; f2 = f1; a2 = a1; f1 = 0;
  for (i = 1; i < n+1; i++) {
    b1 = e[i];
    e[i] = (x[i] - a2) * e[i] - f2 * d[i];
    d[i] = b1;
    f1 = f1 + e[i] * e[i];
  }
  f1 = sqrt(f1);
  for (i = 1; i < n+1; i++)  e[i] = e[i] / f1;
  a1 = 0;
  for (i = 1; i < n+1; i++)  a1 = a1 + x[i] * e[i] * e[i];
  c1 = 0;
  for (i = 1; i < n+1; i++)  c1 = c1 + e[i] * y[i];
  m = m + 1; i = 0;
e15: l = m - i; b2 = b[l]; d1 = 0;
  if (l > 1)  d1 = b[l - 1];
  d1 = d1 - a2 * b[l] - f2 * a[l];
  b[l] = d1 / f1; a[l] = b2; i = i + 1;
  if (i != m) goto e15;
  for (i = 1; i < n+1; i++)  v[i] = v[i] + e[i] * c1;
  for (i = 1; i < n1+1; i++) {
    f[i] = f[i] + b[i] * c1;
    c[i] = f[i];
  }
  vv = 0;
  for (i = 1; i < n+1; i++)
	  vv = vv + (v[i] - y[i]) * (v[i] - y[i]);
  //Note the division is by the number of degrees of freedom
  vv = sqrt(vv / (n - l - 1)); l = m;
  if (e1 == 0) goto e20;
  //Test for minimal improvement
  if (fabs(v1 - vv) / vv < e1) goto e50;
  //if error is larger, quit
  if (e1 * vv > e1 * v1) goto e50;
  v1 = vv;
e20: if (m == n1) goto e30;
  goto e10;
e30: //Shift the c[i] down, so c(0) is the constant term
  for (i = 1; i < l+1; i++)  c[i - 1] = c[i];
  c[l] = 0;
  //l is the order of the polynomial fitted
  l = l - 1; dd = vv;
  goto fin;
e50: // Aborted sequence, recover last values
  l = l2; vv = v2;
  for (i = 1; i < l+1; i++)  c[i] = c2[i];
  goto e30;
fin: ;

for (i = 0; i < number_of_points; i++)
{
	polynomial_fit[i] = v[i + 1];
	wxPrintf("After %i = %f\n", i, polynomial_fit[i]);
}

for (i = 0; i < polynomial_order; i++)
{
	polynomial_coefficients[i] = c[i + 1];
}

}
*/
void LS_POLY(float *x_data, float *y_data, int number_of_points, int order_of_polynomial, float *output_smoothed_curve, float *output_coefficients)
{

	double a[order_of_polynomial + 2];
	double b[order_of_polynomial + 2];
	double c[order_of_polynomial + 3];
	double c2[order_of_polynomial + 2];
	double f[order_of_polynomial + 2];

	double v[number_of_points + 1];
	double d[number_of_points + 1];
	double e[number_of_points + 1];
	double x[number_of_points + 1];
	double y[number_of_points + 1];

	int l;//
	int n = number_of_points;
	int m = order_of_polynomial;

	double e1 = 0.0;//
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

	l = 0;
	n1 = m + 1;
	v1 = 1e7;

	for (i = 0; i < number_of_points; i++)
	{
		x[i + 1] = x_data[i];
		y[i + 1] = y_data[i];

		//wxPrintf("Before %i = %f\n", i, y_data[i]);

  }

  // Initialize the arrays
  for (i = 1; i < n1+1; i++) {
    a[i] = 0; b[i] = 0; f[i] = 0;
  };
  for (i = 1; i < n+1; i++) {
    v[i] = 0; d[i] = 0;
  }
  d1 = sqrt(n); w = d1;
  for (i = 1; i < n+1; i++) {
    e[i] = 1 / w;
  }
  f1 = d1; a1 = 0;
  for (i = 1; i < n+1; i++) {
    a1 = a1 + x[i] * e[i] * e[i];
  }
  c1 = 0;
  for (i = 1; i < n+1; i++) {
    c1 = c1 + y[i] * e[i];
  }
  b[1] = 1 / f1; f[1] = b[1] * c1;
  for (i = 1; i < n+1; i++) {
    v[i] = v[i] + e[i] * c1;
  }
  m = 1;
e10: // Save latest results
  for (i = 1; i < l+1; i++)  c2[i] = c[i];
  l2 = l; v2 = v1; f2 = f1; a2 = a1; f1 = 0;
  for (i = 1; i < n+1; i++) {
    b1 = e[i];
    e[i] = (x[i] - a2) * e[i] - f2 * d[i];
    d[i] = b1;
    f1 = f1 + e[i] * e[i];
  }
  f1 = sqrt(f1);
  for (i = 1; i < n+1; i++)  e[i] = e[i] / f1;
  a1 = 0;
  for (i = 1; i < n+1; i++)  a1 = a1 + x[i] * e[i] * e[i];
  c1 = 0;
  for (i = 1; i < n+1; i++)  c1 = c1 + e[i] * y[i];
  m = m + 1; i = 0;
e15: l = m - i; b2 = b[l]; d1 = 0;
  if (l > 1)  d1 = b[l - 1];
  d1 = d1 - a2 * b[l] - f2 * a[l];
  b[l] = d1 / f1; a[l] = b2; i = i + 1;
  if (i != m) goto e15;
  for (i = 1; i < n+1; i++)  v[i] = v[i] + e[i] * c1;
  for (i = 1; i < n1+1; i++) {
    f[i] = f[i] + b[i] * c1;
    c[i] = f[i];
  }
  vv = 0;
  for (i = 1; i < n+1; i++)
	  vv = vv + (v[i] - y[i]) * (v[i] - y[i]);
  //Note the division is by the number of degrees of freedom
  vv = sqrt(vv / (n - l - 1)); l = m;
  if (e1 == 0) goto e20;
  //Test for minimal improvement
  if (fabs(v1 - vv) / vv < e1) goto e50;
  //if error is larger, quit
  if (e1 * vv > e1 * v1) goto e50;
  v1 = vv;
e20: if (m == n1) goto e30;
  goto e10;
e30: //Shift the c[i] down, so c(0) is the constant term
  for (i = 1; i < l+1; i++)  c[i - 1] = c[i];
  c[l] = 0;
  //l is the order of the polynomial fitted
  l = l - 1; dd = vv;
  goto fin;
e50: // Aborted sequence, recover last values
  l = l2; vv = v2;
  for (i = 1; i < l+1; i++)  c[i] = c2[i];
  goto e30;
fin: ;

for (i = 0; i < number_of_points; i++)
{
	output_smoothed_curve[i] = v[i + 1];
//	output_smoothed_curve[i] = y[i + 1];
	//wxPrintf("After %i = %f\n", i, output_smoothed_curve[i]);
}

// coefficient 0: constant
// coefficient 1: linear
// coefficient 2: square
// coefficient 3: cube
// ...
for (i = 0; i <= order_of_polynomial; i++)
{
	output_coefficients[i] = c[i];
}

}



