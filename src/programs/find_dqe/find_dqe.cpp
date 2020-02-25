#include "../../core/core_headers.h"

class
FindDQE : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

int 	padding = 3;			// Amount of super-sampling to simulate detector before pixel binning
int 	margin = 200;			// Margin in pixels used to taper the edges of the images
bool	two_sided_mtf = true;	// Determine the MTF on both sides of the edge and take the average (experimental)
bool	debug = false;			// Output additional images and fitted parameters for debugging

class CurveComparison
{
public:

	Curve	*curve_to_be_fitted;
	Curve	*nps_fit;
	Image	*experimental_image;
	Image	*model_image;
	Image	*model_image_fft;
//	Image	*model_image_real;
	Image	*difference_image;
	Image	*edge_mask;
	int		parameters_to_fit;
	float	best_score;
	int		busy_state;
//	bool	is_a_counting_detector;
	bool	reset_shadow;
	bool	reset_background;
	double	average_shadow;
	double	average_background;
};

float SincFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);
	int i;
	double residual, average;
	float function_value;

	average = 0.0;
	for (i = 100; i < 990; i++) average += comparison_object->curve_to_be_fitted->data_y[i];
	average /= (990 - 100 + 1);

	residual = 0.0;
	for (i = 100; i < 990; i++)
	{
		function_value = (array_of_values[0] - array_of_values[1] * sinc(fabsf(array_of_values[2]) * comparison_object->curve_to_be_fitted->data_x[i])) * expf(-powf(array_of_values[3] * comparison_object->curve_to_be_fitted->data_x[i], 2));
		residual += powf(fabsf(comparison_object->curve_to_be_fitted->data_y[i] / average - function_value), 0.25f);
//		function_value = comparison_object->curve_to_be_fitted->data_y[i] - function_value;
//		if (fabsf(function_value / comparison_object->curve_to_be_fitted->data_y[i]) > 0.1 * average) residual += average;
//		else residual += powf(function_value, 2);
//		wxPrintf("y, fit = %i %g %g\n", i, comparison_object->curve_to_be_fitted->data_y[i], function_value);
	}

//	wxPrintf("x1, x2, x3, x4, residual = %g %g %g %g %g\n", array_of_values[0], array_of_values[1], array_of_values[2], array_of_values[3], float(residual));
	return float(residual);
}

float LogisticFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);
	int i;
	double residual, average;
	float function_value;

	average = 0.0;
	for (i = 100; i < 990; i++) average += comparison_object->curve_to_be_fitted->data_y[i];
	average /= (990 - 100 + 1);

	residual = 0.0;
	for (i = 100; i < 990; i++)
	{
		function_value = array_of_values[0] / (1.0f + expf((array_of_values[1] - comparison_object->curve_to_be_fitted->data_x[i]) / array_of_values[2])) + array_of_values[3];
		residual += powf(fabsf(comparison_object->curve_to_be_fitted->data_y[i] / average - function_value), 0.25f);
//		wxPrintf("y, fit = %i %g %g\n", i, comparison_object->curve_to_be_fitted->data_y[i], function_value);
	}

//	wxPrintf("x1, x2, x3, x4, residual = %g %g %g %g %g\n", array_of_values[0], array_of_values[1], array_of_values[2], array_of_values[3], float(residual));
	return float(residual);
}
/*
float ThreeGaussianSincFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);
	int i, j;
	double residual;
	float function_value;
//	float sum_of_coefficients;

	residual = 0.0f;
//	sum_of_coefficients = 0.0f;
//	for (j = 1; j < comparison_object->parameters_to_fit; j += 2) sum_of_coefficients += fabsf(array_of_values[j]);
	for (i = 0; i < comparison_object->curve_to_be_fitted->number_of_points; i++)
	{
		function_value = 0.0f;
		for (j = 0; j < std::min(comparison_object->parameters_to_fit, 10); j += 2) function_value += fabsf(array_of_values[j + 1]) * exp(-fabsf(array_of_values[j]) * powf(comparison_object->curve_to_be_fitted->data_x[i], 2));
		if (comparison_object->parameters_to_fit > 10) function_value -= sinc(fabsf(array_of_values[10]) * comparison_object->curve_to_be_fitted->data_x[i]) * fabsf(array_of_values[9]) * exp(-fabsf(array_of_values[8]) * powf(comparison_object->curve_to_be_fitted->data_x[i], 2));
//		function_value /= sum_of_coefficients;
		residual += powf(comparison_object->curve_to_be_fitted->data_y[i] - function_value, 2);
	}
//	wxPrintf("a0-10, residual = %g %g %g %g %g %g %g %g %g %g %g %20.6f\n", array_of_values[0], fabsf(array_of_values[1]), array_of_values[2], \
//		fabsf(array_of_values[3]), array_of_values[4], fabsf(array_of_values[5]), array_of_values[6], fabsf(array_of_values[7]), array_of_values[8], \
//		fabsf(array_of_values[9]), fabsf(array_of_values[10]), residual);

	return float(residual);
}

float FiveGaussianFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);
	int i, j;
	double residual;
	float sum_of_gaussians;
//	float sum_of_coefficients;

	residual = 0.0f;
//	sum_of_coefficients = 0.0f;
//	for (j = 1; j < comparison_object->parameters_to_fit; j += 2) sum_of_coefficients += fabsf(array_of_values[j]);
	for (i = 0; i < comparison_object->curve_to_be_fitted->number_of_points; i++)
	{
		sum_of_gaussians = 0.0f;
		for (j = 0; j < comparison_object->parameters_to_fit; j += 2) sum_of_gaussians += fabsf(array_of_values[j + 1]) * exp(-fabsf(array_of_values[j]) * powf(comparison_object->curve_to_be_fitted->data_x[i], 2));
//		sum_of_gaussians /= sum_of_coefficients;
		residual += powf(comparison_object->curve_to_be_fitted->data_y[i] - sum_of_gaussians, 2);
	}
//	wxPrintf("a0-9, residual = %g %g %g %g %g %g %g %g %g %g %20.6f\n", array_of_values[0], fabsf(array_of_values[1]), array_of_values[2], \
//		fabsf(array_of_values[3]), array_of_values[4], fabsf(array_of_values[5]), array_of_values[6], fabsf(array_of_values[7]), array_of_values[8], \
//		fabsf(array_of_values[9]), residual);

	return float(residual);
}
*/
float MTFFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);

	MyDebugAssertTrue(comparison_object->model_image_fft->is_in_real_space == false, "model_image_fft not in Fourier space");
//	MyDebugAssertTrue(comparison_object->model_image_real->is_in_real_space == true, "model_image_real not in real space");
	MyDebugAssertTrue(comparison_object->model_image->is_in_real_space == true, "model_image not in real space");

//	Image binned_model_image;
	Curve mtf_curve;
	int i, j;
	int pointer;
	int pixels_shadow, pixels_background;
	double residual;
	float sum_of_gaussians;
	float sum_of_coefficients;
	float function_max = -FLT_MAX;
	float scale;
//	float threshold = 0.5f * (comparison_object->average_background + comparison_object->average_shadow);
	float threshold = comparison_object->average_shadow + (comparison_object->average_background - comparison_object->average_shadow) / 2.0f;
	double average_shadow_model, average_background_model;
	double average_shadow_image, average_background_image;

	mtf_curve.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((comparison_object->model_image_fft->logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
//	mtf_curve.SetupXAxis(0.0, 0.5f / 1000.0f * 1415.0f, 1416);

	sum_of_coefficients = 0.0f;
	for (j = 1; j < comparison_object->parameters_to_fit; j += 2) sum_of_coefficients += fabsf(array_of_values[j]);
	for (i = 0; i < mtf_curve.number_of_points; i++)
	{
		if (comparison_object->parameters_to_fit == 1) sum_of_gaussians = exp(-fabsf(array_of_values[0]) * powf(mtf_curve.data_x[i] * padding, 2));
		else
		{
			sum_of_gaussians = 0.0f;
			for (j = 0; j < comparison_object->parameters_to_fit; j += 2) sum_of_gaussians += fabsf(array_of_values[j + 1]) * exp(-fabsf(array_of_values[j]) * powf(mtf_curve.data_x[i] * padding, 2));
			sum_of_gaussians /= sum_of_coefficients;
		}
		if (mtf_curve.data_x[i] * padding <= 0.5f) mtf_curve.data_y[i] = sum_of_gaussians * sqrtf(comparison_object->nps_fit->ReturnLinearInterpolationFromX(mtf_curve.data_x[i] * padding));
		else mtf_curve.data_y[i] = sum_of_gaussians * sqrtf(comparison_object->nps_fit->ReturnLinearInterpolationFromX(0.5f));
//		mtf_curve.data_y[i] = sum_of_gaussians;
//		if (comparison_object->is_a_counting_detector) mtf_curve.data_y[i] *= sqrtf(comparison_object->nps_fit->ReturnLinearInterpolationFromX(mtf_curve.data_x[i]));
		scale = mtf_curve.data_y[i] * sinc(mtf_curve.data_x[i] * padding * PI);
		if (scale > function_max && mtf_curve.data_x[i] * padding <= 0.5f) function_max = scale;
//		if (comparison_object->nps_fit->ReturnLinearInterpolationFromX(mtf_curve.data_x[i]) > function_max) function_max = comparison_object->nps_fit->ReturnLinearInterpolationFromX(mtf_curve.data_x[i]);
	}
//	for (i = 0; i < mtf_curve.number_of_points; i++) {mtf_curve.data_y[i] *= sqrtf(comparison_object->nps_fit->ReturnLinearInterpolationFromX(mtf_curve.data_x[i]) / comparison_object->nps_fit->data_y[0]);}

	comparison_object->difference_image->CopyFrom(comparison_object->model_image_fft);
	comparison_object->difference_image->ApplyCurveFilterUninterpolated(&mtf_curve);
	comparison_object->difference_image->BackwardFFT();

	if (padding != 1) comparison_object->difference_image->RealSpaceBinning(padding, padding, 1, true);
//	comparison_object->difference_image->CosineRectangularMask(comparison_object->difference_image->physical_address_of_box_center_x - 0.5f * margin, comparison_object->difference_image->physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, comparison_object->average_background);

	if (comparison_object->reset_shadow)
	{
		// Replace shadow with unblurred version
		for (i = 0; i < comparison_object->difference_image->real_memory_allocated; i++) {if (comparison_object->model_image->real_values[i] < threshold) comparison_object->difference_image->real_values[i] = comparison_object->model_image->real_values[i];}
	}
	if (comparison_object->reset_background)
	{
		// Replace background with unblurred version
		for (i = 0; i < comparison_object->difference_image->real_memory_allocated; i++) {if (comparison_object->model_image->real_values[i] > threshold) comparison_object->difference_image->real_values[i] = comparison_object->model_image->real_values[i];}
	}
	if (comparison_object->busy_state < 0) {return 0.0f;}

	// Recalculate averages in background and shadow
	average_shadow_model = 0.0;
	average_shadow_image = 0.0;
	pixels_shadow = 0;
	average_background_model = 0.0;
	average_background_image = 0.0;
	pixels_background = 0;
	pointer = 0;
	for (j = 0; j < comparison_object->difference_image->logical_y_dimension; j++)
	{
		for (i = 0; i < comparison_object->difference_image->logical_x_dimension; i++)
		{
			if (i >= margin && i <= comparison_object->difference_image->logical_x_dimension - margin && j >= margin && j <= comparison_object->difference_image->logical_y_dimension - margin)
			{
				if (comparison_object->edge_mask->real_values[pointer] == 0.0f)
				{
					if (comparison_object->difference_image->real_values[pointer] <= threshold)
					{
						average_shadow_model += comparison_object->difference_image->real_values[pointer];
//						average_shadow_model = std::min(average_shadow_model, (double)comparison_object->difference_image->real_values[pointer]);
						average_shadow_image += comparison_object->experimental_image->real_values[pointer];
						pixels_shadow++;
					}
					else
					{
						average_background_model += comparison_object->difference_image->real_values[pointer];
//						average_background_model = std::max(average_shadow_model, (double)comparison_object->difference_image->real_values[pointer]);
						average_background_image += comparison_object->experimental_image->real_values[pointer];
						pixels_background++;
					}
				}
			}
			pointer++;
		}
		pointer += comparison_object->difference_image->padding_jump_value;
	}
	average_shadow_model /= pixels_shadow;
	average_background_model /= pixels_background;
	average_shadow_image /= pixels_shadow;
	average_background_image /= pixels_background;
//	average_background_image = comparison_object->average_background;
//	average_shadow_image = comparison_object->average_shadow;
//	wxPrintf("back, shad model, back, shad image = %g %g %g %g\n", average_background_model, average_shadow_model, average_background_image, average_shadow_image);
	comparison_object->difference_image->AddMultiplyAddConstant(-average_shadow_model, (average_background_image - average_shadow_image) / (average_background_model - average_shadow_model), average_shadow_image);

	if (debug)
	{
		comparison_object->difference_image->QuickAndDirtyWriteSlice("model.mrc", 1);
		comparison_object->experimental_image->QuickAndDirtyWriteSlice("experimental.mrc", 1);
		comparison_object->edge_mask->QuickAndDirtyWriteSlice("edge_mask.mrc", 1);
	}
//	comparison_object->difference_image->CosineRectangularMask(comparison_object->difference_image->physical_address_of_box_center_x - 0.5f * margin, comparison_object->difference_image->physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, comparison_object->average_background);

	comparison_object->difference_image->SubtractImage(comparison_object->experimental_image);
	residual = comparison_object->difference_image->ReturnSumOfSquares();
	if (debug) comparison_object->difference_image->QuickAndDirtyWriteSlice("difference.mrc", 1);
//	exit(0);
	if (debug)
	{
		if (comparison_object->parameters_to_fit < 2) wxPrintf("a0, residual = %g %20.10f\n", fabsf(array_of_values[0]), residual);
		else if (comparison_object->parameters_to_fit < 3) wxPrintf("a0-1, residual = %g %g %20.10f\n", fabsf(array_of_values[0]), fabsf(array_of_values[1]), residual);
		else if (comparison_object->parameters_to_fit < 5) wxPrintf("a0-3, residual = %g %g %g %g %20.10f\n", fabsf(array_of_values[0]), \
				fabsf(array_of_values[1]), fabsf(array_of_values[2]), fabsf(array_of_values[3]), residual);
		else if (comparison_object->parameters_to_fit < 7) wxPrintf("a0-5, residual = %g %g %g %g %g %g %20.10f\n", fabsf(array_of_values[0]), \
				fabsf(array_of_values[1]), fabsf(array_of_values[2]), fabsf(array_of_values[3]), fabsf(array_of_values[4]), fabsf(array_of_values[5]), \
				residual);
		else if (comparison_object->parameters_to_fit < 9) wxPrintf("a0-7, residual = %g %g %g %g %g %g %g %g %20.10f\n", fabsf(array_of_values[0]), \
				fabsf(array_of_values[1]), fabsf(array_of_values[2]), fabsf(array_of_values[3]), fabsf(array_of_values[4]), fabsf(array_of_values[5]), \
				fabsf(array_of_values[6]), fabsf(array_of_values[7]), residual);
		else wxPrintf("a0-9, residual = %g %g %g %g %g %g %g %g %g %g %20.10f\n", fabsf(array_of_values[0]), \
				fabsf(array_of_values[1]), fabsf(array_of_values[2]), fabsf(array_of_values[3]), fabsf(array_of_values[4]), fabsf(array_of_values[5]), \
				fabsf(array_of_values[6]), fabsf(array_of_values[7]), fabsf(array_of_values[8]), fabsf(array_of_values[9]), residual);
	}
	if (residual < comparison_object->best_score)
	{
//		comparison_object->difference_image->QuickAndDirtyWriteSlice("diff.mrc", 1);
		comparison_object->best_score = residual;
		scale = 1.0f / function_max;
		if (padding > 1 ) wxPrintf("  MTF at 0.25x, 0.5x, 1x Nyquist, residual = %10.6f %10.6f %10.6f %18.9f\r", mtf_curve.ReturnLinearInterpolationFromX(0.125f / padding) * sinc(0.125f * PI) * scale, \
			mtf_curve.ReturnLinearInterpolationFromX(0.25f / padding) * sinc(0.25f * PI) * scale, mtf_curve.ReturnLinearInterpolationFromX(0.5f / padding) * sinc(0.5f * PI) * scale, residual);
		else wxPrintf("  MTF at 0.25x, 0.5x, 1x Nyquist, residual = %10.6f %10.6f %10.6f %18.9f\r", mtf_curve.ReturnLinearInterpolationFromX(0.125f / padding) * scale, \
			mtf_curve.ReturnLinearInterpolationFromX(0.25f / padding) * scale, mtf_curve.ReturnLinearInterpolationFromX(0.5f / padding) * scale, residual);
		if (debug) wxPrintf("\n");
	}
	if (! debug)
	{
		if (comparison_object->busy_state == 0) wxPrintf("-\r");
		if (comparison_object->busy_state == 1) wxPrintf("\\\r");
		if (comparison_object->busy_state == 2) wxPrintf("/\r");
		comparison_object->busy_state++;
		if (comparison_object->busy_state > 2) comparison_object->busy_state = 0;
	}
	fflush(stdout);

	return float(residual);
}
/*
float MTFFitSinc(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);

	MyDebugAssertTrue(comparison_object->model_image->is_in_real_space == false, "Model image not in Fourier space");

	Curve mtf_curve;
	int i, j;
	double residual;
	float function_value;
	float function_max;

	mtf_curve.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((comparison_object->model_image->logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
//	mtf_curve.SetupXAxis(0.0, 0.5f / 1000.0f * 1415.0f, 1416);

	function_max = -FLT_MAX;
	for (i = 0; i < mtf_curve.number_of_points; i++)
	{
		function_value = 0.0f;
		for (j = 0; j < std::min(comparison_object->parameters_to_fit, 10); j += 2) function_value += fabsf(array_of_values[j + 1]) * exp(-fabsf(array_of_values[j]) * powf(mtf_curve.data_x[i] * padding, 2));
		if (comparison_object->parameters_to_fit > 10) function_value += sinc(fabsf(array_of_values[10]) * mtf_curve.data_x[i]) * fabsf(array_of_values[9]) * exp(-fabsf(array_of_values[8]) * powf(mtf_curve.data_x[i], 2));
		mtf_curve.data_y[i] = function_value;
		if (function_value > function_max) function_max = function_value;
	}
	for (i = 0; i < mtf_curve.number_of_points; i++) mtf_curve.data_y[i] /= function_max;

	comparison_object->difference_image->CopyFrom(comparison_object->model_image);
	comparison_object->difference_image->ApplyCurveFilterUninterpolated(&mtf_curve);
	comparison_object->difference_image->BackwardFFT();
	if (padding != 1) comparison_object->difference_image->RealSpaceBinning(padding, padding, 1, true);
	comparison_object->difference_image->CosineRectangularMask(comparison_object->difference_image->physical_address_of_box_center_x - 0.5f * margin, comparison_object->difference_image->physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, comparison_object->average_background);

	comparison_object->difference_image->SubtractImage(comparison_object->experimental_image);
	residual = comparison_object->difference_image->ReturnSumOfSquares();
	if (residual < comparison_object->best_score)
	{
//		comparison_object->difference_image->QuickAndDirtyWriteSlice("diff.mrc", 1);
		comparison_object->best_score = residual;
		if (padding > 1 ) wxPrintf("  MTF at 0.25x, 0.5x, 1x Nyquist, residual = %10.6f %10.6f %10.6f %15.6f\r", mtf_curve.ReturnLinearInterpolationFromX(0.125f / padding) * sinc(0.125f * PI), \
			mtf_curve.ReturnLinearInterpolationFromX(0.25f / padding) * sinc(0.25f * PI), mtf_curve.ReturnLinearInterpolationFromX(0.5f / padding) * sinc(0.5f * PI), residual);
		else wxPrintf("  MTF at 0.25x, 0.5x, 1x Nyquist, residual = %10.6f %10.6f %10.6f %15.6f\r", mtf_curve.ReturnLinearInterpolationFromX(0.125f / padding), \
			mtf_curve.ReturnLinearInterpolationFromX(0.25f / padding), mtf_curve.ReturnLinearInterpolationFromX(0.5f / padding), residual);
	}
	if (comparison_object->busy_state == 0) wxPrintf("-\r");
	if (comparison_object->busy_state == 1) wxPrintf("\\\r");
	if (comparison_object->busy_state == 2) wxPrintf("/\r");
	comparison_object->busy_state++;
	if (comparison_object->busy_state > 2) comparison_object->busy_state = 0;
	fflush(stdout);

	return float(residual);
}
*/
float RampFit(void *scoring_parameters, float *array_of_values)
{
	CurveComparison *comparison_object = reinterpret_cast < CurveComparison *> (scoring_parameters);

	MyDebugAssertTrue(comparison_object->experimental_image->is_in_real_space, "Experimental image not in real space");
	MyDebugAssertTrue(comparison_object->model_image->is_in_real_space == true, "Model image not in real space");

	int i, j;
	int ix, iy;
	int pointer = 0;
	double residual;
	float ramp;

	residual = 0.0f;
	for (j = 0; j < comparison_object->experimental_image->logical_y_dimension; j++)
	{
		iy = (j - comparison_object->experimental_image->physical_address_of_box_center_y);
		for (i = 0; i < comparison_object->experimental_image->logical_x_dimension; i++)
		{
			ix = (i - comparison_object->experimental_image->physical_address_of_box_center_x);
			ramp = array_of_values[0] + array_of_values[1] * ix + array_of_values[2] * iy;
			if (comparison_object->parameters_to_fit > 3) ramp += array_of_values[3] * ix * ix + array_of_values[4] * iy * iy;
			if (comparison_object->parameters_to_fit > 5) ramp += array_of_values[5] * ix * ix * ix + array_of_values[6] * iy * iy * iy;
			residual += powf((comparison_object->experimental_image->real_values[pointer] - ramp) * comparison_object->model_image->real_values[pointer], 2);
			pointer++;
		}
		pointer += comparison_object->experimental_image->padding_jump_value;
	}

	if (residual < comparison_object->best_score)
	{
		comparison_object->best_score = residual;
		wxPrintf("  Residual = %15.6f\r", residual);
//		wxPrintf("  a0-6, residual = %g %g %g %g %g %g %g %15.6f\r", array_of_values[0], array_of_values[1], array_of_values[2], array_of_values[3], \
//				array_of_values[4], array_of_values[5], array_of_values[6], residual);
	}
	if (comparison_object->busy_state == 0) wxPrintf("-\r");
	if (comparison_object->busy_state == 1) wxPrintf("\\\r");
	if (comparison_object->busy_state == 2) wxPrintf("/\r");
	comparison_object->busy_state++;
	if (comparison_object->busy_state > 2) comparison_object->busy_state = 0;
	fflush(stdout);

	return float(residual);
}

void FindThreshold(Image &input_image, float &threshold, double &average_background, double &sigma_background, \
		int &pixels_background, double &average_shadow, double &sigma_shadow, int &pixels_shadow)
{
	int i, j, k;
	int pointer;

//	threshold = 0.5f * input_image.ReturnAverageOfMaxN(10000);
	threshold = 0.5f * input_image.ReturnAverageOfRealValues();
	for (k = 0; k < 3; k++)
	{
		pointer = 0;
		pixels_background = 0;
		pixels_shadow = 0;
		average_background = 0.0f;
		sigma_background = 0.0f;
		average_shadow = 0.0f;
		sigma_shadow = 0.0f;
		for (j = 0; j < input_image.logical_y_dimension; j++)
		{
			for (i = 0; i < input_image.logical_x_dimension; i++)
			{
				if (input_image.real_values[pointer] > threshold)
				{
					average_background += input_image.real_values[pointer];
					sigma_background += input_image.real_values[pointer] * input_image.real_values[pointer];
					pixels_background++;
				}
				else
				{
					average_shadow += input_image.real_values[pointer];
					sigma_shadow += input_image.real_values[pointer] * input_image.real_values[pointer];
					pixels_shadow++;
				}
				pointer++;
			}
			pointer += input_image.padding_jump_value;
		}
		if (pixels_background > 0) {average_background /= pixels_background; sigma_background /= pixels_background;}
		if (pixels_shadow > 0) {average_shadow /= pixels_shadow; sigma_shadow /= pixels_shadow;}
		threshold = average_shadow + (average_background - average_shadow) / 2.0f;
	}
	sigma_background = sigma_background - average_background * average_background;
	sigma_shadow = sigma_shadow - average_shadow * average_shadow;

	// Increase threshold a little to move edge slightly further outside the shadow to make CCD MTFs symmerical
//	threshold = average_shadow + 1.1f * (average_background - average_shadow) / 2.0f;

	return;
}

IMPLEMENT_APP(FindDQE)

// override the DoInteractiveUserInput

void FindDQE::DoInteractiveUserInput()
{
	UserInput *my_input = new UserInput("FindDQE", 1.0);
//	float exposure = 0.0f;

	std::string input_pointer_image		= my_input->GetFilenameFromUser("Input shadow image 1", "Filename of input image 1 showing shadow of pointer or aperture", "input1.mrc", true );
	std::string input_pointer_image2	= my_input->GetFilenameFromUser("Input shadow image 2", "Filename of input image 2 showing shadow of pointer or aperture", "input2.mrc", false );
	bool use_both_images				= my_input->GetYesNoFromUser("Use both images", "Should both images be used in the calculation for more reliable results?", "Yes");
	std::string output_table			= my_input->GetFilenameFromUser("Output DQE table", "Filename of output table listing MTF, NPS and DQE values", "table.txt", false );
	std::string output_diagnostic_image	= my_input->GetFilenameFromUser("Output diagnostic image", "Filename of output image showing residuals after MTF fit", "residual.mrc", false );
	bool is_a_counting_detector			= my_input->GetYesNoFromUser("Counting detector", "Were the input images recorded on a counting detector?", "Yes");
//	float counts_multiplier				= my_input->GetFloatFromUser("Counts multiplier", "Factor to be applied to camera counts to undo multiplication by the data collection software", "1.0", 0.1f, 10.0f);
	float exposure						= my_input->GetFloatFromUser("Exposure per pixel", "The number of electrons per pixel used to collect the input image", "50.0", 10.0f, 10000.0f);

	delete my_input;

//	my_current_job.Reset(5);
//	my_current_job.ManualSetArguments("ttbff", input_pointer_image.c_str(), output_diagnostic_image.c_str(), is_a_counting_detector, counts_multiplier, exposure);
//	my_current_job.Reset(6);
	my_current_job.ManualSetArguments("ttbttbf", input_pointer_image.c_str(), input_pointer_image2.c_str(), use_both_images, output_table.c_str(), output_diagnostic_image.c_str(), is_a_counting_detector, exposure);
}

// override the do calculation method which will be what is actually run..

bool FindDQE::DoCalculation()
{

	std::string	input_pointer_image 	= my_current_job.arguments[0].ReturnStringArgument();
	std::string	input_pointer_image2 	= my_current_job.arguments[1].ReturnStringArgument();
	bool use_both_images				= my_current_job.arguments[2].ReturnBoolArgument();
	std::string output_table				= my_current_job.arguments[3].ReturnStringArgument();
	std::string	output_diagnostic_image = my_current_job.arguments[4].ReturnStringArgument();
	bool is_a_counting_detector			= my_current_job.arguments[5].ReturnBoolArgument();
//	float counts_multiplier				= my_current_job.arguments[3].ReturnFloatArgument();
	float exposure						= my_current_job.arguments[6].ReturnFloatArgument();

	int i, j, l, m;
	int ix, iy;
	int pointer;
	int pixels_background;
	int pixels_shadow;
	int pixels_background2;
	int pixels_shadow2;
	int discrepancy_pixels = 0;
	int cycle;
	long counter1;
	int outliers;
	int edge_pixels = 0;
	int x_min, x_max, y_min, y_max;
	int x_mid, y_mid;
	int box_size_x, box_size_y;
	int fit_window_size;
	int bin_size = 50;
	int bin_size_range = 5;
	float threshold;
	float threshold1;
	float threshold2;
	double threshold3;
	float temp_float;
	float average;
	float function_value;
	float function_max;
	float low_pass_filter_constant = 0.05f;
	float gain_conversion_factor;
	float dqe, dqe0;
	float nps0, nps02, nps09, nps1;
	float mask_volume;
	float average_background_float;
	float average_shadow_float;
	float ramp;
	float two_image_factor = 1.0f;
	float allowed_discrepancy = 0.01f;
	float sum_of_coefficients;
	float variance;
	float sizing_threshold = 1000.0f;
	float score;
	float best_mtf_score;
	float offset;
	float slope;
	double average_background;
	double sigma_background;
	double average_shadow;
	double sigma_shadow;
	double average_background2;
	double sigma_background2;
	double average_shadow2;
	double sigma_shadow2;
//	bool is_a_counting_detector = true;

	ImageFile input_file(input_pointer_image,false);
	if (input_file.ReturnZSize() > 1)
	{
		wxPrintf("\nERROR: Input image must be 2D image. Aborting...\n");
		exit(1);
	}
	if (input_file.ReturnXSize() < 5 * margin || input_file.ReturnYSize() < 5 * margin)
	{
		wxPrintf("\nERROR: Input image too small. Aborting...\n");
		exit(1);
	}
	if (use_both_images)
	{
		if (! DoesFileExist(input_pointer_image2))
		{
			wxPrintf("\nERROR: Input image 2 does not exist. Aborting...\n");
			exit(1);
		}
	}

	MRCFile output_file(output_diagnostic_image,true);
	FILE *table_file;
	Image input_image1, input_image2, difference_image;
	Image input_image, temp_image;
	Image threshold_image;
	Image threshold_image_fft;
	Image threshold_image_small;
//	Image threshold_image_real;
	Image mask_image;
//	Image outlier_image;
	Image background_image;
	Image background_mask_image;
//	Curve nps_fit_spectrum;
//	Curve noise_whitening_spectrum;
	Curve mtf, nps, nps_fit, fit_window1;
	Curve number_of_terms, fit_window2;
	Curve fit_window3;
	CurveComparison comparison_object;
	ConjugateGradient conjugate_gradient_minimizer;
	rle3d runlenth3d;
	float cg_starting_point[11];
	float cg_accuracy[11];
	float saved_parameters[11];
	float best_parameters[11];
	float *fitted_parameters;

	threshold_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	threshold_image_small.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	mask_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
//	outlier_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	background_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	background_mask_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

	table_file = fopen(output_table.c_str(), "w");
	if (table_file == NULL)
	{
		wxPrintf("\nERROR: Cannot open output DQE table (%s) for write\n", output_table);
		exit(1);
	}

	wxPrintf("\nReading input image 1...  ");
	input_image.ReadSlice(&input_file, 1);
	wxPrintf("Dimensions (x,y) = %i %i\n", input_image.logical_x_dimension, input_image.logical_y_dimension);
	// Remove crazy outliers
	input_image.SetMaximumValue(5.0f * sqrtf(input_image.ReturnVarianceOfRealValues()) + input_image.ReturnAverageOfMaxN(10000));

	background_image.CopyFrom(&input_image);

	if (use_both_images)
	{
		wxPrintf("Reading input image 2...  ");
		ImageFile input_file2(input_pointer_image2,false);
		wxPrintf("Dimensions (x,y) = %i %i\n", input_image.logical_x_dimension, input_image.logical_y_dimension);
		if (input_file.ReturnXSize() != input_file2.ReturnXSize() || input_file.ReturnYSize() != input_file2.ReturnYSize() || input_file.ReturnZSize() != input_file2.ReturnZSize())
		{
			wxPrintf("\nERROR: Image 2 size must match Image 1. Aborting...\n");
			exit(1);
		}
		input_image2.ReadSlice(&input_file2, 1);
		// Remove crazy outliers
		input_image2.SetMaximumValue(5.0f * sqrtf(input_image2.ReturnVarianceOfRealValues()) + input_image2.ReturnAverageOfMaxN(10000));
		FindThreshold(input_image, threshold1, average_background, sigma_background, pixels_background, average_shadow, sigma_shadow, pixels_shadow);
		FindThreshold(input_image2, threshold2, average_background2, sigma_background2, pixels_background2, average_shadow2, sigma_shadow2, pixels_shadow2);
		if ((fabsf(threshold1 - threshold2) / threshold1 > allowed_discrepancy) || \
			(fabs(average_background - average_background2) / average_background > allowed_discrepancy))
		{
			wxPrintf("\nERROR: Image 2 exposure differs from Image 1. Aborting...\n");
			exit(1);
		}
		// 9.23194e-05, 7.12311e-05, 0.00328548, 1.45924e-05, 0.000838832, 0.000659571, 1.2993e-05
//		(float(abs(pixels_background - pixels_background2)) / pixels_background > allowed_discrepancy) || \
//		(fabs(average_shadow - average_shadow2) / average_shadow > 5.0f * allowed_discrepancy) || \
//		(fabs(sigma_shadow - sigma_shadow2) / sigma_shadow > 5.0f * allowed_discrepancy) || \
//		(float(abs(pixels_shadow - pixels_shadow2)) / pixels_shadow > allowed_discrepancy)) || \
//		(fabs(sigma_background - sigma_background2) / sigma_background > allowed_discrepancy)

//		wxPrintf("%g, %g, %g, %g, %g, %g, %g\n", fabsf(threshold - threshold2) / threshold, fabs(average_background - average_background2) / average_background, \
//				fabs(sigma_background - sigma_background2) / sigma_background, float(abs(pixels_background - pixels_background2)) / pixels_background, \
//				fabs(average_shadow - average_shadow2) / average_shadow, fabs(sigma_shadow - sigma_shadow2) / sigma_shadow, float(abs(pixels_shadow - pixels_shadow2)) / pixels_shadow);
//		wxPrintf("%g, %g, %g, %i, %g, %g, %i\n", threshold2, average_background2, sigma_background2, pixels_background2, average_shadow2, sigma_shadow2, pixels_shadow2);
//		wxPrintf("discrepancy = %i\n", discrepancy_pixels);
//		input_image2.RealSpaceIntegerShift(1,0,0);
//		input_image2.ForwardFFT();
//		input_image2.CalculateDerivative();
//		input_image2.BackwardFFT();
//		input_image2.SetMinimumValue(0.0f);
//		background_image.ForwardFFT();
//		background_image.CalculateDerivative();
//		background_image.BackwardFFT();
//		background_image.SetMinimumValue(0.0f);
		background_image.SubtractImage(&input_image2);
//		for (i = 0; i < background_image.real_memory_allocated; i++) background_image.real_values[i] = fabsf(background_image.real_values[i]);
//		background_image.QuickAndDirtyWriteSlice("background_image.mrc", 1);
//		input_image.QuickAndDirtyWriteSlice("input_image.mrc", 1);
//		input_image2.QuickAndDirtyWriteSlice("input_image2.mrc", 1);
//		exit(0);
		input_image1.CopyFrom(&input_image);
		input_image.AddImage(&input_image2);
		two_image_factor = 2.0f;
		exposure *= two_image_factor;
	}

	wxPrintf("\nThresholding image...  ");
	FindThreshold(input_image, threshold, average_background, sigma_background, pixels_background, average_shadow, sigma_shadow, pixels_shadow);
	if (sigma_background <= 0.0f)
	{
		wxPrintf("\nERROR: Background in input image is zero. Aborting...\n");
		exit(1);
	}
	if (pixels_shadow <= 0.0f)
	{
		wxPrintf("\nERROR: Shadow area in input image is zero. Aborting...\n");
		exit(1);
	}
	sigma_background = sqrtf(sigma_background);
	sigma_shadow = sqrtf(sigma_shadow);
	wxPrintf("Threshold = %g\n\n", threshold / two_image_factor);
	wxPrintf("No of pixels, average, sigma shadow     = %12i %12.4f %12.4f\n", pixels_shadow, average_shadow / two_image_factor, sigma_shadow / sqrtf(two_image_factor));
	wxPrintf("No of pixels, average, sigma background = %12i %12.4f %12.4f\n", pixels_background, average_background / two_image_factor, sigma_background / sqrtf(two_image_factor));
	gain_conversion_factor = average_background / exposure;

	input_image.SetMaximumValue(average_background + 5.0f * sigma_background);
	average_background_float = average_background;
	average_shadow_float = average_shadow;
	threshold_image_small.SetToConstant(average_shadow_float);
	for (i = 0; i < input_image.real_memory_allocated; i++) if (input_image.real_values[i] > threshold) threshold_image_small.real_values[i] = average_background_float;
//	threshold_image_small.CopyFrom(&threshold_image);

	wxPrintf("\nDetecting pixel outliers...  ");
	threshold_image.CopyFrom(&input_image);
	threshold_image.Binarise(threshold);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image4.mrc", 1);
	runlenth3d.EncodeFrom(threshold_image);
	runlenth3d.ConnectedSizeDecodeTo(threshold_image);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image5.mrc", 1);
	threshold_image.MultiplyByConstant(-1.0f);
	threshold_image.Binarise(-sizing_threshold);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image2.mrc", 1);
//	temp_image.CopyFrom(&input_image);
//	temp_image.MultiplyByConstant(-1.0f);
//	temp_image.Binarise(-threshold);
//	rle3d rle3d2(temp_image);
//	rle3d2.ConnectedSizeDecodeTo(temp_image);
//	temp_image.QuickAndDirtyWriteSlice("threshold_image3.mrc", 1);
	runlenth3d.EncodeFrom(threshold_image);
	runlenth3d.ConnectedSizeDecodeTo(threshold_image);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image3.mrc", 1);
//	max_background = threshold_image.ReturnMaximumValue() - 1.0f;
//	max_shadow = temp_image.ReturnMaximumValue() - 1.0f;
//	threshold_image.Binarise(threshold_image.ReturnMaximumValue() - 1.0f);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image.mrc", 1);
//	exit(0);
//	outliers = 0;
	for (i = 0; i < input_image.real_memory_allocated; i++)
	{
//		if (threshold_image.real_values[i] < max_background && temp_image.real_values[i] < max_shadow)
//		{
//			outliers++;
//			if (threshold_image.real_values[i] < max_background) threshold_image.real_values[i] = average_background_float;
//			else threshold_image.real_values[i] = average_shadow_float;
////			threshold_image.real_values[i] = 1.0f;
//		}
//		else threshold_image.real_values[i] = 0.0f;
		if (threshold_image.real_values[i] < sizing_threshold) threshold_image.real_values[i] = average_background_float;
		else threshold_image.real_values[i] = average_shadow_float;
//		else
//		{
//			wxPrintf("\nERROR: Could not remove outliers. Aborting...\n");
//			exit(1);
//		}

	}
//	threshold_image_small.CopyFrom(&threshold_image);
//	threshold_image.QuickAndDirtyWriteSlice("threshold_image.mrc", 1);
//	threshold_image_small.QuickAndDirtyWriteSlice("threshold_image1.mrc", 1);
//	exit(0);
	outliers = 0;
	pointer = 0;
	for (j = 0; j < threshold_image.logical_y_dimension; j++)
	{
		for (i = 0; i < threshold_image.logical_x_dimension; i++)
		{
			if (fabsf(threshold_image.real_values[pointer] - threshold_image_small.real_values[pointer]) > 0.1f)
			{
				outliers++;
//				threshold_image_small.real_values[pointer] = 1.0f;
//				threshold_image_small.real_values[pointer] = threshold_image.real_values[pointer];
				input_image.real_values[pointer] = threshold_image.real_values[pointer];
			}
//			else threshold_image_small.real_values[pointer] = 0.0f;
			pointer++;
		}
		pointer += threshold_image.padding_jump_value;
	}
	threshold_image_small.CopyFrom(&threshold_image);
//	threshold_image_small.QuickAndDirtyWriteSlice("threshold_image2.mrc", 1);
//	exit(0);

/*
	outliers = 0;
	pointer = 0;
	outlier_image.SetToConstant(0.0f);
	for (j = 0; j < threshold_image.logical_y_dimension; j++)
	{
		for (i = 0; i < threshold_image.logical_x_dimension; i++)
		{
			counter1 = 0;
			counter2 = 0;
			for (m = -1; m <= 1; m++)
			{
				for (l = -1; l <= 1; l++)
				{
					ix = i + l;
					iy = j + m;
					if (ix >= 0 && ix < threshold_image.logical_x_dimension && iy >= 0 && iy < threshold_image.logical_y_dimension && ! (l == 0 && m == 0))
					{
						counter1++;
						temp_float = threshold_image.ReturnRealPixelFromPhysicalCoord(ix, iy, 0);
						if (threshold_image.real_values[pointer] <= threshold && temp_float > threshold) counter2++;
						if (threshold_image.real_values[pointer] > threshold && temp_float <= threshold) counter2++;
					}
				}
			}
			if (float(counter2) / float(counter1) > 0.6)
			{
				outliers++;
				if (threshold_image.real_values[pointer] <= threshold)
				{
					outlier_image.real_values[pointer] = average_background;
					threshold_image_small.real_values[pointer] = average_background;
				}
				else
				{
					outlier_image.real_values[pointer] = average_shadow;
					threshold_image_small.real_values[pointer] = average_shadow;
				}
			}
			pointer++;
		}
		pointer += threshold_image.padding_jump_value;
	}
*/
	wxPrintf("Number of pixel outliers = %9i\n", outliers);
	if (outliers > 0.02f * threshold_image.number_of_real_space_pixels) wxPrintf("Try a higher exposure to reduce outliers\n");

	wxPrintf("\nDetecting edge...  ");
	temp_image.CopyFrom(&threshold_image_small);
	temp_image.ForwardFFT();
	temp_image.CalculateDerivative();
	temp_image.BackwardFFT();
	mask_image.CopyFrom(&temp_image);
	temp_image.SetMinimumValue(0.0f);
	for (i = 0; i < mask_image.real_memory_allocated; i++) if (mask_image.real_values[i] < 0.0f) mask_image.real_values[i] = - mask_image.real_values[i];
	mask_image.ForwardFFT();
	mask_image.GaussianLowPassFilter(low_pass_filter_constant);
	mask_image.BackwardFFT();

	pointer = 0;
	temp_float = 0.5f * threshold;
	threshold3 = 0.0f;
	for (j = 0; j < temp_image.logical_y_dimension; j++)
	{
		for (i = 0; i < temp_image.logical_x_dimension; i++)
		{
			if (i >= margin && i <= temp_image.logical_x_dimension - margin && j >= margin && j <= temp_image.logical_y_dimension - margin)
			{
				if (temp_image.real_values[pointer] > temp_float)
				{
					threshold3 += mask_image.real_values[pointer];
					edge_pixels++;
				}
			}
			pointer++;
		}
		pointer += temp_image.padding_jump_value;
	}
	wxPrintf("Number of usable pixels along edge = %9i\n", edge_pixels);
	if (edge_pixels == 0)
	{
		wxPrintf("\nERROR: No edge detected. Aborting...\n");
		exit(1);
	}
	threshold3 /= edge_pixels;

	counter1 = 0;
	temp_float = 0.002f * threshold3;
	pointer = 0;
	for (j = 0; j < mask_image.logical_y_dimension; j++)
	{
		for (i = 0; i < mask_image.logical_x_dimension; i++)
		{
			if (mask_image.real_values[pointer] > temp_float) {mask_image.real_values[pointer] /= threshold3; counter1++;}
			else mask_image.real_values[pointer] = 0.0f;
			pointer++;
		}
		pointer += mask_image.padding_jump_value;
	}

	mask_image.CosineRectangularMask(mask_image.physical_address_of_box_center_x - 0.5f * margin, mask_image.physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, 0.0f);
	wxPrintf("Pixels inside edge mask = %li\n", counter1);
//	mask_image.QuickAndDirtyWriteSlice("filtered_edge.mrc", 1);

	// Test if edge moved between images 1 and 2
	if (use_both_images)
	{
		pointer = 0;
		for (j = 0; j < input_image1.logical_y_dimension; j++)
		{
			for (i = 0; i < input_image1.logical_x_dimension; i++)
			{
				if (mask_image.real_values[pointer] > 0.0f)
				{
					if (input_image1.real_values[pointer] > threshold1 && input_image2.real_values[pointer] < threshold2) discrepancy_pixels++;
					else if (input_image1.real_values[pointer] < threshold1 && input_image2.real_values[pointer] > threshold2) discrepancy_pixels++;
				}
				pointer++;
			}
			pointer += input_image1.padding_jump_value;
		}
//		wxPrintf("disc, edge = %i %i\n", discrepancy_pixels, edge_pixels);
		if (discrepancy_pixels > edge_pixels)
		{
			wxPrintf("\nEdge in Image 2 offset from Image 1, or edge too noisy. Aborting...\n");
			exit(1);
		}
	}

	// Finding min, max x,y coordinates of shadow to cut image
	y_min = input_image.logical_y_dimension;
	y_max = 0;
	for (j = 0; j < input_image.logical_y_dimension; j++)
	{
		counter1 = 0;
		for (i = 0; i < input_image.logical_x_dimension; i++)
		{
			if (mask_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) > 0.5f) counter1++;
		}
		if (counter1 > 10)
		{
			if (j > y_max) y_max = j;
			if (j < y_min) y_min = j;
		}
	}
	x_min = input_image.logical_x_dimension;
	x_max = 0;
	for (i = 0; i < input_image.logical_x_dimension; i++)
	{
		counter1 = 0;
		for (j = 0; j < input_image.logical_y_dimension; j++)
		{
			if (mask_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) > 0.5f) counter1++;
		}
		if (counter1 > 10)
		{
			if (i > x_max) x_max = i;
			if (i < x_min) x_min = i;
		}
	}
	box_size_x = ReturnClosestFactorizedUpper(x_max - x_min + 2 * margin, 5, true);
	box_size_y = ReturnClosestFactorizedUpper(y_max - y_min + 2 * margin, 5, true);
	if (box_size_x > input_image.logical_x_dimension)
	{
		box_size_x = input_image.logical_x_dimension;
		if (IsOdd(box_size_x)) box_size_x--;
	}
	if (box_size_y > input_image.logical_y_dimension)
	{
		box_size_y = input_image.logical_y_dimension;
		if (IsOdd(box_size_y)) box_size_y--;
	}

	x_mid = (x_min + x_max - input_image.logical_x_dimension) / 2;
	y_mid = (y_min + y_max - input_image.logical_y_dimension) / 2;
	if (x_mid > (input_image.logical_x_dimension - box_size_x) / 2) x_mid = (input_image.logical_x_dimension - box_size_x) / 2;
	if (x_mid < (box_size_x - input_image.logical_x_dimension) / 2) x_mid = (box_size_x - input_image.logical_x_dimension) / 2;
	if (y_mid > (input_image.logical_y_dimension - box_size_y) / 2) y_mid = (input_image.logical_y_dimension - box_size_y) / 2;
	if (y_mid < (box_size_y - input_image.logical_y_dimension) / 2) y_mid = (box_size_y - input_image.logical_y_dimension) / 2;

	// Recalculate averages excluding edge
	average_shadow = 0.0;
	pixels_shadow = 0;
	average_background = 0.0;
	pixels_background = 0;
	pointer = 0;
	for (j = 0; j < input_image.logical_y_dimension; j++)
	{
		for (i = 0; i < input_image.logical_x_dimension; i++)
		{
			if (i >= margin && i <= input_image.logical_x_dimension - margin && j >= margin && j <= input_image.logical_y_dimension - margin)
			{
				if (mask_image.real_values[pointer] == 0.0f)
				{
					if (threshold_image_small.real_values[pointer] <= threshold)
					{
						average_shadow += input_image.real_values[pointer];
						pixels_shadow++;
					}
					else
					{
						average_background += input_image.real_values[pointer];
						pixels_background++;
					}
				}
			}
			pointer++;
		}
		pointer += input_image.padding_jump_value;
	}
	average_shadow /= pixels_shadow;
	average_background /= pixels_background;

	// Calculate background image
	temp_image.CopyFrom(&threshold_image_small);
//	background_image.CopyFrom(&input_image);
	temp_image.SetToConstant(0.0f);
	for (i = 0; i < input_image.real_memory_allocated; i++)
	{
		if (mask_image.real_values[i] == 0.0f && threshold_image_small.real_values[i] > threshold) temp_float = 1.0f;
		else {temp_float = 0.0f; temp_image.real_values[i] = 1.0f;}
		background_mask_image.real_values[i] = temp_float;
	}

	// Dilate mask
	temp_image.ForwardFFT();
	temp_image.GaussianLowPassFilter(0.01f);
	temp_image.BackwardFFT();
	for (i = 0; i < input_image.real_memory_allocated; i++)
	{
		if (temp_image.real_values[i] > 0.1f) temp_image.real_values[i] = 1.0f;
		if (temp_image.real_values[i] < 0.0f) temp_image.real_values[i] = 0.0f;
	}
	// Smooth mask
	temp_image.ForwardFFT();
	temp_image.GaussianLowPassFilter(0.01f);
	temp_image.BackwardFFT();
	for (i = 0; i < input_image.real_memory_allocated; i++)
	{
		if (temp_image.real_values[i] > 1.0f) temp_image.real_values[i] = 1.0f;
		if (temp_image.real_values[i] < 0.0f) temp_image.real_values[i] = 0.0f;
	}
	// Apply mask to background image to smooth shadow edge
	for (i = 0; i < input_image.real_memory_allocated; i++)
	{
		if (mask_image.real_values[i] == 0.0f && threshold_image_small.real_values[i] > threshold)
		{
			if (temp_image.real_values[i] > 0.01f) background_mask_image.real_values[i] = (1.0f - temp_image.real_values[i]);
		}
	}
	background_mask_image.CosineRectangularMask(background_mask_image.physical_address_of_box_center_x - 0.5f * margin, background_mask_image.physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, 0.0f);

	// Crop images for faster MTF fitting
	temp_image.CopyFrom(&threshold_image);
	threshold_image.Resize(box_size_x, box_size_y, 1);
	temp_image.ClipInto(&threshold_image, 0.0f, false, 0.0f, x_mid, y_mid);
	temp_image.CopyFrom(&input_image);
	input_image.Resize(box_size_x, box_size_y, 1);
	temp_image.ClipInto(&input_image, 0.0f, false, 0.0f, x_mid, y_mid);
	temp_image.CopyFrom(&mask_image);
	mask_image.Resize(box_size_x, box_size_y, 1);
	temp_image.ClipInto(&mask_image, 0.0f, false, 0.0f, x_mid, y_mid);
//	temp_image.CopyFrom(&outlier_image);
//	outlier_image.Resize(box_size_x, box_size_y, 1);
//	temp_image.ClipInto(&outlier_image, 0.0f, false, 0.0f, x_mid, y_mid);
	difference_image.Allocate(box_size_x, box_size_y, true);

//	if (! use_both_images)
//	{
//		wxPrintf("\nRemoving intensity ramp from background image...\n\n");
//		comparison_object.experimental_image = &background_image;
//		comparison_object.model_image = &background_mask_image;
//		comparison_object.parameters_to_fit = 7;
//		comparison_object.best_score = FLT_MAX;
//		comparison_object.busy_state = 0;
//
//		cg_starting_point[0] = average_background;
//		cg_accuracy[0] = 0.1f;
////		cg_accuracy[1] = 0.001f;
////		cg_accuracy[2] = 0.001f;
//		for (i = 1; i < 7; i++) cg_starting_point[i] = 0.0f;
//		for (i = 1; i < 7; i++) cg_accuracy[i] = 0.01f;
//		conjugate_gradient_minimizer.Init(&RampFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);
//		conjugate_gradient_minimizer.Run(5);
//		fitted_parameters = conjugate_gradient_minimizer.GetPointerToBestValues();
//		for (i = 0; i < comparison_object.parameters_to_fit; i++) cg_starting_point[i] = fitted_parameters[i];
//		for (i = 1; i < comparison_object.parameters_to_fit; i++) cg_accuracy[i] = 1000.0f * fitted_parameters[i];
//		conjugate_gradient_minimizer.Init(&RampFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);
//		conjugate_gradient_minimizer.Run(5);
//		fitted_parameters = conjugate_gradient_minimizer.GetPointerToBestValues();
//		pointer = 0;
//		for (j = 0; j < background_image.logical_y_dimension; j++)
//		{
//			iy = (j - background_image.physical_address_of_box_center_y);
//			for (i = 0; i < background_image.logical_x_dimension; i++)
//			{
//				ix = (i - background_image.physical_address_of_box_center_x);
//				ramp = fitted_parameters[0] - average_background + fitted_parameters[1] * ix + fitted_parameters[2] * iy;
//				if (comparison_object.parameters_to_fit > 3) ramp += fitted_parameters[3] * ix * ix + fitted_parameters[4] * iy * iy;
//				if (comparison_object.parameters_to_fit > 5) ramp += fitted_parameters[5] * ix * ix * ix + fitted_parameters[6] * iy * iy * iy;
//				background_image.real_values[pointer] -= ramp;
//				pointer++;
//			}
//			pointer += background_image.padding_jump_value;
//		}
//		wxPrintf("\n");
//	}

	// Mask background image
	mask_volume = 0.0f;
	pointer = 0;
	average_background_float = average_background;
	if (use_both_images) average_background_float = 0.0f;
	for (j = 0; j < background_mask_image.logical_y_dimension; j++)
	{
		for (i = 0; i < background_mask_image.logical_x_dimension; i++)
		{
			mask_volume += powf(background_mask_image.real_values[pointer], 2);
			background_image.real_values[pointer] = background_image.real_values[pointer] * background_mask_image.real_values[pointer] + average_background_float * (1.0f - background_mask_image.real_values[pointer]);
			pointer++;
		}
		pointer += background_mask_image.padding_jump_value;
	}
	mask_volume /= background_image.number_of_real_space_pixels;
	if (debug) background_image.QuickAndDirtyWriteSlice("background_image.mrc", 1);

//	bin_size = myroundint(float(std::min(background_image.logical_x_dimension, background_image.logical_y_dimension)) / 5.0f);
	nps0 = 0.0f;
	for (i = bin_size - bin_size_range; i <= bin_size + bin_size_range; i++)
	{
		j = std::max(1, i);
		temp_image.CopyFrom(&background_image);
		temp_image.RealSpaceBinning(j, j, 1, false, true);
		variance = temp_image.ReturnVarianceOfRealValues();
		temp_image.CopyFrom(&background_mask_image);
		temp_image.RealSpaceBinning(j, j, 1, false, true);
		temp_float = 0.0f;
		pointer = 0;
		for (m = 0; m < temp_image.logical_y_dimension; m++)
		{
			for (l = 0; l < temp_image.logical_x_dimension; l++)
			{
				temp_float += powf(temp_image.real_values[pointer], 2);
				pointer++;
			}
			pointer += temp_image.padding_jump_value;
		}
		temp_float /= temp_image.number_of_real_space_pixels;
//		wxPrintf("bin, std = %i %g\n", j, sqrtf(variance / temp_float / two_image_factor) * j);
		nps0 += variance / temp_float * j * j;
	}
	nps0 /= (2.0f * bin_size_range + 1.0f);

	// Calculate NPS
	background_image.ForwardFFT();
	temp_image.CopyFrom(&background_image);
	temp_image.ConjugateMultiplyPixelWise(background_image);
	nps.SetupXAxis(0.0, 0.5f / 1000.0f * 1415.0f, 1416);
	number_of_terms.SetupXAxis(0.0, 0.5f / 1000.0f * 1415.0f, 1416);
	temp_image.Compute1DRotationalAverage(nps, number_of_terms);
	nps.MultiplyByConstant(float(background_image.number_of_real_space_pixels) / mask_volume);

	nps1 = 0.0f;
	for (i = 700; i < 1000; i++) nps1 += nps.data_y[i];
	nps1 /= 300.0f;
//	wxPrintf("NPS1 = %g\n", sqrtf(nps0 / two_image_factor));

//	temp_image.CopyFrom(&background_image);
//	noise_whitening_spectrum = nps;
//	noise_whitening_spectrum.SquareRoot();
//	noise_whitening_spectrum.Reciprocal();
//	fit_window_size = myroundint(std::min(float(background_image.logical_x_dimension) / 50.0f, float(background_image.logical_y_dimension) / 50.0f));
//	if (IsEven(fit_window_size)) fit_window_size++;

//	for (i = 200; i < nps.number_of_points - 10; i++) fit_window1.AddPoint(nps.data_x[i], nps.data_y[i]);
//	fit_window_size = 101;
//	fit_window1.FitSavitzkyGolayToData(fit_window_size, 1);
//	slope = FLT_MAX;
//	for (i = nps.number_of_points - 1; i >= 0; i--)
//	{
//		if (nps.data_x[i] > 0.25f)
//		{
//			temp_float = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//		}
//		else
//		{
//			if (slope > 0.0f)
//			{
//				slope = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i + 1]) - fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//				offset = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//			}
//			if (slope > 0.0f) temp_float = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//			else temp_float = offset;
////			if (offset > fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i])) offset = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
////			if (offset < fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i])) temp_float = offset;
////			else temp_float = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//		}
//		fit_window2.AddPoint(nps.data_x[i], temp_float);
//	}
//	for (i = 0; i < nps.number_of_points; i++) nps_fit.AddPoint(nps.data_x[i], fit_window2.data_y[nps.number_of_points - i - 1]);

	for (i = 10; i < nps.number_of_points - 10; i++) fit_window1.AddPoint(nps.data_x[i], nps.data_y[i]);
	fit_window_size = 21;
	fit_window1.FitSavitzkyGolayToData(fit_window_size, 2);
	for (i = 10; i < nps.number_of_points - 10; i++) fit_window2.AddPoint(nps.data_x[i], nps.data_y[i]);
	fit_window_size = 201;
	fit_window2.FitSavitzkyGolayToData(fit_window_size, 1);
	for (i = 10; i < nps.number_of_points - 10; i++)
	{
		temp_float = fabsf(nps.data_y[i] - fit_window2.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]));
		if ((temp_float / fit_window2.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]) > 0.1f) && (nps.data_x[i] > 0.1f))
		{
			fit_window3.AddPoint(nps.data_x[i], fit_window2.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]));
		}
		else
		{
			fit_window3.AddPoint(nps.data_x[i], nps.data_y[i]);
		}
	}
	fit_window3.FitSavitzkyGolayToData(fit_window_size, 1);
//	for (i = 10; i < nps.number_of_points - 10; i++) fit_window2.AddPoint(nps.data_x[i], nps.data_y[i]);
//	offset = fit_window1.ReturnSavitzkyGolayInterpolationFromX(0.05f);
//	for (i = 0; i < nps.number_of_points; i++)
//	{
//		if (nps.data_x[i] < 0.05f)
//		{
//			temp_float = offset;
//		}
//		else temp_float = nps.data_y[i];
//		fit_window2.AddPoint(nps.data_x[i], temp_float);
//	}
//	fit_window2.FitPolynomialToData(4);
	for (i = 0; i < nps.number_of_points; i++)
	{
//		if (nps.data_x[i] < 0.01f)
//		{
//			temp_float = nps.data_y[i];
//		}
//		else if (nps.data_x[i] < 0.02f)
//		{
//			temp_float = (1.0f - (nps.data_x[i] - 0.01f) / 0.01f) * nps.data_y[i] \
//					+ (nps.data_x[i] - 0.01f) / 0.01f * fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
////			temp_float = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//		}
//		else if (nps.data_x[i] < 0.05f)
		if (nps.data_x[i] < 0.05f)
		{
			temp_float = fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
		}
		else if (nps.data_x[i] < 0.1f)
		{
			temp_float = (1.0f - (nps.data_x[i] - 0.05f) / 0.05f) * fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]) \
					+ (nps.data_x[i] - 0.05f) / 0.05f * fit_window3.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//			temp_float = (1.0f - (nps.data_x[i] - 0.05f) / 0.15f) * fit_window1.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]) \
//					+ (nps.data_x[i] - 0.05f) / 0.15f * fit_window2.polynomial_fit[i];
		}
		else temp_float = fit_window3.ReturnSavitzkyGolayInterpolationFromX(nps.data_x[i]);
//		else temp_float = fit_window2.polynomial_fit[i];
		nps_fit.AddPoint(nps.data_x[i], temp_float);
	}

//	comparison_object.parameters_to_fit = 4;
//	comparison_object.curve_to_be_fitted = &nps;
//	average = 0.0;
//	for (i = 100; i < 990; i++) average += nps.data_y[i];
//	average /= (990 - 100 + 1);

//	cg_starting_point[0] = 1.0f;
//	cg_starting_point[1] = 0.5f;
//	cg_starting_point[2] = 10.0f;
//	cg_starting_point[3] = 1.0f;
//	cg_accuracy[0] = 0.01f;
//	cg_accuracy[1] = 0.01f;
//	cg_accuracy[2] = 0.01f;
//	cg_accuracy[3] = 0.01f;
//	conjugate_gradient_minimizer.Init(&SincFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);

//	cg_starting_point[0] = 1.0f;
//	cg_starting_point[1] = 0.1f;
//	cg_starting_point[2] = 0.1f;
//	cg_starting_point[3] = 0.4f;
//	cg_accuracy[0] = 0.01f;
//	cg_accuracy[1] = 0.02f;
//	cg_accuracy[2] = 0.01f;
//	cg_accuracy[3] = 0.01f;
//	conjugate_gradient_minimizer.Init(&LogisticFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);
//	conjugate_gradient_minimizer.Run(5);
//	fitted_parameters = conjugate_gradient_minimizer.GetPointerToBestValues();
//	for (i = 0; i < nps.number_of_points; i++)
//	{
//		temp_float = average * (fitted_parameters[0] / (1.0f + expf((fitted_parameters[1] - nps.data_x[i]) / fitted_parameters[2])) + fitted_parameters[3]);
//		if (fabsf(nps.data_y[i] - temp_float) / temp_float < 0.1f) temp_float = nps.data_y[i];
//		fit_window1.AddPoint(nps.data_x[i], temp_float);
//	}
//	comparison_object.curve_to_be_fitted = &fit_window1;
//	cg_starting_point[0] = fitted_parameters[0];
//	cg_starting_point[1] = fitted_parameters[1];
//	cg_starting_point[2] = fitted_parameters[2];
//	cg_starting_point[3] = fitted_parameters[3];
//	conjugate_gradient_minimizer.Init(&LogisticFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);
//	conjugate_gradient_minimizer.Run(5);
//	fitted_parameters = conjugate_gradient_minimizer.GetPointerToBestValues();
////	for (i = 0; i < nps.number_of_points; i++) nps_fit.AddPoint(nps.data_x[i], average * (fitted_parameters[0] - fitted_parameters[1] * sinc(fabsf(fitted_parameters[2]) * nps.data_x[i])) * exp(-powf(fitted_parameters[3] * nps.data_x[i], 2)));
//	for (i = 0; i < nps.number_of_points; i++) nps_fit.AddPoint(nps.data_x[i], average * (fitted_parameters[0] / (1.0f + expf((fitted_parameters[1] - nps.data_x[i]) / fitted_parameters[2])) + fitted_parameters[3]));

//	nps.WriteToFile("nps.txt");
//	nps_fit.WriteToFile("nps_fit.txt");
//	fit_window1.WriteToFile("fit_window1.txt");
//	background_image.QuickAndDirtyWriteSlice("background.mrc", 1);
//	exit(0);
//	temp_float = FLT_MAX;
//	for (i = 25; i < 999 - 25; i++) if (noise_whitening_spectrum.savitzky_golay_fit[i] < temp_float) temp_float = noise_whitening_spectrum.savitzky_golay_fit[i];
//	noise_whitening_spectrum.MultiplyByConstant(1.0f / temp_float);
//	noise_whitening_spectrum.data_y[0] = 1.0f;
//	temp_image.ApplyCurveFilter(&noise_whitening_spectrum);
//	temp_image.BackwardFFT();
	wxPrintf("\nGain (counts/e) from exposure and background average = %10.4f\n", gain_conversion_factor);
	if (is_a_counting_detector)
	{
		dqe0 = pow(average_background, 2) / exposure / nps1;
		wxPrintf("Sqrt of noise power at Nyquist frequency (NPS1)      = %10.4f\n", sqrtf(nps1 / two_image_factor));
		wxPrintf("DQE0 based on gain and NPS1                          = %10.4f\n", dqe0);
	}
	else
	{
		dqe0 = pow(average_background, 2) / exposure / nps0;
		wxPrintf("Sqrt of noise power at 0 frequency (NPS0)            = %10.4f\n", sqrtf(nps0 / two_image_factor));
		wxPrintf("DQE0 based on gain and NPS0                          = %10.4f\n", dqe0);
	}

//	nps_fit.GetYMinMax(temp_float, nps_max);
//	nps.MultiplyByConstant(1.0f / nps_max);
//	nps_fit.MultiplyByConstant(1.0f / nps_max);
	nps.MultiplyByConstant(1.0f / nps_fit.data_y[0]);
	nps_fit.MultiplyByConstant(1.0f / nps_fit.data_y[0]);

	// Test if this is a counting detector
	nps02 = 0.0f;
	nps09 = 0.0f;
	for (i = 175; i < 225; i++) nps02 += nps.data_y[i];
	nps02 /= 50.0f;
	for (i = 875; i < 925; i++) nps09 += nps.data_y[i];
	nps09 /= 50.0f;
//	if (nps02 / nps09 < 2.0f)
//	{
		// This is currently not used
//		wxPrintf("\nUsing NPS scaling for counting detector\n");
//		wxPrintf("\nThis is a counting detector: MTFs will be fitted independently on both sides of the edge.\n");
//		is_a_counting_detector = true;
////		nps.MultiplyByConstant(1.0f / nps09);
//	}
//	else nps.MultiplyByConstant(float(background_image.number_of_real_space_pixels) / mask_volume / nps0);

	// Spectrum always has huge peak at origin. Reset to value at [1]
	nps.FlattenBeforeIndex(5);

	if (padding > 1) wxPrintf("\nCreating %ix super-sampled model image...\n", padding);
//	temp_image.CopyFrom(&input_image);
//	temp_image.ForwardFFT();
//	temp_image.Resize(padding * input_image.logical_x_dimension, padding * input_image.logical_y_dimension, 1);
//	temp_image.BackwardFFT();
//	threshold_image.Resize(padding * input_image.logical_x_dimension, padding * input_image.logical_y_dimension, 1);
//	threshold_image.SetToConstant(average_shadow);
	threshold_image.CopyFrom(&input_image);
	threshold_image.ForwardFFT();
	threshold_image.Resize(padding * input_image.logical_x_dimension, padding * input_image.logical_y_dimension, 1);
	threshold_image.BackwardFFT();
	threshold_image.Binarise(threshold);
	runlenth3d.EncodeFrom(threshold_image);
	runlenth3d.ConnectedSizeDecodeTo(threshold_image);
	threshold_image.MultiplyByConstant(-1.0f);
	threshold_image.Binarise(-sizing_threshold);
	runlenth3d.EncodeFrom(threshold_image);
	runlenth3d.ConnectedSizeDecodeTo(threshold_image);
	pointer = 0;
	for (j = 0; j < threshold_image.logical_y_dimension; j++)
	{
		iy = int(float(j) / padding);
		for (i = 0; i < threshold_image.logical_x_dimension; i++)
		{
			ix = int(float(i) / padding);
			if (threshold_image.real_values[pointer] < sizing_threshold) threshold_image.real_values[pointer] = average_background;
			else threshold_image.real_values[pointer] = average_shadow;
//			temp_float = outlier_image.ReturnRealPixelFromPhysicalCoord(ix, iy, 0);
//			if (temp_float != 0.0f) threshold_image.real_values[pointer] = temp_float;
			pointer++;
		}
		pointer += threshold_image.padding_jump_value;
	}
//	threshold_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
//	exit(0);

	// Update threshold_image with new averages
//	for (i = 0; i < threshold_image.real_memory_allocated; i++)
//	{
//		if (threshold_image.real_values[i] > threshold) threshold_image.real_values[i] = average_background;
//		else threshold_image.real_values[i] = average_shadow;
//	}

	// Dilate shadow
//	threshold_image.ForwardFFT();
//	threshold_image.GaussianLowPassFilter(0.01f);
//	threshold_image.BackwardFFT();
//	for (i = 0; i < threshold_image.real_memory_allocated; i++)
//	{
//		if (threshold_image.real_values[i] > 1.0f * threshold) threshold_image.real_values[i] = average_background;
//		else threshold_image.real_values[i] = average_shadow;
//	}

	temp_image.CopyFrom(&input_image);
	temp_image.CosineRectangularMask(temp_image.physical_address_of_box_center_x - 0.5f * margin, temp_image.physical_address_of_box_center_y - 0.5f * margin, 0.0f, margin, false, true, average_background);
	threshold_image.CosineRectangularMask(threshold_image.physical_address_of_box_center_x - 0.5f * padding * margin, threshold_image.physical_address_of_box_center_y - 0.5f * padding * margin, 0.0f, padding * margin, false, true, average_background);
//	threshold_image_real.CopyFrom(&threshold_image);
	threshold_image_fft.CopyFrom(&threshold_image);
	threshold_image_fft.ForwardFFT();
	if (padding != 1) threshold_image.RealSpaceBinning(padding, padding, 1, true);

//	wxPrintf("\nFitting MTF\n");
	wxPrintf("\nMin,max x,y coordinates for shadow   = %8i %8i %8i %8i\n", x_min + 1, x_max + 1, y_min + 1, y_max + 1);
	wxPrintf("Using box size %8i x %8i\n\n", box_size_x, box_size_y);
	fflush(stdout);

	comparison_object.model_image = &threshold_image;
	comparison_object.experimental_image = &temp_image;
//	comparison_object.model_image_real = &threshold_image_real;
	comparison_object.model_image_fft = &threshold_image_fft;
	comparison_object.difference_image = &difference_image;
	comparison_object.edge_mask = &mask_image;
	comparison_object.average_background = average_background;
	comparison_object.average_shadow = average_shadow;
//	comparison_object.is_a_counting_detector = is_a_counting_detector;
	comparison_object.nps_fit = &nps_fit;

	for (cycle = 0; cycle < 2; cycle++)
	{
		if (! two_sided_mtf)
		{
			wxPrintf("Fitting MTF...\n\n");
			comparison_object.reset_shadow = false;
			comparison_object.reset_background = false;
//			comparison_object.model_image = &threshold_image;
		}
		else if (cycle == 0)
		{
			wxPrintf("Fitting MTF inside shadow...\n\n");
//			comparison_object.reset_shadow = true;
//			comparison_object.reset_background = false;
			comparison_object.reset_shadow = false;
			comparison_object.reset_background = true;
//			comparison_object.model_image = &threshold_image;
		}
		else
		{
			wxPrintf("\n\nFitting MTF outside shadow...\n\n");
//			comparison_object.reset_shadow = false;
//			comparison_object.reset_background = true;
			comparison_object.reset_shadow = true;
			comparison_object.reset_background = false;
		}

		for (i = 0; i < 10; i += 2) cg_accuracy[i] = 0.1f;
		for (i = 1; i < 10; i += 2) cg_accuracy[i] = 0.01f;
		cg_accuracy[10] = 0.1f;

//		comparison_object.experimental_image = &temp_image;
//		comparison_object.model_image_real = &threshold_image_real;
//		comparison_object.model_image_fft = &threshold_image_fft;
//		comparison_object.difference_image = &difference_image;
//		comparison_object.edge_mask = &mask_image;
//		comparison_object.parameters_to_fit = 11;
//		comparison_object.average_background = average_background;
//		comparison_object.average_shadow = average_shadow;
		comparison_object.best_score = FLT_MAX;
		comparison_object.busy_state = 0;
//		comparison_object.is_a_counting_detector = is_a_counting_detector;
//		comparison_object.nps_fit = &nps_fit;

		// F416 200 kV
//		fitted_parameters = new float [10];
//		fitted_parameters[0] = -2.99066;
//		fitted_parameters[1] = 14.9696;
//		fitted_parameters[2] = 103.689;
//		fitted_parameters[3] = 6.31202;
//		fitted_parameters[4] = 34738.4;
//		fitted_parameters[5] = -0.236264;
//		fitted_parameters[6] = 41176.2;
//		fitted_parameters[7] = -0.299999;
//		fitted_parameters[8] = 41211.1;
//		fitted_parameters[9] = -0.309835;

		best_mtf_score = FLT_MAX;
		ZeroFloatArray(best_parameters, 11);
		for (j = 0; j < 5; j++)
//		for (j = 4; j < 5; j++)
//		for (j = 0; j < 0; j++)
		{
			if (j == 0) comparison_object.parameters_to_fit = 1;
			else comparison_object.parameters_to_fit = 2 * j + 2;
//			cg_starting_point[0] = 10.0f + 200.0f * j;
			cg_starting_point[0] = 1.0f + 10.0f * j;
//			sum_of_coefficients = 0.0f;
//			for (i = 1; i < 2 * j; i += 2) sum_of_coefficients += fabsf(fitted_parameters[i]);
//			if (j > 0) cg_starting_point[1] = 0.1f * sum_of_coefficients;
//			else cg_starting_point[1] = 1.0f;
			if (j > 0)
			{
				cg_starting_point[1] = 0.1f;
//				cg_starting_point[1] = 0.0f;
				sum_of_coefficients = 0.0f;
				for (i = 1; i < 2 * j + 2; i += 2) sum_of_coefficients += fabsf(cg_starting_point[i]);
				for (i = 1; i < 2 * j + 2; i += 2) cg_starting_point[i] /= sum_of_coefficients;
			}
			else cg_starting_point[1] = 1.0f;
//			if (j == 4)
//			{
//				comparison_object.parameters_to_fit++;
//				cg_starting_point[10] = 1.0f;
//			}
//			conjugate_gradient_minimizer.Init(&MTFFitSinc, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);

//			cg_starting_point[0] = -2.99066;
//			cg_starting_point[1] = 14.9696;
//			cg_starting_point[2] = 103.689;
//			cg_starting_point[3] = 6.31202;
//			cg_starting_point[4] = 34738.4;
//			cg_starting_point[5] = -0.236264;
//			cg_starting_point[6] = 41176.2;
//			cg_starting_point[7] = -0.299999;
//			cg_starting_point[8] = 41211.1;
//			cg_starting_point[9] = -0.309835;

			conjugate_gradient_minimizer.Init(&MTFFit, &comparison_object, comparison_object.parameters_to_fit, cg_starting_point, cg_accuracy);
			score = conjugate_gradient_minimizer.Run(50);
			fitted_parameters = conjugate_gradient_minimizer.GetPointerToBestValues();

			if (j == 0)
			{
				if (score < best_mtf_score)
				{
					best_mtf_score = score;
					best_parameters[0] = fitted_parameters[0];
					best_parameters[1] = 1.0f;
				}
//				cg_starting_point[2 * j + 2] = best_parameters[2 * j];
//				cg_starting_point[2 * j + 3] = 1.0f;
			}
			else if (j < 4)
			{
				if (score < best_mtf_score)
				{
					best_mtf_score = score;
					for (i = 0; i < comparison_object.parameters_to_fit; i++) best_parameters[i] = fitted_parameters[i];
				}
//				cg_starting_point[2 * j + 2] = fitted_parameters[2 * j];
//				cg_starting_point[2 * j + 3] = fitted_parameters[2 * j + 1];
			}
			for (i = 0; i < comparison_object.parameters_to_fit; i++) cg_starting_point[i + 2] = best_parameters[i];
//			for (i = 0; i < comparison_object.parameters_to_fit; i++) wxPrintf("Cycle %i: fitted_parameters[%1i] = %g;\n", j, i, fitted_parameters[i]);

			if (debug)
			{
				wxPrintf("\nIteration %i\n", j);
				for (i = 0; i < comparison_object.parameters_to_fit; i++) wxPrintf("fitted_parameters[%1i] = %g;\n", i, fabsf(fitted_parameters[i]));
				wxPrintf("\n");
			}
	    }

		if (cycle == 0 && two_sided_mtf)
		{
			comparison_object.busy_state = -1;
			score = MTFFit(&comparison_object, best_parameters);
			threshold_image.CopyFrom(comparison_object.difference_image);
//			threshold_image_fft.CopyFrom(comparison_object.difference_image);
//			threshold_image_fft.ForwardFFT();
//			for (i = 0; i < comparison_object.parameters_to_fit; i++) saved_parameters[i] = fitted_parameters[i];
			for (i = 0; i < 11; i++) saved_parameters[i] = best_parameters[i];
		}
		else
		{
		// calculate difference image with best parameters
		//	MTFFitSinc(&comparison_object, fitted_parameters);
			score = MTFFit(&comparison_object, best_parameters);
			if (two_sided_mtf) wxPrintf("\n\nUsing the average MTF for DQE calculation.\n");
		}
		if (! two_sided_mtf) break;
	}

//	fitted_parameters = new float [11];
	// US4000
//	fitted_parameters[0] = -8.39287;
//	fitted_parameters[1] = -0.550055;
//	fitted_parameters[2] = 10300.3;
//	fitted_parameters[3] = 0.0759119;
//	fitted_parameters[4] = 36.4889;
//	fitted_parameters[5] = 3.00088e-05;
//	fitted_parameters[6] = 40.9219;
//	fitted_parameters[7] = -0.143997;
//	fitted_parameters[8] = 154.697;
//	fitted_parameters[9] = -1.00049;
//	fitted_parameters[10] = 10.00049;

	mtf.SetupXAxis(0.0, 0.5f / 1000.0f * 1415.0f, 1416);
	mtf.number_of_points = 1001;
	nps.number_of_points = mtf.number_of_points;
	nps_fit.number_of_points = mtf.number_of_points;
	sum_of_coefficients = 0.0f;
	for (j = 1; j < 10; j += 2) sum_of_coefficients += fabsf(best_parameters[j]);
	if (two_sided_mtf) {for (j = 1; j < 10; j += 2) sum_of_coefficients += fabsf(saved_parameters[j]);}
	for (i = 0; i < mtf.number_of_points; i++)
	{
		function_value = 0.0f;
		for (j = 0; j < 10; j += 2) function_value += fabsf(best_parameters[j + 1]) * exp(-fabsf(best_parameters[j]) * powf(mtf.data_x[i], 2));
		if (two_sided_mtf) {for (j = 0; j < 10; j += 2) function_value += fabsf(saved_parameters[j + 1]) * exp(-fabsf(saved_parameters[j]) * powf(mtf.data_x[i], 2));}
//		if (comparison_object.parameters_to_fit > 10) function_value += sinc(fabsf(fitted_parameters[10]) * mtf.data_x[i]) * fabsf(fitted_parameters[9]) * exp(-fabsf(fitted_parameters[8]) * powf(mtf.data_x[i], 2));
		mtf.data_y[i] = function_value / sum_of_coefficients * sqrtf(nps_fit.data_y[i]);
//		mtf.data_y[i] = function_value / sum_of_coefficients;
//		if (is_a_counting_detector) mtf.data_y[i] *= sqrtf(nps_fit.data_y[i]);
//		mtf.data_y[i] = function_value;
//		if (function_value > function_max) function_max = function_value;
//		if (mtf.data_y[i] > function_max) function_max = mtf.data_y[i];
	}
	// Scale MTF to have maximum of 1.0 (convention). Also need to apply this scale (squared) to NPS to leave DQE unchanged
//	mtf.MultiplyByConstant(1.0f / function_max);
//	nps.MultiplyByConstant(1.0f / powf(function_max, 2));
//	nps_fit.MultiplyByConstant(1.0f / powf(function_max, 2));

//	for (i = 0; i < mtf.number_of_points; i++) mtf.data_y[i] /= function_max;
//	if (is_a_counting_detector)
//	{
//		for (i = 0; i < mtf.number_of_points; i++) {mtf.data_y[i] *= sqrtf(nps_fit.data_y[i] / function_max);}
//	}
	if (padding > 1) for (i = 0; i < mtf.number_of_points; i++) mtf.data_y[i] *= sinc(PI * mtf.data_x[i]);
	function_max = -FLT_MAX;
	for (i = 0; i < mtf.number_of_points; i++) if (mtf.data_y[i] > function_max) function_max = mtf.data_y[i];
	mtf.MultiplyByConstant(1.0f / function_max);
	nps.MultiplyByConstant(1.0f / powf(function_max, 2));
	nps_fit.MultiplyByConstant(1.0f / powf(function_max, 2));
	mtf.MultiplyXByConstant(2.0f);
	nps.MultiplyXByConstant(2.0f);
	nps_fit.MultiplyXByConstant(2.0f);

//	mtf.MultiplyByConstant(1.0f / sqrtf(nps_fit.data_y[100]));
//	nps.MultiplyByConstant(1.0f / nps_fit.data_y[100]);
//	nps_fit.MultiplyByConstant(1.0f / nps_fit.data_y[100]);

//	sxx = 0.0f;
//	sxy = 0.0f;
//	for (i = fit_bins_start; i <= fit_bins_end; i++)
//	{
//		sxx += powf(mtf.data_y[i], 4);
//		sxy += nps.data_y[i] * powf(mtf.data_y[i], 2);
//	}
//	scale = sxy / sxx;

//	nps_fit_spectrum.SetupXAxis(0.0, 1.0f / 1000.0f * 1415.0f, 1416);
//	for (i = 0; i < nps.number_of_points; i++)
//	{
//		if (i <= fit_bins_start) nps_fit_spectrum.data_y[i] = scale * powf(mtf.data_y[i], 2);
//		else if (i < fit_bins_end) nps_fit_spectrum.data_y[i] = float(i - fit_bins_start) / float(fit_bins_end - fit_bins_start) * nps.data_y[i] + \
//				(1.0f - float(i - fit_bins_start) / float(fit_bins_end - fit_bins_start)) * scale * powf(mtf.data_y[i], 2);
//		else nps_fit_spectrum.data_y[i] = nps.data_y[i];
//	}

//	if (is_a_counting_detector) mtf.MultiplyByConstant(sqrtf(nps_fit.data_y[0]));
//	for (i = 0; i < mtf.number_of_points; i++) if (mtf.data_x[i] <= 1.0001f) j = i;
//	mtf.number_of_points = j + 1;
//	mtf.WriteToFile("MTF.txt");
//	nps.number_of_points = j + 1;
//	nps_fit.number_of_points = j + 1;
//	nps.WriteToFile("NPS.txt");
//	nps_fit_spectrum.number_of_points = j + 1;
//	nps_fit_spectrum.WriteToFile("NPS_fit.txt");

	wxPrintf("\n\nWriting diagnostic output image...\n\n");
	difference_image.WriteSlice(&output_file, 1);

	printf(" Frequency         MTF        √NPS    √NPS_Fit         DQE\n");
	fprintf(table_file, " Frequency         MTF        √NPS    √NPS_Fit         DQE\n");
	for (i = 0; i < mtf.number_of_points; i++)
	{
		dqe = dqe0 * powf(mtf.data_y[i], 2) / nps_fit.data_y[i];
//		dqe = average_background * gain_conversion_factor * powf(mtf.data_y[i], 2) / nps.data_y[i] * mask_volume / background_image.number_of_real_space_pixels;
//		dqe = powf(sigma_background * mtf.data_y[i], 2) / exposure / nps.data_y[i] * mask_volume;
		wxPrintf("%10.6f %11.7f %11.7f %11.7f %11.7f\n", mtf.data_x[i], mtf.data_y[i], sqrtf(nps.data_y[i]), sqrtf(nps_fit.data_y[i]), dqe);
		fprintf(table_file, "%10.6f %11.7f %11.7f %11.7f %11.7f\n", mtf.data_x[i], mtf.data_y[i], sqrtf(nps.data_y[i]), sqrtf(nps_fit.data_y[i]), dqe);
	}
	fclose(table_file);

	wxPrintf("\nFindDQE finished cleanly!\n\n");

	return true;
}
