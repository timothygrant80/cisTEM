#include "../../core/core_headers.h"

class
NikoTestApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput()
{

}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation()
{
	int i,j;
	int count;
	int padded_dimensions_x;
	int padded_dimensions_y;
	int pad_factor = 6;
	float sigma;
	float peak;
	float sum_of_peaks;

	MRCFile input_file_3d("input3d.mrc", false);
	MRCFile input_file_2d("input2d.mrc", false);
	MRCFile output_file("output.mrc", true);
	Image input_volume;
	Image input_image;
	Image padded_image;
	Image output_image;
	Image temp_image;
	Image temp_image2;

	padded_dimensions_x = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnXSize(), 3);
	padded_dimensions_y = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnYSize(), 3);
	input_volume.Allocate(input_file_3d.ReturnXSize(), input_file_3d.ReturnYSize(), input_file_3d.ReturnZSize(), true);
	input_image.Allocate(input_file_2d.ReturnXSize(), input_file_2d.ReturnYSize(), true);
	padded_image.Allocate(padded_dimensions_x, padded_dimensions_y, true);

	input_volume.ReadSlices(&input_file_3d, 1, input_file_3d.ReturnZSize());
	input_image.ReadSlice(&input_file_2d, 1);
	sigma = sqrtf(input_image.ReturnVarianceOfRealValues());
	input_image.ClipIntoLargerRealSpace2D(&padded_image);
	padded_image.AddGaussianNoise(10.0 * sigma);
	padded_image.WriteSlice(&output_file, 1);
	padded_image.ForwardFFT();

	input_image.AddSlices(input_volume);
//	input_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
	count = 1;
	output_image.CopyFrom(&padded_image);
	temp_image.CopyFrom(&input_image);
	temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//	temp_image2.CopyFrom(&input_image);
//	temp_image2.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//	temp_image2.RealSpaceIntegerShift(input_image.logical_x_dimension, input_image.logical_y_dimension, 0);
//	temp_image.AddImage(&temp_image2);
//	temp_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
	temp_image.ForwardFFT();
	output_image.ConjugateMultiplyPixelWise(temp_image);
	output_image.SwapRealSpaceQuadrants();
	output_image.BackwardFFT();
	peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
	wxPrintf("\nPeak with whole projection = %g background = %g\n\n", peak, output_image.ReturnVarianceOfRealValues(float(2 * input_image.logical_x_dimension), 0.0, 0.0, 0.0, true));
//	wxPrintf("\nPeak with whole projection = %g\n\n", output_image.ReturnMaximumValue());
	output_image.WriteSlice(&output_file, 2);
	sum_of_peaks = 0.0;
	for (i = 1; i <= 3; i += 2)
	{
		for (j = 1; j <= 3; j += 2)
		{
			output_image.CopyFrom(&padded_image);
			temp_image.CopyFrom(&input_image);
			temp_image.SquareMaskWithValue(float(input_image.logical_x_dimension) / 2.0, 0.0, false, i * input_image.logical_x_dimension / 4, j * input_image.logical_x_dimension / 4);
			temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
			temp_image.ForwardFFT();
			output_image.ConjugateMultiplyPixelWise(temp_image);
			output_image.SwapRealSpaceQuadrants();
			output_image.BackwardFFT();
			peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
//			peak = output_image.ReturnMaximumValue();
			wxPrintf("Quarter peak = %i %i %g\n", i, j, peak);
			sum_of_peaks += peak;
			count++;
			output_image.WriteSlice(&output_file, 1 + count);
		}
	}
	wxPrintf("\nSum of quarter peaks = %g\n\n", sum_of_peaks);

	sum_of_peaks = 0.0;
	for (i = 0; i < 4; i ++)
	{
		output_image.CopyFrom(&padded_image);
		input_image.AddSlices(input_volume, i * input_volume.logical_z_dimension / 4 + 1, (i + 1) * input_volume.logical_z_dimension / 4);
//		input_image.QuickAndDirtyWriteSlice("junk.mrc", i + 1);
		temp_image.CopyFrom(&input_image);
		temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
		temp_image.ForwardFFT();
		output_image.ConjugateMultiplyPixelWise(temp_image);
		output_image.SwapRealSpaceQuadrants();
		output_image.BackwardFFT();
		peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
//		peak = output_image.ReturnMaximumValue();
		wxPrintf("Slice peak = %i %g\n", i + 1, peak);
		sum_of_peaks += peak;
		count++;
		output_image.WriteSlice(&output_file, 1 + count);
	}
	wxPrintf("\nSum of slice peaks = %g\n", sum_of_peaks);

/*	wxPrintf("\nDoing 1000 FFTs %i x %i\n", output_image.logical_x_dimension, output_image.logical_y_dimension);
	for (i = 0; i < 1000; i++)
	{
		output_image.is_in_real_space = false;
		output_image.SetToConstant(1.0);
		output_image.BackwardFFT();
	}
	wxPrintf("\nFinished\n");
*/

/*	int i, j;
	int slice_thickness;
	int first_slice, last_slice, middle_slice;
	long offset;
	long pixel_counter;
	float bfactor = 20.0;
	float mask_radius = 75.0;
	float pixel_size = 1.237;
//	float pixel_size = 0.97;
	float bfactor_res_limit = 8.0;
	float resolution_limit = 3.8;
//	float resolution_limit = 3.0;
	float cosine_edge = 5.0;
	float bfactor_pixels;

	MRCFile input_file("input.mrc", false);
	MRCFile output_file_2D("output2D.mrc", true);
	MRCFile output_file_3D("output3D.mrc", true);
	Image input_image;
	Image output_image;
	Image output_image_3D;

	Curve power_spectrum;
	Curve number_of_terms;

	UserInput my_input("NikoTest", 1.00);
	pixel_size = my_input.GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	mask_radius = my_input.GetFloatFromUser("Mask radius (A)", "Radius of mask to be applied to input 3D map, in Angstroms", "100.0", 0.0);
	bfactor = my_input.GetFloatFromUser("B-Factor (A^2)", "B-factor to be applied to dampen the 3D map after spectral flattening, in Angstroms squared", "20.0");
	bfactor_res_limit = my_input.GetFloatFromUser("Low resolution limit for spectral flattening (A)", "The resolution at which spectral flattening starts being applied, in Angstroms", "8.0", 0.0);
	resolution_limit = my_input.GetFloatFromUser("High resolution limit (A)", "Resolution of low-pass filter applied to final output maps, in Angstroms", "3.0", 0.0);

	slice_thickness = myroundint(resolution_limit / pixel_size);
	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), slice_thickness, true);
	output_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	output_image_3D.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

	wxPrintf("\nCalculating 3D spectrum...\n");

	power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	output_image_3D.ReadSlices(&input_file, 1, input_file.ReturnZSize());
	output_image_3D.CosineMask(mask_radius / pixel_size, 10.0 / pixel_size);

	first_slice = int((input_file.ReturnZSize() - slice_thickness + 1) / 2.0);
	last_slice = first_slice + slice_thickness;
	pixel_counter = 0;
	for (j = first_slice; j < last_slice; j++)
	{
		offset = j * (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension;
		for (i = 0; i < (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension; i++) {input_image.real_values[pixel_counter] = output_image_3D.real_values[i + offset]; pixel_counter++;}
	}

	output_image_3D.ForwardFFT();
	output_image_3D.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
	power_spectrum.SquareRoot();
	wxPrintf("Done with 3D spectrum. Starting slice estimation...\n");

//	input_image.ReadSlices(&input_file, first_slice, last_slice);
	bfactor_pixels = bfactor / pixel_size / pixel_size;
	input_image.ForwardFFT();
	input_image.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
//	input_image.ApplyBFactor(bfactor_pixels);
//	input_image.CosineMask(pixel_size / resolution_limit, cosine_edge / input_file.ReturnXSize());
	input_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
	input_image.BackwardFFT();

	middle_slice = int(slice_thickness / 2.0);
	offset = middle_slice * (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize();
	pixel_counter = 0;
	for (i = 0; i < (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize(); i++) {output_image.real_values[pixel_counter] = input_image.real_values[i + offset]; pixel_counter++;}
//	output_image.ForwardFFT();
//	output_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
//	output_image.BackwardFFT();
	output_image.WriteSlice(&output_file_2D, 1);
	wxPrintf("Done with slices. Starting 3D B-factor application...\n");

	output_image_3D.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
	output_image_3D.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
	output_image_3D.BackwardFFT();
	output_image_3D.WriteSlices(&output_file_3D, 1, input_file.ReturnZSize());
	wxPrintf("Done with 3D B-factor application.\n");
/*
	int i;
	int min_class;
	int max_class;
	int count;
	float temp_float;
	float input_parameters[17];

	MRCFile input_file("input.mrc", false);
	MRCFile output_file("output.mrc", true);
	Image input_image;
	Image padded_image;
	Image ctf_image;
	Image sum_image;
	CTF ctf;
	AnglesAndShifts rotation_angle;

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	padded_image.Allocate(4 * input_file.ReturnXSize(), 4 * input_file.ReturnYSize(), true);
	ctf_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
	sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
	sum_image.SetToConstant(0.0);

	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
	input_par_file.ReadFile();

//	count = 0;
//	for (i = 1; i <= input_par_file.number_of_lines; i++)
//	{
//		if (i % 100 == 1) wxPrintf("Working on line %i\n", i);
//		input_par_file.ReadLine(input_parameters);
//		input_image.ReadSlice(&input_file, int(input_parameters[0] + 0.5));
//		count++;
//		input_image.WriteSlice(&output_file, count);
//	}

	for (i = 1; i <= input_par_file.number_of_lines; i++)
	{
		if (i % 100 == 1) wxPrintf("Rotating image %i\n", i);
		input_par_file.ReadLine(input_parameters);
		input_image.ReadSlice(&input_file, i);
		input_image.RealSpaceIntegerShift(-input_parameters[4], -input_parameters[5]);
		input_image.ForwardFFT();
		input_image.ClipInto(&padded_image);
		padded_image.BackwardFFT();
		rotation_angle.GenerateRotationMatrix2D(-input_parameters[1]);
		padded_image.Rotate2DSample(input_image, rotation_angle);
		input_image.WriteSlice(&output_file, i);
		if (input_parameters[7] == 2)
		{
			ctf.Init(300.0, 0.0, 0.1, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, 1.0, input_parameters[11]);
			ctf_image.CalculateCTFImage(ctf);
			input_image.ForwardFFT();
			input_image.PhaseFlipPixelWise(ctf_image);
			sum_image.AddImage(&input_image);
		}
//		if (i == 1001) break;
	}
	sum_image.QuickAndDirtyWriteSlice("sum.mrc", 1);
*/

/*	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
	FrealignParameterFile output_par_file("output.par", OPEN_TO_WRITE);
	input_par_file.ReadFile(true);
	input_par_file.ReduceAngles();
	min_class = myroundint(input_par_file.ReturnMin(7));
	max_class = myroundint(input_par_file.ReturnMax(7));
	for (i = min_class; i <= max_class; i++)
	{
		temp_float = input_par_file.ReturnDistributionMax(2, i);
		if (temp_float != 0.0) wxPrintf("theta max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(2, temp_float, i));
//		input_par_file.SetParameters(2, temp_float, i);
		temp_float = input_par_file.ReturnDistributionMax(3, i);
		if (temp_float != 0.0) wxPrintf("phi max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(3, temp_float, i));
//		input_par_file.SetParameters(3, temp_float, i);
	} */
//	for (i = 1; i <= input_par_file.number_of_lines; i++)
//	{
//		input_par_file.ReadLine(input_parameters);
//		output_par_file.WriteLine(input_parameters);
//	}

//	MRCFile input_file("input.mrc", false);
//	MRCFile output_file("output.mrc", true);
//	Image input_image;
//	Image filtered_image;
//	Image kernel;

//	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	filtered_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	kernel.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	input_image.ReadSlices(&input_file,1,input_image.logical_z_dimension);

/*	kernel.SetToConstant(1.0);
	kernel.CosineMask(8.0, 8.0, false, true, 0.0);
//	kernel.real_values[0] = 1.0;
	temp_float = kernel.ReturnAverageOfRealValues() * kernel.number_of_real_space_pixels;
//	wxPrintf("average = %g\n", temp_float);
//	kernel.WriteSlices(&output_file,1,input_image.logical_z_dimension);
	kernel.ForwardFFT();
	kernel.SwapRealSpaceQuadrants();
	kernel.MultiplyByConstant(float(kernel.number_of_real_space_pixels) / temp_float);
//	kernel.CosineMask(0.03, 0.03, true);

	input_image.SetMinimumValue(0.0);
	filtered_image.CopyFrom(&input_image);
	filtered_image.ForwardFFT();
	filtered_image.MultiplyPixelWise(kernel);
//	filtered_image.CosineMask(0.01, 0.02);
	filtered_image.BackwardFFT();
//	filtered_image.MultiplyByConstant(0.3);
	input_image.SubtractImage(&filtered_image);
*/
//	input_image.SetToConstant(1.0);
//	input_image.CorrectSinc(45.0, 1.0, true, 0.0);
//	for (i = 0; i < input_image.real_memory_allocated; i++) if (input_image.real_values[i] < 0.0) input_image.real_values[i] = -log(-input_image.real_values[i] + 1.0);
//	input_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);
//	temp_float = -420.5; wxPrintf("%g\n", fmodf(temp_float, 360.0));
//	filtered_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);

	return true;
}
