#include "../../core/core_headers.h"

class
SharpenMap : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(SharpenMap)

// override the DoInteractiveUserInput

void SharpenMap::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("SharpenMap", 1.0);

	wxString input_volume	= my_input->GetFilenameFromUser("Input volume file name", "Name of input image file", "input.mrc", true );
	wxString output_volume	= my_input->GetFilenameFromUser("Output sharpened volume file name", "Name of sharpened output volume", "output.mrc", false );
	wxString input_mask		= my_input->GetFilenameFromUser("Input mask file name", "Name of input image file", "mask.mrc", true );
	float pixel_size		= my_input->GetFloatFromUser("Pixel size (A)", "Pixel size of the map in Angstroms", "1.0", 0.000001);
	float mask_radius		= my_input->GetFloatFromUser("Mask radius (A)", "Radius of mask to be applied to input map, in Angstroms", "100.0", 0.0);
	float bfactor			= my_input->GetFloatFromUser("B-Factor (A^2)", "B-factor to be applied to dampen the map after spectral flattening, in Angstroms squared", "20.0");
	float bfactor_res_limit	= my_input->GetFloatFromUser("Low resolution limit for spectral flattening (A)", "The resolution at which spectral flattening starts being applied, in Angstroms", "8.0", 0.0);
	float resolution_limit	= my_input->GetFloatFromUser("High resolution limit (A)", "Resolution of low-pass filter applied to final output maps, in Angstroms", "3.0", 0.0);
	bool use_mask			= my_input->GetYesNoFromUser("Use 3D mask", "Should the 3D mask be used to mask the input map before sharpening?", "No");
	bool invert_hand		= my_input->GetYesNoFromUser("Invert handedness", "Should the map handedness be inverted?", "No");

	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("tttfffffbb", input_volume.ToUTF8().data(), output_volume.ToUTF8().data(), input_mask.ToUTF8().data(), pixel_size, mask_radius, bfactor, bfactor_res_limit, resolution_limit, use_mask, invert_hand);
}

// override the do calculation method which will be what is actually run..

bool SharpenMap::DoCalculation()
{

	wxString input_volume	= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_volume	= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_mask		= my_current_job.arguments[2].ReturnStringArgument();
	float pixel_size		= my_current_job.arguments[3].ReturnFloatArgument();
	float mask_radius		= my_current_job.arguments[4].ReturnFloatArgument();
	float bfactor			= my_current_job.arguments[5].ReturnFloatArgument();
	float bfactor_res_limit	= my_current_job.arguments[6].ReturnFloatArgument();
	float resolution_limit	= my_current_job.arguments[7].ReturnFloatArgument();
	bool use_mask			= my_current_job.arguments[8].ReturnBoolArgument();
	bool invert_hand		= my_current_job.arguments[9].ReturnBoolArgument();

	MRCFile input_file(input_volume.ToStdString(), false);
	MRCFile output_file(output_volume.ToStdString(), true);
	MRCFile *input_mask_file;

	int i, j;
	long offset, pixel_counter;
	float cosine_edge = 10.0;
	float slice_thickness;
	Image input_map;
	Image mask_volume;
	Image output_map;

	Curve power_spectrum;
	Curve number_of_terms;

	if (input_file.ReturnZSize() <= 1)
	{
		MyPrintWithDetails("Error: Input map is not a volume\n");
		abort();
	}

	if (use_mask)
	{
		input_mask_file = new MRCFile(input_mask.ToStdString(), false);
		if (input_file.ReturnXSize() != input_mask_file->ReturnXSize() || input_file.ReturnYSize() != input_mask_file->ReturnYSize() || input_file.ReturnZSize() != input_mask_file->ReturnZSize())
		{
			wxPrintf("\nVolume and mask file have different dimensions\n");
			abort();
		}
	}

	slice_thickness = myroundint(resolution_limit / pixel_size);
	input_map.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	output_map.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

//	wxPrintf("\nCalculating 3D spectrum...\n");

	power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_map.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_map.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	input_map.ReadSlices(&input_file, 1, input_file.ReturnZSize());
	output_map.CopyFrom(&input_map);
	output_map.ForwardFFT();

	if (use_mask)
	{
		mask_volume.ReadSlices(input_mask_file, 1, input_mask_file->ReturnNumberOfSlices());
		wxPrintf("\nMask volume = %g\n\n", input_map.ApplyMask(mask_volume, cosine_edge / pixel_size, 0.0, 0.0, 0.0));
	}
	else input_map.CosineMask(mask_radius / pixel_size, cosine_edge / pixel_size);

	input_map.ForwardFFT();
	input_map.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
	power_spectrum.SquareRoot();
//	wxPrintf("Done with 3D spectrum. Starting slice estimation...\n");

	output_map.ApplyBFactorAndWhiten(power_spectrum, bfactor / pixel_size / pixel_size, pixel_size / bfactor_res_limit, pixel_size / resolution_limit);
	output_map.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
	output_map.BackwardFFT();

	if (invert_hand)
	{
		pixel_counter = 0;
		for (j = output_map.logical_z_dimension - 1; j >= 0; j--)
		{
			offset = j * (output_map.logical_x_dimension + output_map.padding_jump_value) * output_map.logical_y_dimension;
			for (i = 0; i < (output_map.logical_x_dimension + output_map.padding_jump_value) * output_map.logical_y_dimension; i++)
			{
				input_map.real_values[pixel_counter] = output_map.real_values[i + offset];
				pixel_counter++;
			}
		}
		input_map.is_in_real_space = true;
		input_map.WriteSlices(&output_file, 1, input_file.ReturnZSize());
	}
	else output_map.WriteSlices(&output_file, 1, input_file.ReturnZSize());

	if (use_mask) delete input_mask_file;

//	wxPrintf("Done with 3D B-factor application.\n");

	return true;
}
