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
	wxString input_mask		= my_input->GetFilenameFromUser("Input mask file name", "Name of input image file", "mask.mrc", false );
	wxString res_statistics	= my_input->GetFilenameFromUser("Input reconstruction statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	bool use_statistics		= my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	float pixel_size		= my_input->GetFloatFromUser("Pixel size (A)", "Pixel size of the map in Angstroms", "1.0", 0.000001);
	float inner_mask_radius	= my_input->GetFloatFromUser("Inner mask radius (A)", "Inner radius of mask to be applied to the input map, in Angstroms", "0.0", 0.0);
	float outer_mask_radius	= my_input->GetFloatFromUser("Outer mask radius (A)", "Outer radius of mask to be applied to the input map, in Angstroms", "100.0", 0.0);
	float bfactor_low		= my_input->GetFloatFromUser("Low-res B-Factor (A^2)", "B-factor to be applied to the non-flattened part of the amplitude spectrum, in Angstroms squared", "0.0");
	float bfactor_high		= my_input->GetFloatFromUser("High-res B-Factor (A^2)", "B-factor to be applied to the flattened part of the amplitude spectrum, in Angstroms squared", "0.0");
	float bfactor_res_limit	= my_input->GetFloatFromUser("Low resolution limit for spectral flattening (A)", "The resolution at which spectral flattening starts being applied, in Angstroms", "8.0", 0.0);
	float resolution_limit	= my_input->GetFloatFromUser("High resolution limit (A)", "Resolution of low-pass filter applied to final output maps, in Angstroms", "3.0", 0.0);
	float filter_edge		= my_input->GetFloatFromUser("Filter edge width (A)", "Cosine edge used with the low-pass filter, in Angstroms", "20.0", 0.0);
	float fudge_SSNR		= my_input->GetFloatFromUser("Part_SSNR scale factor", "Scale the Part_SSNR curve to account for disordered regions in the map", "1.0", 0.1, 10.0);
//	float fudge_FSC			= my_input->GetFloatFromUser("Statistics curve scale factor", "Resample the Part_FSC curve to extend or limit resolution", "1.0", 0.5, 1.5);
	bool use_mask			= my_input->GetYesNoFromUser("Use 3D mask", "Should the 3D mask be used to mask the input map before sharpening?", "No");
	bool invert_hand		= my_input->GetYesNoFromUser("Invert handedness", "Should the map handedness be inverted?", "No");

	delete my_input;

	my_current_job.Reset(16);
	my_current_job.ManualSetArguments("ttttbfffffffffbb", input_volume.ToUTF8().data(), output_volume.ToUTF8().data(), input_mask.ToUTF8().data(), res_statistics.ToUTF8().data(),
			use_statistics, pixel_size, inner_mask_radius, outer_mask_radius, bfactor_low, bfactor_high, bfactor_res_limit, resolution_limit, filter_edge, fudge_SSNR, use_mask, invert_hand);
}

// override the do calculation method which will be what is actually run..

bool SharpenMap::DoCalculation()
{

	wxString input_volume	= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_volume	= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_mask		= my_current_job.arguments[2].ReturnStringArgument();
	wxString res_statistics	= my_current_job.arguments[3].ReturnStringArgument();
	bool use_statistics		= my_current_job.arguments[4].ReturnBoolArgument();
	float pixel_size		= my_current_job.arguments[5].ReturnFloatArgument();
	float inner_mask_radius	= my_current_job.arguments[6].ReturnFloatArgument();
	float outer_mask_radius	= my_current_job.arguments[7].ReturnFloatArgument();
	float bfactor_low		= my_current_job.arguments[8].ReturnFloatArgument();
	float bfactor_high		= my_current_job.arguments[9].ReturnFloatArgument();
	float bfactor_res_limit	= my_current_job.arguments[10].ReturnFloatArgument();
	float resolution_limit	= my_current_job.arguments[11].ReturnFloatArgument();
	float filter_edge		= my_current_job.arguments[12].ReturnFloatArgument();
	float fudge_SSNR		= my_current_job.arguments[13].ReturnFloatArgument();
//	float fudge_FSC			= my_current_job.arguments[13].ReturnFloatArgument();
	bool use_mask			= my_current_job.arguments[14].ReturnBoolArgument();
	bool invert_hand		= my_current_job.arguments[15].ReturnBoolArgument();

	MRCFile input_file(input_volume.ToStdString(), false);
	MRCFile output_file(output_volume.ToStdString(), true);
	MRCFile *input_mask_file;

	int i, j;
	long offset, pixel_counter;
	float cosine_edge = 10.0;
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
		else mask_volume.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	}

	input_map.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	output_map.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

	ResolutionStatistics input_statistics(pixel_size, input_map.logical_y_dimension);
	if (use_statistics)
	{
		input_statistics.ReadStatisticsFromFile(res_statistics);
//		if (fudge_FSC != 1.0) input_statistics.part_FSC.ResampleCurve(&input_statistics.part_FSC, myroundint(input_statistics.part_FSC.number_of_points * fudge_FSC));
		if (fudge_SSNR != 1.0) input_statistics.rec_SSNR.MultiplyByConstant(fudge_SSNR);
	}

//	wxPrintf("\nCalculating 3D spectrum...\n");

	power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_map.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_map.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	input_map.ReadSlices(&input_file, 1, input_file.ReturnZSize());
	output_map.CopyFrom(&input_map);
	output_map.ForwardFFT();

	if (outer_mask_radius == 0.0) outer_mask_radius = input_map.logical_x_dimension / 2.0;
	if (use_mask)
	{
		mask_volume.ReadSlices(input_mask_file, 1, input_mask_file->ReturnNumberOfSlices());
		wxPrintf("\nMask volume = %g voxels\n\n", input_map.ApplyMask(mask_volume, cosine_edge / pixel_size, 0.0, 0.0, 0.0));
	}
	else input_map.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, cosine_edge / pixel_size);
//	else input_map.CosineMask(mask_radius / pixel_size, cosine_edge / pixel_size);

	input_map.ForwardFFT();
	input_map.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
	power_spectrum.SquareRoot();
//	wxPrintf("Done with 3D spectrum. Starting slice estimation...\n");

	output_map.ApplyBFactorAndWhiten(power_spectrum, bfactor_low / pixel_size / pixel_size, bfactor_high / pixel_size / pixel_size, pixel_size / bfactor_res_limit);
//	if (use_statistics) output_map.OptimalFilterFSC(input_statistics.part_FSC);
	if (use_statistics) output_map.OptimalFilterSSNR(input_statistics.rec_SSNR);
	output_map.CosineMask(pixel_size / resolution_limit - pixel_size / 2.0 / filter_edge, pixel_size / filter_edge);
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

	output_file.SetPixelSize(pixel_size);
	output_file.WriteHeader();

	return true;
}
