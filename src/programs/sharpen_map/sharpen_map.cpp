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

//	my_current_job.Reset(16);
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

	Image input_map;
	Image *mask_volume = NULL;

	MRCFile input_file(input_volume.ToStdString(), false);
	MRCFile output_file(output_volume.ToStdString(), true);

	MRCFile *input_mask_file = NULL;
	ResolutionStatistics *input_statistics = NULL;

	if (input_file.ReturnZSize() <= 1)
	{
		MyPrintWithDetails("Error: Input map is not a volume\n");
		DEBUG_ABORT;
	}


	if (use_mask)
	{
		mask_volume = new Image;
		input_mask_file = new MRCFile(input_mask.ToStdString(), false);
		if (input_file.ReturnXSize() != input_mask_file->ReturnXSize() || input_file.ReturnYSize() != input_mask_file->ReturnYSize() || input_file.ReturnZSize() != input_mask_file->ReturnZSize())
		{
			wxPrintf("\nVolume and mask file have different dimensions\n");
			DEBUG_ABORT;
		}
		else mask_volume->Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	}

	input_map.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

	if (use_statistics)
	{
		input_statistics = new ResolutionStatistics(pixel_size, input_map.logical_y_dimension);
		input_statistics->ReadStatisticsFromFile(res_statistics);
	}

	input_map.ReadSlices(&input_file, 1, input_file.ReturnZSize());

	if (use_mask)
	{
		mask_volume->ReadSlices(input_mask_file, 1, input_mask_file->ReturnNumberOfSlices());
	}

	input_map.SharpenMap(pixel_size, resolution_limit, invert_hand, inner_mask_radius, outer_mask_radius , bfactor_res_limit, bfactor_low, bfactor_high, filter_edge, mask_volume, input_statistics, fudge_SSNR);
	input_map.is_in_real_space = true;
	input_map.WriteSlices(&output_file, 1, input_file.ReturnZSize());

//	wxPrintf("Done with 3D B-factor application.\n");

	output_file.SetPixelSize(pixel_size);
	output_file.WriteHeader();

	return true;

	if (use_mask == true)
	{
		delete input_mask_file;
		delete mask_volume;
	}
	if (use_statistics == true) delete input_statistics;
}
