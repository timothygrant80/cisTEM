#include "../../core/core_headers.h"

class
CalculateFSC : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(CalculateFSC)

// override the DoInteractiveUserInput

void CalculateFSC::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("CalculateFSC", 1.0);

	wxString output_reconstruction_1	= my_input->GetFilenameFromUser("Input reconstruction 1", "The first input 3D reconstruction used for FSC calculation", "my_reconstruction_1.mrc", false);
	wxString output_reconstruction_2	= my_input->GetFilenameFromUser("Input reconstruction 2", "The second input 3D reconstruction used for FSC calculation", "my_reconstruction_2.mrc", false);
	wxString input_mask					= my_input->GetFilenameFromUser("Input mask file name", "Name of input 3D volume to be applied to input reconstructions ", "mask.mrc", false );
	wxString res_statistics				= my_input->GetFilenameFromUser("Output resolution statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	float pixel_size					= my_input->GetFloatFromUser("Pixel size (A)", "Pixel size of the map in Angstroms", "1.0", 0.000001);
	float inner_mask_radius				= my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstructions, in Angstroms", "0.0", 0.0);
	float outer_mask_radius				= my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstructions, in Angstroms", "100.0", inner_mask_radius);
	float molecular_mass_in_kDa			= my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	bool use_mask						= my_input->GetYesNoFromUser("Use 3D mask", "Should the 3D mask be used to mask the input reconstructions before FSC calculation?", "No");

	delete my_input;

	my_current_job.Reset(9);
	my_current_job.ManualSetArguments("ttttffffb", output_reconstruction_1.ToUTF8().data(), output_reconstruction_2.ToUTF8().data(), input_mask.ToUTF8().data(), res_statistics.ToUTF8().data(),
			pixel_size, inner_mask_radius, outer_mask_radius, molecular_mass_in_kDa, use_mask);
}

// override the do calculation method which will be what is actually run..

bool CalculateFSC::DoCalculation()
{

	wxString output_reconstruction_1	= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_reconstruction_2	= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_mask					= my_current_job.arguments[2].ReturnStringArgument();
	wxString res_statistics				= my_current_job.arguments[3].ReturnStringArgument();
	float pixel_size					= my_current_job.arguments[4].ReturnFloatArgument();
	float inner_mask_radius				= my_current_job.arguments[5].ReturnFloatArgument();
	float outer_mask_radius				= my_current_job.arguments[6].ReturnFloatArgument();
	float molecular_mass_in_kDa			= my_current_job.arguments[7].ReturnFloatArgument();
	bool use_mask						= my_current_job.arguments[8].ReturnBoolArgument();

	MRCFile reconstruction_1(output_reconstruction_1.ToStdString(), false);
	MRCFile reconstruction_2(output_reconstruction_2.ToStdString(), false);
	NumericTextFile output_statistics_file(res_statistics, OPEN_TO_WRITE, 7);
	MRCFile *input_mask_file;

	int i, j;
	float cosine_edge = 10.0;
	float mask_volume_in_voxels;
	Image density_map_1;
	Image density_map_2;
	Image mask_volume;

	if (reconstruction_1.ReturnZSize() <= 1)
	{
		MyPrintWithDetails("Error: Input reconstruction 1 is not a volume\n");
		abort();
	}

	if (reconstruction_1.ReturnXSize() != reconstruction_1.ReturnYSize() && reconstruction_1.ReturnXSize() != reconstruction_1.ReturnZSize() && reconstruction_1.ReturnYSize() != reconstruction_1.ReturnZSize())
	{
		MyPrintWithDetails("Error: Input reconstruction 1 is not a cube\n");
		abort();
	}

	if (reconstruction_1.ReturnXSize() != reconstruction_2.ReturnXSize() || reconstruction_1.ReturnYSize() != reconstruction_2.ReturnYSize() || reconstruction_1.ReturnZSize() != reconstruction_2.ReturnZSize())
	{
		wxPrintf("\nInput reconstructions have different dimensions\n");
		abort();
	}

	if (use_mask)
	{
		input_mask_file = new MRCFile(input_mask.ToStdString(), false);
		if (reconstruction_1.ReturnXSize() != input_mask_file->ReturnXSize() || reconstruction_1.ReturnYSize() != input_mask_file->ReturnYSize() || reconstruction_1.ReturnZSize() != input_mask_file->ReturnZSize())
		{
			wxPrintf("\nVolume and mask file have different dimensions\n");
			abort();
		}
		else mask_volume.Allocate(reconstruction_1.ReturnXSize(), reconstruction_1.ReturnYSize(), reconstruction_1.ReturnZSize(), true);
	}

	density_map_1.Allocate(reconstruction_1.ReturnXSize(), reconstruction_1.ReturnYSize(), reconstruction_1.ReturnZSize(), true);
	density_map_2.Allocate(reconstruction_2.ReturnXSize(), reconstruction_2.ReturnYSize(), reconstruction_2.ReturnZSize(), true);
	density_map_1.ReadSlices(&reconstruction_1, 1, reconstruction_1.ReturnZSize());
	density_map_2.ReadSlices(&reconstruction_2, 1, reconstruction_2.ReturnZSize());

	if (use_mask)
	{
		mask_volume.ReadSlices(input_mask_file, 1, input_mask_file->ReturnNumberOfSlices());
		mask_volume_in_voxels = density_map_1.ApplyMask(mask_volume, cosine_edge / pixel_size, 0.0, 0.0, 0.0);
		mask_volume_in_voxels = density_map_2.ApplyMask(mask_volume, cosine_edge / pixel_size, 0.0, 0.0, 0.0);
		wxPrintf("\nMask volume = %g voxels\n\n", mask_volume_in_voxels);
	}
	else
	{
		mask_volume_in_voxels = density_map_1.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, cosine_edge / pixel_size);
		mask_volume_in_voxels = density_map_2.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, cosine_edge / pixel_size);
	}

	density_map_1.ForwardFFT();
	density_map_2.ForwardFFT();

	ResolutionStatistics statistics(pixel_size, density_map_1.logical_x_dimension);
	statistics.CalculateFSC(density_map_1, density_map_2, true);
	statistics.CalculateParticleFSCandSSNR(mask_volume_in_voxels, molecular_mass_in_kDa);
	statistics.part_SSNR.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((density_map_1.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	statistics.PrintStatistics();

	statistics.WriteStatisticsToFile(output_statistics_file);

	if (use_mask) delete input_mask_file;

	return true;
}
