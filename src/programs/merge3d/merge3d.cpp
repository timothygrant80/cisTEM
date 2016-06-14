#include "../../core/core_headers.h"

class
Merge3DApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

IMPLEMENT_APP(Merge3DApp)

// override the DoInteractiveUserInput

void Merge3DApp::DoInteractiveUserInput()
{
	wxString	output_reconstruction_1;
	wxString	output_reconstruction_2;
	wxString	output_reconstruction_filtered;
	wxString	output_resolution_statistics;
	float		molecular_mass_kDa = 1000.0;
	float		inner_mask_radius = 0.0;
	float		outer_mask_radius = 100.0;
	wxString	dump_file_seed_1;
	wxString	dump_file_seed_2;

	UserInput *my_input = new UserInput("Merge3D", 1.00);

	output_reconstruction_1 = my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
	output_reconstruction_2 = my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
	output_resolution_statistics = my_input->GetFilenameFromUser("Output resolution statistics", "The text file with the resolution statistics for the final reconstruction", "my_statistics.txt", false);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	dump_file_seed_1 = my_input->GetFilenameFromUser("Seed for input dump filenames for odd particles", "The name of the first dump file with the intermediate reconstruction arrays", "dump_file_seed_1_.dat", false);
	dump_file_seed_2 = my_input->GetFilenameFromUser("Seed for input dump filenames for even particles", "The name of the second dump file with the intermediate reconstruction arrays", "dump_file_seed_2_.dat", false);

	delete my_input;

	my_current_job.Reset(9);
	my_current_job.ManualSetArguments("ttttffftt",	output_reconstruction_1.ToUTF8().data(),
													output_reconstruction_2.ToUTF8().data(),
													output_reconstruction_filtered.ToUTF8().data(),
													output_resolution_statistics.ToUTF8().data(),
													molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
													dump_file_seed_1.ToUTF8().data(),
													dump_file_seed_2.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Merge3DApp::DoCalculation()
{
	wxString output_reconstruction_1			= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_reconstruction_2			= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_reconstruction_filtered		= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_resolution_statistics		= my_current_job.arguments[3].ReturnStringArgument();
	float 	 molecular_mass_kDa					= my_current_job.arguments[4].ReturnFloatArgument();
	float    inner_mask_radius					= my_current_job.arguments[5].ReturnFloatArgument();
	float    outer_mask_radius					= my_current_job.arguments[6].ReturnFloatArgument();
	wxString dump_file_seed_1 					= my_current_job.arguments[7].ReturnStringArgument();
	wxString dump_file_seed_2 					= my_current_job.arguments[8].ReturnStringArgument();

	ReconstructedVolume output_3d;
	ReconstructedVolume output_3d1;
	ReconstructedVolume output_3d2;

	int			logical_x_dimension;
	int			logical_y_dimension;
	int			logical_z_dimension;
	int			count;
	float		mask_volume_fraction;
	float		pssnr_scale_factor;
	float		mask_falloff = 10.0;
	float		pixel_size;
	float		average_occupancy;
	float		average_score;
	float		score_bfactor_conversion;
	float		particle_area_in_pixels;
	float		scale;
	wxString	my_symmetry;
	wxDateTime	my_time_in;
	wxFileName	dump_file_name = wxFileName::FileName(dump_file_seed_1);
	wxString	extension = dump_file_name.GetExt();
	wxString	dump_file;
	bool		insert_even;

	MRCFile output_file;
	NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);

	my_time_in = wxDateTime::Now();
	output_statistics_file.WriteCommentLine("C Merge3D run date and time:               " + my_time_in.FormatISOCombined(' '));
	output_statistics_file.WriteCommentLine("C Output reconstruction 1:                 " + output_reconstruction_1);
	output_statistics_file.WriteCommentLine("C Output reconstruction 2:                 " + output_reconstruction_2);
	output_statistics_file.WriteCommentLine("C Output filtered reconstruction:          " + output_reconstruction_filtered);
	output_statistics_file.WriteCommentLine("C Output resolution statistics:            " + output_resolution_statistics);
	output_statistics_file.WriteCommentLine("C Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
	output_statistics_file.WriteCommentLine("C Inner mask radius (A):                   " + wxString::Format("%f", inner_mask_radius));
	output_statistics_file.WriteCommentLine("C Outer mask radius (A):                   " + wxString::Format("%f", outer_mask_radius));
	output_statistics_file.WriteCommentLine("C Seed for dump files for odd particles:   " + dump_file_seed_1);
	output_statistics_file.WriteCommentLine("C Seed for dump files for even particles:  " + dump_file_seed_2);
	output_statistics_file.WriteCommentLine("C");

	dump_file = wxFileName::StripExtension(dump_file_seed_1) + wxString::Format("%i", 1) + "." + extension;
	if (! DoesFileExist(dump_file))
	{
		MyPrintWithDetails("Error: Dump file %s not found\n", dump_file);
		abort();
	}

	Reconstruct3D temp_reconstruction;
	temp_reconstruction.ReadArrayHeader(dump_file, logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_score, score_bfactor_conversion, my_symmetry, insert_even);
	wxPrintf("\nReconstruction dimensions = %i, %i, %i, pixel size = %f, symmetry = %s\n", logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, my_symmetry);
	temp_reconstruction.Init(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_score, score_bfactor_conversion);
	Reconstruct3D my_reconstruction_1(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_score, score_bfactor_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_score, score_bfactor_conversion, my_symmetry);

	wxPrintf("\nReading reconstruction arrays...\n\n");

	count = 1;
	while (DoesFileExist(dump_file))
	{
		wxPrintf("%s\n", dump_file);
		temp_reconstruction.ReadArrays(dump_file);
		my_reconstruction_1 += temp_reconstruction;
		count++;
		dump_file = wxFileName::StripExtension(dump_file_seed_1) + wxString::Format("%i", count) + "." + extension;
	}

	count = 1;
	dump_file = wxFileName::StripExtension(dump_file_seed_2) + wxString::Format("%i", count) + "." + extension;
	while (DoesFileExist(dump_file))
	{
		wxPrintf("%s\n", dump_file);
		temp_reconstruction.ReadArrays(dump_file);
		my_reconstruction_2 += temp_reconstruction;
		count++;
		dump_file = wxFileName::StripExtension(dump_file_seed_2) + wxString::Format("%i", count) + "." + extension;
	}
	wxPrintf("\nFinished reading arrays\n\n");

	output_3d1.InitWithReconstruct3D(my_reconstruction_1, pixel_size);
	output_3d1.Calculate3DSimple(my_reconstruction_1);
	output_3d1.density_map.SwapRealSpaceQuadrants();
	output_3d1.density_map.BackwardFFT();
	pssnr_scale_factor = output_3d1.Correct3D(outer_mask_radius / pixel_size);
	output_3d1.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	output_file.OpenFile(output_reconstruction_1.ToStdString(), true);
	output_3d1.density_map.WriteSlices(&output_file,1,output_3d1.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();
	output_3d1.density_map.ForwardFFT();

	output_3d2.InitWithReconstruct3D(my_reconstruction_2, pixel_size);
	output_3d2.Calculate3DSimple(my_reconstruction_2);
	output_3d2.density_map.SwapRealSpaceQuadrants();
	output_3d2.density_map.BackwardFFT();
	pssnr_scale_factor = output_3d2.Correct3D(outer_mask_radius / pixel_size);
	output_3d2.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_file.OpenFile(output_reconstruction_2.ToStdString(), true);
	output_3d2.density_map.WriteSlices(&output_file,1,output_3d2.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();
	output_3d2.density_map.ForwardFFT();

	output_3d.InitWithReconstruct3D(my_reconstruction_1, pixel_size);
	output_3d.statistics.CalculateFSC(output_3d1.density_map, output_3d2.density_map);
//	my_reconstruction_1 += my_reconstruction_2;
//	pssnr_scale_factor = my_reconstruction_1.Correct3DCTF(output_3d2.density_map);
//	my_reconstruction_1.image_reconstruction.BackwardFFT();
//	pssnr_scale_factor = my_reconstruction_1.image_reconstruction.Correct3D();
//	my_reconstruction_1.image_reconstruction.ForwardFFT();

	output_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	output_3d.statistics.CalculateParticleFSCandSSNR(output_3d1.mask_volume_in_voxels, molecular_mass_kDa);
	// mask_volume_2d / output_3d1.density_map.logical_x_dimension / output_3d1.density_map.logical_y_dimension = fraction of input image inside 2D mask applied on input.
	// (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) = volume (number of voxels) inside particle envelope.
	// (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) / output_3d1.mask_volume_in_voxels = fraction of voxels in the 3D mask applied to the final reconstruction that are inside the particle envelope.
	// The first line below lowers the SSNR to account for the increase in the calculated FSC/SSNR due to the 2D masking applied on input.
	// The second line below increases the SSNR to account for the noise inside the 3D mask in the voxels that are outside the particle envelope.
	particle_area_in_pixels = PI * powf(3.0 * (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) / 4.0 / PI, 2.0 / 3.0);
	mask_volume_fraction = output_3d1.density_map.logical_x_dimension * output_3d1.density_map.logical_y_dimension / particle_area_in_pixels
			* output_3d1.mask_volume_in_voxels / output_3d1.density_map.ReturnVolumeInRealSpace();
	wxPrintf("particle_area_in_pixels = %g, mask_volume_fraction = %g\n", particle_area_in_pixels, mask_volume_fraction);
	output_3d.statistics.CalculateParticleSSNR(my_reconstruction_1.image_reconstruction, my_reconstruction_1.ctf_reconstruction, mask_volume_fraction);
	output_3d.has_statistics = true;
	output_3d.PrintStatistics();
	output_3d.WriteStatisticsToFile(output_statistics_file);

	my_reconstruction_1 += my_reconstruction_2;
//	~my_reconstruction_2;
	pssnr_scale_factor = output_3d1.density_map.ReturnVolumeInRealSpace() / (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3))
			* particle_area_in_pixels / output_3d1.density_map.logical_x_dimension / output_3d1.density_map.logical_y_dimension;
	output_3d.Calculate3DOptimal(my_reconstruction_1, pssnr_scale_factor);
	output_3d.density_map.SwapRealSpaceQuadrants();
	output_3d.density_map.CosineMask(0.5, 1.0 / 20.0);
	output_3d.density_map.BackwardFFT();
	output_3d.Correct3D();
	output_3d.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_file.OpenFile(output_reconstruction_filtered.ToStdString(), true);
	output_3d.density_map.WriteSlices(&output_file,1,output_3d.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();

	wxPrintf("\nMerge3D: Normal termination\n\n");

	return true;
}
