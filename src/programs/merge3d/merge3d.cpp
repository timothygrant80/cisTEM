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

	UserInput *my_input = new UserInput("Merge3D", 1.01);

	output_reconstruction_1 = my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
	output_reconstruction_2 = my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
	output_resolution_statistics = my_input->GetFilenameFromUser("Output resolution statistics", "The text file with the resolution statistics for the final reconstruction", "my_statistics.txt", false);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	dump_file_seed_1 = my_input->GetFilenameFromUser("Seed for input dump filenames for odd particles", "The seed name of the first dump files with the intermediate reconstruction arrays", "dump_file_seed_1_.dat", false);
	dump_file_seed_2 = my_input->GetFilenameFromUser("Seed for input dump filenames for even particles", "The seed name of the second dump files with the intermediate reconstruction arrays", "dump_file_seed_2_.dat", false);

	delete my_input;

	int class_number_for_gui = 1;
	bool save_orthogonal_views_image = false;
	wxString orthogonal_views_filename = "";
	my_current_job.Reset(12);
	my_current_job.ManualSetArguments("ttttfffttibt",	output_reconstruction_1.ToUTF8().data(),
													output_reconstruction_2.ToUTF8().data(),
													output_reconstruction_filtered.ToUTF8().data(),
													output_resolution_statistics.ToUTF8().data(),
													molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
													dump_file_seed_1.ToUTF8().data(),
													dump_file_seed_2.ToUTF8().data(),
													class_number_for_gui,
													save_orthogonal_views_image,
													orthogonal_views_filename.ToUTF8().data());
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
	int class_number_for_gui					= my_current_job.arguments[9].ReturnIntegerArgument();
	bool save_orthogonal_views_image			= my_current_job.arguments[10].ReturnBoolArgument();
	wxString orthogonal_views_filename			= my_current_job.arguments[11].ReturnStringArgument();

	ResolutionStatistics *gui_statistics = NULL;

	if (is_running_locally == false) gui_statistics = new ResolutionStatistics;

	ReconstructedVolume output_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d1(molecular_mass_kDa);
	ReconstructedVolume output_3d2(molecular_mass_kDa);

	int			i;
	int			logical_x_dimension;
	int			logical_y_dimension;
	int			logical_z_dimension;
	int			original_x_dimension;
	int			original_y_dimension;
	int			original_z_dimension;
	int			count;
	int 		intermediate_box_size;
	int			images_processed;
	float		mask_volume_fraction;
	float		mask_falloff = 10.0;
	float		pixel_size;
	float		original_pixel_size;
	float		average_occupancy;
	float		average_sigma;
	float		sigma_bfactor_conversion;
	float		particle_area_in_pixels;
	float		scale;
	float		binning_factor;
	wxString	my_symmetry;
	wxDateTime	my_time_in;
	wxFileName	dump_file_name = wxFileName::FileName(dump_file_seed_1);
	wxString	extension = dump_file_name.GetExt();
	wxString	dump_file;
	bool		insert_even;
	bool		center_mass;
	bool		crop_images;

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
	temp_reconstruction.ReadArrayHeader(dump_file, logical_x_dimension, logical_y_dimension, logical_z_dimension,
			original_x_dimension, original_y_dimension, original_z_dimension, images_processed, pixel_size, original_pixel_size,
			average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry, insert_even, center_mass);
	wxPrintf("\nReconstruction dimensions = %i, %i, %i, pixel size = %f, symmetry = %s\n", logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, my_symmetry);
	temp_reconstruction.Init(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion);
	Reconstruct3D my_reconstruction_1(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);

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
	wxPrintf("\nFinished reading arrays\n");

	output_3d1.FinalizeSimple(my_reconstruction_1, original_x_dimension, original_pixel_size, pixel_size,
			inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_1);
	output_3d2.FinalizeSimple(my_reconstruction_2, original_x_dimension, original_pixel_size, pixel_size,
			inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_2);

	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	my_reconstruction_1 += my_reconstruction_2;
	my_reconstruction_2.FreeMemory();

	output_3d.FinalizeOptimal(my_reconstruction_1, output_3d1.density_map, output_3d2.density_map,
			original_pixel_size, pixel_size, inner_mask_radius, outer_mask_radius, mask_falloff,
			center_mass, output_reconstruction_filtered, output_statistics_file, gui_statistics);

	if (save_orthogonal_views_image == true)
	{
		Image orth_image;
		orth_image.Allocate(output_3d.density_map.logical_x_dimension * 3, output_3d.density_map.logical_y_dimension * 2, 1, true);
		output_3d.density_map.CreateOrthogonalProjectionsImage(&orth_image);
		orth_image.QuickAndDirtyWriteSlice(orthogonal_views_filename.ToStdString(), 1);
	}


	if (is_running_locally == false)
	{
		int number_of_points = gui_statistics->FSC.number_of_points;
		int array_size = (number_of_points * 5) + 2;
		wxPrintf("number of points = %i, class is %i, array size = %i\n", number_of_points, class_number_for_gui, array_size);

	    float *statistics = new float [array_size];
	    gui_statistics->WriteStatisticsToFloatArray(statistics, class_number_for_gui);
	    my_result.SetResult(array_size, statistics);
	    delete statistics;
	}

	wxPrintf("\nMerge3D: Normal termination\n\n");

	return true;
}
