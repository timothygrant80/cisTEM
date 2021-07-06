#include "../../core/core_headers.h"

class
	Generate_Local_Res_App : public MyApp
{
public:
	bool DoCalculation();
	void DoInteractiveUserInput();

private:
};

IMPLEMENT_APP(Generate_Local_Res_App)

void Generate_Local_Res_App::DoInteractiveUserInput()
{

	wxString half_map_1;
	wxString half_map_2;
	wxString output_reconstruction;
	wxString mask_image;
	int first_slice;
	int last_slice;
	float pixel_size;
	float inner_mask_radius;
	float outer_mask_radius;
	float molecular_mass_kDa;
	int number_of_threads;
	wxString symmetry;
	float measured_global_resolution;

	UserInput *my_input = new UserInput("Generate_Local_Res", 1.01);

	half_map_1 = my_input->GetFilenameFromUser("Half Map 1", "The first output 3D reconstruction, calculated from half the data", "my_reconstruction_1.mrc", false);
	half_map_2 = my_input->GetFilenameFromUser("Half Map 2", "The second output 3D reconstruction, calculated from half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction = my_input->GetFilenameFromUser("Output reconstruction", "The final 3D reconstruction, containing all data", "my_reconstruction.mrc", false);
	mask_image = my_input->GetFilenameFromUser("Mask Image", "The filename of the mask", "mask_image.mrc", false);
	first_slice = my_input->GetIntFromUser("Starting Slice", "The slice to start from", "1", false);
	last_slice = my_input->GetIntFromUser("Last Slice", "The slice to end with", "1", false);
	inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");
	pixel_size = my_input->GetFloatFromUser("Pixel size", "In Angstroms", "1.0", 0.0);
	measured_global_resolution = my_input->GetFloatFromUser("Measured Global Resolution", "In Angstroms", "5.0", 0.0);

#ifdef _OPENMP
	number_of_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	number_of_threads = 1;
#endif

	delete my_input;

	my_current_job.ManualSetArguments("ttttiiffftfif", half_map_1.ToUTF8().data(),
									  half_map_2.ToUTF8().data(),
									  output_reconstruction.ToUTF8().data(),
									  mask_image.ToUTF8().data(),
									  first_slice,
									  last_slice,
									  inner_mask_radius,
									  outer_mask_radius,
									  molecular_mass_kDa,
									  symmetry.ToUTF8().data(),
									  pixel_size,
									  number_of_threads,
									  measured_global_resolution);
}

bool Generate_Local_Res_App::DoCalculation()
{
	//TODO
	// need to read in slices that are relavant
	// need to ask for mask file name, and only right slices of mask
	wxString half_map_1 = my_current_job.arguments[0].ReturnStringArgument();
	wxString half_map_2 = my_current_job.arguments[1].ReturnStringArgument();
	wxString output_reconstruction = my_current_job.arguments[2].ReturnStringArgument();
	wxString mask_image_name = my_current_job.arguments[3].ReturnStringArgument();
	int first_slice = my_current_job.arguments[4].ReturnIntegerArgument();
	int last_slice = my_current_job.arguments[5].ReturnIntegerArgument();
	float inner_mask_radius = my_current_job.arguments[12].ReturnFloatArgument();
	float outer_mask_radius = my_current_job.arguments[11].ReturnFloatArgument();
	float molecular_mass_kDa = my_current_job.arguments[6].ReturnFloatArgument();
	wxString symmetry = my_current_job.arguments[7].ReturnStringArgument();
	float original_pixel_size = my_current_job.arguments[8].ReturnFloatArgument();
	int num_threads = my_current_job.arguments[9].ReturnIntegerArgument();
	float measured_global_resolution = my_current_job.arguments[10].ReturnFloatArgument();

	if (is_running_locally == false)
		num_threads = number_of_threads_requested_on_command_line;

	int max_width = ceil(18 / original_pixel_size);
	int num_slices = 0;

	ImageFile half_map_1_imagefile(half_map_1.ToStdString());
	ImageFile half_map_2_imagefile(half_map_2.ToStdString());
	ImageFile mask_image_imagefile(mask_image_name.ToStdString());
	int num_slices_half_map_1 = half_map_1_imagefile.ReturnNumberOfSlices();

	//read slices (keeping in mind window)
	Image half_map_1_image;
	Image half_map_2_image;
	Image mask_image;

	if (first_slice - max_width < 1)
	{
		if (last_slice + max_width > num_slices_half_map_1)
		{
			half_map_1_image.ReadSlices(&half_map_1_imagefile, 1, num_slices_half_map_1);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, 1, num_slices_half_map_1);
			mask_image.ReadSlices(&mask_image_imagefile, 1, num_slices_half_map_1);
			num_slices = num_slices_half_map_1;
		}
		else
		{
			half_map_1_image.ReadSlices(&half_map_1_imagefile, 1, last_slice + max_width);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, 1, last_slice + max_width);
			mask_image.ReadSlices(&mask_image_imagefile, 1, last_slice + max_width);
			num_slices = last_slice + max_width;
		}
	}
	else
	{
		if (last_slice + max_width > num_slices_half_map_1)
		{
			half_map_1_image.ReadSlices(&half_map_1_imagefile, first_slice - max_width, num_slices_half_map_1);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, first_slice - max_width, num_slices_half_map_1);
			mask_image.ReadSlices(&mask_image_imagefile, first_slice - max_width, num_slices_half_map_1);
			num_slices = num_slices_half_map_1 - (first_slice - max_width) + 1;
		}
		else
		{
			half_map_1_image.ReadSlices(&half_map_1_imagefile, first_slice - max_width, last_slice + max_width);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, first_slice - max_width, last_slice + max_width);
			mask_image.ReadSlices(&mask_image_imagefile, first_slice - max_width, last_slice + max_width);
			num_slices = last_slice + max_width - (first_slice - max_width) + 1;
		}
	}

	if (num_threads > num_slices)
	{
		num_threads = num_slices;
	}

	//combie the two image files (need to do this in gui for correct mask)
	Image combined_images(half_map_1_image);
	combined_images.AddImage(&half_map_2_image);
	combined_images.DivideByConstant(2);

	float mask_falloff = 10.0;

	if (is_running_locally == false)
		num_threads = number_of_threads_requested_on_command_line;

	//initialize images
	Image local_resolution_volume;
	local_resolution_volume.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
	local_resolution_volume.SetToConstant(0.0f);

	Image local_resolution_volume_all;
	local_resolution_volume_all.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
	local_resolution_volume_all.SetToConstant(0.0f);

	Image original_volume;
	original_volume.CopyFrom(&combined_images);

	//declare and initialize constants and other variables
	int box_size;
	box_size = 18.0f / original_pixel_size;
	const float threshold_snr = 1;
	const float threshold_confidence = 2.0;
	float fixed_fsc_threshold = .9;
	const bool use_fixed_threshold = true;
	int number_averaged = 0;
	float alignment_res = 5.0f;

	int total_slices = last_slice - first_slice;
	float slices_per_thread = total_slices / float(num_threads);

	for (float current_res = 18.0f; current_res < 37.0f; current_res += 6.0f)
	{
		//	float current_res = 24;
		box_size = current_res / original_pixel_size;

		if (alignment_res > 15)
			fixed_fsc_threshold = 0.75;
		else if (alignment_res > 8)
			fixed_fsc_threshold = 0.85;
		else if (alignment_res > 6)
			fixed_fsc_threshold = 0.9;
		else
			fixed_fsc_threshold = 0.95f;

		local_resolution_volume.SetToConstant(0.0f);

#pragma omp parallel default(shared) num_threads(num_threads)
		{

			int first_slice_p = (first_slice - 1) + myroundint(ReturnThreadNumberOfCurrentThread() * slices_per_thread) + 1;
			int last_slice_p = (last_slice - 1) + myroundint((ReturnThreadNumberOfCurrentThread() + 1) * slices_per_thread);

			Image local_resolution_volume_local;
			Image input_volume_one_local;
			Image input_volume_two_local;

			input_volume_one_local.CopyFrom(&half_map_1_image);
			input_volume_two_local.CopyFrom(&half_map_2_image);

			local_resolution_volume_local.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
			local_resolution_volume_local.SetToConstant(0.0f);

			LocalResolutionEstimator *estimator = new LocalResolutionEstimator();
			estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &mask_image, first_slice_p, last_slice_p, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold, symmetry, true, 2);
			estimator->EstimateLocalResolution(&local_resolution_volume_local);
			delete estimator;

#pragma omp critical
			{
				for (long pixel_counter = 0; pixel_counter < local_resolution_volume.number_of_real_space_pixels; pixel_counter++)
				{
					if (local_resolution_volume_local.real_values[pixel_counter] != 0.0f)
						local_resolution_volume.real_values[pixel_counter] = local_resolution_volume_local.real_values[pixel_counter];
				}
			}
		}

		float max_res = local_resolution_volume.ReturnMaximumValue();
		MyPrintfRed("Max_Res: %f", max_res);

		for (long pixel_counter = 0; pixel_counter < local_resolution_volume.real_memory_allocated; pixel_counter++)
		{
			if (local_resolution_volume.real_values[pixel_counter] < 0.5f)
				local_resolution_volume.real_values[pixel_counter] = max_res;
		}

		local_resolution_volume_all.AddImage(&local_resolution_volume);
		number_averaged++;
	}

	// divide and copy

	local_resolution_volume_all.DivideByConstant(number_averaged);
	local_resolution_volume.CopyFrom(&local_resolution_volume_all);

	local_resolution_volume.WriteSlicesAndFillHeader("local_res_vol_test.mrc", original_pixel_size);

	//scaling code

	int number_of_top_pixels_to_use = local_resolution_volume.number_of_real_space_pixels * 0.00001;
	if (number_of_top_pixels_to_use < 50)
		number_of_top_pixels_to_use = 50;

	float highest_resolution = local_resolution_volume.ReturnAverageOfMinN(number_of_top_pixels_to_use);
	// measured_global_resolution = resolution_statistics->ReturnEstimatedResolution(true); //arg measured global res

	float average_resolution = 0.0f;
	long voxels_in_the_mask = 0;

	int i, j, k;
	long pixel_counter = 0;

	for (k = 0; k < local_resolution_volume.logical_z_dimension; k++)
	{
		for (j = 0; j < local_resolution_volume.logical_y_dimension; j++)
		{
			for (i = 0; i < local_resolution_volume.logical_x_dimension; i++)
			{
				if (mask_image.real_values[pixel_counter] == 1.0f)
				{
					//if (local_resolution_volume.real_values[pixel_counter] < highest_resolution) highest_resolution = local_resolution_volume.real_values[pixel_counter];
					average_resolution += local_resolution_volume.real_values[pixel_counter];
					voxels_in_the_mask++;
				}

				pixel_counter++;
			}
			pixel_counter += local_resolution_volume.padding_jump_value;
		}
	}

	average_resolution /= voxels_in_the_mask;
	wxPrintf("Local high / Measured Average / Local Average = %.2f / %.2f / %.2f\n", highest_resolution, measured_global_resolution, average_resolution);

	if (highest_resolution != 8.0f && measured_global_resolution != 8.0f)
	{
		float scaler = (8.0f - measured_global_resolution) / (8.0f - highest_resolution);

		pixel_counter = 0;
		for (k = 0; k < local_resolution_volume.logical_z_dimension; k++)
		{
			for (j = 0; j < local_resolution_volume.logical_y_dimension; j++)
			{
				for (i = 0; i < local_resolution_volume.logical_x_dimension; i++)
				{
					if (mask_image.real_values[pixel_counter] == 1.0f)
					{
						if (local_resolution_volume.real_values[pixel_counter] < 8.0f)
						{
							if (scaler > 1.0f)
								local_resolution_volume.real_values[pixel_counter] = ((local_resolution_volume.real_values[pixel_counter] - highest_resolution) * scaler) + measured_global_resolution;
							if (local_resolution_volume.real_values[pixel_counter] < 5.0f)
							{
								local_resolution_volume.real_values[pixel_counter] = measured_global_resolution;
							}
						}
					}

					pixel_counter++;
				}
				pixel_counter += local_resolution_volume.padding_jump_value;
			}
		}
	}

	local_resolution_volume.SetMaximumValue(20.0f);

	//local_resolution_volume.WriteSlicesAndFillHeader(output_reconstruction.ToStdString(), original_pixel_size);

	MyDebugPrint("About to apply locres filter\n");

	int number_of_levels = box_size;
	combined_images.ApplyLocalResolutionFilter(local_resolution_volume, original_pixel_size, number_of_levels);

	for (long address = 0; address < combined_images.real_memory_allocated; address++)
	{
		//	if (size_image.real_values[address] == 0.0f) output_3d.density_map->real_values[address] = original_average_value;

		// go back to original density if high res..

		if (local_resolution_volume.real_values[address] <= measured_global_resolution + (measured_global_resolution * 0.1))
		{
			combined_images.real_values[address] = original_volume.real_values[address];
		}
	}

	//output_3d.density_map->SetMinimumValue(original_average_value);
	combined_images.CosineMask(outer_mask_radius / original_pixel_size, 1.0, false, true, 0.0);

	//passing back more than asked for! TODO
	//combined_images.QuickAndDirtyWriteSlices(output_reconstruction.ToStdString(), first_slice, last_slice);
	combined_images.WriteSlicesAndFillHeader(output_reconstruction.ToStdString(), original_pixel_size);

	return true;
}
