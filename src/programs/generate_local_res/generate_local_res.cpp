#include "../../core/core_headers.h"

class AggregatedLocalResResult
{
public:
	int result_id;
	int start_slice;
	int end_slice;
	int x_dimension;
	int y_dimension;
	int z_dimension;
	int num_float_values; //WTW should be long?
	int number_of_meta_data_values;
	float pixel_size;
	float *float_values;

	AggregatedLocalResResult();
	~AggregatedLocalResResult();
};

WX_DECLARE_OBJARRAY(AggregatedLocalResResult, ArrayOfAggregatedLocalResResults);
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAggregatedLocalResResults);

class
	Generate_Local_Res_App : public MyApp
{
public:
	bool DoCalculation();
	void DoInteractiveUserInput();
	void MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results);

	wxString output_reconstruction;

	ArrayOfAggregatedLocalResResults aggregated_results;

private:
};

IMPLEMENT_APP(Generate_Local_Res_App)

void Generate_Local_Res_App::DoInteractiveUserInput()
{
	int number_of_threads;

	UserInput *my_input = new UserInput("Generate_Local_Res", 1.01);

	wxString half_map_1 = my_input->GetFilenameFromUser("Half Map 1", "The first output 3D reconstruction, calculated from half the data", "my_reconstruction_1.mrc", false);
	wxString half_map_2 = my_input->GetFilenameFromUser("Half Map 2", "The second output 3D reconstruction, calculated from half the data", "my_reconstruction_2.mrc", false);
	wxString output_reconstruction = my_input->GetFilenameFromUser("Output reconstruction", "The final 3D reconstruction, containing all data", "my_reconstruction.mrc", false);
	wxString mask_image = my_input->GetFilenameFromUser("Mask Image", "The filename of the mask", "mask_image.mrc", false);
	int first_slice = my_input->GetIntFromUser("Starting Slice", "The slice to start from", "1", false);
	int last_slice = my_input->GetIntFromUser("Last Slice", "The slice to end with", "1", false);
	float inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	float outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	float molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	wxString symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");
	float pixel_size = my_input->GetFloatFromUser("Pixel size", "In Angstroms", "1.0", 0.0);

#ifdef _OPENMP
	number_of_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	number_of_threads = 1;
#endif

	float measured_global_resolution = my_input->GetFloatFromUser("Measured Global Resolution", "In Angstroms", "5.0", 0.0);

	int image_number_for_gui = 0;
	int number_of_jobs_per_image_in_gui = 0;

	wxString directory_for_collated_results = "/dev/null/";
	wxString result_filename = "/dev/null"; // shouldn't be used in interactive

	delete my_input;

	//TODO change autorefine3d panel to match
	my_current_job.ManualSetArguments("ttttiiffftfifiitt", half_map_1.ToUTF8().data(),
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
									  measured_global_resolution,
									  image_number_for_gui,
									  number_of_jobs_per_image_in_gui,
									  directory_for_collated_results.ToUTF8().data(),
									  result_filename.ToUTF8().data());
}

bool Generate_Local_Res_App::DoCalculation()
{
	wxString half_map_1 = my_current_job.arguments[0].ReturnStringArgument();
	wxString half_map_2 = my_current_job.arguments[1].ReturnStringArgument();
	output_reconstruction = my_current_job.arguments[2].ReturnStringArgument();
	wxString mask_image_name = my_current_job.arguments[3].ReturnStringArgument();
	int first_slice = my_current_job.arguments[4].ReturnIntegerArgument();
	int last_slice = my_current_job.arguments[5].ReturnIntegerArgument();
	float inner_mask_radius = my_current_job.arguments[6].ReturnFloatArgument();
	float outer_mask_radius = my_current_job.arguments[7].ReturnFloatArgument();
	float molecular_mass_kDa = my_current_job.arguments[8].ReturnFloatArgument();
	wxString symmetry = my_current_job.arguments[9].ReturnStringArgument();
	float original_pixel_size = my_current_job.arguments[10].ReturnFloatArgument();
	int num_threads = my_current_job.arguments[11].ReturnIntegerArgument();
	float measured_global_resolution = my_current_job.arguments[12].ReturnFloatArgument();
	int image_number_for_gui = my_current_job.arguments[13].ReturnIntegerArgument();
	int number_of_jobs_per_image_in_gui = my_current_job.arguments[14].ReturnIntegerArgument();
	wxString directory_for_collated_results = my_current_job.arguments[15].ReturnStringArgument();
	wxString result_filename = my_current_job.arguments[16].ReturnStringArgument();

	if (is_running_locally == false)
		num_threads = number_of_threads_requested_on_command_line;

	int max_width = ceil(18 / original_pixel_size);
	wxPrintf("DEBUG WTW MAX WIDTH:%i:\n", max_width);
	wxPrintf("WTW DEBUG FIRST SLICE:%i:LAST SLICE:%i:\n", first_slice, last_slice);
	wxPrintf("WTW DEBUG mask name :%s:\n", mask_image_name.ToStdString());
	int num_slices = 0;

	ImageFile half_map_1_imagefile(half_map_1.ToStdString());
	ImageFile half_map_2_imagefile(half_map_2.ToStdString());
	ImageFile mask_image_imagefile(mask_image_name.ToStdString());
	int num_slices_half_map_1 = half_map_1_imagefile.ReturnNumberOfSlices();

	//read slices (keeping in mind window)
	Image half_map_1_image;
	Image half_map_2_image;
	Image mask_image;

	int first_used_slice = -1;
	int last_used_slice = -1;

	if (first_slice - max_width < 1)
	{
		if (last_slice + max_width > num_slices_half_map_1)
		{

			first_used_slice = 1;
			last_used_slice = num_slices_half_map_1;
			half_map_1_image.ReadSlices(&half_map_1_imagefile, 1, num_slices_half_map_1);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, 1, num_slices_half_map_1);
			mask_image.ReadSlices(&mask_image_imagefile, 1, num_slices_half_map_1);
			num_slices = num_slices_half_map_1;
		}
		else
		{

			first_used_slice = 1;
			last_used_slice = last_slice + max_width;
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

			first_used_slice = first_slice - max_width;
			last_used_slice = num_slices_half_map_1;
			half_map_1_image.ReadSlices(&half_map_1_imagefile, first_slice - max_width, num_slices_half_map_1);
			half_map_2_image.ReadSlices(&half_map_2_imagefile, first_slice - max_width, num_slices_half_map_1);
			mask_image.ReadSlices(&mask_image_imagefile, first_slice - max_width, num_slices_half_map_1);
			num_slices = num_slices_half_map_1 - (first_slice - max_width) + 1;
		}
		else
		{

			first_used_slice = first_slice - max_width;
			last_used_slice = last_slice + max_width;
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

		////DEBUGGING WTW
		for (int i = 0; i < 64; i++)
		{
			int first_slice_d = (first_slice - 1) + myroundint(i * slices_per_thread) + 1;
			int last_slice_d = (first_slice - 1) + myroundint((i + 1) * slices_per_thread);

			wxPrintf("WTW DEBUG FIRST SLICE P:%i:LAST SLICE P:%i:\n", first_slice_d, last_slice_d);
		}
		exit(1);
		////

#pragma omp parallel default(shared) num_threads(num_threads)
		{

			int first_slice_p = (first_slice - 1) + myroundint(ReturnThreadNumberOfCurrentThread() * slices_per_thread) + 1;
			int last_slice_p = (first_slice - 1) + myroundint((ReturnThreadNumberOfCurrentThread() + 1) * slices_per_thread);

			wxPrintf("WTW DEBUG FIRST SLICE P:%i:LAST SLICE P:%i:\n", first_slice_p, last_slice_p);

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

	MyDebugPrint("About to proccess data for SendProgramDefinedResultToMaster\n");

	//TODO there might be 2 times as many floats as expected
	int number_of_meta_data_values = 8;
	int num_slices_for_results = last_slice - (first_slice - 1);
	long number_of_valid_floats = num_slices_for_results * (combined_images.real_memory_allocated / combined_images.logical_z_dimension);
	long offest_to_valid_values = (first_slice - first_used_slice) * (combined_images.real_memory_allocated / combined_images.logical_z_dimension);
	long number_of_result_floats = number_of_meta_data_values + number_of_valid_floats;
	float *result = new float[number_of_result_floats];

	MyDebugPrint("Number of Floats:%li:Combined Images Real Mem Allocated:%li:\n", number_of_result_floats, combined_images.real_memory_allocated);
	MyDebugPrint("DEBUG 1 for SendProgramDefinedResultToMaster\n");

	result[0] = first_slice;
	result[1] = last_slice;
	result[2] = combined_images.logical_x_dimension;
	result[3] = combined_images.logical_y_dimension;
	result[4] = num_slices_for_results;
	result[5] = number_of_result_floats; //long to float?
	result[6] = number_of_meta_data_values;
	result[7] = original_pixel_size;

	MyDebugPrint("DEBUG 2 for SendProgramDefinedResultToMaster\n");

	for (int result_array_counter = 0; result_array_counter < number_of_valid_floats; result_array_counter++)
	{
		result[result_array_counter + number_of_meta_data_values] = combined_images.real_values[offest_to_valid_values + result_array_counter];
	}

	//WTW debug
	Image result_image;
	result_image.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, num_slices_for_results, true, true);

	for (int result_array_counter = 0; result_array_counter < number_of_valid_floats; result_array_counter++)
	{
		result_image.real_values[result_array_counter] = combined_images.real_values[offest_to_valid_values + result_array_counter];
	}

	MyDebugPrint("First Slice:%i:LastSlice:%i:\n", first_slice, last_slice);
	MyDebugPrint("num_slices_for_results:%i:combined_images.logical_z_dimension:%i:\n", num_slices_for_results, combined_images.logical_z_dimension);
	MyDebugPrint("DEBUG WTW MAX WIDTH:%i:\n", max_width);
	result_image.WriteSlicesAndFillHeader("/data/wtwoods/test_local_filtering/debug_test_result_images.mrc", original_pixel_size);
	//WTW debug

	MyDebugPrint("About to pass data back with SendProgramDefinedResultToMaster\n");

	SendProgramDefinedResultToMaster(result, number_of_result_floats, image_number_for_gui, number_of_jobs_per_image_in_gui);

	MyDebugPrint("\nGenerate Local Res: Normal termination\n\n");

	return true;
}

AggregatedLocalResResult::AggregatedLocalResResult()
{
	result_id = -1;
	start_slice = -1;
	end_slice = -1;
	x_dimension = -1;
	y_dimension = -1;
	z_dimension = -1;
	num_float_values = -1;
	number_of_meta_data_values = -1;
	pixel_size = -1;
	float_values = NULL;
}

AggregatedLocalResResult::~AggregatedLocalResResult()
{
	// is this neccesary WTW
	// if (result != NULL)
	// 	delete[] result;
}

void Generate_Local_Res_App::MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results)
{
	MyDebugPrint("WTW DEBUG MASTER 1\n");
	AggregatedLocalResResult result_to_add;
	aggregated_results.Add(result_to_add);
	aggregated_results[aggregated_results.GetCount() - 1].result_id = result_number;
	aggregated_results[aggregated_results.GetCount() - 1].start_slice = result_array[0];
	aggregated_results[aggregated_results.GetCount() - 1].end_slice = result_array[1];
	aggregated_results[aggregated_results.GetCount() - 1].x_dimension = result_array[2];
	aggregated_results[aggregated_results.GetCount() - 1].y_dimension = result_array[3];
	aggregated_results[aggregated_results.GetCount() - 1].z_dimension = result_array[4];
	aggregated_results[aggregated_results.GetCount() - 1].num_float_values = result_array[5];
	aggregated_results[aggregated_results.GetCount() - 1].number_of_meta_data_values = result_array[6];
	aggregated_results[aggregated_results.GetCount() - 1].pixel_size = result_array[7];
	aggregated_results[aggregated_results.GetCount() - 1].float_values = &result_array[8]; //should act like a sub array of all non-metadata values
																						   //may need to copy them? mem disapears? is it on the heap ig
	int array_location = aggregated_results.GetCount() - 1;

	MyDebugPrint("WTW DEBUG MASTER 2\n");

	if (aggregated_results.GetCount() == number_of_expected_results)
	{
		MyDebugPrint("WTW DEBUG MASTER 3\n");
		//get total z dim
		int x_dim = aggregated_results[aggregated_results.GetCount() - 1].x_dimension;
		int y_dim = aggregated_results[aggregated_results.GetCount() - 1].y_dimension;
		int z_dim = 0;
		long num_floats_per_slice = x_dim * y_dim;
		int min_start_slice = aggregated_results[aggregated_results.GetCount() - 1].start_slice;

		MyDebugPrint("WTW DEBUG MASTER 4\n");

		for (int results_counter = 0; results_counter < aggregated_results.GetCount(); results_counter++)
		{
			z_dim += aggregated_results[results_counter].z_dimension;
			if (aggregated_results[results_counter].start_slice < min_start_slice)
			{
				min_start_slice = aggregated_results[results_counter].start_slice;
			}
		}

		MyDebugPrint("WTW DEBUG MASTER 5\n");

		Image result_image;
		result_image.Allocate(x_dim, y_dim, z_dim, true, true);
		MyDebugPrint("x:%i:, y:%i:, z:%i:\n", x_dim, y_dim, z_dim);

		for (int results_counter = 0; results_counter < aggregated_results.GetCount(); results_counter++)
		{

			// Image individual_image;
			// result_image.Allocate(x_dim, y_dim, aggregated_results[results_counter].z_dimension, true, true);
			// result_image.real_values = aggregated_results[results_counter].float_values; //should this be a deep copy

			int start_slice_temp = aggregated_results[results_counter].start_slice - min_start_slice;
			MyDebugPrint("start_slice_temp:%i:num_floats_per_slice:%li:aggregated_results[results_counter].num_float_values:%i:\n", start_slice_temp, num_floats_per_slice, aggregated_results[results_counter].num_float_values);
			MyDebugPrint("result_image.real_memory_allocated:%li:\n", result_image.real_memory_allocated);

			MyDebugPrint("WTW DEBUG MASTER 6\n");

			int number_of_non_meta_floats = aggregated_results[results_counter].num_float_values - aggregated_results[results_counter].number_of_meta_data_values;
			for (int float_counter = 0; float_counter < number_of_non_meta_floats; float_counter++)
			{
				result_image.real_values[(start_slice_temp * num_floats_per_slice) + float_counter] = aggregated_results[results_counter].float_values[float_counter];
			}
		}

		MyDebugPrint("WTW DEBUG MASTER 7\n");
		result_image.WriteSlicesAndFillHeader(output_reconstruction.ToStdString(), aggregated_results[aggregated_results.GetCount() - 1].pixel_size);
	}
}
