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
    int first_slice;
	int last_slice;
    float pixel_size;
    float inner_mask_radius;
    float outer_mask_radius;
    float molecular_mass_kDa;
    int number_of_threads;
    wxString symmetry;
	

    UserInput *my_input = new UserInput("Generate_Local_Res", 1.01);

    half_map_1 = my_input->GetFilenameFromUser("Half Map 1", "The first output 3D reconstruction, calculated from half the data", "my_reconstruction_1.mrc", false);
	half_map_2 = my_input->GetFilenameFromUser("Half Map 2", "The second output 3D reconstruction, calculated from half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction = my_input->GetFilenameFromUser("Output reconstruction", "The final 3D reconstruction, containing from all data", "my_reconstruction.mrc", false);
    first_slice = my_input->GetIntFromUser("Starting Slice", "The slice to start from", "1", false);
	last_slice = my_input->GetIntFromUser("Last Slice", "The slice to end with", "1", false);
    inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
    molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
    symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");
	pixel_size = my_input->GetFloatFromUser("Pixel size","In Angstroms","1.0",0.0);

	#ifdef _OPENMP
	number_of_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	number_of_threads = 1;
#endif

    delete my_input;

    my_current_job.ManualSetArguments("tttiiffftfi", half_map_1.ToUTF8().data(), half_map_2.ToUTF8().data(), output_reconstruction.ToUTF8().data(), first_slice, last_slice, inner_mask_radius, outer_mask_radius, molecular_mass_kDa, symmetry.ToUTF8().data(), pixel_size, number_of_threads);

}

bool Generate_Local_Res_App::DoCalculation() 
{
    //TODO
	// 4. pixel size vs original pixel size
    wxString half_map_1				= my_current_job.arguments[0].ReturnStringArgument();
	wxString half_map_2				= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_reconstruction 	= my_current_job.arguments[2].ReturnStringArgument();
    int first_slice			    	= my_current_job.arguments[3].ReturnIntegerArgument();
	int last_slice			    	= my_current_job.arguments[4].ReturnIntegerArgument();
    float inner_mask_radius     	= my_current_job.arguments[5].ReturnFloatArgument();
    float outer_mask_radius     	= my_current_job.arguments[6].ReturnFloatArgument();
    float molecular_mass_kDa    	= my_current_job.arguments[7].ReturnFloatArgument();
    wxString symmetry           	= my_current_job.arguments[8].ReturnStringArgument();
	float original_pixel_size		= my_current_job.arguments[9].ReturnFloatArgument();
	int	 num_threads				= my_current_job.arguments[10].ReturnIntegerArgument();

	//get num slices (may need to use MRCFile?)
	ImageFile half_map_1_imagefile(half_map_1.ToStdString());
	int num_slices_half_map_1 = half_map_1_imagefile.ReturnNumberOfSlices();
	ImageFile half_map_2_imagefile(half_map_2.ToStdString());
	int num_slices_half_map_2 = half_map_2_imagefile.ReturnNumberOfSlices();

	//read slices 
	Image half_map_1_image; 
	Image half_map_2_image;
	half_map_1_image.ReadSlices(&half_map_1_imagefile, 1, num_slices_half_map_1);
	half_map_2_image.ReadSlices(&half_map_2_imagefile, 1, num_slices_half_map_2);

	//combie the two image files 
	Image combined_images(half_map_1_image);
	combined_images.AddImage(&half_map_2_image);
	combined_images.DivideByConstant(2); 

	float		mask_falloff = 10.0;

	if (is_running_locally == false) num_threads = number_of_threads_requested_on_command_line;

    //initialize images
    Image local_resolution_volume;
    local_resolution_volume.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
	local_resolution_volume.SetToConstant(0.0f);

    Image local_resolution_volume_all;
	local_resolution_volume_all.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
	local_resolution_volume_all.SetToConstant(0.0f);

	// int first_slice_with_data;
	// int last_slice_with_data;

	// Image slice_image;

	// for (int counter = 1; counter <= local_resolution_volume.logical_z_dimension; counter++)
	// {
	// 	slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
	// 	if (slice_image.IsConstant() == false)
	// 	{
	// 		first_slice_with_data = counter;
	// 		break;
	// 	}
	// }

	// for (int counter = local_resolution_volume.logical_z_dimension; counter >= 1; counter--)
	// {
	// 	slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
	// 	if (slice_image.IsConstant() == false)
	// 	{
	// 		last_slice_with_data = counter;
	// 		break;
	// 	}
	// }

	// int slices_with_data = last_slice_with_data - first_slice_with_data;
	// float slices_per_thread = slices_with_data / float(num_threads);
	// int number_averaged = 0;

    //declare and initialize constants and other variables 
    int box_size;
    box_size = 18.0f / original_pixel_size;
    const float threshold_snr = 1;
    const float threshold_confidence = 2.0;
    float fixed_fsc_threshold = .9;
    const bool use_fixed_threshold = true;
    int number_averaged = 0;
	float alignment_res = 5.0f;

	for (float current_res = 18.0f; current_res < 37.0f; current_res += 6.0f)
	{
	//	float current_res = 24;
		box_size = current_res / original_pixel_size;

		if (alignment_res > 15) fixed_fsc_threshold = 0.75;
		else
		if (alignment_res > 8) fixed_fsc_threshold = 0.85;
		else
		if (alignment_res > 6) fixed_fsc_threshold = 0.9;
		else
		fixed_fsc_threshold = 0.95f;

		local_resolution_volume.SetToConstant(0.0f);
    
		#pragma omp parallel default(shared) num_threads(num_threads)
		{
			Image local_resolution_volume_local;
			Image input_volume_one_local;
			Image input_volume_two_local;

			input_volume_one_local.CopyFrom(&half_map_1_image); 
			input_volume_two_local.CopyFrom(&half_map_2_image);

			Image mask_image;
			mask_image.CopyFrom(&combined_images);
			float original_average_value = mask_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
			mask_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f);

			local_resolution_volume_local.Allocate(combined_images.logical_x_dimension, combined_images.logical_y_dimension, combined_images.logical_z_dimension);
			local_resolution_volume_local.SetToConstant(0.0f);

			LocalResolutionEstimator *estimator = new LocalResolutionEstimator();
			estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &mask_image, first_slice, num_slices_half_map_1, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold,symmetry,true,2); //paralelization bug here TODO
			estimator->EstimateLocalResolution(&local_resolution_volume_local);
			delete estimator;

			#pragma omp critical
			{
				for (long pixel_counter = 0; pixel_counter < local_resolution_volume.number_of_real_space_pixels; pixel_counter++)
				{
					if (local_resolution_volume_local.real_values[pixel_counter] != 0.0f) local_resolution_volume.real_values[pixel_counter] = local_resolution_volume_local.real_values[pixel_counter];
				}

			}
		} 

		float max_res = local_resolution_volume.ReturnMaximumValue();

		for (long pixel_counter = 0; pixel_counter < local_resolution_volume.real_memory_allocated; pixel_counter++)
		{
			if (local_resolution_volume.real_values[pixel_counter] < 0.5f) local_resolution_volume.real_values[pixel_counter] = max_res;
		}

		local_resolution_volume_all.AddImage(&local_resolution_volume);
		number_averaged++;

	}

	// divide and copy

	local_resolution_volume_all.DivideByConstant(number_averaged);
	local_resolution_volume.CopyFrom(&local_resolution_volume_all);

	local_resolution_volume.WriteSlicesAndFillHeader(output_reconstruction.ToStdString(), original_pixel_size);

	return true;
}

