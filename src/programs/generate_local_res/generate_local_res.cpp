#include "../../core/core_headers.h"

class
Generate_Local_Res_App : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

//IMPLEMENT_APP(Generate_Local_Res_App)

void Generate_Local_Res_App::DoInteractiveUserInput()
{

    wxString half_map_1;
    wxString half_map_2;
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
    first_slice = my_input->GetIntFromUser("Starting Slice", "The slice to start from", "1", false);
    last_slice = my_input->GetIntFromUser("Ending Slice", "The slice to end with", "2", false);
    pixel_size = my_input->GetFloatFromUser("Pixel size of reconstruction (A)", "Pixel size of the output reconstruction in Angstroms", "1.0", 0.0);\
    inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
    molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
    symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");

    //TODO get number of threads from the gui? 

    delete my_input;

    my_current_job.ManualSetArguments("ttiifffft", half_map_1.ToUTF8().data(), half_map_2.ToUTF8().data(), first_slice, last_slice, pixel_size, inner_mask_radius, outer_mask_radius, molecular_mass_kDa, symmetry);

}

bool Generate_Local_Res_App::DoCalculation() 
{
    //TODO
    // 1. WHAT TO DO WITH HALF MAPS 
    // 2. NUMBER OF THREADS FROM THE GUI 
    // 3. MY_RECONSTRUCTION_1
    wxString half_map_1			= my_current_job.arguments[0].ReturnStringArgument();
	wxString half_map_2			= my_current_job.arguments[1].ReturnStringArgument();
    int first_slice			    = my_current_job.arguments[2].ReturnIntegerArgument();
    int last_slice			    = my_current_job.arguments[3].ReturnIntegerArgument();
    float original_pixel_size   = my_current_job.arguments[4].ReturnFloatArgument();
    float inner_mask_radius     = my_current_job.arguments[5].ReturnFloatArgument();
    float outer_mask_radius     = my_current_job.arguments[6].ReturnFloatArgument();
    float molecular_mass_kDa    = my_current_job.arguments[7].ReturnFloatArgument();
    wxString symmetry           = my_current_job.arguments[8].ReturnStringArgument();

    //uh? 
    ReconstructedVolume output_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d1(molecular_mass_kDa);
	ReconstructedVolume output_3d2(molecular_mass_kDa);
    //

    Image local_resolution_volume;
    local_resolution_volume.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
	local_resolution_volume.SetToConstant(0.0f);

    Image original_volume;

    Image local_resolution_volume_all;
	local_resolution_volume_all.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
	local_resolution_volume_all.SetToConstant(0.0f);

    int box_size;
    box_size = 18.0f / original_pixel_size;
    const float threshold_snr = 1;
    const float threshold_confidence = 2.0;
    float fixed_fsc_threshold = .9;
    const bool use_fixed_threshold = true;
    int number_averaged = 0;

    #pragma omp parallel default(shared) num_threads(number_of_threads)
	{
        Image local_resolution_volume_local;
        Image input_volume_one_local;
        Image input_volume_two_local;

        input_volume_one_local.CopyFrom(output_3d1.density_map);
        input_volume_two_local.CopyFrom(output_3d2.density_map);

        Image mask_image;
        mask_image.CopyFrom(output_3d.density_map);
        float original_average_value = mask_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
        mask_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f);

        local_resolution_volume_local.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
        local_resolution_volume_local.SetToConstant(0.0f);

        LocalResolutionEstimator *estimator = new LocalResolutionEstimator();
        estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &mask_image, first_slice,last_slice, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold,symmetry,true,2);
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

	// divide and copy

	local_resolution_volume_all.DivideByConstant(number_averaged);
	local_resolution_volume.CopyFrom(&local_resolution_volume_all);

    // Image local_resolution_volume;
	// Image original_volume;
	// int box_size;
    // box_size = 18.0f / original_pixel_size;

	// wxPrintf("Will estimate local resolution using a box size of %i\n",box_size);
	// const float threshold_snr = 1;
	// const float threshold_confidence = 2.0;
	// float fixed_fsc_threshold = .9;
	// const bool use_fixed_threshold = true;

	// MyDebugPrint("About to estimate loc res, with %.2f cutoff\n", fixed_fsc_threshold);
	// if (! output_3d1.density_map->is_in_real_space) output_3d1.density_map->BackwardFFT();
	// if (! output_3d2.density_map->is_in_real_space) output_3d2.density_map->BackwardFFT();

	// original_volume.CopyFrom(output_3d.density_map);
	// output_3d.density_map->QuickAndDirtyWriteSlices("/tmp/locres_original.mrc", 1, output_3d.density_map->logical_z_dimension, true, original_pixel_size);
    // Image mask_image;
    // mask_image.CopyFrom(output_3d.density_map);

    // float original_average_value = mask_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
    // mask_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f);

	// local_resolution_volume.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
	// local_resolution_volume.SetToConstant(0.0f);

	// Image local_resolution_volume_all;
	// local_resolution_volume_all.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
	// local_resolution_volume_all.SetToConstant(0.0f);

	// int first_slice_with_data;
	// int last_slice_with_data;

	// Image slice_image;

	// for (int counter = 1; counter <= local_resolution_volume.logical_z_dimension; counter++)
	// {
	// 	slice_image.AllocateAsPointingToSliceIn3D(&mask_image, counter);
	// 	if (slice_image.IsConstant() == false)
	// 	{
	// 		first_slice_with_data = counter;
	// 		break;
	// 	}
	// }

	// for (int counter = local_resolution_volume.logical_z_dimension; counter >= 1; counter--)
	// {
	// 	slice_image.AllocateAsPointingToSliceIn3D(&mask_image, counter);
	// 	if (slice_image.IsConstant() == false)
	// 	{
	// 		last_slice_with_data = counter;
	// 		break;
	// 	}
	// }

	// int slices_with_data = last_slice_with_data - first_slice_with_data;
	// float slices_per_thread = slices_with_data / float(number_of_threads);
	// int number_averaged = 0;

	// for (float current_res = 18.0f; current_res < 37.0f; current_res += 6.0f)
	// {
	// //	float current_res = 24;
	// 	box_size = current_res / original_pixel_size;

	// 	if (alignment_res > 15) fixed_fsc_threshold = 0.75;
	// 	else
	// 	if (alignment_res > 8) fixed_fsc_threshold = 0.85;
	// 	else
	// 	if (alignment_res > 6) fixed_fsc_threshold = 0.9;
	// 	else
	// 	fixed_fsc_threshold = 0.95f;

	// 	local_resolution_volume.SetToConstant(0.0f);

	// 	#pragma omp parallel default(shared) num_threads(number_of_threads)
	// 	{ // for omp

	// 		int first_slice = (first_slice_with_data - 1) + myroundint(ReturnThreadNumberOfCurrentThread() * slices_per_thread) + 1;
	// 		int last_slice = (first_slice_with_data - 1) +  myroundint((ReturnThreadNumberOfCurrentThread() + 1) * slices_per_thread);

	// 		Image local_resolution_volume_local;
	// 		Image input_volume_one_local;
	// 		Image input_volume_two_local;

	// 		input_volume_one_local.CopyFrom(output_3d1.density_map);
	// 		input_volume_two_local.CopyFrom(output_3d2.density_map);

	// 		local_resolution_volume_local.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
	// 		local_resolution_volume_local.SetToConstant(0.0f);
	// 		LocalResolutionEstimator *estimator = new LocalResolutionEstimator();
	// 		estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &mask_image, first_slice,last_slice, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold,my_reconstruction_1.symmetry_matrices.symmetry_symbol,true,2);
	// 		estimator->EstimateLocalResolution(&local_resolution_volume_local);
	// 		delete estimator;

	// 		#pragma omp critical
	// 		{
	// 			for (long pixel_counter = 0; pixel_counter < local_resolution_volume.number_of_real_space_pixels; pixel_counter++)
	// 			{
	// 				if (local_resolution_volume_local.real_values[pixel_counter] != 0.0f) local_resolution_volume.real_values[pixel_counter] = local_resolution_volume_local.real_values[pixel_counter];
	// 			}

	// 		}
	// 	} // end omp

	// 	local_resolution_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/local_res_%i", int(current_res)).ToStdString(), 1, local_resolution_volume.logical_z_dimension);

	// 	// fill in gaps..

	// 	float max_res = local_resolution_volume.ReturnMaximumValue();

	// 	for (long pixel_counter = 0; pixel_counter < local_resolution_volume.real_memory_allocated; pixel_counter++)
	// 	{
	// 		if (local_resolution_volume.real_values[pixel_counter] < 0.5f) local_resolution_volume.real_values[pixel_counter] = max_res;
	// 	}

	// 	local_resolution_volume_all.AddImage(&local_resolution_volume);
	// 	number_averaged++;
	// }

	// // divide and copy

	// local_resolution_volume_all.DivideByConstant(number_averaged);
	// local_resolution_volume.CopyFrom(&local_resolution_volume_all);


}