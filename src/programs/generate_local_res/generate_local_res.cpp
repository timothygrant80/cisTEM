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
    inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
    molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
    symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");
	pixel_size = my_input->GetFloatFromUser("Pixel Size", "Pixel Size of the image", "1.50", 1.50);

	#ifdef _OPENMP
	number_of_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	number_of_threads = 1;
#endif

    delete my_input;

    my_current_job.ManualSetArguments("ttiiffftfi", half_map_1.ToUTF8().data(), half_map_2.ToUTF8().data(), first_slice, last_slice, inner_mask_radius, outer_mask_radius, molecular_mass_kDa, symmetry, pixel_size, number_of_threads);

}

bool Generate_Local_Res_App::DoCalculation() 
{
    //TODO
    // 1. WHAT TO DO WITH HALF MAPS 
    // 3. MY_RECONSTRUCTION_1
	// 4. pixel size vs original pixel size 
	// information from array header? where can I get that if not from "dump file"
    wxString half_map_1			= my_current_job.arguments[0].ReturnStringArgument();
	wxString half_map_2			= my_current_job.arguments[1].ReturnStringArgument();
    int first_slice			    = my_current_job.arguments[2].ReturnIntegerArgument();
    int last_slice			    = my_current_job.arguments[3].ReturnIntegerArgument();
    float inner_mask_radius     = my_current_job.arguments[4].ReturnFloatArgument();
    float outer_mask_radius     = my_current_job.arguments[5].ReturnFloatArgument();
    float molecular_mass_kDa    = my_current_job.arguments[6].ReturnFloatArgument();
    wxString symmetry           = my_current_job.arguments[7].ReturnStringArgument();
	float original_pixel_size	= my_current_job.arguments[8].ReturnFloatArgument();
	int	 num_threads			= my_current_job.arguments[9].ReturnIntegerArgument();

	//need to read halfmap1 and halfmap2 into an image (read slices)
	//so have to get num slices then do read 1..num slices into an image 
	//that image is a density map, then combine and divide by 2 then use that 

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
	combined_images.DivideByConstant(2); //this is now a density map? 

	float		mask_falloff = 10.0;

	if (is_running_locally == false) num_threads = number_of_threads_requested_on_command_line;

    //uh? dump file? user input? evrything until END might be gargbage// yea this is probably all grarbage  
    // ReconstructedVolume output_3d(molecular_mass_kDa);
	// ReconstructedVolume output_3d1(molecular_mass_kDa);
	// ReconstructedVolume output_3d2(molecular_mass_kDa);

	// int			logical_x_dimension;
	// int			logical_y_dimension;
	// int			logical_z_dimension;
	// int			original_x_dimension;
	// int			original_y_dimension;
	// int			original_z_dimension;
	// int			images_processed;
	// float		pixel_size;
	// float		original_pixel_size;  
	// float		average_occupancy;
	// float		average_sigma;
	// float		sigma_bfactor_conversion;
	// wxString	my_symmetry;
	// bool		insert_even;
	// bool		center_mass;
	// wxString dump_file = "dump_file.dat";

	// Reconstruct3D temp_reconstruction;
	// temp_reconstruction.ReadArrayHeader(dump_file, logical_x_dimension, logical_y_dimension, logical_z_dimension,
	// 		original_x_dimension, original_y_dimension, original_z_dimension, images_processed, pixel_size, original_pixel_size,
	// 		average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry, insert_even, center_mass);

	// temp_reconstruction.Init(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion);
	// Reconstruct3D my_reconstruction_1(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);
	// Reconstruct3D my_reconstruction_2(logical_x_dimension, logical_y_dimension, logical_z_dimension, pixel_size, average_occupancy, average_sigma, sigma_bfactor_conversion, my_symmetry);

	// output_3d1.FinalizeSimple(my_reconstruction_1, original_x_dimension, original_pixel_size, pixel_size,
	// 		inner_mask_radius, outer_mask_radius, mask_falloff, half_map_1);
	// output_3d2.FinalizeSimple(my_reconstruction_2, original_x_dimension, original_pixel_size, pixel_size,
	// 		inner_mask_radius, outer_mask_radius, mask_falloff, half_map_2);

    // // END flag 

    //initialize images
    Image local_resolution_volume;
    local_resolution_volume.Allocate((&combined_images)->logical_x_dimension, (&combined_images)->logical_y_dimension, (&combined_images)->logical_z_dimension);
	local_resolution_volume.SetToConstant(0.0f);

    Image local_resolution_volume_all;
	local_resolution_volume_all.Allocate((&combined_images)->logical_x_dimension, (&combined_images)->logical_y_dimension, (&combined_images)->logical_z_dimension);
	local_resolution_volume_all.SetToConstant(0.0f);

    //declare and initialize constants and other variables 
    int box_size;
    box_size = 18.0f / original_pixel_size;
    const float threshold_snr = 1;
    const float threshold_confidence = 2.0;
    float fixed_fsc_threshold = .9;
    const bool use_fixed_threshold = true;
    int number_averaged = 0;

    
    #pragma omp parallel default(shared) num_threads(num_threads)
	{
        Image local_resolution_volume_local;
        Image input_volume_one_local;
        Image input_volume_two_local;

        input_volume_one_local.CopyFrom(&combined_images); 
        input_volume_two_local.CopyFrom(&combined_images);

        Image mask_image;
        mask_image.CopyFrom(&combined_images);
        float original_average_value = mask_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
        mask_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f);

        local_resolution_volume_local.Allocate((&combined_images)->logical_x_dimension, (&combined_images)->logical_y_dimension, (&combined_images)->logical_z_dimension);
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


	return true;
}

