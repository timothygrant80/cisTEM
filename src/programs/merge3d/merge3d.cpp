#include "cistem_config.h" // include to ensure that the macros are found; must do before core_headers.h in this particular scenario
// because of weirdness with libtorch

// These torch includes must be included before the core headers, else compilation errors result
#include "../../core/core_headers.h"

// There are type conflicts with the LibTorch libraries
// within defines.h; they must be undefined to be able
// to compile successfully with LibTorch.
#ifdef BLUSH
#ifdef NONE
#undef NONE
#endif
#ifdef FLOAT
#undef FLOAT
#endif
#ifdef LONG
#undef LONG
#endif
#ifdef N_
#undef N_
#endif
#include <torch/nn/functional/conv.h>
#include <torch/script.h>
#include "../blush_refinement/blush_model.h"
#include "../blush_refinement/blush_helpers.h"
#include "../blush_refinement/block_iterator.h"
#endif

class
        Merge3DApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(Merge3DApp)

// override the DoInteractiveUserInput

void Merge3DApp::DoInteractiveUserInput( ) {
    wxString output_reconstruction_1;
    wxString output_reconstruction_2;
    wxString output_reconstruction_filtered;
    wxString output_resolution_statistics;
    float    molecular_mass_kDa = 1000.0;
    float    inner_mask_radius  = 0.0;
    float    outer_mask_radius  = 100.0;
    wxString dump_file_seed_1;
    wxString dump_file_seed_2;
    int      number_of_dump_files;

    UserInput* my_input = new UserInput("Merge3D", 1.01);

    output_reconstruction_1        = my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
    output_reconstruction_2        = my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
    output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
    output_resolution_statistics   = my_input->GetFilenameFromUser("Output resolution statistics", "The text file with the resolution statistics for the final reconstruction", "my_statistics.txt", false);
    molecular_mass_kDa             = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
    inner_mask_radius              = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
    outer_mask_radius              = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
    dump_file_seed_1               = my_input->GetFilenameFromUser("Seed for input dump filenames for odd particles", "The seed name of the first dump files with the intermediate reconstruction arrays", "dump_file_seed_1_.dat", false);
    dump_file_seed_2               = my_input->GetFilenameFromUser("Seed for input dump filenames for even particles", "The seed name of the second dump files with the intermediate reconstruction arrays", "dump_file_seed_2_.dat", false);
    number_of_dump_files           = my_input->GetIntFromUser("Number of dump files", "The number of dump files that should be read from disk and merged", "1", 1);

    delete my_input;

    int      class_number_for_gui        = 1;
    bool     save_orthogonal_views_image = false;
    wxString orthogonal_views_filename   = "";
    float    weiner_nominator            = 1.0f;
    float    alignment_res               = 5.0f;
    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("ttttfffttibtiff", output_reconstruction_1.ToUTF8( ).data( ),
                                      output_reconstruction_2.ToUTF8( ).data( ),
                                      output_reconstruction_filtered.ToUTF8( ).data( ),
                                      output_resolution_statistics.ToUTF8( ).data( ),
                                      molecular_mass_kDa,
                                      inner_mask_radius,
                                      outer_mask_radius,
                                      dump_file_seed_1.ToUTF8( ).data( ),
                                      dump_file_seed_2.ToUTF8( ).data( ),
                                      class_number_for_gui,
                                      save_orthogonal_views_image,
                                      orthogonal_views_filename.ToUTF8( ).data( ),
                                      number_of_dump_files,
                                      weiner_nominator,
                                      alignment_res);
}

// override the do calculation method which will be what is actually run..

bool Merge3DApp::DoCalculation( ) {
    wxString output_reconstruction_1        = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_reconstruction_2        = my_current_job.arguments[1].ReturnStringArgument( );
    wxString output_reconstruction_filtered = my_current_job.arguments[2].ReturnStringArgument( );
    wxString output_resolution_statistics   = my_current_job.arguments[3].ReturnStringArgument( );
    float    molecular_mass_kDa             = my_current_job.arguments[4].ReturnFloatArgument( );
    float    inner_mask_radius              = my_current_job.arguments[5].ReturnFloatArgument( );
    float    outer_mask_radius              = my_current_job.arguments[6].ReturnFloatArgument( );
    wxString dump_file_seed_1               = my_current_job.arguments[7].ReturnStringArgument( );
    wxString dump_file_seed_2               = my_current_job.arguments[8].ReturnStringArgument( );
    int      class_number_for_gui           = my_current_job.arguments[9].ReturnIntegerArgument( );
    bool     save_orthogonal_views_image    = my_current_job.arguments[10].ReturnBoolArgument( );
    wxString orthogonal_views_filename      = my_current_job.arguments[11].ReturnStringArgument( );
    int      number_of_dump_files           = my_current_job.arguments[12].ReturnIntegerArgument( );
    float    weiner_nominator               = my_current_job.arguments[13].ReturnFloatArgument( );
    // FOR LOCRES HACK..
    float alignment_res = my_current_job.arguments[14].ReturnFloatArgument( );

    // Only used when doing blush
    float particle_diameter     = my_current_job.arguments[15].ReturnFloatArgument( );
    bool  apply_blush_denoising = my_current_job.arguments[16].ReturnBoolArgument( );

    ResolutionStatistics* resolution_statistics = NULL;
    resolution_statistics                       = new ResolutionStatistics;

    ReconstructedVolume output_3d(molecular_mass_kDa);
    ReconstructedVolume output_3d1(molecular_mass_kDa);
    ReconstructedVolume output_3d2(molecular_mass_kDa);

    int        i;
    int        logical_x_dimension;
    int        logical_y_dimension;
    int        logical_z_dimension;
    int        original_x_dimension;
    int        original_y_dimension;
    int        original_z_dimension;
    int        count;
    int        intermediate_box_size;
    int        images_processed;
    float      mask_volume_fraction;
    float      mask_falloff = 10.0;
    float      pixel_size;
    float      original_pixel_size;
    float      average_occupancy;
    float      average_sigma;
    float      sigma_bfactor_conversion;
    float      particle_area_in_pixels;
    float      scale;
    float      binning_factor;
    wxString   my_symmetry;
    wxDateTime my_time_in;
    wxFileName dump_file_name = wxFileName::FileName(dump_file_seed_1);
    wxString   extension      = dump_file_name.GetExt( );
    wxString   dump_file;
    bool       insert_even;
    bool       center_mass;
    bool       crop_images;

    NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);

    my_time_in = wxDateTime::Now( );
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

    if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
    {
        //
    }
    else {
        SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
        exit(-1);
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

    for ( count = 1; count <= number_of_dump_files; count++ ) {
        dump_file = wxFileName::StripExtension(dump_file_seed_1) + wxString::Format("%i", count) + "." + extension;
        wxPrintf("%s\n", dump_file);
        if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
        {
            temp_reconstruction.ReadArrays(dump_file);
            my_reconstruction_1 += temp_reconstruction;
        }
        else {
            SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
            exit(-1);
        }
    }

    for ( count = 1; count <= number_of_dump_files; count++ ) {
        dump_file = wxFileName::StripExtension(dump_file_seed_2) + wxString::Format("%i", count) + "." + extension;
        wxPrintf("%s\n", dump_file);
        if ( (is_running_locally && DoesFileExist(dump_file)) || (! is_running_locally && DoesFileExistWithWait(dump_file, 90)) ) // C++ standard says if LHS of OR is true, RHS never gets evaluated
        {
            temp_reconstruction.ReadArrays(dump_file);
            my_reconstruction_2 += temp_reconstruction;
        }
        else {
            SendError(wxString::Format("Error: Dump file %s not found\n", dump_file));
            exit(-1);
        }
    }

    wxPrintf("\nFinished reading arrays\n");

    output_3d1.FinalizeSimple(my_reconstruction_1, original_x_dimension, original_pixel_size, pixel_size,
                              inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_1);
    output_3d2.FinalizeSimple(my_reconstruction_2, original_x_dimension, original_pixel_size, pixel_size,
                              inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_2);

    output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
    my_reconstruction_1 += my_reconstruction_2;
    my_reconstruction_2.FreeMemory( );

    output_3d.FinalizeOptimal(my_reconstruction_1, output_3d1.density_map, output_3d2.density_map,
                              original_pixel_size, pixel_size, inner_mask_radius, outer_mask_radius, mask_falloff,
                              center_mass, output_reconstruction_filtered, output_statistics_file, resolution_statistics, weiner_nominator);

    //float orientation_distribution_efficiency = output_3d.ComputeOrientationDistributionEfficiency(my_reconstruction_1);
    //SendInfo(wxString::Format("Orientation distribution efficiency: %0.2f\n",orientation_distribution_efficiency));

    // LOCAL RESOLUTION HACK - REMOVE!!

    ///////////// LOCAL RES HACK!! TODO REMOVE!

    // MASKING

    const bool test_locres_filtering = false;
    const int  number_of_threads     = 44;
    if ( test_locres_filtering ) {

        /*
		 * Parameters for local resolution estimation & filtering
		 */

        /*
		 * Compute a mask
		 */

        // remove disconnected density
        /* old_mask
 *

			Image buffer_image;

			buffer_image.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);

			buffer_image.CopyFrom(output_3d.density_map);
			buffer_image.SetMinimumValue(original_average_value);
			buffer_image.ForwardFFT();
			buffer_image.CosineMask(original_pixel_size / 50.0f, original_pixel_size / 10.0f);
			buffer_image.BackwardFFT();

			float average_value = buffer_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
			float average_of_100_max = buffer_image.ReturnAverageOfMaxN(500, outer_mask_radius / original_pixel_size);
			float threshold_value = average_value + ((average_of_100_max - average_value) * 0.03);

			buffer_image.CosineMask(outer_mask_radius / original_pixel_size, 1.0, false, true, 0.0);
			buffer_image.Binarise(threshold_value);

			MyDebugPrint("About to compute size image\n");

			rle3d my_rle3d(buffer_image);
			size_image.Allocate(buffer_image.logical_x_dimension, buffer_image.logical_y_dimension, buffer_image.logical_z_dimension, true);
			my_rle3d.ConnectedSizeDecodeTo(size_image);

			MyDebugPrint("About to compute mask\n");

			size_image.Binarise(size_image.ReturnMaximumValue() - 1.0f);
	#ifdef DEBUG
	size_image.QuickAndDirtyWriteSlices("/tmp/locres_mask.mrc", 1, size_image.logical_z_dimension);
		#endif

			for (long address = 0; address < output_3d.density_map->real_memory_allocated; address++)
			{
				if (size_image.real_values[address] == 0.0f) output_3d.density_map->real_values[address] = original_average_value;
			}

	//		output_3d.density_map->SetMinimumValue(original_average_value);
	//		output_3d.density_map->CosineMask(outer_mask_radius / original_pixel_size, 1.0, false, true, 0.0);

*/

        /*
		 * Apply a local resolution filter to the reconstruction
		 */
        {
            Image local_resolution_volume;
            Image original_volume;
            //const float krr = 5.0;
            int box_size;
            //box_size = int(krr * gui_statistics->ReturnEstimatedResolution() / original_pixel_size);
            box_size = 18.0f / original_pixel_size;
            //if (box_size < 15) box_size = 15;

            wxPrintf("Will estimate local resolution using a box size of %i\n", box_size);
            const float threshold_snr        = 1;
            const float threshold_confidence = 2.0;
            float       fixed_fsc_threshold  = .9;
            const bool  use_fixed_threshold  = true;

            MyDebugPrint("About to estimate loc res, with %.2f cutoff\n", fixed_fsc_threshold);
            if ( ! output_3d1.density_map->is_in_real_space )
                output_3d1.density_map->BackwardFFT( );
            if ( ! output_3d2.density_map->is_in_real_space )
                output_3d2.density_map->BackwardFFT( );

            original_volume.CopyFrom(output_3d.density_map);
            output_3d.density_map->QuickAndDirtyWriteSlices("/tmp/locres_original.mrc", 1, output_3d.density_map->logical_z_dimension, true, original_pixel_size);
            Image size_image;
            size_image.CopyFrom(output_3d.density_map);

#ifdef DEBUG
            size_image.QuickAndDirtyWriteSlices("/tmp/locres_filtered_input.mrc", 1, size_image.logical_z_dimension, true, original_pixel_size);
#endif

            float original_average_value = size_image.ReturnAverageOfRealValues(outer_mask_radius / original_pixel_size, true);
            size_image.ConvertToAutoMask(original_pixel_size, outer_mask_radius, original_pixel_size * 2.0f, 0.2f);

#ifdef DEBUG
            size_image.QuickAndDirtyWriteSlices("/tmp/locres_mask.mrc", 1, size_image.logical_z_dimension, true, original_pixel_size);
#endif

            local_resolution_volume.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
            local_resolution_volume.SetToConstant(0.0f);

            Image local_resolution_volume_all;
            local_resolution_volume_all.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
            local_resolution_volume_all.SetToConstant(0.0f);

            int first_slice_with_data;
            int last_slice_with_data;

            Image slice_image;

            for ( int counter = 1; counter <= local_resolution_volume.logical_z_dimension; counter++ ) {
                slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
                if ( slice_image.IsConstant( ) == false ) {
                    first_slice_with_data = counter;
                    break;
                }
            }

            for ( int counter = local_resolution_volume.logical_z_dimension; counter >= 1; counter-- ) {
                slice_image.AllocateAsPointingToSliceIn3D(&size_image, counter);
                if ( slice_image.IsConstant( ) == false ) {
                    last_slice_with_data = counter;
                    break;
                }
            }

            int   slices_with_data  = last_slice_with_data - first_slice_with_data;
            float slices_per_thread = slices_with_data / float(number_of_threads);
            int   number_averaged   = 0;

            for ( float current_res = 18.0f; current_res < 37.0f; current_res += 6.0f ) {
                //	float current_res = 24;
                box_size = current_res / original_pixel_size;

                if ( alignment_res > 15 )
                    fixed_fsc_threshold = 0.75;
                else if ( alignment_res > 8 )
                    fixed_fsc_threshold = 0.85;
                else if ( alignment_res > 6 )
                    fixed_fsc_threshold = 0.9;
                else
                    fixed_fsc_threshold = 0.95f;

                local_resolution_volume.SetToConstant(0.0f);

#pragma omp parallel default(shared) num_threads(number_of_threads)
                { // for omp

                    int first_slice = (first_slice_with_data - 1) + myroundint(ReturnThreadNumberOfCurrentThread( ) * slices_per_thread) + 1;
                    int last_slice  = (first_slice_with_data - 1) + myroundint((ReturnThreadNumberOfCurrentThread( ) + 1) * slices_per_thread);

                    Image local_resolution_volume_local;
                    Image input_volume_one_local;
                    Image input_volume_two_local;

                    input_volume_one_local.CopyFrom(output_3d1.density_map);
                    input_volume_two_local.CopyFrom(output_3d2.density_map);

                    local_resolution_volume_local.Allocate(output_3d.density_map->logical_x_dimension, output_3d.density_map->logical_y_dimension, output_3d.density_map->logical_z_dimension);
                    local_resolution_volume_local.SetToConstant(0.0f);
                    LocalResolutionEstimator* estimator = new LocalResolutionEstimator( );
                    estimator->SetAllUserParameters(&input_volume_one_local, &input_volume_two_local, &size_image, first_slice, last_slice, 1, original_pixel_size, box_size, threshold_snr, threshold_confidence, use_fixed_threshold, fixed_fsc_threshold, my_reconstruction_1.symmetry_matrices.symmetry_symbol, true, 2);
                    estimator->EstimateLocalResolution(&local_resolution_volume_local);
                    delete estimator;

#pragma omp critical
                    {
                        for ( long pixel_counter = 0; pixel_counter < local_resolution_volume.number_of_real_space_pixels; pixel_counter++ ) {
                            if ( local_resolution_volume_local.real_values[pixel_counter] != 0.0f )
                                local_resolution_volume.real_values[pixel_counter] = local_resolution_volume_local.real_values[pixel_counter];
                        }
                    }
                } // end omp

                local_resolution_volume.QuickAndDirtyWriteSlices(wxString::Format("/tmp/local_res_%i", int(current_res)).ToStdString( ), 1, local_resolution_volume.logical_z_dimension);

                // fill in gaps..

                float max_res = local_resolution_volume.ReturnMaximumValue( );

                for ( long pixel_counter = 0; pixel_counter < local_resolution_volume.real_memory_allocated; pixel_counter++ ) {
                    if ( local_resolution_volume.real_values[pixel_counter] < 0.5f )
                        local_resolution_volume.real_values[pixel_counter] = max_res;
                }

                local_resolution_volume_all.AddImage(&local_resolution_volume);
                number_averaged++;
            }

            //			MRCFile junk("/tmp/locres.mrc");
            //			local_resolution_volume.ReadSlices(&junk, 1, junk.ReturnNumberOfSlices());

            // divide and copy

            local_resolution_volume_all.DivideByConstant(number_averaged);
            local_resolution_volume.CopyFrom(&local_resolution_volume_all);

#ifdef DEBUG
            local_resolution_volume.QuickAndDirtyWriteSlices("/tmp/locres.mrc", 1, size_image.logical_z_dimension, true, original_pixel_size);
#endif

            // get scaler for resolution

            int number_of_top_pixels_to_use = local_resolution_volume.number_of_real_space_pixels * 0.00001;
            if ( number_of_top_pixels_to_use < 50 )
                number_of_top_pixels_to_use = 50;

            float highest_resolution  = local_resolution_volume.ReturnAverageOfMinN(number_of_top_pixels_to_use);
            float measured_resolution = resolution_statistics->ReturnEstimatedResolution(true);

            float average_resolution = 0.0f;
            long  voxels_in_the_mask = 0;

            int  i, j, k;
            long pixel_counter = 0;

            for ( k = 0; k < local_resolution_volume.logical_z_dimension; k++ ) {
                for ( j = 0; j < local_resolution_volume.logical_y_dimension; j++ ) {
                    for ( i = 0; i < local_resolution_volume.logical_x_dimension; i++ ) {
                        if ( size_image.real_values[pixel_counter] == 1.0f ) {
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
            wxPrintf("Local high / Measured Average / Local Average = %.2f / %.2f / %.2f\n", highest_resolution, measured_resolution, average_resolution);

            if ( highest_resolution != 8.0f && measured_resolution != 8.0f ) {
                float scaler = (8.0f - measured_resolution) / (8.0f - highest_resolution);

                pixel_counter = 0;
                for ( k = 0; k < local_resolution_volume.logical_z_dimension; k++ ) {
                    for ( j = 0; j < local_resolution_volume.logical_y_dimension; j++ ) {
                        for ( i = 0; i < local_resolution_volume.logical_x_dimension; i++ ) {
                            if ( size_image.real_values[pixel_counter] == 1.0f ) {
                                if ( local_resolution_volume.real_values[pixel_counter] < 8.0f ) {
                                    if ( scaler > 1.0f )
                                        local_resolution_volume.real_values[pixel_counter] = ((local_resolution_volume.real_values[pixel_counter] - highest_resolution) * scaler) + measured_resolution;
                                    if ( local_resolution_volume.real_values[pixel_counter] < 5.0f ) {
                                        local_resolution_volume.real_values[pixel_counter] = measured_resolution;
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

#ifdef DEBUG
            local_resolution_volume.QuickAndDirtyWriteSlices("/tmp/locres_scaled.mrc", 1, size_image.logical_z_dimension, true, pixel_size);
#endif

            // Apply filter mask..

            /*		Image filter_mask;
			MRCFile filter_mask_file("/tmp/fab_filter_mask.mrc");

			filter_mask.ReadSlices(&filter_mask_file, 1, filter_mask_file.ReturnNumberOfSlices());

			pixel_counter = 0;
			for ( k = 0; k < local_resolution_volume.logical_z_dimension; k ++ )
			{
				for ( j = 0; j < local_resolution_volume.logical_y_dimension; j ++ )
				{
					for ( i = 0; i < local_resolution_volume.logical_x_dimension; i ++ )
					{
						if (size_image.real_values[pixel_counter] == 1.0f)
						{
							if (filter_mask.real_values[pixel_counter] > local_resolution_volume.real_values[pixel_counter])
							{
								local_resolution_volume.real_values[pixel_counter] = filter_mask.real_values[pixel_counter];
							}
						}
						pixel_counter++;
					}
					pixel_counter += local_resolution_volume.padding_jump_value;
				}
			}
			*/

#ifdef DEBUG
            local_resolution_volume.QuickAndDirtyWriteSlices("/tmp/locres_scaled_filter_mask.mrc", 1, size_image.logical_z_dimension, true, pixel_size);
#endif

            MyDebugPrint("About to apply locres filter\n");

            int number_of_levels = box_size;
            output_3d.density_map->ApplyLocalResolutionFilter(local_resolution_volume, original_pixel_size, number_of_levels);

            for ( long address = 0; address < output_3d.density_map->real_memory_allocated; address++ ) {
                //	if (size_image.real_values[address] == 0.0f) output_3d.density_map->real_values[address] = original_average_value;

                // go back to original density if high res..

                if ( local_resolution_volume.real_values[address] <= measured_resolution + (measured_resolution * 0.1) ) {
                    output_3d.density_map->real_values[address] = original_volume.real_values[address];
                }
            }

            //output_3d.density_map->SetMinimumValue(original_average_value);

            //  MAKE BACKGROUND ZERO - MIGHT BE BAD!!!
            //output_3d.density_map->AddConstant(-original_average_value);

            output_3d.density_map->CosineMask(outer_mask_radius / original_pixel_size, 1.0, false, true, 0.0);
        }

        output_3d.density_map->WriteSlicesAndFillHeader(output_reconstruction_filtered.ToStdString( ), original_pixel_size);
    }
    /////////////////////// END HACK..

#ifdef BLUSH
    if ( apply_blush_denoising ) {
        SendInfo("Running Blush - this can take several minutes...\n");

        wxDateTime overall_start = wxDateTime::Now( );
        wxDateTime overall_finish;

        // NOTE: this is a no_grad guard, so that the Blush model does not track gradients in a computation graph for the forward pass.
        // If it were to track gradients, it would consume a lot of memory (sometimes greater than 32 GB).
        // These computation graphs are used for backpropagation during neural network training, which is not needed in this case as this
        // is using the model in inference mode (i.e., for making predictions rather than training), along with loaded weights.
        torch::NoGradGuard no_grad;
        // TODO: insert the same logic for running blush that is used in blush_refinement.cpp

        constexpr float model_voxel_size{1.5f}; // The expected voxel/pixel size of inputs to the blush model
        constexpr int   model_block_size{64};
        constexpr int   strides{20};
        constexpr int   in_channels{2};
        constexpr int   batch_size{1};
        constexpr float mask_edge_in_angstr{10.0f}; // Edge width of the mask in Angstroms; this is the same as the Python model uses

        torch::set_num_threads(1); // Set to 1 to avoid issues with parallelism in the model

        BlushModel model(2, 2);
        try {
            model.load_weights("blush_weights.dat");
        } catch ( std::exception& e ) {
            wxPrintf("Blush error - Error loading model weights: %s\n", e.what( ));
            SendErrorAndCrash(wxString::Format("Blush error - Error loading model weights. It is likely that the blush_weights.dat file is not present in the directory from which merge3d is being run. %s\n", e.what( )));
        }

        model.eval( );
        float     scale_factor;
        const int original_box_size = output_3d.density_map->logical_x_dimension;
        bool      must_resample     = false;
        int       new_box_size;
        {
            float           wanted_sf = original_pixel_size / model_voxel_size;
            constexpr float tolerance = 1e-2f;
            new_box_size              = static_cast<int>(std::floor(output_3d.density_map->logical_x_dimension * wanted_sf + 0.5f));
            if ( new_box_size % 2 != 0 )
                new_box_size++;

            scale_factor  = static_cast<float>(new_box_size) / static_cast<float>(output_3d.density_map->logical_x_dimension);
            must_resample = (std::abs(scale_factor - 1.0f) > tolerance);

            if ( must_resample ) {
                output_3d.density_map->ForwardFFT( );
                output_3d.density_map->Resize(new_box_size, new_box_size, new_box_size);
                output_3d.density_map->BackwardFFT( );
            }
        }

        // Must remove padding because model as implemented expects none -- it is possible to account for the padding, it's just not implemented here for simplifying integration
        // of the blush model into cisTEM.

        torch::Tensor                 blocks;
        std::vector<std::vector<int>> coords;
        torch::Tensor                 volume_tensor;
        torch::Tensor                 local_std_dev;
        torch::Tensor                 weights;
        torch::Tensor                 infer_grid;
        torch::Tensor                 count_grid;
        torch::Tensor                 mask_tensor;

        // Set up tensors to be used for model and model setup
        try {
            blocks        = torch::zeros({batch_size, model_block_size, model_block_size, model_block_size});
            coords        = std::vector(batch_size, std::vector<int>(3, 0));
            volume_tensor = torch::zeros({new_box_size, new_box_size, new_box_size}, torch::kFloat32);
            local_std_dev = torch::zeros({new_box_size, new_box_size, new_box_size}, torch::kFloat32);
        } catch ( std::exception& e ) {
            wxPrintf("Blush error - Error setting up tensors: %s\n", e.what( ));
            SendErrorAndCrash(wxString::Format("Blush error - Error setting up tensors: %s\n", e.what( )));
        }

        output_3d.density_map->RemoveFFTWPadding( );

        volume_tensor = torch::from_blob(output_3d.density_map->real_values, {new_box_size, new_box_size, new_box_size}, torch::kFloat32).clone( ).contiguous( );
        volume_tensor = volume_tensor.permute({2, 1, 0}).contiguous( ); // Change to (z, y, x) order for LibTorch

        weights    = make_weight_box(model_block_size, 10);
        infer_grid = torch::zeros({new_box_size, new_box_size, new_box_size}, torch::kFloat32);
        count_grid = torch::zeros({new_box_size, new_box_size, new_box_size}, torch::kFloat32);

        try {
            torch::Tensor tmp_volume_clone   = volume_tensor.clone( ).contiguous( );
            torch::Tensor tmp_local_std_dev  = get_local_std_dev(tmp_volume_clone.unsqueeze(0), 10).squeeze(0).contiguous( ).clone( );
            torch::Tensor local_std_dev_mean = tmp_local_std_dev.mean( );
            torch::Tensor volume_mean        = volume_tensor.mean( );
            torch::Tensor volume_std         = volume_tensor.std( );

            local_std_dev = tmp_local_std_dev / local_std_dev_mean;
            volume_tensor = (volume_tensor - volume_mean) / (volume_std + 1e-8);
        } catch ( std::exception& e ) {
            wxPrintf("Blush error - Error getting localized standard deviation and normalizing the volume tensor: %s\n", e.what( ));
            SendErrorAndCrash(wxString::Format("Blush error - Error getting localized standard deviation and normalizing the volume tensor: %s\n", e.what( )));
            // TODO: Return or otherwise exit?
        }

        // Generate and apply mask
        int mask_edge_width = static_cast<int>(20.0f / model_voxel_size);

        float radius = particle_diameter * model_voxel_size + mask_edge_width / 2;
        radius       = std::min(radius, (new_box_size - mask_edge_width) / 2.0f + 1.0f);
        mask_tensor  = generate_radial_mask(new_box_size, radius, mask_edge_width);

        volume_tensor *= mask_tensor;
        volume_tensor = volume_tensor.unsqueeze(0);
        local_std_dev *= mask_tensor;
        local_std_dev = local_std_dev.unsqueeze(0);

        // Set up done, now pass to the model
        try {
            BlockIterator it({new_box_size, new_box_size, new_box_size}, model_block_size, strides);
            int           bi = 0;

            for ( auto it_coords : it ) {
                torch::Tensor volume_block  = torch::zeros({1, model_block_size, model_block_size, model_block_size}, torch::kFloat32);
                torch::Tensor std_dev_block = torch::zeros({1, model_block_size, model_block_size, model_block_size}, torch::kFloat32);

                int x = std::get<0>(it_coords);
                int y = std::get<1>(it_coords);
                int z = std::get<2>(it_coords);

                // Here -1 is just a method for determining if iterations have finished; could probably improve clarity,
                // but for now this note is enough.
                if ( x > -1 ) {
                    torch::Tensor current_slice = mask_tensor.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                    float         mask_mean     = current_slice.mean( ).item<float>( );

                    // Skip this block if the mask mean is quite low as there must not be much density here.
                    if ( mask_mean < 0.3f ) {
                        continue;
                    }

                    volume_block  = volume_tensor.slice(1, z, z + model_block_size, 1).slice(2, y, y + model_block_size, 1).slice(3, x, x + model_block_size, 1).clone( );
                    std_dev_block = local_std_dev.slice(1, z, z + model_block_size, 1).slice(2, y, y + model_block_size, 1).slice(3, x, x + model_block_size, 1).clone( );
                    bi++;
                }

                if ( bi == batch_size ) {
                    auto          initial_output = model.forward(volume_block, std_dev_block);
                    torch::Tensor vol_output     = std::get<0>(initial_output);

                    for ( int i = 0; i < batch_size; i++ ) {
                        torch::Tensor infer_grid_slice = infer_grid.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                        auto          update           = vol_output[i] * weights;
                        infer_grid_slice += update;

                        torch::Tensor count_grid_slice = count_grid.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                        count_grid_slice += weights;
                    }
                    bi = 0;
                }
            }
            infer_grid = torch::where(count_grid > 0, infer_grid / count_grid, infer_grid);
            infer_grid = torch::where(count_grid < 1e-1f, 0, infer_grid); // Set values where count_grid is less than 0.1 to 0
            infer_grid *= mask_tensor;

            infer_grid = infer_grid * (volume_tensor.std( ) + 1e-8) + volume_tensor.mean( ); // Normalize the inference grid
        } catch ( std::exception& e ) {
            wxPrintf("Blush error - Error running the blush model: %s\n", e.what( ));
            SendErrorAndCrash("Blush error - Error running the blush model: " + wxString::FromUTF8(e.what( )));
        }

        // Finally, put the result back into the output_3d.density_map, and handle resampling if needed
        infer_grid *= mask_tensor;
        infer_grid = infer_grid.permute({2, 1, 0}).contiguous( ); // Change back to (x, y, z) order for MRC output
        std::memcpy(output_3d.density_map->real_values, infer_grid.data_ptr<float>( ), sizeof(float) * std::pow(new_box_size, 3));
        output_3d.density_map->AddFFTWPadding( );
        if ( must_resample ) {
            output_3d.density_map->ForwardFFT( );
            output_3d.density_map->Resize(original_box_size, original_box_size, original_box_size);
            output_3d.density_map->BackwardFFT( );
        }

        // Write out again to save the blush changes
        MRCFile output_file;
        output_file.OpenFile(output_reconstruction_filtered.ToStdString( ), true);
        output_3d.density_map->WriteSlices(&output_file, 1, output_3d.density_map->logical_z_dimension);
        output_file.SetPixelSize(original_pixel_size);
        EmpiricalDistribution<double> density_distribution;
        output_3d.density_map->UpdateDistributionOfRealValues(&density_distribution);
        output_file.SetDensityStatistics(density_distribution.GetMinimum( ), density_distribution.GetMaximum( ), density_distribution.GetSampleMean( ), sqrtf(density_distribution.GetSampleVariance( )));
        output_file.CloseFile( );

        overall_finish      = wxDateTime::Now( );
        wxTimeSpan duration = overall_finish.Subtract(overall_start);
        wxPrintf("Total blush runtime:         %s\n", duration.Format( ));
    }
#endif

    if ( save_orthogonal_views_image == true ) {
        Image orth_image;
        orth_image.Allocate(output_3d.density_map->logical_x_dimension * 3, output_3d.density_map->logical_y_dimension * 2, 1, true);
        output_3d.density_map->CreateOrthogonalProjectionsImage(&orth_image);
        orth_image.QuickAndDirtyWriteSlice(orthogonal_views_filename.ToStdString( ), 1);
    }

    if ( is_running_locally == false ) {
        int number_of_points = resolution_statistics->FSC.NumberOfPoints( );
        int array_size       = (number_of_points * 5) + 2;
        wxPrintf("number of points = %i, class is %i, array size = %i\n", number_of_points, class_number_for_gui, array_size);

        float* statistics = new float[array_size];
        resolution_statistics->WriteStatisticsToFloatArray(statistics, class_number_for_gui);
        my_result.SetResult(array_size, statistics);
        delete[] statistics;
    }

    wxPrintf("\nMerge3D: Normal termination\n\n");

    delete resolution_statistics;
    return true;
}
