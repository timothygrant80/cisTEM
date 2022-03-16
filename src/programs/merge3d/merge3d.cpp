#include "../../core/core_headers.h"

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

    if ( save_orthogonal_views_image == true ) {
        Image orth_image;
        orth_image.Allocate(output_3d.density_map->logical_x_dimension * 3, output_3d.density_map->logical_y_dimension * 2, 1, true);
        output_3d.density_map->CreateOrthogonalProjectionsImage(&orth_image);
        orth_image.QuickAndDirtyWriteSlice(orthogonal_views_filename.ToStdString( ), 1);
    }

    if ( is_running_locally == false ) {
        int number_of_points = resolution_statistics->FSC.number_of_points;
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
