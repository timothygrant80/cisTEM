#include "core_headers.h"

// Use this define to dump intermediate files
#ifdef DEBUG
#define dump_intermediate_files
#endif

// We probably don't need to write out PLT files
//#define write_out_plt_file

ParticleFinder::ParticleFinder( ) {
    // User parameters
    micrograph_filename                         = "";
    original_micrograph_pixel_size              = 0.0;
    acceleration_voltage_in_keV                 = 0.0;
    spherical_aberration_in_mm                  = 0.0;
    amplitude_contrast                          = 0.0;
    additional_phase_shift_in_radians           = 0.0;
    defocus_1_in_angstroms                      = 0.0;
    defocus_2_in_angstroms                      = 0.0;
    astigmatism_angle_in_degrees                = 0.0;
    already_have_templates                      = false;
    templates_filename                          = "";
    average_templates_radially                  = false;
    number_of_template_rotations                = 0;
    typical_radius_in_angstroms                 = 0.0;
    maximum_radius_in_angstroms                 = 0.0;
    highest_resolution_to_use                   = 0.0;
    output_stack_filename                       = "";
    output_stack_box_size                       = 0;
    minimum_distance_from_edges_in_pixels       = 0;
    minimum_peak_height_for_candidate_particles = 0.0;
    low_variance_threshold_in_fwhm              = 0.0;
    high_variance_threshold_in_fwhm             = 0.0;
    avoid_low_variance_areas                    = false;
    avoid_high_variance_areas                   = false;
    avoid_high_low_mean_areas                   = false;
    algorithm_to_find_background                = 0;
    number_of_background_boxes                  = 0;
    number_of_templates                         = 0;
    particles_are_white                         = false;

    // Other
    number_of_background_boxes_to_skip   = 0;
    write_out_plt                        = false;
    pixel_size                           = 0.0;
    minimum_box_size_for_picking         = 0;
    minimum_box_size_for_object_with_psf = 0;
    maximum_radius_in_pixels             = 0.0;
    typical_radius_in_pixels             = 0.0;
    new_template_dimension               = 0;
    new_micrograph_dimension_x           = 0;
    new_micrograph_dimension_y           = 0;
    my_progress_bar                      = NULL;
    local_sigma_mode                     = 0.0;
    local_sigma_fwhm                     = 0.0;
    local_mean_mode                      = 0.0;
    local_mean_fwhm                      = 0.0;
    micrograph_mean                      = 0.0;
    template_image                       = NULL;
}

ParticleFinder::~ParticleFinder( ) {
    DeallocateTemplateImages( );
    CloseImageFiles( );
}

void ParticleFinder::DoItAll( ) {
    OpenMicrographAndUpdateDimensions( );

    UpdatePixelSizeFromMicrographDimensions( );

    OpenTemplatesAndUpdateDimensions( );

    UpdateCTF( );

    UpdateMinimumBoxSize( );

    SetupCurveObjects( );

    // If the user is supplying templates, read them in. If not, generate a single template image.
    AllocateTemplateImages( );
    if ( already_have_templates ) {
        ReadTemplatesFromDisk( );
    }
    else // User did not supply a template, we will generate one
    {
        GenerateATemplate( );
    }

#ifdef dump_intermediate_files
    template_power_spectrum.WriteToFile("dbg_template_power.txt");
#endif

    // Read in the micrograph and resample it
    ReadAndResampleMicrograph( );

    // Band-pass filter the micrograph to emphasize features similar to the templates
    BandPassMicrograph( );

    // Compute local mean, local sigma and get stats on these local statistics
    UpdateLocalMeanAndSigma( );

    // Whiten the background
    WhitenMicrographBackground( );

    // Compute the target function (i.e. do template matching)
    DoTemplateMatching( );

    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );

    // Cleanup
    DeallocateTemplateImages( );
    CloseImageFiles( );
}

void ParticleFinder::RedoWithNewHighestResolution( ) {
    // When the highest resolution changes, I think we need to redo everything
    DoItAll( );
}

void ParticleFinder::RedoWithNewMinimumDistanceFromEdges( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewAvoidLowVarianceAreas( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewAvoidHighVarianceAreas( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewLowVarianceThreshold( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewHighVarianceThreshold( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewAvoidAbnormalLocalMeanAreas( ) {
    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );
}

void ParticleFinder::RedoWithNewNumberOfBackgroundBoxes( ) {
    /*
	wxPrintf("Warning: doing it all\n");
	OpenMicrographAndUpdateDimensions();

	UpdatePixelSizeFromMicrographDimensions();
*/
    OpenTemplatesAndUpdateDimensions( );
    /*
	UpdateCTF();

	UpdateMinimumBoxSize();

	SetupCurveObjects();
	*/

    // If the user is supplying templates, read them in. If not, generate a single template image.
    AllocateTemplateImages( );
    if ( already_have_templates ) {
        ReadTemplatesFromDisk( );
    }
    else // User did not supply a template, we will generate one
    {
        GenerateATemplate( );
    }
    /*

	// Read in the micrograph and resample it
	ReadAndResampleMicrograph();

	// Band-pass filter the micrograph to emphasize features similar to the templates
	BandPassMicrograph();

	// Compute local mean, local sigma and get stats on these local statistics
	UpdateLocalMeanAndSigma();
	*/

    // Whiten the background
    WhitenMicrographBackground( );

    // Compute the target function (i.e. do template matching)
    DoTemplateMatching( );

    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );

    // Cleanup
    //DeallocateTemplateImages();
    //CloseImageFiles();
}

void ParticleFinder::RedoWithNewAlgorithmToFindBackground( ) {
    RedoWithNewNumberOfBackgroundBoxes( );
}

void ParticleFinder::RedoWithNewMinimumPeakHeight( ) {
    /*
	OpenMicrographAndUpdateDimensions();

	UpdatePixelSizeFromMicrographDimensions();

	OpenTemplatesAndUpdateDimensions();

	UpdateCTF();

	UpdateMinimumBoxSize();

	SetupCurveObjects();


	// If the user is supplying templates, read them in. If not, generate a single template image.
	AllocateTemplateImages();
	if (already_have_templates)
	{
		ReadTemplatesFromDisk();
	}
	else // User did not supply a template, we will generate one
	{
		GenerateATemplate();
	}

#ifdef dump_intermediate_files
	template_power_spectrum.WriteToFile("dbg_template_power.txt");
#endif


	// Read in the micrograph and resample it
	ReadAndResampleMicrograph();

	// Band-pass filter the micrograph to emphasize features similar to the templates
	BandPassMicrograph();

	// Compute local mean, local sigma and get stats on these local statistics
	UpdateLocalMeanAndSigma();

	// Whiten the background
	WhitenMicrographBackground();

	// Compute the target function (i.e. do template matching)
	DoTemplateMatching();

	*/

    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );

    // Cleanup
    //DeallocateTemplateImages();
    //CloseImageFiles();
}

void ParticleFinder::RedoWithNewTypicalRadius( ) {
    UpdateTypicalRadiusInPixels( );

    //OpenMicrographAndUpdateDimensions();

    //UpdatePixelSizeFromMicrographDimensions();

    OpenTemplatesAndUpdateDimensions( );

    //UpdateCTF();

    //UpdateMinimumBoxSize();

    //SetupCurveObjects();

    // If the user is supplying templates, read them in. If not, generate a single template image.
    AllocateTemplateImages( );
    if ( already_have_templates ) {
        //ReadTemplatesFromDisk();
    }
    else // User did not supply a template, we will generate one
    {
        GenerateATemplate( );
    }

#ifdef dump_intermediate_files
    template_power_spectrum.WriteToFile("dbg_template_power.txt");
#endif

    // Read in the micrograph and resample it
    //ReadAndResampleMicrograph();

    // Band-pass filter the micrograph to emphasize features similar to the templates
    BandPassMicrograph( );

    // Compute local mean, local sigma and get stats on these local statistics
    UpdateLocalMeanAndSigma( );

    // Whiten the background
    WhitenMicrographBackground( );

    // Compute the target function (i.e. do template matching)
    DoTemplateMatching( );

    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );

    // Cleanup
    DeallocateTemplateImages( );
    CloseImageFiles( );
}

void ParticleFinder::RedoWithNewMaximumRadius( ) {

    UpdateMaximumRadiusInPixels( );

    //OpenMicrographAndUpdateDimensions();

    //UpdatePixelSizeFromMicrographDimensions();

    OpenTemplatesAndUpdateDimensions( );

    //UpdateCTF();

    UpdateMinimumBoxSize( );

    SetupCurveObjects( );

    // If the user is supplying templates, read them in. If not, generate a single template image.
    AllocateTemplateImages( );
    if ( already_have_templates ) {
        ReadTemplatesFromDisk( );
    }
    else // User did not supply a template, we will generate one
    {
        GenerateATemplate( );
    }

#ifdef dump_intermediate_files
    template_power_spectrum.WriteToFile("dbg_template_power.txt");
#endif

    // Read in the micrograph and resample it
    //ReadAndResampleMicrograph();

    // Band-pass filter the micrograph to emphasize features similar to the templates
    BandPassMicrograph( );

    // Compute local mean, local sigma and get stats on these local statistics
    UpdateLocalMeanAndSigma( );

    // Whiten the background
    WhitenMicrographBackground( );

    // Compute the target function (i.e. do template matching)
    DoTemplateMatching( );

    // Look for peaks and extract particles
    FindPeaksAndExtractParticles( );

    // Cleanup
    DeallocateTemplateImages( );
    CloseImageFiles( );
}

void ParticleFinder::CloseImageFiles( ) {
    micrograph_file.CloseFile( );
    template_file.CloseFile( );
}

ArrayOfParticlePositionAssets ParticleFinder::ReturnArrayOfParticlePositionAssets( ) {
    ParticlePositionAsset         temp_asset;
    ArrayOfParticlePositionAssets array_of_assets;
    int                           address_within_results = 0;

    temp_asset.pick_job_id = 0;
    temp_asset.asset_id    = 0;

    temp_asset.parent_id  = 0;
    temp_asset.picking_id = 0;

    // Loop over picked coordinates
    for ( int particle_counter = 0; particle_counter < results_x_y.NumberOfPoints( ); particle_counter++ ) {
        // Finish setting up the asset. We use an ID that hasn't been used for any other position asset previously.
        temp_asset.asset_id++;
        temp_asset.x_position         = results_x_y.data_x[particle_counter];
        temp_asset.y_position         = results_x_y.data_y[particle_counter];
        temp_asset.peak_height        = results_height_template.data_x[particle_counter];
        temp_asset.parent_template_id = myroundint(results_height_template.data_y[particle_counter]);

        //
        array_of_assets.Add(temp_asset);
    }

    return array_of_assets;
}

void ParticleFinder::SetAllUserParameters(wxString wanted_micrograph_filename,
                                          float    wanted_original_micrograph_pixel_size,
                                          float    wanted_acceleration_voltage_in_keV,
                                          float    wanted_spherical_aberration_in_mm,
                                          float    wanted_amplitude_contrast,
                                          float    wanted_additional_phase_shift_in_radians,
                                          float    wanted_defocus_1_in_angstroms,
                                          float    wanted_defocus_2_in_angstroms,
                                          float    wanted_astigmatism_angle_in_degrees,
                                          bool     wanted_already_have_templates,
                                          wxString wanted_templates_filename,
                                          bool     wanted_average_templates_radially,
                                          int      wanted_number_of_template_rotations,
                                          float    wanted_typical_radius_in_angstroms,
                                          float    wanted_maximum_radius_in_angstroms,
                                          float    wanted_highest_resolution_to_use,
                                          wxString wanted_output_stack_filename,
                                          int      wanted_output_stack_box_size,
                                          int      wanted_minimum_distance_from_edges_in_pixels,
                                          float    wanted_minimum_peak_height_for_candidate_particles,
                                          bool     wanted_avoid_low_variance_areas,
                                          bool     wanted_avoid_high_variance_areas,
                                          float    wanted_low_variance_threshold_in_fwhm,
                                          float    wanted_high_variance_threshold_in_fwhm,
                                          bool     wanted_avoid_high_low_mean_areas,
                                          int      wanted_algorithm_to_find_background,
                                          int      wanted_number_of_background_boxes,
                                          bool     wanted_particles_are_white) {
    micrograph_filename                         = wanted_micrograph_filename;
    original_micrograph_pixel_size              = wanted_original_micrograph_pixel_size;
    acceleration_voltage_in_keV                 = wanted_acceleration_voltage_in_keV;
    spherical_aberration_in_mm                  = wanted_spherical_aberration_in_mm;
    amplitude_contrast                          = wanted_amplitude_contrast;
    additional_phase_shift_in_radians           = wanted_additional_phase_shift_in_radians;
    defocus_1_in_angstroms                      = wanted_defocus_1_in_angstroms;
    defocus_2_in_angstroms                      = wanted_defocus_2_in_angstroms;
    astigmatism_angle_in_degrees                = wanted_astigmatism_angle_in_degrees;
    already_have_templates                      = wanted_already_have_templates;
    templates_filename                          = wanted_templates_filename;
    average_templates_radially                  = wanted_average_templates_radially;
    number_of_template_rotations                = wanted_number_of_template_rotations;
    typical_radius_in_angstroms                 = wanted_typical_radius_in_angstroms;
    maximum_radius_in_angstroms                 = wanted_maximum_radius_in_angstroms;
    highest_resolution_to_use                   = wanted_highest_resolution_to_use;
    output_stack_filename                       = wanted_output_stack_filename;
    output_stack_box_size                       = wanted_output_stack_box_size;
    minimum_distance_from_edges_in_pixels       = wanted_minimum_distance_from_edges_in_pixels;
    minimum_peak_height_for_candidate_particles = wanted_minimum_peak_height_for_candidate_particles;
    avoid_low_variance_areas                    = wanted_avoid_low_variance_areas;
    avoid_high_variance_areas                   = wanted_avoid_high_variance_areas;
    low_variance_threshold_in_fwhm              = wanted_low_variance_threshold_in_fwhm;
    high_variance_threshold_in_fwhm             = wanted_high_variance_threshold_in_fwhm;
    avoid_high_low_mean_areas                   = wanted_avoid_high_low_mean_areas;
    algorithm_to_find_background                = wanted_algorithm_to_find_background;
    number_of_background_boxes                  = wanted_number_of_background_boxes;
    particles_are_white                         = wanted_particles_are_white;
}

void ParticleFinder::FindPeaksAndExtractParticles( ) {
    maximum_score_modified = maximum_score;

    // We may want to avoid some areas
    RemoveHighLowMeanAreasFromTargetFunction( );
    RemoveHighLowVarianceAreasFromTargetFunction( );

    // Let's find peaks in our scoring function and box candidate particles out
    int       index_of_matching_template;
    float     rotation_of_matching_template;
    const int minimum_distance_from_edges_in_rescaled_pixels = minimum_distance_from_edges_in_pixels * original_micrograph_pixel_size / pixel_size + 1;
    int       coo_to_ignore_x, coo_to_ignore_y;
    const int min_x = minimum_distance_from_edges_in_rescaled_pixels;
    const int min_y = minimum_distance_from_edges_in_rescaled_pixels;
    const int max_x = maximum_score.logical_x_dimension - minimum_distance_from_edges_in_rescaled_pixels;
    const int max_y = maximum_score.logical_y_dimension - minimum_distance_from_edges_in_rescaled_pixels;
    Image     box;
    box.Deallocate( );
    if ( output_stack_box_size > 0 )
        box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);
    if ( output_stack_box_size > 0 )
        micrograph.ReadSlice(&micrograph_file, 1);
    Peak             my_peak;
    int              number_of_candidate_particles = 0;
    NumericTextFile* output_coos_file;
    MRCFile          output_stack;
    float            highest_peak;
    results_x_y.ClearData( );
    results_height_template.ClearData( );
    results_rotation.ClearData( );
    wxPrintf("\nFinding peaks & extracting particle images...\n");
    float temp_float[6];
    my_progress_bar = new ProgressBar(100);

#ifdef DEBUG
    Image           junk_image;
    Image           junk_image_rotate;
    AnglesAndShifts myangle;
#endif
    while ( true ) {
        // We allow ourselves to find peaks beyond our boundary (that way, if something is found just beyond the boundary, we will blank that area and we won't end up
        // picking the edge of the object which falls within the boundary)
        my_peak = maximum_score_modified.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, std::max(0, minimum_distance_from_edges_in_rescaled_pixels - int(maximum_radius_in_pixels) - 1));
        if ( my_peak.value < minimum_peak_height_for_candidate_particles )
            break;
        if ( number_of_candidate_particles == 0 )
            highest_peak = my_peak.value;
        // Zero an area around this peak to ensure we don't pick again near there
        coo_to_ignore_x = int(my_peak.x) + maximum_score_modified.physical_address_of_box_center_x;
        coo_to_ignore_y = int(my_peak.y) + maximum_score_modified.physical_address_of_box_center_y;
        SetCircularAreaToIgnore(maximum_score_modified, coo_to_ignore_x, coo_to_ignore_y, 2.0 * maximum_radius_in_pixels, 0.0);
        //maximum_score.QuickAndDirtyWriteSlice("dbg_latest_maximum_score.mrc",number_of_candidate_particles);
        // Now we do the actual check as to whether the peak is within our boundaries
        if ( coo_to_ignore_x >= min_x && coo_to_ignore_x <= max_x && coo_to_ignore_y >= min_y && coo_to_ignore_y <= max_y ) {
            // We have found a candidate particle
            number_of_candidate_particles++;
            if ( number_of_candidate_particles == 1 ) {
                if ( output_stack_box_size > 0 )
                    output_stack.OpenFile(output_stack_filename.ToStdString( ), true);
                if ( write_out_plt ) {
                    wxPrintf("about to open numeric text file with filename %s\n", FilenameReplaceExtension(output_stack_filename.ToStdString( ), "plt"));
                    output_coos_file = new NumericTextFile(FilenameReplaceExtension(output_stack_filename.ToStdString( ), "plt"), OPEN_TO_WRITE, 6);
                }
            }
            if ( output_stack_box_size > 0 )
                micrograph.ClipInto(&box, micrograph_mean, false, 1.0, -int(my_peak.x * pixel_size / original_micrograph_pixel_size), -int(my_peak.y * pixel_size / original_micrograph_pixel_size), 0); // - in front of coordinates I think is because micrograph was conjugate multiplied, i.e. reversed order in real space
            if ( output_stack_box_size > 0 )
                box.WriteSlice(&output_stack, number_of_candidate_particles);
            //wxPrintf("Boxed out particle %i at %i, %i, peak height = %f, coo to ignore = %i, %i\n",number_of_candidate_particles,int(my_peak.x),int(my_peak.y),my_peak.value,coo_to_ignore_x,coo_to_ignore_y);

            // Find the matching template
            index_of_matching_template    = template_giving_maximum_score.real_values[my_peak.physical_address_within_image];
            rotation_of_matching_template = template_rotation_giving_maximum_score.real_values[my_peak.physical_address_within_image];

            /*
#ifdef DEBUG
			junk_image.CopyFrom(&template_image[index_of_matching_template]);

			if (junk_image.is_in_real_space)
			{
				junk_image.ForwardFFT(false);
				junk_image.NormalizeFT();

			}

			if (junk_image.object_is_centred_in_box)
			{
				junk_image.SwapRealSpaceQuadrants();
			}

			junk_image_rotate.CopyFrom(&junk_image);
			myangle.Init(0.0,0.0,rotation_of_matching_template,0.0,0.0);
			junk_image.RotateFourier2D(junk_image_rotate,myangle,1.0,false);
			junk_image_rotate.SwapRealSpaceQuadrants();
			junk_image_rotate.QuickAndDirtyWriteSlice("/tmp/templates.mrc", number_of_candidate_particles);
#endif
			 */

            if ( write_out_plt ) {
                temp_float[1] = maximum_score.logical_x_dimension - (maximum_score.physical_address_of_box_center_x + (my_peak.x));
                temp_float[0] = maximum_score.physical_address_of_box_center_y + (my_peak.y);
                temp_float[1] = temp_float[1] * pixel_size / original_micrograph_pixel_size + 1.0;
                temp_float[0] = temp_float[0] * pixel_size / original_micrograph_pixel_size + 1.0;
                temp_float[2] = 1.0;
                temp_float[3] = float(index_of_matching_template);
                temp_float[4] = rotation_of_matching_template;
                temp_float[5] = my_peak.value;
                output_coos_file->WriteLine(temp_float);
            }

            // Remember results
            results_x_y.AddPoint(pixel_size * (float(maximum_score_modified.physical_address_of_box_center_x) - my_peak.x), pixel_size * (float(maximum_score_modified.physical_address_of_box_center_y) - my_peak.y));
            results_height_template.AddPoint(my_peak.value, float(index_of_matching_template));
            results_rotation.AddPoint(rotation_of_matching_template, 0.0);

            /*
			// Get ready for template matching
			PrepareTemplateForMatching(&template_image[index_of_matching_template],template_medium,rotation_of_matching_template,&micrograph_ctf,&background_whitening_filter);

			#ifdef dump_intermediate_files
			template_medium.QuickAndDirtyWriteSlice("dbg_candidate_matched_templates.mrc",number_of_candidate_particles);
			#endif

			template_medium.MultiplyByConstant(my_peak.value);
			box.SubtractImage(&template_medium);
			#ifdef dump_intermediate_files
			box.QuickAndDirtyWriteSlice("dbg_candidate_particle_residuals.mrc",number_of_candidate_particles);
			#endif
	 	 	 */

            //
            my_progress_bar->Update(long((highest_peak - my_peak.value) / (highest_peak - minimum_peak_height_for_candidate_particles) * 100.0) + 1);
        }
    }
    delete my_progress_bar;
    if ( write_out_plt )
        delete output_coos_file;
    wxPrintf("\nFound %i candidate particles\n", number_of_candidate_particles);
}

void ParticleFinder::RemoveHighLowMeanAreasFromTargetFunction( ) {
    if ( avoid_high_low_mean_areas ) {
        long  address;
        long  address_in_score;
        float threshold_high = local_mean_mode + 2.0 * local_mean_fwhm;
        float threshold_low  = local_mean_mode - 2.0 * local_mean_fwhm;
        address              = 0;
        address_in_score     = maximum_score_modified.real_memory_allocated;
        for ( int j = 0; j < maximum_score_modified.logical_y_dimension; j++ ) {
            address_in_score -= maximum_score_modified.padding_jump_value;
            for ( int i = 0; i < maximum_score_modified.logical_x_dimension; i++ ) {
                if ( local_mean.real_values[address] > threshold_high || local_mean.real_values[address] < threshold_low ) {
                    maximum_score_modified.real_values[address_in_score] = 0.0;
                }
                address++;
                address_in_score--;
            }
            address += local_mean.padding_jump_value;
        }

#ifdef dump_intermediate_files
        maximum_score.QuickAndDirtyWriteSlice("dbg_maximum_score_3.mrc", 1);
#endif
    }
}

void ParticleFinder::RemoveHighLowVarianceAreasFromTargetFunction( ) {
    if ( avoid_high_variance_areas || avoid_low_variance_areas ) {
        /*
		 * Decide what we call a "high-variance area"
		 * Normally, 2 sigmas over the mode is OK, but when the micrograph
		 * is from a phase plate, the particles are super high-contrast, so we need a larger value
		 * The following may be tweaked heuristically:
		 * - the phase shift range over which high contrast is assumed
		 * - the number of sigmas above the mode for the threshold when in high-contrast mode
		 *
		 * Update Dec 2018: we will now let the user/GUI set this
		 */
        float threshold_high;
        float threshold_low;

        if ( avoid_high_variance_areas ) {
            threshold_high = local_sigma_mode + high_variance_threshold_in_fwhm * local_sigma_fwhm;
        }
        else {
            threshold_high = local_sigma_mode + 9999.9 * local_sigma_fwhm;
        }
        if ( avoid_low_variance_areas ) {
            threshold_low = local_sigma_mode + low_variance_threshold_in_fwhm * local_sigma_fwhm;
        }
        else {
            threshold_low = local_sigma_mode - 9999.9 * local_sigma_fwhm;
        }

        long address;
        long address_in_score;
        // this is slightly complicated because the correlation map and the variance map are in reverse order

        address          = 0;
        address_in_score = maximum_score_modified.real_memory_allocated;
        for ( int j = 0; j < maximum_score_modified.logical_y_dimension; j++ ) {
            address_in_score -= maximum_score_modified.padding_jump_value;
            for ( int i = 0; i < maximum_score_modified.logical_x_dimension; i++ ) {
                if ( local_sigma.real_values[address] < threshold_low || local_sigma.real_values[address] > threshold_high ) {
                    maximum_score_modified.real_values[address_in_score] = 0.0;
                }
                address++;
                address_in_score--;
            }
            address += local_sigma.padding_jump_value;
        }

#ifdef dump_intermediate_files
        maximum_score.QuickAndDirtyWriteSlice("dbg_maximum_score_2.mrc", 1);
#endif
    }
}

void ParticleFinder::DoTemplateMatching( ) {

    Image template_medium;
    Image template_large;

    // Now we can look for the templates in the background-whitened micrograph
    wxPrintf("\nTemplate matching...\n");
    my_progress_bar = new ProgressBar(number_of_templates);
    float  template_b_value[number_of_templates];
    float  expected_density_of_false_positives[number_of_templates];
    double b_numerator, b_denominator;

    template_medium.Allocate(minimum_box_size_for_object_with_psf, minimum_box_size_for_object_with_psf, true);
    template_medium.SetToConstant(0.0);
    template_large.Allocate(micrograph_whitened.logical_x_dimension, micrograph_whitened.logical_y_dimension, true);
    maximum_score.Allocate(micrograph_whitened.logical_x_dimension, micrograph_whitened.logical_y_dimension, true);
    maximum_score.SetToConstant(0.0);
    template_giving_maximum_score.Allocate(micrograph_whitened.logical_x_dimension, micrograph_whitened.logical_y_dimension, true);
    template_giving_maximum_score.SetToConstant(0.0);
    template_rotation_giving_maximum_score = template_giving_maximum_score;
    for ( int template_counter = 0; template_counter < number_of_templates; template_counter++ ) {
        wxPrintf("Working on template %i\n", template_counter);

        // Ideally, one would pad the template image to the micrograph dimensions before applying the CTF,
        // so that one wouldn't have to worry about PSF spread, or at least one would pad them large enough
        // to allow for proper CTF correction
        // For performance however, one does the CTF and filtering on a small box before padding

        for ( int rotation_counter = 0; rotation_counter < number_of_template_rotations; rotation_counter++ ) {

            // Prepare the template for matching
            // (rotate it, pad it, apply CTF, apply whitening filter, pad to micrograph dimensions
            PrepareTemplateForMatching(&template_image[template_counter], template_medium, 360.0 / number_of_template_rotations * rotation_counter, &micrograph_ctf, &background_whitening_filter);

            // Clip into micrograph-sized image
            MyDebugAssertTrue(template_medium.is_in_real_space, "template_medium should be in real space");
            //template_medium.ClipInto(&template_large);
            template_medium.ClipIntoLargerRealSpace2D(&template_large);

            // We want to compute the statistic B (Eqn 5 of Sigworth 2004) to help estimate the expected
            // rate of false negatives later on
            if ( rotation_counter == 0 ) {
                template_medium.ForwardFFT( );
                template_medium.NormalizeFT( );
                template_medium.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
                b_numerator   = 0.0;
                b_denominator = 0.0;
                for ( int curve_counter = 0; curve_counter < current_power_spectrum.NumberOfPoints( ); curve_counter++ ) {
                    b_numerator += pow(current_power_spectrum.data_x[curve_counter], 2) * current_power_spectrum.data_y[curve_counter];
                    b_denominator += current_power_spectrum.data_y[curve_counter];
                }
                template_b_value[template_counter] = b_numerator / b_denominator;
                //wxPrintf("B value for template %i = %f nm-2\n",template_counter+1,template_b_value[template_counter] / pixel_size / pixel_size * 100.0);

                // Expected density of false positives (Eqn 4 of Sigworth 2004), per square micron
                expected_density_of_false_positives[template_counter] = 100000000.0 * sqrtf(2.0 * PI) * template_b_value[template_counter] * minimum_peak_height_for_candidate_particles * expf(-powf(minimum_peak_height_for_candidate_particles, 2) * 0.5);
                //wxPrintf("Expected density of spurious peaks (per squared micron) = %g\n",expected_density_of_false_positives[template_counter]);
            }

            //wxPrintf("After clipping to large but before normalization, the template variance is %f, std %f (medium dim = %i; large dim = %i)\n",template_large.ReturnVarianceOfRealValues(),sqrtf(template_large.ReturnVarianceOfRealValues()),template_medium.logical_x_dimension,template_large.logical_x_dimension);

            // Here are the conditions which should be met by the template according to
            // Sigworth (2004):
            // filtered by |CTF|, filtered by background-whitening filter,
            // sum of squares = 1.0, non-zero only

//#define extra_check
#ifdef extra_check
            my_dist = template_large.ReturnDistributionOfRealValues( );
            //template_large.QuickAndDirtyWriteSlice("dbg_template_large.mrc",template_counter * number_of_template_rotations + rotation_counter + 1);
            MyDebugAssertTrue(fabsf(template_large.ReturnAverageOfRealValuesOnEdges( )) < 0.01, "Ooops, template is not 0.0 on edges, it is %g\n", template_large.ReturnAverageOfRealValuesOnEdges( ));
            MyDebugAssertTrue(fabsf(my_dist.GetSampleSumOfSquares( ) - 1.0) < 0.01, "Large template sum of squares is not 1.0, it is %f\n", my_dist.GetSampleSumOfSquares( ));
            //MyDebugAssertTrue(fabsf(my_dist.GetSampleVariance() - 1.0) < 0.01,"Large template variance is not 1.0, it is %f\n",my_dist.GetSampleVariance());
#endif

            // Go to Fourier space
            template_large.ForwardFFT(false);
            template_large.NormalizeFT( );

            // Cross correlation (matched filter)
            template_large.ConjugateMultiplyPixelWise(micrograph_whitened);
            template_large.BackwardFFT( );
            //template_large.NormalizeFT(); // This is necessary for the scaling to be correct
            //template_large.QuickAndDirtyWriteSlice("dbg_cc.mrc",template_counter * number_of_template_rotations + rotation_counter + 1);

            // Keep track of the best score for every pixel and the template which gave this best score
            long address = 0;
            for ( int j = 0; j < maximum_score.logical_y_dimension; j++ ) {
                for ( int i = 0; i < maximum_score.logical_x_dimension; i++ ) {
                    if ( template_large.real_values[address] > maximum_score.real_values[address] ) {
                        maximum_score.real_values[address]                          = template_large.real_values[address];
                        template_giving_maximum_score.real_values[address]          = float(template_counter);
                        template_rotation_giving_maximum_score.real_values[address] = 360.0 / number_of_template_rotations * rotation_counter;
                    }
                    address++;
                }
                address += maximum_score.padding_jump_value;
            }
        }

        my_progress_bar->Update(template_counter + 1);
    }
    delete my_progress_bar;
    maximum_score.SwapRealSpaceQuadrants( );
    maximum_score.object_is_centred_in_box = true;
#ifdef dump_intermediate_files
    maximum_score.QuickAndDirtyWriteSlice("dbg_maximum_score.mrc", 1);
#endif
    template_giving_maximum_score.SwapRealSpaceQuadrants( );
    template_giving_maximum_score.object_is_centred_in_box = true;
    template_rotation_giving_maximum_score.SwapRealSpaceQuadrants( );
    template_rotation_giving_maximum_score.object_is_centred_in_box = true;
#ifdef dump_intermediate_files
    template_giving_maximum_score.QuickAndDirtyWriteSlice("dbg_template_giving_maximum_score.mrc", 1);
#endif
}

void ParticleFinder::WhitenMicrographBackground( ) {

    long  address;
    Image local_sigma_modified;
    local_sigma_modified = local_sigma;

    switch ( algorithm_to_find_background ) {
        case (0):
            // Let's look for the areas of lowest variance, which we will assume are plain ice, so we can work out a whitening filter later on
            // WARNING; this is liable to bias the whitening filter against the templates
            local_sigma_modified.MultiplyByConstant(-1.0);
            break;
        case (1):
            // fold the values around such that the mode becomes the maximum
            address = 0;
            for ( int j = 0; j < local_sigma.logical_y_dimension; j++ ) {
                for ( int i = 0; i < local_sigma.logical_x_dimension; i++ ) {
                    local_sigma_modified.real_values[address] = -1.0 * fabs(local_sigma.real_values[address] - local_sigma_mode);
                    address++;
                }
                address += local_sigma_modified.padding_jump_value;
            }
            break;
        default:
            MyDebugAssertTrue(false, "Oops, bad algorithm number : %i\n", algorithm_to_find_background);
    }

#ifdef dump_intermediate_files
    NumericTextFile temp_coos_file("dbg_background_box.plt", OPEN_TO_WRITE, 3);
#endif

    Image box;
    box.Allocate(maximum_radius_in_pixels * 2 + 2, maximum_radius_in_pixels * 2 + 2, 1);
    background_power_spectrum.ZeroYData( );

    wxPrintf("\nEstimating background whitening filter...\n");
    my_progress_bar = new ProgressBar(number_of_background_boxes);
    for ( int background_box_counter = 0; background_box_counter < number_of_background_boxes; background_box_counter++ ) {
        // Find the area to be boxed out
#ifdef dump_intermediate_files
        local_sigma_modified.QuickAndDirtyWriteSlice("dbg_latest_variance.mrc", background_box_counter + 1);
#endif
        Peak my_peak = local_sigma_modified.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, box.physical_address_of_box_center_x + 1);

        if ( background_box_counter >= number_of_background_boxes_to_skip ) {
            // Box out an image from the micrograph at that location
            if ( ! micrograph.is_in_real_space )
                micrograph.BackwardFFT( );
            micrograph.ClipInto(&box, 0.0, false, 1.0, int(my_peak.x), int(my_peak.y), 0);
            //wxPrintf("Boxed out background at position %i, %i = %i, %i; peak value = %f\n",int(my_peak.x),int(my_peak.y),int(my_peak.x)+local_sigma.physical_address_of_box_center_x,int(my_peak.y)+local_sigma.physical_address_of_box_center_y,my_peak.value);
#ifdef dump_intermediate_files
            float temp_float[3];
            box.QuickAndDirtyWriteSlice("dbg_background_box.mrc", background_box_counter + 1 - number_of_background_boxes_to_skip);
            temp_float[1] = micrograph.logical_x_dimension - (micrograph.physical_address_of_box_center_x - (my_peak.x));
            temp_float[0] = micrograph.physical_address_of_box_center_y - (my_peak.y);
            temp_float[1] = temp_float[1] * pixel_size / original_micrograph_pixel_size + 1.0;
            temp_float[0] = temp_float[0] * pixel_size / original_micrograph_pixel_size + 1.0;
            temp_float[2] = 1.0;
            temp_coos_file.WriteLine(temp_float);
#endif
            box.ForwardFFT(false);
            box.NormalizeFT( );
            box.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
            background_power_spectrum.AddWith(&current_power_spectrum);
        }

        // Before we look for the next background box, we need to set the pixels we have already extracted from
        //  the variance map to a terrible value so they don't get picked again
        SetAreaToIgnore(local_sigma_modified, int(my_peak.x) + local_sigma_modified.physical_address_of_box_center_x, int(my_peak.y) + local_sigma_modified.physical_address_of_box_center_y, &box, -99999.99); // TODO: use a better value, such as the minimum value found in the image

        my_progress_bar->Update(background_box_counter + 1);
    }
    background_power_spectrum.MultiplyByConstant(1.0 / float(number_of_background_boxes - number_of_background_boxes_to_skip));
#ifdef dump_intermediate_files
    background_power_spectrum.WriteToFile("dbg_background_spectrum.txt");
#endif
    delete my_progress_bar;

    // average_amplitude_spectrum should now contain a decent estimate of the the input micrograph's noise spectrum

    // TODO: look into fitting an analytical function to the whitening filter, a la Sigworth (2004)

    // Next, we need to whiten the noise in the micrograph and ensure that at each pixel
    // it has a variance of 1.0
    background_whitening_filter = background_power_spectrum;
    for ( int counter = 0; counter < background_power_spectrum.NumberOfPoints( ); counter++ ) {
        if ( background_power_spectrum.data_y[counter] > 0.0 ) {
            background_whitening_filter.data_y[counter] = 1.0 / sqrtf(background_power_spectrum.data_y[counter]);
        }
    }
#ifdef dump_intermediate_files
    background_whitening_filter.WriteToFile("dbg_background_whitening_filter.txt");
#endif
    micrograph_whitened = micrograph;
    micrograph_whitened.ForwardFFT(false);
    micrograph_whitened.NormalizeFT( );
    micrograph_whitened.ApplyCurveFilter(&background_whitening_filter);
#ifdef dump_intermediate_files
    micrograph_whitened.BackwardFFT( );
    micrograph_whitened.NormalizeFT( );
    micrograph_whitened.QuickAndDirtyWriteSlice("dbg_micrograph_whitened.mrc", 1);
    micrograph_whitened.ForwardFFT(false);
    micrograph_whitened.NormalizeFT( );
#endif
    //wxPrintf("DBG: micrograph var, std = %f, %f\n",micrograph.ReturnVarianceOfRealValues(),sqrt(micrograph.ReturnVarianceOfRealValues()));

#ifdef check_whitening_worked
    // Check the micrograph amplitude spectrum
    micrograph_whitened.ForwardFFT(false);
    micrograph_whitened.NormalizeFT( );
    micrograph_whitened.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
#ifdef dump_intermediate_files
    current_power_spectrum.WriteToFile("dbg_micrograph_whitened_spectrum.txt");
#endif

    // Check the background boxes again, recompute their average amplitude spectrum
    wxPrintf("\nChecking whitening worked correctly (debug)...\n");
    my_progress_bar = new ProgressBar(number_of_background_boxes);
    temp_curve.ZeroYData( );
    EmpiricalDistribution dist;
    for ( int background_box_counter = 0; background_box_counter < number_of_background_boxes; background_box_counter++ ) {
        if ( background_box_counter >= number_of_background_boxes_to_skip ) {
            box.QuickAndDirtyReadSlice("dbg_background_box.mrc", background_box_counter + 1 - number_of_background_boxes_to_skip);
            dist = box.ReturnDistributionOfRealValues( );
            //wxPrintf("Background box %i of %i, mean = %f, std = %f\n",background_box_counter+1, number_of_background_boxes,dist.GetSampleMean(),sqrtf(dist.GetSampleVariance()));
            box.ForwardFFT(false);
            box.NormalizeFT( );
            box.ApplyCurveFilter(&background_whitening_filter);
            box.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
            temp_curve.AddWith(&current_power_spectrum);
            box.BackwardFFT( );
            box.NormalizeFT( );
            dist = box.ReturnDistributionOfRealValues( );
            box.QuickAndDirtyWriteSlice("dbg_background_box_whitened.mrc", background_box_counter + 1 - number_of_background_boxes_to_skip);
        }
        my_progress_bar->Update(background_box_counter + 1);
    }
    temp_curve.MultiplyByConstant(1.0 / float(number_of_background_boxes - number_of_background_boxes_to_skip));
#ifdef dump_intermediate_files
    temp_curve.WriteToFile("dbg_whitened_background_spectrum.txt");
#endif
    delete my_progress_bar;
#endif
}

/*
 * Call if
 * - new micrograph or micrograph_bp
 * - change in maximum radius
 */
void ParticleFinder::UpdateLocalMeanAndSigma( ) {
    // We will need a few images with the same dimensions as the micrograph
    Image mask_image;
    mask_image.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension, 1);
    local_mean.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension, 1);
    local_sigma.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension, 1);

    // Prepare a mask
    mask_image.Allocate(micrograph.logical_x_dimension, micrograph.logical_y_dimension, 1);
    mask_image.SetToConstant(1.0);
    //mask_image.CircleMaskWithValue(mask_radius_in_pixels,0.0);
    mask_image.SquareMaskWithValue(maximum_radius_in_pixels * 2, 0.0);
    long number_of_pixels_within_mask = mask_image.ReturnAverageOfRealValues( ) * mask_image.logical_x_dimension * mask_image.logical_y_dimension;
#ifdef dump_intermediate_files
    mask_image.QuickAndDirtyWriteSlice("dbg_mask.mrc", 1);
#endif

    // Compute local average and local sigma
    micrograph_bp.ComputeLocalMeanAndVarianceMaps(&local_mean, &local_sigma, &mask_image, number_of_pixels_within_mask);
#ifdef dump_intermediate_files
    local_mean.QuickAndDirtyWriteSlice("dbg_local_average.mrc", 1);
    local_sigma.QuickAndDirtyWriteSlice("dbg_local_variance.mrc", 1);
#endif
    local_sigma.SetMinimumValue(0.0);
    local_sigma.SquareRootRealValues( );
    MyDebugAssertFalse(local_sigma.HasNan( ), "Oops, local_sigma has NaN\n");
    //local_sigma.ForwardFFT();
    //local_sigma.ApplyCurveFilter(&template_amplitude_spectrum);
    //local_sigma.BackwardFFT();

#ifdef dump_intermediate_files
    // Debug dumps
    local_mean.QuickAndDirtyWriteSlice("dbg_local_average.mrc", 1);
    local_sigma.QuickAndDirtyWriteSlice("dbg_local_sigma.mrc", 1);
#endif

    // Get some statistics on the local_sigma image
    Curve local_sigma_histogram;
    local_sigma.ComputeHistogramOfRealValuesCurve(&local_sigma_histogram);
    float local_sigma_histogram_max_value;
    local_sigma_histogram.ComputeMaximumValueAndMode(local_sigma_histogram_max_value, local_sigma_mode);
    local_sigma_fwhm = local_sigma_histogram.ReturnFullWidthAtGivenValue(local_sigma_histogram_max_value * 0.5);

    // Get some statistics on the local_mean image
    Curve local_mean_histogram;
    local_mean.ComputeHistogramOfRealValuesCurve(&local_mean_histogram);
    float local_mean_histogram_max_value;
    local_mean_histogram.ComputeMaximumValueAndMode(local_mean_histogram_max_value, local_mean_mode);
    local_mean_fwhm = local_mean_histogram.ReturnFullWidthAtGivenValue(local_mean_histogram_max_value * 0.5);

#ifdef dump_intermediate_files
    local_sigma_histogram.WriteToFile("dbg_local_sigma_histogram.txt");
    local_mean_histogram.WriteToFile("dbg_local_mean_histogram.txt");
#endif

    // keep a copy of the unmodified variance
    //local_sigma_modified = local_sigma;
}

/*
 * Call if
 * - change in template power spectrum
 */
void ParticleFinder::BandPassMicrograph( ) {
    micrograph_bp = micrograph;
    if ( micrograph_bp.is_in_real_space )
        micrograph_bp.ForwardFFT(false);
    micrograph_bp.NormalizeFT( );
    micrograph_bp.ApplyCurveFilter(&template_power_spectrum);
    micrograph_bp.BackwardFFT( );
    micrograph_bp.NormalizeFT( );
#ifdef dump_intermediate_files
    micrograph_bp.QuickAndDirtyWriteSlice("dbg_micrograph_filtered.mrc", 1);
#endif
}

/*
 * Call if
 * - new micrograph filename
 * - change in resampling (e.g. new high resolution)
 */
void ParticleFinder::ReadAndResampleMicrograph( ) {
    Image temp_image;

    temp_image.ReadSlice(&micrograph_file, 1);
    temp_image.ForwardFFT(false);
    temp_image.NormalizeFT( );
    micrograph_mean = temp_image.real_values[0] / sqrtf(float(temp_image.number_of_real_space_pixels));
    micrograph.Allocate(new_micrograph_dimension_x, new_micrograph_dimension_y, 1, false);
    temp_image.ClipInto(&micrograph);

    // Phase flip
    micrograph.ApplyCTFPhaseFlip(micrograph_ctf);

    // High pass filter to remove density ramps (due to variations in ice thickness)
    micrograph.CosineMask(10.0 / float(micrograph.logical_x_dimension), 20.0 / float(micrograph.logical_x_dimension), true);

    // Back to real space
    micrograph.BackwardFFT( );
    if ( particles_are_white ) {
        micrograph.NormalizeFTAndInvertRealValues( );
    }
    else {
        micrograph.NormalizeFT( );
    }

    // Write the raw micrograph's spectrum to disk
#ifdef dump_intermediate_files
    micrograph.ForwardFFT(false);
    micrograph.NormalizeFT( );
    micrograph.Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
    micrograph.BackwardFFT( );
    micrograph.NormalizeFT( );
    current_power_spectrum.WriteToFile("dbg_micrograph_power.txt");
#endif
}

/*
 * Call when
 * - minimum box size for picking changes
 */
void ParticleFinder::SetupCurveObjects( ) {
    // We will estimate the amplitude spectrum of the templates using curve objects
    // with spatial frequency (0.5 is Nyquist) on the X axis
    template_power_spectrum.SetupXAxis(0.0, sqrtf(2.0) * 0.5, minimum_box_size_for_picking);
    background_power_spectrum          = template_power_spectrum;
    current_power_spectrum             = template_power_spectrum;
    current_number_of_fourier_elements = template_power_spectrum;
    temp_curve                         = template_power_spectrum;
}

void ParticleFinder::GenerateATemplate( ) {
    MyDebugAssertTrue(number_of_templates == 1, "Bad number of templates: %i", number_of_templates);

    template_image[0].Allocate(minimum_box_size_for_picking, minimum_box_size_for_picking, 1);
    template_image[0].SetToConstant(1.0);
    template_image[0].CosineMask(typical_radius_in_pixels, 5.0, false, true, 0.0);
#ifdef dump_intermediate_files
    template_image[0].QuickAndDirtyWriteSlice("dbg_template.mrc", 1);
#endif
    template_image[0].ForwardFFT(false);
    template_image[0].NormalizeFT( );
    template_image[0].Compute1DPowerSpectrumCurve(&template_power_spectrum, &current_number_of_fourier_elements);

    // Normalize the curve to turn it into a band-pass filter
    template_power_spectrum.NormalizeMaximumValue( );
    template_power_spectrum.SquareRoot( );
}

void ParticleFinder::ReadTemplatesFromDisk( ) {
    MyDebugAssertTrue(already_have_templates, "Should only be called when templates are given as input");

    Image temp_image;

    // TODO: check the template dimensions are sufficient to accomodate for CTF correction
    wxPrintf("\nEstimating template power spectrum...\n");
    my_progress_bar = new ProgressBar(number_of_templates);
    for ( int template_counter = 0; template_counter < number_of_templates; template_counter++ ) {
        temp_image.ReadSlice(&template_file, template_counter + 1);
        temp_image.ForwardFFT(false);
        temp_image.NormalizeFT( );
        template_image[template_counter].Allocate(new_template_dimension, new_template_dimension, 1, false);
        temp_image.ClipInto(&template_image[template_counter]);
        if ( average_templates_radially ) {
            if ( ! template_image[template_counter].is_in_real_space ) {
                template_image[template_counter].BackwardFFT( );
                template_image[template_counter].NormalizeFT( );
            }
            template_image[template_counter].AverageRadially( );
        }
        if ( template_image[template_counter].is_in_real_space ) {
            template_image[template_counter].ForwardFFT( );
            template_image[template_counter].NormalizeFT( );
        }
        template_image[template_counter].Compute1DPowerSpectrumCurve(&current_power_spectrum, &current_number_of_fourier_elements);
        template_power_spectrum.AddWith(&current_power_spectrum);
        // Now go back to real space and pad to be big enough to accomodate PSF
        template_image[template_counter].BackwardFFT( );
        template_image[template_counter].NormalizeFT( );
        temp_image.Allocate(minimum_box_size_for_object_with_psf, minimum_box_size_for_object_with_psf, 1, true);
        template_image[template_counter].AddConstant(-template_image[template_counter].ReturnAverageOfRealValuesOnEdges( ));
        template_image[template_counter].ClipInto(&temp_image);
        //template_image[template_counter].Consume(&temp_image); // This is buggy - can't work out why
        template_image[template_counter] = temp_image;
#ifdef dump_intermediate_files
        template_image[template_counter].QuickAndDirtyWriteSlice("dbg_template.mrc", template_counter + 1);
#endif
        my_progress_bar->Update(template_counter + 1);
    }
    template_power_spectrum.MultiplyByConstant(1.0 / float(template_file.ReturnNumberOfSlices( )));
    delete my_progress_bar;

#ifdef sum_all_templates
    for ( int template_counter = 1; template_counter < number_of_templates; template_counter++ ) {
        template_image[0].AddImage(&template_image[template_counter]);
    }
#ifdef dump_intermediate_files
    template_image[0].QuickAndDirtyWriteSlice("dbg_template.mrc", 1);
#endif
    number_of_templates = 1;
#endif

    // Normalize the curve to turn it into a band-pass filter
    template_power_spectrum.NormalizeMaximumValue( );
    template_power_spectrum.SquareRoot( );
}

void ParticleFinder::AllocateTemplateImages( ) {
    template_image = new Image[number_of_templates];
}

void ParticleFinder::DeallocateTemplateImages( ) {
    for ( int counter = 0; counter < number_of_templates; counter++ ) {
        template_image[counter].Deallocate( );
    }
    if ( number_of_templates > 0 )
        delete[] template_image;
    number_of_templates = 0;
}

/*
 * Call this if
 * - new pixel_size
 * - new CTF
 * - new highest_resolution
 */
void ParticleFinder::UpdateMinimumBoxSize( ) {
    // Let's decide on a box size for picking (on the resampled micrograph)
    minimum_box_size_for_object_with_psf = 2 * (maximum_radius_in_pixels + int(std::max(micrograph_ctf.GetDefocus1( ), micrograph_ctf.GetDefocus2( )) * micrograph_ctf.GetWavelength( ) / highest_resolution_to_use * pixel_size));
    minimum_box_size_for_object_with_psf = std::max(4, minimum_box_size_for_object_with_psf);
    minimum_box_size_for_picking         = minimum_box_size_for_object_with_psf;

    const int minimum_box_size_for_picking_unbinned = int(float(minimum_box_size_for_picking) * pixel_size / original_micrograph_pixel_size) + 1;

    if ( output_stack_box_size < minimum_box_size_for_picking_unbinned && output_stack_box_size > 0 ) {
        wxPrintf("Warning: user-supplied box size (%i) is smaller than minimum recommended box size given the max radius and the defocus (%i)\n", output_stack_box_size, minimum_box_size_for_picking_unbinned);
    }
}

/*
 * Call this if
 * - new templates
 * - already_have_templates changes
 */
void ParticleFinder::OpenTemplatesAndUpdateDimensions( ) {

    if ( already_have_templates ) {
        template_file.OpenFile(templates_filename.ToStdString( ), false);
        number_of_templates = template_file.ReturnNumberOfSlices( );
        MyDebugAssertTrue(template_file.ReturnXSize( ) == template_file.ReturnYSize( ), "Oops, template is not in a square box");
        new_template_dimension        = myroundint(float(template_file.ReturnXSize( )) / pixel_size * original_micrograph_pixel_size);
        float new_template_pixel_size = original_micrograph_pixel_size * float(template_file.ReturnXSize( )) / float(new_template_dimension);

        // Check for any pixel size mismatches between the micrograph and the templates
        float pixel_size_mistmatch_due_to_rescaling = abs(pixel_size - new_template_pixel_size) / pixel_size * 100.0;
        if ( pixel_size_mistmatch_due_to_rescaling > 0.5 ) {
            wxPrintf("Warning: internal image resampling led to significant scaling mistmatch between micrograph and template, of %f %\n", pixel_size_mistmatch_due_to_rescaling);
            MyDebugAssertTrue(false, "Problematic resampling");
        }
    }
    else {
        number_of_templates = 1;
    }
}

void ParticleFinder::UpdateMaximumRadiusInPixels( ) {
    maximum_radius_in_pixels = maximum_radius_in_angstroms / pixel_size;
}

void ParticleFinder::UpdateTypicalRadiusInPixels( ) {
    typical_radius_in_pixels = typical_radius_in_angstroms / pixel_size;
}

void ParticleFinder::UpdatePixelSizeFromMicrographDimensions( ) {
    pixel_size = original_micrograph_pixel_size * micrograph_file.ReturnXSize( ) / new_micrograph_dimension_x;

    // When the pixel size changes, these need to be updated
    UpdateMaximumRadiusInPixels( );
    UpdateTypicalRadiusInPixels( );
}

CTF ParticleFinder::ReturnMicrographCTFWithOriginalPixelSize( ) {
    CTF ctf_with_original_pixel_size;
    ctf_with_original_pixel_size.CopyFrom(micrograph_ctf);
    ctf_with_original_pixel_size.ChangePixelSize(pixel_size, original_micrograph_pixel_size);
    return ctf_with_original_pixel_size;
}

/*
 * Call when
 * - new micrograph filename
 * - change of highest resolution
 */
void ParticleFinder::OpenMicrographAndUpdateDimensions( ) {
    // Open input files so we know dimensions
    micrograph_file.OpenFile(micrograph_filename.ToStdString( ), false);
    MyDebugAssertTrue(micrograph_file.ReturnNumberOfSlices( ) == 1, "Input micrograph file should only contain one image for now");

    // First we look for a nice factorizable micrograph dimension which gives approximately the desired pixel size
    if ( micrograph_file.ReturnXSize( ) == micrograph_file.ReturnYSize( ) ) {
        new_micrograph_dimension_x = ReturnClosestFactorizedUpper(int(micrograph_file.ReturnXSize( ) * original_micrograph_pixel_size / highest_resolution_to_use * 2.0), 5, true);
        new_micrograph_dimension_y = new_micrograph_dimension_x;
    }
    else if ( micrograph_file.ReturnXSize( ) > micrograph_file.ReturnYSize( ) ) {
        new_micrograph_dimension_y = ReturnClosestFactorizedUpper(int(micrograph_file.ReturnYSize( ) * original_micrograph_pixel_size / highest_resolution_to_use * 2.0), 5, true);
        new_micrograph_dimension_x = myroundint(float(new_micrograph_dimension_y) / float(micrograph_file.ReturnYSize( )) * float(micrograph_file.ReturnXSize( )));
    }
    else {
        new_micrograph_dimension_x = ReturnClosestFactorizedUpper(int(micrograph_file.ReturnXSize( ) * original_micrograph_pixel_size / highest_resolution_to_use * 2.0), 5, true);
        new_micrograph_dimension_y = myroundint(float(new_micrograph_dimension_x) / float(micrograph_file.ReturnXSize( )) * float(micrograph_file.ReturnYSize( )));
    }

    // Check for any distortions due to rescaling, warn the user if any significant
    float new_micrograph_pixel_size_x            = original_micrograph_pixel_size * float(micrograph_file.ReturnXSize( )) / float(new_micrograph_dimension_x);
    float new_micrograph_pixel_size_y            = original_micrograph_pixel_size * float(micrograph_file.ReturnYSize( )) / float(new_micrograph_dimension_y);
    float micrograph_distortion_due_to_rescaling = float(abs(new_micrograph_pixel_size_x - new_micrograph_pixel_size_y)) / float(new_micrograph_pixel_size_x) * 100.0;
    if ( micrograph_distortion_due_to_rescaling > 1.0 ) {
        wxPrintf("Warning: internal resampling of the micrograph led to significant scaling distortion of %f %\n", micrograph_distortion_due_to_rescaling);
    }
}

/*
 * Call this if
 * - CTF parameters are changed
 * - pixel size is changed
 */
void ParticleFinder::UpdateCTF( ) {
    // Set up a CTF object
    micrograph_ctf.Init(acceleration_voltage_in_keV, spherical_aberration_in_mm, amplitude_contrast, defocus_1_in_angstroms, defocus_2_in_angstroms, astigmatism_angle_in_degrees, pixel_size, additional_phase_shift_in_radians);
}

void ParticleFinder::PrepareTemplateForMatching(Image* template_image, Image& prepared_image, float in_plane_rotation, CTF* micrograph_ctf, Curve* whitening_filter) {

    MyDebugAssertTrue(template_image->is_in_memory, "template not allocated");
    MyDebugAssertTrue(prepared_image.is_in_memory, "prepared image not allocated");
    MyDebugAssertTrue(template_image->HasSameDimensionsAs(&prepared_image), "images don't have same dimensions");

    AnglesAndShifts template_rotation;

    Image temporary_image;

    EmpiricalDistribution my_dist;

    temporary_image = template_image;

    // Zero float the background
    temporary_image.AddConstant(-temporary_image.ReturnAverageOfRealValuesOnEdges( ));

    // Rotate the template
    if ( in_plane_rotation != 0.0 ) {
        if ( in_plane_rotation != 0.0 && fabs(in_plane_rotation - 90.0) > 0.0001 && fabs(in_plane_rotation - 180.0) > 0.001 && fabs(in_plane_rotation - 270.0) > 0.001 ) {
            if ( ! temporary_image.is_in_real_space ) {
                temporary_image.BackwardFFT( );
                temporary_image.NormalizeFT( );
            }
            temporary_image.CorrectSinc( );
        }

        if ( temporary_image.is_in_real_space ) {
            temporary_image.ForwardFFT(false);
            temporary_image.NormalizeFT( );
        }

        if ( temporary_image.object_is_centred_in_box ) {
            temporary_image.SwapRealSpaceQuadrants( );
        }

        template_rotation.Init(0.0, 0.0, in_plane_rotation, 0.0, 0.0);
        temporary_image.RotateFourier2D(prepared_image, template_rotation, 1.0, false);
        prepared_image.SwapRealSpaceQuadrants( );
    }
    else {
        prepared_image = template_image;
    }

    if ( ! prepared_image.is_in_real_space ) {
        prepared_image.BackwardFFT( );
        prepared_image.NormalizeFT( );
    }

    // Get ready to apply CTF
    prepared_image.ForwardFFT(false);
    prepared_image.NormalizeFT( );

    // Multiply the template by the |CTF|
    prepared_image.ApplyCTF(*micrograph_ctf, true);

    // Apply the background whitening filter
    // NOTE: this only really makes sense if the templates were generated from the micrographs, I think
    prepared_image.ApplyCurveFilter(whitening_filter);

    // Make sure background goes to 0.0, and that total power = 1.0
    prepared_image.BackwardFFT( );
    prepared_image.NormalizeFT( );
    prepared_image.AddConstant(-prepared_image.ReturnAverageOfRealValuesOnEdges( ));
    my_dist = prepared_image.ReturnDistributionOfRealValues( );
    prepared_image.DivideByConstant(sqrtf(my_dist.GetSampleSumOfSquares( )));
}

// TODO: make this faster by only looping over relevant area of my_image?
void ParticleFinder::SetCircularAreaToIgnore(Image& my_image, const int central_pixel_address_x, const int central_pixel_address_y, const float wanted_radius, const float wanted_value) {

    const float wanted_radius_sq = powf(wanted_radius, 2);

    float sq_dist_x, sq_dist_y;
    long  address = 0;
    for ( int j = 0; j < my_image.logical_y_dimension; j++ ) {
        sq_dist_y = float(pow(j - central_pixel_address_y, 2));
        for ( int i = 0; i < my_image.logical_x_dimension; i++ ) {
            sq_dist_x = float(pow(i - central_pixel_address_x, 2));
            // The square centered at the pixel
            if ( sq_dist_x + sq_dist_y <= wanted_radius_sq ) {
                my_image.real_values[address] = wanted_value;
            }
            address++;
        }
        address += my_image.padding_jump_value;
    }
}

// The box_image is just used to get dimensions and for addressing convenience
void ParticleFinder::SetAreaToIgnore(Image& my_image, int central_pixel_address_x, int central_pixel_address_y, Image* box_image, float wanted_value) {

    const int box_lbound_x = central_pixel_address_x - box_image->physical_address_of_box_center_x;
    const int box_ubound_x = box_lbound_x + box_image->logical_x_dimension - 1;

    const int box_lbound_y = central_pixel_address_y - box_image->physical_address_of_box_center_y;
    const int box_ubound_y = box_lbound_y + box_image->logical_y_dimension - 1;

    long address = 0;
    for ( int j = 0; j < my_image.logical_y_dimension; j++ ) {
        for ( int i = 0; i < my_image.logical_x_dimension; i++ ) {
            // The square centered at the pixel
            if ( i >= box_lbound_x && i <= box_ubound_x && j >= box_lbound_y && j <= box_ubound_y ) {
                my_image.real_values[address] = wanted_value;
            }
            address++;
        }
        address += my_image.padding_jump_value;
    }
}

/*

// Implementation of Equation 8 of Scheres (JSB 2015)
void ComputeScheresPickingFunction(Image *micrograph, Image *micrograph_local_mean, Image *micrograph_local_stdev, Image *template_image, float mask_radius, long number_of_pixels_within_mask, Image *scoring_function)
{
	// We assume the template has been normalized such that its mean is 0.0 and its stdev 1.0 outside the mask
	// (I'm not sure that this is exactly what Sjors does, nor whether this is the correct thing to do)
#ifdef DEBUG
	EmpiricalDistribution template_values_outside_radius = template_image->ReturnDistributionOfRealValues(mask_radius,true);
	MyDebugAssertTrue(fabs(template_values_outside_radius.GetSampleMean()) < 0.001,"Template should be normalized to have mean value of 0.0 outside radius");
	const float template_sum_outside_of_mask = template_values_outside_radius.GetSampleSum();
	const float template_sum_of_squares_outside_of_mask = template_values_outside_radius.GetSampleSumOfSquares();
#endif
	EmpiricalDistribution template_values_inside_radius  = template_image->ReturnDistributionOfRealValues(mask_radius,false);
	const float template_sum_inside_of_mask = template_values_inside_radius.GetSampleSum();
	const float template_sum_of_squares_inside_of_mask = template_values_inside_radius.GetSampleSumOfSquares();
	wxPrintf("Template sum of squares inside of mask = %e\n", template_sum_of_squares_inside_of_mask);

	Image template_image_large; // TODO: don't allocate this within this subroutine, pass the memory around

	// We need a version of the masked template padded to the same dimensions as the micrograph
	template_image_large.Allocate(micrograph->logical_x_dimension,micrograph->logical_y_dimension,1);
	template_image->ClipIntoLargerRealSpace2D(&template_image_large,template_image->ReturnAverageOfRealValuesOnEdges());

	//
	long number_of_pixels_in_template = template_image->logical_x_dimension * template_image->logical_y_dimension;

	// Cross-correlation
	if (micrograph->is_in_real_space) micrograph->ForwardFFT(false);
	//template_image_large.DivideByConstant(number_of_pixels_within_mask);
	template_image_large.ForwardFFT(false);
	scoring_function->CopyFrom(micrograph);
	scoring_function->ConjugateMultiplyPixelWise(template_image_large);
	scoring_function->SwapRealSpaceQuadrants();
	scoring_function->BackwardFFT();




	// Equation 6 & equation 8
	long pixel_counter=0;
	for (int counter_y=0; counter_y < micrograph->logical_y_dimension; counter_y++)
	{
		for (int counter_x=0; counter_x < micrograph->logical_x_dimension; counter_x++)
		{
			// Equation 6
			scoring_function->real_values[pixel_counter] =   exp( scoring_function->real_values[pixel_counter] / micrograph_local_stdev->real_values[pixel_counter]
															    - micrograph_local_mean->real_values[pixel_counter] * template_sum_inside_of_mask / micrograph_local_stdev->real_values[pixel_counter]
															    - 0.5 * template_sum_of_squares_inside_of_mask);

			// Equation 8
			//scoring_function->real_values[pixel_counter] = (exp(scoring_function->real_values[pixel_counter]) - 1.0) / (exp(template_sum_of_squares_inside_of_mask/(2.0*number_of_pixels_within_mask)) - 1.0);


			// Equation 8 (with an extra normalization by the number of pixels in the template - I'm not sure why) (this gives a function very very similar to the normalized CCF
			//scoring_function->real_values[pixel_counter] = (exp(scoring_function->real_values[pixel_counter]/number_of_pixels_in_template) - 1.0) / (exp(template_sum_of_squares_inside_of_mask/(2.0*number_of_pixels_within_mask)) - 1.0);


			pixel_counter++;
		}
		pixel_counter += micrograph->padding_jump_value;
	}

}


// It is assumed that the template image has been normalized and masked
void ComputeNormalizedCrossCorrelationFunction(Image *micrograph, Image *micrograph_local_stdev, Image *template_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *nccf)
{

	MyDebugAssertTrue(template_image->is_in_real_space, "Template image must be in real space");

	Image template_image_large; // TODO: don't allocate this within this subroutine, pass the memory around

	// We need a version of the masked template padded to the same dimensions as the micrograph
	template_image_large.Allocate(micrograph->logical_x_dimension,micrograph->logical_y_dimension,1);
	template_image->ClipIntoLargerRealSpace2D(&template_image_large,template_image->ReturnAverageOfRealValuesOnEdges());

	// Let's compute the local normalized correlation
	// First, convolve the masked tempate with the micrograph
	// Then divide the result by the local std dev times the number of pixels within mask
	if (micrograph->is_in_real_space) micrograph->ForwardFFT();
	nccf->CopyFrom(micrograph);
	if (template_image_large.is_in_real_space) template_image_large.ForwardFFT(false);
	MyDebugAssertFalse(micrograph->is_in_real_space,"Micrograph must be in Fourier space");
	MyDebugAssertFalse(template_image_large.is_in_real_space,"Template must be in Fourier space");
	nccf->ConjugateMultiplyPixelWise(template_image_large);
	nccf->SwapRealSpaceQuadrants();
	nccf->BackwardFFT();
	nccf->DivideByConstant(number_of_pixels_within_mask);
	nccf->DividePixelWise(*micrograph_local_stdev);

}
*/

// Compute the local standard deviation in the image, a la Roseman (Ultramicroscopy, 2003), Eqn 6.
void ParticleFinder::ComputeLocalMeanAndStandardDeviation(Image* micrograph, Image* mask_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image* micrograph_local_mean, Image* micrograph_local_stdev) {
    micrograph_local_mean->CopyFrom(micrograph);

    MyDebugAssertFalse(micrograph_local_mean->is_in_real_space, "Need to be in Fourier space (local average)");
    MyDebugAssertFalse(mask_image->is_in_real_space, "Need to be in Fourier space (mask image)");
    micrograph_local_mean->MultiplyPixelWise(*mask_image);
    micrograph_local_mean->SwapRealSpaceQuadrants( );
    micrograph_local_mean->BackwardFFT( );
    micrograph_local_mean->DivideByConstant(number_of_pixels_within_mask);

    // The square of the local average and the square of the micrograph are now needed in preparation
    // for computing the local variance of the micrograph
    MyDebugAssertFalse(micrograph_local_stdev->is_in_real_space, "Thought this was in F space");
    micrograph_local_stdev->CopyFrom(micrograph);
    micrograph_local_stdev->BackwardFFT( );
    micrograph_local_stdev->SquareRealValues( );

    // Convolute the squared micrograph with the mask image
    MyDebugAssertTrue(micrograph_local_stdev->is_in_real_space, "Thought this would be in R space");
    MyDebugAssertFalse(mask_image->is_in_real_space, "Thought mask was already in Fourier space");
    micrograph_local_stdev->ForwardFFT( );
    micrograph_local_stdev->MultiplyPixelWise(*mask_image);
    micrograph_local_stdev->SwapRealSpaceQuadrants( );
    micrograph_local_stdev->BackwardFFT( );

    // Compute the local variance (Eqn 10 in Roseman 2003)
    micrograph_local_stdev->DivideByConstant(number_of_pixels_within_mask);
    micrograph_local_stdev->SubtractSquaredImage(micrograph_local_mean);

    // Square root to get local standard deviation
    micrograph_local_stdev->SetMinimumValue(0.0); // Otherwise, the image is not save for sqrt
    micrograph_local_stdev->SquareRootRealValues( );
}
