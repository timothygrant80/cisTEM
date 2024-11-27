#include "core_headers.h"

ReconstructedVolume::ReconstructedVolume(float wanted_molecular_mass_in_kDa) {
    volume_initialized     = false;
    projection_initialized = false;
    mask_volume_in_voxels  = 0.0;
    molecular_mass_in_kDa  = wanted_molecular_mass_in_kDa;
    mask_radius            = 0.0;
    has_masked_applied     = false;
    was_corrected          = false;
    //	has_statistics = false;
    has_been_filtered        = false;
    pixel_size               = 0.0;
    symmetry_symbol          = "C1";
    current_resolution_limit = -1.0;
    current_ctf              = 0.0;
    current_phi              = 0.0;
    current_theta            = 0.0;
    current_psi              = 0.0;
    current_shift_x          = 0.0;
    current_shift_y          = 0.0;
    current_mask_radius      = 0.0;
    current_mask_falloff     = 0.0;
    current_whitening        = false;
    current_swap_quadrants   = false;
    whitened_projection      = false;
    density_map              = NULL;

    //	MyPrintWithDetails("Error: Constructor must be called with volume dimensions and pixel size");
    //	DEBUG_ABORT;
}

ReconstructedVolume::~ReconstructedVolume( ) {
    Deallocate( );
}

ReconstructedVolume& ReconstructedVolume::operator=(const ReconstructedVolume& other_volume) {
    *this = &other_volume;
    return *this;
}

ReconstructedVolume& ReconstructedVolume::operator=(const ReconstructedVolume* other_volume) {
    // Check for self assignment
    if ( this != other_volume ) {
        MyDebugAssertTrue(other_volume->density_map != NULL, "Other volume has not been initialized");

        //		if (density_map != NULL && volume_initialized == true)
        //		{
        //
        //			if (density_map->logical_x_dimension != other_volume->density_map->logical_x_dimension || density_map->logical_y_dimension != other_volume->density_map->logical_y_dimension || density_map->logical_z_dimension != other_volume->density_map->logical_z_dimension)
        //			{
        //				Deallocate();
        //				InitWithDimensions(other_volume->density_map->logical_x_dimension, other_volume->density_map->logical_y_dimension, other_volume->density_map->logical_z_dimension, other_volume->pixel_size, other_volume->symmetry_symbol);
        //			}
        //		}
        //		else
        //		{
        //			InitWithDimensions(other_volume->density_map->logical_x_dimension, other_volume->density_map->logical_y_dimension, other_volume->density_map->logical_z_dimension, other_volume->pixel_size, other_volume->symmetry_symbol);
        //		}

        // by here the memory allocation should be OK...

        if ( other_volume->volume_initialized ) {
            if ( density_map == NULL )
                density_map = new Image;
            //			density_map->CopyFrom(other_volume->density_map);
            *density_map = *other_volume->density_map;
        }
        if ( other_volume->projection_initialized )
            current_projection = other_volume->current_projection;
        //		*density_map = *other_volume->density_map;
        //		current_projection = other_volume->current_projection;

        mask_volume_in_voxels    = other_volume->mask_volume_in_voxels;
        molecular_mass_in_kDa    = other_volume->molecular_mass_in_kDa;
        mask_radius              = other_volume->mask_radius;
        has_masked_applied       = other_volume->has_masked_applied;
        was_corrected            = other_volume->was_corrected;
        has_been_filtered        = other_volume->has_been_filtered;
        pixel_size               = other_volume->pixel_size;
        symmetry_symbol          = other_volume->symmetry_symbol;
        current_resolution_limit = other_volume->current_resolution_limit;
        current_ctf              = other_volume->current_ctf;
        current_phi              = other_volume->current_phi;
        current_theta            = other_volume->current_theta;
        current_psi              = other_volume->current_psi;
        current_shift_x          = other_volume->current_shift_x;
        current_shift_y          = other_volume->current_shift_y;
        current_mask_radius      = other_volume->current_mask_radius;
        current_mask_falloff     = other_volume->current_mask_falloff;
        current_whitening        = other_volume->current_whitening;
        current_swap_quadrants   = other_volume->current_swap_quadrants;
        whitened_projection      = other_volume->whitened_projection;
        symmetry_matrices        = other_volume->symmetry_matrices;
        volume_initialized       = other_volume->volume_initialized;
        projection_initialized   = other_volume->projection_initialized;
        has_masked_applied       = other_volume->has_masked_applied;
        was_corrected            = other_volume->was_corrected;
        has_been_filtered        = other_volume->has_been_filtered;
        whitened_projection      = other_volume->whitened_projection;
    }

    return *this;
}

void ReconstructedVolume::CopyAllButVolume(const ReconstructedVolume* other_volume) {
    // Check for self assignment
    if ( this != other_volume ) {
        mask_volume_in_voxels    = other_volume->mask_volume_in_voxels;
        molecular_mass_in_kDa    = other_volume->molecular_mass_in_kDa;
        mask_radius              = other_volume->mask_radius;
        has_masked_applied       = other_volume->has_masked_applied;
        was_corrected            = other_volume->was_corrected;
        has_been_filtered        = other_volume->has_been_filtered;
        pixel_size               = other_volume->pixel_size;
        symmetry_symbol          = other_volume->symmetry_symbol;
        current_resolution_limit = other_volume->current_resolution_limit;
        current_ctf              = other_volume->current_ctf;
        current_phi              = other_volume->current_phi;
        current_theta            = other_volume->current_theta;
        current_psi              = other_volume->current_psi;
        current_shift_x          = other_volume->current_shift_x;
        current_shift_y          = other_volume->current_shift_y;
        current_mask_radius      = other_volume->current_mask_radius;
        current_mask_falloff     = other_volume->current_mask_falloff;
        current_whitening        = other_volume->current_whitening;
        current_swap_quadrants   = other_volume->current_swap_quadrants;
        whitened_projection      = other_volume->whitened_projection;
        symmetry_matrices        = other_volume->symmetry_matrices;
        //		volume_initialized = other_volume->volume_initialized;
        has_masked_applied  = other_volume->has_masked_applied;
        was_corrected       = other_volume->was_corrected;
        has_been_filtered   = other_volume->has_been_filtered;
        whitened_projection = other_volume->whitened_projection;
        current_projection  = other_volume->current_projection;
        if ( other_volume->projection_initialized )
            current_projection = other_volume->current_projection;
        projection_initialized = other_volume->projection_initialized;
    }
}

void ReconstructedVolume::Deallocate( ) {
    if ( density_map != NULL ) {
        if ( volume_initialized )
            delete density_map;
        density_map = NULL;
    }
    if ( projection_initialized ) {
        current_projection.Deallocate( );
        projection_initialized = false;
    }
}

void ReconstructedVolume::InitWithReconstruct3D(Reconstruct3D& image_reconstruction, float wanted_pixel_size) {
    if ( density_map == NULL ) {
        density_map = new Image;
    }
    density_map->Allocate(image_reconstruction.logical_x_dimension, image_reconstruction.logical_y_dimension, image_reconstruction.logical_z_dimension, false);
    density_map->object_is_centred_in_box = false;
    volume_initialized                    = true;
    pixel_size                            = wanted_pixel_size;
    symmetry_symbol                       = image_reconstruction.symmetry_matrices.symmetry_symbol;
    symmetry_matrices.Init(image_reconstruction.symmetry_matrices.symmetry_symbol);
    //	statistics.Init(wanted_pixel_size);
    current_projection.Allocate(image_reconstruction.logical_x_dimension, image_reconstruction.logical_y_dimension, 1, false);
    current_projection.object_is_centred_in_box = false;
    projection_initialized                      = true;
}

void ReconstructedVolume::InitWithDimensions(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString wanted_symmetry_symbol) {
    if ( density_map == NULL ) {
        density_map = new Image;
    }
    density_map->Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, false);
    density_map->object_is_centred_in_box = false;
    volume_initialized                    = true;
    pixel_size                            = wanted_pixel_size;
    symmetry_symbol                       = wanted_symmetry_symbol;
    symmetry_matrices.Init(wanted_symmetry_symbol);
    //	statistics.Init(wanted_pixel_size);
    current_projection.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);
    current_projection.object_is_centred_in_box = false;
    projection_initialized                      = true;
}

//void ReconstructedVolume::PrepareForProjections(float resolution_limit, bool approximate_binning, bool apply_binning)
void ReconstructedVolume::PrepareForProjections(float low_resolution_limit, float high_resolution_limit, bool approximate_binning, bool apply_binning) {
    int   fourier_size_x;
    int   fourier_size_y;
    int   fourier_size_z;
    float binning_factor;
    //	float average_density;

    //	density_map->CorrectSinc();
    // Correct3D amplifies noise at the edges. Maybe it is better not to do this...
    Correct3D(mask_radius / pixel_size);
    //	if (mask_radius > 0.0) density_map->CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
    //	else density_map->CosineMask(0.45 * density_map->logical_x_dimension, 10.0 / pixel_size, false, true, 0.0);
    //	density_map->CorrectSinc();
    //	if (mask_radius > 0.0) density_map->AddConstant(- density_map->ReturnAverageOfRealValues(mask_radius, true));
    //	else density_map->AddConstant(- density_map->ReturnAverageOfRealValues(0.45 * density_map->logical_x_dimension, true));
    density_map->ForwardFFT( );

    if ( apply_binning && high_resolution_limit > 0.0 ) {
        binning_factor = high_resolution_limit / pixel_size / 2.0;

        if ( approximate_binning ) {
            fourier_size_x = ReturnClosestFactorizedUpper(ReturnSafeBinnedBoxSize(density_map->logical_x_dimension, binning_factor), 3, true);
            fourier_size_y = ReturnClosestFactorizedUpper(ReturnSafeBinnedBoxSize(density_map->logical_y_dimension, binning_factor), 3, true);
            fourier_size_z = ReturnClosestFactorizedUpper(ReturnSafeBinnedBoxSize(density_map->logical_z_dimension, binning_factor), 3, true);
        }
        else {
            fourier_size_x = int(density_map->logical_x_dimension / binning_factor + 0.5);
            if ( ! IsEven(fourier_size_x) )
                fourier_size_x++;
            //			fourier_size_x += 2;
            fourier_size_y = int(density_map->logical_y_dimension / binning_factor + 0.5);
            if ( ! IsEven(fourier_size_y) )
                fourier_size_y++;
            //			fourier_size_y += 2;
            fourier_size_z = int(density_map->logical_z_dimension / binning_factor + 0.5);
            if ( ! IsEven(fourier_size_z) )
                fourier_size_z++;
            //			fourier_size_z += 2;
        }
        // The following line assumes that we have a cubic volume
        binning_factor = float(density_map->logical_x_dimension) / float(fourier_size_x);
        if ( binning_factor != 1.0 ) {
            density_map->Resize(fourier_size_x, fourier_size_y, fourier_size_z);
            pixel_size *= binning_factor;
        }
    }

    if ( high_resolution_limit > 0.0 )
        density_map->CosineMask(pixel_size / high_resolution_limit, pixel_size / 100.0);
    if ( low_resolution_limit > 0.0 )
        density_map->CosineMask(pixel_size / low_resolution_limit, pixel_size / 100.0, true);

    /*	density_map->BackwardFFT();
//	density_map->AddConstant(- density_map->ReturnAverageOfRealValuesOnEdges());

//	average_density = density_map->ReturnAverageOfRealValues(mask_radius / pixel_size, true);
//	density_map->CosineMask(mask_radius / pixel_size, 10.0 / pixel_size, false, true, average_density);

	density_map->CosineMask(mask_radius / pixel_size, 10.0 / pixel_size, false, true, 0.0);

	average_density = density_map->ReturnAverageOfMaxN(100, mask_radius / pixel_size);
	density_map->SetMinimumValue(-0.3 * average_density);
	density_map->ForwardFFT(); */
    density_map->SwapRealSpaceQuadrants( );
    // The following is important to avoid interpolation artifacts near the Fourier space origin
    density_map->complex_values[0] = 0.0f + I * 0.0f;
}

void ReconstructedVolume::CalculateProjection(Image& projection, Image& CTF, AnglesAndShifts& angles_and_shifts_of_projection,
                                              float mask_radius, float mask_falloff, float resolution_limit, bool swap_quadrants, bool apply_shifts, bool whiten, bool apply_ctf, bool abolute_ctf, bool calculate_projection) {
    //	MyDebugAssertTrue(projection.logical_x_dimension == density_map->logical_x_dimension && projection.logical_y_dimension == density_map->logical_y_dimension, "Error: Images have different sizes");
    MyDebugAssertTrue(CTF.logical_x_dimension == projection.logical_x_dimension && CTF.logical_y_dimension == projection.logical_y_dimension, "Error: CTF image has different size");
    MyDebugAssertTrue(projection.logical_z_dimension == 1, "Error: attempting to extract 3D image from 3D reconstruction");
    MyDebugAssertTrue(projection.is_in_memory, "Memory not allocated for receiving image");
    MyDebugAssertTrue(density_map->IsCubic( ), "Image volume to project is not cubic (%i, %i, %i)", density_map->logical_x_dimension, density_map->logical_y_dimension, density_map->logical_z_dimension);
    MyDebugAssertTrue(! density_map->object_is_centred_in_box, "Image volume quadrants not swapped");

    if ( current_phi != angles_and_shifts_of_projection.ReturnPhiAngle( ) || current_theta != angles_and_shifts_of_projection.ReturnThetaAngle( ) || current_psi != angles_and_shifts_of_projection.ReturnPsiAngle( ) || current_resolution_limit != resolution_limit ) {
        if ( calculate_projection )
            density_map->ExtractSlice(projection, angles_and_shifts_of_projection, resolution_limit);
        current_projection.CopyFrom(&projection);
        current_phi              = angles_and_shifts_of_projection.ReturnPhiAngle( );
        current_theta            = angles_and_shifts_of_projection.ReturnThetaAngle( );
        current_psi              = angles_and_shifts_of_projection.ReturnPsiAngle( );
        current_shift_x          = angles_and_shifts_of_projection.ReturnShiftX( );
        current_shift_y          = angles_and_shifts_of_projection.ReturnShiftY( );
        current_resolution_limit = resolution_limit;
        current_ctf              = CTF.real_values[10];
        current_mask_radius      = mask_radius;
        current_mask_falloff     = mask_falloff;
        current_swap_quadrants   = swap_quadrants;
        current_whitening        = whiten;

        if ( whiten ) {
            //			var_A = projection.ReturnSumOfSquares();
            //			projection.MultiplyByConstant(sqrtf(projection.number_of_real_space_pixels / var_A));
            projection.Whiten(resolution_limit);
            //			projection.PhaseFlipPixelWise(CTF);
            //			projection.BackwardFFT();
            //			projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
            //			projection.ForwardFFT();
        }
        if ( apply_ctf ) {
            //			projection.BackwardFFT();
            //			projection.SetToConstant(1.0);
            //			projection.real_values[0] = 1.0;
            //			projection.CosineMask(20.0, 1.0, false, true, 0.0);
            //			projection.ForwardFFT();
            //			projection.MultiplyPixelWiseReal(CTF, false);
            //			projection.SwapRealSpaceQuadrants();
            //			projection.QuickAndDirtyWriteSlice("proj_20_flipped.mrc", 1);
            //			exit(0);
            projection.MultiplyPixelWiseReal(CTF, abolute_ctf);

            if ( mask_radius > 0.0 ) {
                projection.BackwardFFT( );
                projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
                projection.ForwardFFT( );
            }
        }

        if ( apply_shifts )
            projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX( ) / pixel_size, angles_and_shifts_of_projection.ReturnShiftY( ) / pixel_size);
        if ( swap_quadrants )
            projection.SwapRealSpaceQuadrants( );
    }
    else {
        if ( current_ctf != CTF.real_values[10] || current_shift_x != angles_and_shifts_of_projection.ReturnShiftX( ) || current_shift_y != angles_and_shifts_of_projection.ReturnShiftY( ) || current_mask_radius != mask_radius || current_mask_falloff != mask_falloff || current_swap_quadrants != swap_quadrants || current_whitening != whiten ) {
            current_shift_x        = angles_and_shifts_of_projection.ReturnShiftX( );
            current_shift_y        = angles_and_shifts_of_projection.ReturnShiftY( );
            current_ctf            = CTF.real_values[10];
            current_mask_radius    = mask_radius;
            current_mask_falloff   = mask_falloff;
            current_swap_quadrants = swap_quadrants;
            current_whitening      = whiten;

            projection.CopyFrom(&current_projection);

            if ( whiten ) {
                //				var_A = projection.ReturnSumOfSquares();
                //				projection.MultiplyByConstant(sqrtf(projection.number_of_real_space_pixels / var_A));
                projection.Whiten(resolution_limit);
                //				projection.PhaseFlipPixelWise(CTF);
                //				projection.BackwardFFT();
                //				projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
                //				projection.ForwardFFT();
            }
            if ( apply_ctf ) {
                projection.MultiplyPixelWiseReal(CTF, abolute_ctf);

                if ( mask_radius > 0.0 ) {
                    projection.BackwardFFT( );
                    projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
                    projection.ForwardFFT( );
                }
            }

            if ( apply_shifts )
                projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX( ) / pixel_size, angles_and_shifts_of_projection.ReturnShiftY( ) / pixel_size);
            if ( swap_quadrants )
                projection.SwapRealSpaceQuadrants( );
        }
    }

    whitened_projection = whiten;
}

void ReconstructedVolume::Calculate3DSimple(Reconstruct3D& reconstruction) {
    MyDebugAssertTrue(density_map != NULL, "Error: reconstruction volume has not been initialized");

    int i;
    int j;
    int k;

    long pixel_counter = 0;

    reconstruction.CompleteEdges( );

    // Now do the division by the CTF volume
    for ( k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++ ) {
        for ( j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++ ) {
            for ( i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++ ) {
                if ( reconstruction.ctf_reconstruction[pixel_counter] != 0.0 ) {
                    //					if (reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_X(i)==40 && reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j)==20 && reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k)==4)
                    //					{
                    //						wxPrintf("counter = %li, image = %g, ctf = %g, weights = %g, n = %i\n", pixel_counter, cabsf(reconstruction.image_reconstruction.complex_values[pixel_counter]),
                    //						reconstruction.ctf_reconstruction[pixel_counter], reconstruction.weights_reconstruction[pixel_counter], reconstruction.number_of_measurements[pixel_counter]);
                    //					}

                    // Use 100.0 as Wiener constant for 3DSimple since the SSNR near the resolution limit of high-resolution reconstructions is often close to 0.01.
                    // The final result is not strongly dependent on this constant and therefore, a value of 100.0 is a sufficiently good estimate.
                    density_map->complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + 100.0f);
                    //																		/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
                }
                else {
                    density_map->complex_values[pixel_counter] = 0.0;
                }
                pixel_counter++;
            }
        }
    }
}

void ReconstructedVolume::Calculate3DOptimal(Reconstruct3D& reconstruction, ResolutionStatistics& statistics, float weiner_filter_nominator) {
    MyDebugAssertTrue(density_map != NULL, "Error: reconstruction volume has not been initialized");
    //	MyDebugAssertTrue(has_statistics, "Error: 3D statistics have not been calculated");
    MyDebugAssertTrue(int((reconstruction.image_reconstruction.ReturnSmallestLogicalDimension( ) / 2 + 1) * sqrtf(3.0)) + 1 == statistics.part_SSNR.NumberOfPoints( ), "Error: part_SSNR table incompatible with volume");

    int i;
    int j;
    int k;
    int bin;

    long pixel_counter = 0;

    float x;
    float y;
    float z;
    float frequency_squared;
    float particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
    float pssnr_correction_factor = float(density_map->ReturnVolumeInRealSpace( )) / (kDa_to_Angstrom3(molecular_mass_in_kDa) / powf(pixel_size, 3)) * particle_area_in_pixels / float(density_map->logical_x_dimension * density_map->logical_y_dimension);

    int number_of_bins2 = reconstruction.image_reconstruction.ReturnSmallestLogicalDimension( );

    float* wiener_constant = new float[statistics.part_SSNR.NumberOfPoints( )];

    reconstruction.CompleteEdges( );

    // Now do the division by the CTF volume
    pixel_counter = 0;

    for ( i = 0; i < statistics.part_SSNR.NumberOfPoints( ); i++ ) {
        if ( statistics.part_SSNR.data_y[i] > 0.0 ) {
            //			wiener_constant[i] = 1.0 / statistics.part_SSNR.data_y[i];
            //			wiener_constant[i] = 1.0 / pssnr_correction_factor / statistics.part_SSNR.data_y[i];
            wiener_constant[i] = weiner_filter_nominator / pssnr_correction_factor / statistics.part_SSNR.data_y[i];
            //		wiener_constant[i] = 1;
        }
        else
            wiener_constant[i] = 0.0;
        //		wiener_constant[i] = 0.000001;
    }

    for ( k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++ ) {
        z = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * reconstruction.image_reconstruction.fourier_voxel_size_z, 2);

        for ( j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++ ) {
            y = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * reconstruction.image_reconstruction.fourier_voxel_size_y, 2);

            for ( i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++ ) {
                if ( reconstruction.ctf_reconstruction[pixel_counter] != 0.0 ) {
                    x                 = powf(i * reconstruction.image_reconstruction.fourier_voxel_size_x, 2);
                    frequency_squared = x + y + z;

                    // compute radius, in units of physical Fourier pixels
                    bin = int(sqrtf(frequency_squared) * number_of_bins2);

                    if ( statistics.part_SSNR.data_y[bin] != 0.0 ) {
                        density_map->complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + wiener_constant[bin]);
                        //									/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
                    }
                    else {
                        density_map->complex_values[pixel_counter] = 0.0;
                    }
                }
                else {
                    density_map->complex_values[pixel_counter] = 0.0;
                }
                pixel_counter++;
            }
        }
    }

    density_map->is_in_real_space = false;
    delete[] wiener_constant;
}

/*
 * Compute the efficiency of the orientation distribution, following
 * Naydenova & Russo (2017)
 *
 * The final result is between 0.0 and 1.0, where 1.0 is a perfectly isotropic
 * orientation distribution, and >0.5 is desirable.
 *
 * The algorithm described in the paper was modified to use the CTF^2 volume directly.
 * This may be slightly problematic because:
 * - the CTF^2 volume may have been computed with per-particle attenuations as a function of the particle score
 * - (...?)
 *
 * Alexis Rohou, September 2017
 */
float ReconstructedVolume::ComputeOrientationDistributionEfficiency(Reconstruct3D& reconstruction) {
    float efficiency;
    float radius_mean;
    float radius_sigma;
    float radius_0pc;
    float radius_25pc;
    float radius_50pc;
    float radius_75pc;
    float radius_100pc;
    float psf_radial_average_max;

    Image psf;
    Curve psf_radial_average;
    Curve psf_radial_count;

    long pixel_counter = 0;

    float threshold;

    // Turn the ctf_reconstruction into a "proper" image object so we can easily FT it
    psf.Allocate(reconstruction.logical_x_dimension, reconstruction.logical_y_dimension, reconstruction.logical_z_dimension, false);
    for ( pixel_counter = 0; pixel_counter <= reconstruction.image_reconstruction.physical_upper_bound_complex_z * reconstruction.image_reconstruction.physical_upper_bound_complex_y * reconstruction.image_reconstruction.physical_upper_bound_complex_x; pixel_counter++ ) {
        psf.complex_values[pixel_counter] = reconstruction.ctf_reconstruction[pixel_counter];
    }

    // Apply the desired B factor (presumed to reflect the quality of the images, i.e. the rate of loss of contrast as a function of frequency)
    psf.ApplyBFactor(200.0 / reconstruction.pixel_size / reconstruction.pixel_size);

    // Inverse FT of the CTF^2 volume. We now have a point spread function (PSF)
    psf.SwapRealSpaceQuadrants( );
    psf.BackwardFFT( );
    psf.DivideByConstant(reconstruction.images_processed * reconstruction.symmetry_matrices.number_of_matrices);
    //psf.QuickAndDirtyWriteSlices("psf.mrc",1,psf.logical_z_dimension);

    // Binarize with a threshold
    // In the paper, the threshold would be: number of slices inserted (number of particles * symmetry) / e^2
    // (they actually binarize at 1/e^2, but they first normalize by the number of images)
    // I also add a factor of * 0.5, since we are dealing with squared CTF rather than unity
    threshold = exp(-2.0) * 0.5 * psf.ReturnMaximumValue( );
    wxPrintf("threshold = %f\n", threshold);
    MyDebugAssertTrue(threshold > psf.ReturnMinimumValue( ) && threshold < psf.ReturnMaximumValue( ), "Bad threshold value: %f. Min, max of image: %f - %f\n", threshold, psf.ReturnMinimumValue( ), psf.ReturnMaximumValue( ));
    psf.Binarise(threshold);
    //psf.QuickAndDirtyWriteSlices("psf_bin.mrc",1,psf.logical_z_dimension);

    // Measure the radius of the blob, we need to accumulate the average radius and its sigma
    // This is probably complicated to do properly. Let's just take the radial average and assume that the mean
    // radius is the radius at which the radial average gets to half max.
    //psf_radial_average.SetupXAxis(0.0, 0.5*sqrt(2.0)*float(psf.logical_x_dimension), 0.5*sqrt(2.0)*float(psf.logical_x_dimension));
    //psf_radial_count.SetupXAxis(0.0, 0.5*sqrt(2.0)*float(psf.logical_x_dimension), 0.5*sqrt(2.0)*float(psf.logical_x_dimension));
    psf_radial_average.SetupXAxis(0.0, psf.ReturnMaximumDiagonalRadius( ), psf.ReturnMaximumDiagonalRadius( ) * 0.5);
    psf_radial_count.SetupXAxis(0.0, psf.ReturnMaximumDiagonalRadius( ), psf.ReturnMaximumDiagonalRadius( ) * 0.5);
    wxPrintf("psf dim = %i, x axis from 0.0 to %0.3f\n", psf.logical_x_dimension, psf.ReturnMaximumDiagonalRadius( ));
    psf.Compute1DRotationalAverage(psf_radial_average, psf_radial_count, false);
    psf_radial_average.PrintToStandardOut( );
    psf_radial_average_max = psf_radial_average.ReturnMaximumValue( );
    wxPrintf("PSF radial average: max value = %0.3f at %0.3f\n", psf_radial_average_max, psf_radial_average.ReturnMode( ));

    // Now let's walk through the radial average and make a note of the slope from max to 0.0 and how steep it is
    radius_0pc   = -1.0;
    radius_25pc  = -1.0;
    radius_50pc  = -1.0;
    radius_75pc  = -1.0;
    radius_100pc = -1.0;
    //
    for ( pixel_counter = psf_radial_average.NumberOfPoints( ) - 2; pixel_counter >= 0; pixel_counter-- ) {
        if ( psf_radial_average.data_y[pixel_counter] > 0.0 && radius_0pc < 0.0 ) {
            radius_0pc = psf_radial_average.data_x[pixel_counter + 1];
        }
        if ( psf_radial_average.data_y[pixel_counter] > 0.1 * psf_radial_average_max && radius_25pc < 0.0 ) {
            radius_25pc = psf_radial_average.data_x[pixel_counter + 1];
        }
        if ( psf_radial_average.data_y[pixel_counter] > 0.5 * psf_radial_average_max && radius_50pc < 0.0 ) {
            radius_50pc = psf_radial_average.data_x[pixel_counter + 1];
        }
        if ( psf_radial_average.data_y[pixel_counter] > 0.9 * psf_radial_average_max && radius_75pc < 0.0 ) {
            radius_75pc = psf_radial_average.data_x[pixel_counter + 1];
        }
        if ( psf_radial_average.data_y[pixel_counter] > 0.99 * psf_radial_average_max && radius_100pc < 0.0 ) {
            radius_100pc = psf_radial_average.data_x[pixel_counter];
        }
    }

    wxPrintf("Radius 0: %0.3f; 25: %0.3f; 50: %0.3f; 75: %0.3f; 100: %0.3f\n", radius_0pc, radius_25pc, radius_50pc, radius_75pc, radius_100pc);

    // We cheat and come up with pretend values for the radius mean and sigma
    radius_mean  = radius_50pc;
    radius_sigma = 0.5 * (radius_25pc - radius_75pc);

    // Eqn 1 in Naydenova & Russo (2017)
    efficiency = 1.0 - (2.0 * radius_sigma) / radius_mean;

    // Sanity check: the number should be between 0.0 and 1.0
    MyDebugAssertTrue(efficiency > -0.01 && efficiency < 1.01, "Efficiency out of range: %0.3f\n", efficiency);

    psf.Deallocate( );
    psf_radial_average.ClearData( );
    psf_radial_count.ClearData( );

    return efficiency;
}

void ReconstructedVolume::CosineRingMask(float wanted_inner_mask_radius, float wanted_outer_mask_radius, float wanted_mask_edge) {
    mask_volume_in_voxels = density_map->CosineRingMask(wanted_inner_mask_radius, wanted_outer_mask_radius, wanted_mask_edge);
    has_masked_applied    = true;
}

void ReconstructedVolume::CosineMask(float wanted_mask_radius, float wanted_mask_edge) {
    mask_volume_in_voxels = density_map->CosineMask(wanted_mask_radius, wanted_mask_edge);
    has_masked_applied    = true;
}

float ReconstructedVolume::Correct3D(float mask_radius) {
    was_corrected = true;
    //	return density_map->CorrectSinc(mask_radius);
    return density_map->CorrectSinc(mask_radius, 1.0, true, 0.0);
}

void ReconstructedVolume::OptimalFilter(ResolutionStatistics& statistics) {
    density_map->OptimalFilterFSC(statistics.part_FSC);
    was_corrected = true;
}

void ReconstructedVolume::FinalizeSimple(Reconstruct3D& reconstruction, int& original_box_size, float& original_pixel_size, float& pixel_size,
                                         float& inner_mask_radius, float& outer_mask_radius, float& mask_falloff, wxString& output_volume) {
    int     intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
    int     box_size              = reconstruction.logical_x_dimension;
    MRCFile output_file;

    InitWithReconstruct3D(reconstruction, pixel_size);
    Calculate3DSimple(reconstruction);
    density_map->SwapRealSpaceQuadrants( );
    if ( intermediate_box_size != box_size ) {
        density_map->BackwardFFT( );
        density_map->Resize(intermediate_box_size, intermediate_box_size,
                            intermediate_box_size, density_map->ReturnAverageOfRealValuesOnEdges( ));
        density_map->ForwardFFT( );
    }
    if ( pixel_size != original_pixel_size )
        density_map->Resize(original_box_size, original_box_size, original_box_size);
    density_map->BackwardFFT( );
    CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
    //	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
    output_file.OpenFile(output_volume.ToStdString( ), true);
    density_map->WriteSlices(&output_file, 1, density_map->logical_z_dimension);
    output_file.SetPixelSize(original_pixel_size);
    output_file.CloseFile( );
    density_map->ForwardFFT( );
}

void ReconstructedVolume::FinalizeOptimal(Reconstruct3D& reconstruction, Image* density_map_1, Image* density_map_2,
                                          float& original_pixel_size, float& pixel_size, float& inner_mask_radius, float& outer_mask_radius, float& mask_falloff,
                                          bool center_mass, wxString& output_volume, NumericTextFile& output_statistics, ResolutionStatistics* copy_of_statistics, float weiner_filter_nominator) {
    int                  original_box_size     = density_map_1->logical_x_dimension;
    int                  intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
    int                  box_size              = reconstruction.logical_x_dimension;
    float                binning_factor        = pixel_size / original_box_size;
    float                particle_area_in_pixels;
    float                mask_volume_fraction;
    float                resolution_limit = 0.0;
    float                temp_float;
    MRCFile              output_file;
    ResolutionStatistics statistics(original_pixel_size, original_box_size);
    ResolutionStatistics cropped_statistics(pixel_size, box_size);
    ResolutionStatistics temp_statistics(pixel_size, intermediate_box_size);
    Peak                 center_of_mass;
    wxChar               symmetry_type;
    long                 symmetry_number;

    if ( pixel_size != original_pixel_size )
        resolution_limit = 2.0 * pixel_size;

    statistics.CalculateFSC(*density_map_1, *density_map_2, true);
    // TESTING OF LOCAL FILTERING
    const bool test_locres_filtering = false;
    if ( ! test_locres_filtering ) {
        density_map_1->Deallocate( );
        density_map_2->Deallocate( );
    }

    InitWithReconstruct3D(reconstruction, pixel_size);
    statistics.CalculateParticleFSCandSSNR(mask_volume_in_voxels, molecular_mass_in_kDa);
    particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
    mask_volume_fraction    = mask_volume_in_voxels / particle_area_in_pixels / original_box_size;
    if ( intermediate_box_size != box_size && binning_factor != 1.0 ) {
        temp_statistics.CopyFrom(statistics);
        cropped_statistics.ResampleFrom(temp_statistics);
    }
    else {
        cropped_statistics.CopyFrom(statistics);
    }
    cropped_statistics.CalculateParticleSSNR(reconstruction.image_reconstruction, reconstruction.ctf_reconstruction, mask_volume_fraction);
    if ( intermediate_box_size != box_size && binning_factor != 1.0 ) {
        temp_statistics.ResampleParticleSSNR(cropped_statistics);
        statistics.CopyParticleSSNR(temp_statistics);
    }
    else {
        statistics.CopyParticleSSNR(cropped_statistics);
    }
    statistics.ZeroToResolution(resolution_limit);
    statistics.PrintStatistics( );

    statistics.WriteStatisticsToFile(output_statistics);
    if ( copy_of_statistics != NULL ) {
        copy_of_statistics->Init(original_pixel_size, original_box_size);
        copy_of_statistics->CopyFrom(statistics);
    }

    Calculate3DOptimal(reconstruction, cropped_statistics, weiner_filter_nominator);
    density_map->SwapRealSpaceQuadrants( );
    // Check if cropping was used and resize reconstruction accordingly
    if ( intermediate_box_size != box_size ) {
        density_map->BackwardFFT( );
        // Correct3D is necessary to correct the signal in the map but it also amplifies the noise. Try without this...
        //Correct3D(outer_mask_radius / pixel_size);
        // Scaling factor needed to compensate for FFT normalization for different box sizes
        density_map->MultiplyByConstant(float(intermediate_box_size) / float(box_size));
        density_map->Resize(intermediate_box_size, intermediate_box_size,
                            intermediate_box_size, density_map->ReturnAverageOfRealValuesOnEdges( ));
        density_map->ForwardFFT( );
    }
    // Check if binning was used and resize reconstruction accordingly
    if ( pixel_size != original_pixel_size ) {
        //		density_map->CosineMask(0.5 - pixel_size / 20.0, pixel_size / 10.0);
        density_map->CosineMask(0.45, 0.1);
        density_map->Resize(original_box_size, original_box_size, original_box_size);
    }
    else
        density_map->CosineMask(0.45, 0.1);
    //	else density_map->CosineMask(0.5, original_pixel_size / 10.0);
    density_map->BackwardFFT( );
    // Need to run Correct3D if cropping was not used
    // Correct3D is necessary to correct the signal in the map but it also amplifies the noise. Try without this...
    //if (intermediate_box_size == box_size) Correct3D(outer_mask_radius / original_pixel_size);
    // Now we have a full-size map with the final pixel size. Applying mask and center map in box...
    //	CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
    //	CosineMask(density_map->physical_address_of_box_center_x - 3.0 * mask_falloff / original_pixel_size, 3.0 * mask_falloff / original_pixel_size);
    if ( center_mass ) {
        //		temp_float = density_map->ReturnAverageOfRealValuesOnEdges();
        temp_float     = density_map->ReturnAverageOfRealValues( );
        center_of_mass = density_map->CenterOfMass(temp_float, true);
        symmetry_type  = symmetry_symbol.Capitalize( )[0];
        if ( symmetry_type == 'C' && center_of_mass.value > 0.0 ) {
            symmetry_symbol.Mid(1).ToLong(&symmetry_number);
            if ( symmetry_number < 2 )
                density_map->RealSpaceIntegerShift(int(center_of_mass.x), int(center_of_mass.y), int(center_of_mass.z));
            else
                density_map->RealSpaceIntegerShift(0, 0, int(center_of_mass.z));
        }
    }
    output_file.OpenFile(output_volume.ToStdString( ), true);
    density_map->WriteSlices(&output_file, 1, density_map->logical_z_dimension);
    output_file.SetPixelSize(original_pixel_size);
    EmpiricalDistribution<double> density_distribution;
    density_map->UpdateDistributionOfRealValues(&density_distribution);
    output_file.SetDensityStatistics(density_distribution.GetMinimum( ), density_distribution.GetMaximum( ), density_distribution.GetSampleMean( ), sqrtf(density_distribution.GetSampleVariance( )));
    output_file.CloseFile( );
}

void ReconstructedVolume::FinalizeML(Reconstruct3D& reconstruction, Image* density_map_1, Image* density_map_2,
                                     float& original_pixel_size, float& pixel_size, float& inner_mask_radius, float& outer_mask_radius, float& mask_falloff,
                                     wxString& output_volume, NumericTextFile& output_statistics, ResolutionStatistics* copy_of_statistics) {
    int                  original_box_size     = density_map_1->logical_x_dimension;
    int                  intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
    int                  box_size              = reconstruction.logical_x_dimension;
    float                binning_factor        = pixel_size / original_box_size;
    float                particle_area_in_pixels;
    float                mask_volume_fraction;
    float                resolution_limit = 0.0;
    MRCFile              output_file;
    ResolutionStatistics statistics(original_pixel_size, original_box_size);
    ResolutionStatistics cropped_statistics(pixel_size, box_size);
    ResolutionStatistics temp_statistics(pixel_size, intermediate_box_size);

    if ( pixel_size != original_pixel_size )
        resolution_limit = 2.0 * pixel_size;

    InitWithReconstruct3D(reconstruction, pixel_size);
    statistics.CalculateFSC(*density_map_1, *density_map_2, true);
    statistics.CalculateParticleFSCandSSNR(mask_volume_in_voxels, molecular_mass_in_kDa);
    particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
    mask_volume_fraction    = mask_volume_in_voxels / particle_area_in_pixels / original_box_size;
    if ( intermediate_box_size != box_size && binning_factor != 1.0 ) {
        temp_statistics.CopyFrom(statistics);
        cropped_statistics.ResampleFrom(temp_statistics);
    }
    else {
        cropped_statistics.CopyFrom(statistics);
    }
    cropped_statistics.CalculateParticleSSNR(reconstruction.image_reconstruction, reconstruction.ctf_reconstruction, mask_volume_fraction);
    if ( intermediate_box_size != box_size && binning_factor != 1.0 ) {
        temp_statistics.ResampleParticleSSNR(cropped_statistics);
        statistics.CopyParticleSSNR(temp_statistics);
    }
    else {
        statistics.CopyParticleSSNR(cropped_statistics);
    }
    statistics.ZeroToResolution(resolution_limit);
    statistics.PrintStatistics( );

    statistics.WriteStatisticsToFile(output_statistics);
    if ( copy_of_statistics != NULL ) {
        copy_of_statistics->Init(original_pixel_size, original_box_size);
        copy_of_statistics->CopyFrom(statistics);
    }

    // This would have to be added back if FinalizeML was used
    //	if (reconstruction.images_processed > 0) reconstruction.noise_power_spectrum->MultiplyByConstant(1.0 / reconstruction.images_processed);
    Calculate3DML(reconstruction);
    density_map->SwapRealSpaceQuadrants( );
    if ( intermediate_box_size != box_size ) {
        density_map->BackwardFFT( );
        Correct3D(outer_mask_radius / pixel_size);
        density_map->Resize(intermediate_box_size, intermediate_box_size,
                            intermediate_box_size, density_map->ReturnAverageOfRealValuesOnEdges( ));
        density_map->ForwardFFT( );
    }
    if ( pixel_size != original_pixel_size )
        density_map->Resize(original_box_size, original_box_size, original_box_size);
    //	density_map->CosineMask(0.5, 1.0 / 20.0);
    density_map->CosineMask(0.475, 0.05);
    density_map->BackwardFFT( );
    if ( intermediate_box_size == box_size )
        Correct3D(outer_mask_radius / original_pixel_size);
    CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
    output_file.OpenFile(output_volume.ToStdString( ), true);
    density_map->WriteSlices(&output_file, 1, density_map->logical_z_dimension);
    output_file.SetPixelSize(original_pixel_size);
    output_file.CloseFile( );
}

// This calculation is missing the scaling of the CTF2 sums by sigma2 and therefore will not work properly
void ReconstructedVolume::Calculate3DML(Reconstruct3D& reconstruction) {
    MyDebugAssertTrue(density_map != NULL, "Error: reconstruction volume has not been initialized");
    //	MyDebugAssertTrue(has_statistics, "Error: 3D statistics have not been calculated");
    // This would have to be added back if FinalizeML was used
    //	MyDebugAssertTrue(int((reconstruction.image_reconstruction.ReturnSmallestLogicalDimension() / 2 + 1) * sqrtf(3.0)) + 1 == reconstruction.signal_power_spectrum->NumberOfPoints( ), "Error: signal_power_spectrum table incompatible with volume");

    int i;
    int j;
    int k;
    int bin;

    long pixel_counter = 0;

    float x;
    float y;
    float z;
    float frequency_squared;
    //	float particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
    //	float pssnr_correction_factor = density_map->ReturnVolumeInRealSpace() / (kDa_to_Angstrom3(molecular_mass_in_kDa) / powf(pixel_size,3))
    //			* particle_area_in_pixels / density_map->logical_x_dimension / density_map->logical_y_dimension;

    int number_of_bins2 = reconstruction.image_reconstruction.ReturnSmallestLogicalDimension( );

    // This would have to be added back if FinalizeML was used
    //	float *wiener_constant = new float[reconstruction.signal_power_spectrum->NumberOfPoints( )];
    float* wiener_constant = new float[314];

    reconstruction.CompleteEdges( );

    // Now do the division by the CTF volume
    pixel_counter = 0;

    // This would have to be added back if FinalizeML was used
    //	for (i = 0; i < reconstruction.signal_power_spectrum->NumberOfPoints( ); i++)
    //	{
    //		if (reconstruction.signal_power_spectrum->data_y[i] != 0.0) wiener_constant[i] = reconstruction.noise_power_spectrum->data_y[i] / reconstruction.signal_power_spectrum->data_y[i];
    //		else wiener_constant[i] = - 1.0;
    //		wxPrintf("noise, signal, filter = %i %g %g %g\n", i, reconstruction.noise_power_spectrum->data_y[i], reconstruction.signal_power_spectrum->data_y[i], wiener_constant[i]);
    //	}

    for ( k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++ ) {
        z = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * reconstruction.image_reconstruction.fourier_voxel_size_z, 2);

        for ( j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++ ) {
            y = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * reconstruction.image_reconstruction.fourier_voxel_size_y, 2);

            for ( i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++ ) {
                if ( reconstruction.ctf_reconstruction[pixel_counter] != 0.0 ) {
                    x                 = powf(i * reconstruction.image_reconstruction.fourier_voxel_size_x, 2);
                    frequency_squared = x + y + z;

                    // compute radius, in units of physical Fourier pixels
                    bin = int(sqrtf(frequency_squared) * number_of_bins2);

                    if ( wiener_constant[bin] >= 0.0 ) {
                        density_map->complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + wiener_constant[bin]);
                        //						wxPrintf("i j k = %i %i %i bin = %i pow = %g rec = %g ctf2 = %g filt = %g final = %g\n",i,j,k,bin,signal_power_spectrum.data_y[bin],
                        //								cabsf(reconstruction.image_reconstruction.complex_values[pixel_counter]),
                        //								reconstruction.ctf_reconstruction[pixel_counter],wiener_constant[bin], cabsf(density_map->complex_values[pixel_counter]));
                    }
                    else {
                        density_map->complex_values[pixel_counter] = 0.0;
                    }
                }
                else {
                    density_map->complex_values[pixel_counter] = 0.0;
                }
                pixel_counter++;
            }
        }
    }

    delete[] wiener_constant;
}
