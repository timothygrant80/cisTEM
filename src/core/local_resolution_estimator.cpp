#include "core_headers.h"

// Use this define to dump intermediate files
//#define dump_intermediate_files

LocalResolutionEstimator::LocalResolutionEstimator( ) {
    input_volume_one                         = NULL;
    input_volume_two                         = NULL;
    input_original_volume					 = NULL;
    input_volume_mask                        = NULL;
    box_size                                 = 0;
    number_of_fsc_shells                     = 0;
    first_slice                              = 0;
    last_slice                               = 0;
    sampling_step                            = 0;
    number_of_fsc_shells                     = 0;
    pixel_size_in_Angstroms                  = 0.0;
    highest_resolution_expected_in_Angstroms = 0.0;
    maximum_radius_in_Angstroms              = 0.0;
    shell_number_lut_is_allocated            = false;
    fixed_fsc_threshold                      = 0.0;
    use_fixed_fsc_threshold                  = false;
    threshold_snr                            = 0.0;
    threshold_confidence_n_sigma             = 0.0;
    resolution_value_before_first_shell      = 0.0;
    resolution_value_where_wont_estimate     = 0.0;
    symmetry_redundancy                      = 0;
    whiten_half_maps                         = true;
    shell_number_lut                         = NULL;
    padding_factor                           = 1;
}

LocalResolutionEstimator::~LocalResolutionEstimator( ) {
    DeallocateShellNumberLUT( );
    box_one.Deallocate( );
    box_two.Deallocate( );
    box_original.Deallocate( );
    box_combined_half_maps.Deallocate( );
    box_one_no_padding.Deallocate( );
    box_two_no_padding.Deallocate( );
    box_original_no_padding.Deallocate( );
    box_combined_half_maps_no_padding.Deallocate( );
}

void LocalResolutionEstimator::SetAllUserParameters(Image* wanted_input_volume_one, Image* wanted_input_volume_two, Image* wanted_input_orignal_volume, Image* wanted_mask_volume, int wanted_first_slice, int wanted_last_slice, int wanted_sampling_step, float input_pixel_size_in_Angstroms, int wanted_box_size, float wanted_threshold_snr, float wanted_threshold_confidence_n_sigma, bool wanted_use_fixed_fsc_threshold, float wanted_fixed_fsc_threshold, wxString wanted_symmetry_symbol, bool wanted_whiten_half_maps, int wanted_padding_factor) {
    MyDebugAssertTrue(IsEven(box_size), "Box size should be even");
    SetInputVolumes(wanted_input_volume_one, wanted_input_volume_two, wanted_input_orignal_volume, wanted_mask_volume);
    first_slice   = wanted_first_slice;
    last_slice    = wanted_last_slice;
    sampling_step = wanted_sampling_step;
    SetPixelSize(input_pixel_size_in_Angstroms);
    box_size                     = wanted_box_size;
    threshold_snr                = wanted_threshold_snr;
    threshold_confidence_n_sigma = wanted_threshold_confidence_n_sigma;
    use_fixed_fsc_threshold      = wanted_use_fixed_fsc_threshold;
    fixed_fsc_threshold          = wanted_fixed_fsc_threshold;
    SymmetryMatrix sym_matrix;
    sym_matrix.Init(wanted_symmetry_symbol);
    symmetry_redundancy  = sym_matrix.number_of_matrices;
    whiten_half_maps     = wanted_whiten_half_maps;
    padding_factor       = wanted_padding_factor;
    number_of_fsc_shells = box_size * padding_factor / 2;

    // Update based on user-supplied parameter values
 //   resolution_value_before_first_shell  = pixel_size_in_Angstroms * box_size * padding_factor; // TODO: check / think about whether the resolution of a shell is computed properly. On average, the frequencies contributing to a shell are not the frequency of the average radius of the shell... Especially relevant in first couple of shells!
    resolution_value_where_wont_estimate = 0;
}

void LocalResolutionEstimator::EstimateLocalResolution(Image* local_resolution_volume) {
    MyDebugAssertTrue(input_volume_one->IsCubic( ), "This method assumes the input volumes are cubic");
    MyDebugAssertTrue(local_resolution_volume->HasSameDimensionsAs(input_volume_one), "Local res volume does not have expected dimensions");

    AllocateLocalVolumes( );

    /*
	 * Setup a couple of arrays on the stack.
	 * TODO: consider whether doing this on the stack is really helping performance. If not, probably better to make these arrays members of the class (which I think means they have to go on the heap)
	 */
    float number_of_independent_voxels[number_of_fsc_shells];
    float fsc_threshold[number_of_fsc_shells];
    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        number_of_independent_voxels[shell_counter] = 0.0;
        fsc_threshold[shell_counter]                = 0.0;
    }

    // Precompute a mask for the small boxes
    Image box_mask(box_one_no_padding);
    box_mask.SetToConstant(1.0);
    box_mask.CosineMask(box_size / 4, box_size / 2, false, true, 0.0);

    // Precompute, for each Fourier voxel, the shell it will contribute to.
    AllocateShellNumberLUT( );
    PopulateShellNumberLUT( );

    /*
	 * Estimate the number of independent voxels per shell, taking into account real-space apodization.
	 *
	 * Mask factor:
	 * We are imposing an apodization envelope (a raised cosine, aka Hann window). The main lobe
	 * of the Fourier transform of this envelope has a width of 2 voxels (assuming the apodization is over
	 * the full width of the box) (according to Cardone et al), and therefore
	 * neighboring voxels will be convoluted with each other.
	 * The intersection of this convolution sphere with each Fourier shell can be approximated
	 * as a disc of radius 1 voxels and surface area $\pi$.
	 *
	 */
    float symmetry_factor;
    float mask_factor;
    mask_factor     = 1.0 / (PI * float(padding_factor * padding_factor));
    symmetry_factor = 1.0 / float(symmetry_redundancy);
    MyDebugPrint("Scaling factor from mask = %f\nScaling factor from symmetry = %f\n", mask_factor, symmetry_factor);
    CountIndependentVoxelsPerShell(number_of_independent_voxels, mask_factor * symmetry_factor);

    // Compute the FSC threshold
    ComputeFSCThresholdBasedOnUnbiasedSNREstimator(number_of_independent_voxels, fsc_threshold);

    // Set all voxels
    //local_resolution_volume->SetToConstant(resolution_value_where_wont_estimate);

    // Compute FSC locally and compare to the threshold
    ComputeLocalFSCAndCompareToThreshold(fsc_threshold, local_resolution_volume, box_mask);

    // Cleanup
    DeallocateShellNumberLUT( );
}

void LocalResolutionEstimator::AllocateLocalVolumes( ) {
    MyDebugAssertTrue(box_size > 0, "Box size was not set");
    box_one_no_padding.Allocate(box_size, box_size, box_size);
    box_two_no_padding.Allocate(box_size, box_size, box_size);
    box_original_no_padding.Allocate(box_size, box_size, box_size);
    box_combined_half_maps_no_padding.Allocate(box_size, box_size, box_size);
    box_one.Allocate(box_size * padding_factor, box_size * padding_factor, box_size * padding_factor);
    box_two.Allocate(box_size * padding_factor, box_size * padding_factor, box_size * padding_factor);
    box_original.Allocate(box_size * padding_factor, box_size * padding_factor, box_size * padding_factor);
    box_combined_half_maps.Allocate(box_size * padding_factor, box_size * padding_factor, box_size * padding_factor);
    // This below to calm valgrind down
    box_one.SetToConstant(0.0);
    box_two.SetToConstant(0.0);
    box_original.SetToConstant(0.0);
    box_combined_half_maps.SetToConstant(0.0);
    box_one_no_padding.SetToConstant(0.0);
    box_two_no_padding.SetToConstant(0.0);
    box_original_no_padding.SetToConstant(0.0);
    box_combined_half_maps_no_padding.SetToConstant(0.0);
}

void LocalResolutionEstimator::SetInputVolumes(Image* wanted_input_volume_one, Image* wanted_input_volume_two, Image* wanted_input_original_volume, Image* wanted_mask_volume) {
    MyDebugAssertTrue(wanted_input_volume_one->is_in_memory, "Input volume one is not in memory");
    MyDebugAssertTrue(wanted_input_volume_two->is_in_memory, "Input volume two is not in memory");
    MyDebugAssertTrue(wanted_mask_volume->is_in_memory, "Mask volume two is not in memory");
    MyDebugAssertTrue(wanted_input_volume_one->HasSameDimensionsAs(wanted_input_volume_two), "The input volumes do not have the same dimensions");
    MyDebugAssertTrue(wanted_mask_volume->HasSameDimensionsAs(wanted_input_volume_one), "The mask volume does not have the same dimensions as the input volumes");
    input_volume_one  = wanted_input_volume_one;
    input_volume_two  = wanted_input_volume_two;
    input_original_volume  = wanted_input_original_volume;
    input_volume_mask = wanted_mask_volume;
}

void LocalResolutionEstimator::AllocateShellNumberLUT( ) {
    if ( ! shell_number_lut_is_allocated ) {
        MyDebugAssertTrue(box_one.is_in_memory, "Local volume is not allocated - cannot allocate shell number LUT");
        shell_number_lut = new int[box_one.real_memory_allocated / 2];
    }
}

void LocalResolutionEstimator::DeallocateShellNumberLUT( ) {
    if ( shell_number_lut_is_allocated )
        delete[] shell_number_lut;
}

/*
 * Precompute, for each Fourier voxel, the shell it will contribute to.
 * Note that we will not compute the FSC into the corners (past 0.5/pixel).
 * This method will use box_one as a scratch array and overwrite it. Beware.
 */
void LocalResolutionEstimator::PopulateShellNumberLUT( ) {
    box_one.ComputeSpatialFrequencyAtEveryVoxel( );
    for ( long pixel_counter = 0; pixel_counter < box_one.real_memory_allocated / 2; pixel_counter++ ) {
        shell_number_lut[pixel_counter] = myroundint(real(box_one.complex_values[pixel_counter]) * 2.0 * float(number_of_fsc_shells - 1));
        if ( shell_number_lut[pixel_counter] >= number_of_fsc_shells )
            shell_number_lut[pixel_counter] = 0; // we basically ignore the corners of the volume (beyond 0.5 freq)

        // Debugging
        MyDebugAssertTrue(shell_number_lut[pixel_counter] >= 0, "Oops bad shell number");
        MyDebugAssertTrue(shell_number_lut[pixel_counter] < number_of_fsc_shells, "Oops, crazy shell number (%i) at pixel number %li", shell_number_lut[pixel_counter], pixel_counter);
        if ( real(box_one.complex_values[pixel_counter]) <= 0.5 )
            MyDebugAssertTrue(shell_number_lut[pixel_counter] < number_of_fsc_shells, "Oops bad calculation: %f, %i", real(box_one.complex_values[pixel_counter]), shell_number_lut[pixel_counter]);
    }
}

void LocalResolutionEstimator::CountIndependentVoxelsPerShell(float number_of_independent_voxels[], float wanted_scaling_factor) {
    // Initialize
    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        number_of_independent_voxels[shell_counter] = 0.0;
    }

    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        number_of_independent_voxels[shell_counter] = 0.0;
    }
    for ( long pixel_counter = 0; pixel_counter < box_one.real_memory_allocated / 2; pixel_counter++ ) {
        // Keep track of number of voxels contributing to each shell
        number_of_independent_voxels[shell_number_lut[pixel_counter]] += 1.0;
    }

    // It could be that e.g. a real-space mask was applied, meaning that not all Fourier voxels are independent from each other
    MyDebugAssertTrue(wanted_scaling_factor >= 0.0 && wanted_scaling_factor <= 1.0, "Bad value of scaling factor for independent voxels");
    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        number_of_independent_voxels[shell_counter] *= wanted_scaling_factor;
    }

    number_of_independent_voxels[0] = 0.0;
}

void LocalResolutionEstimator::ComputeFSCThresholdBasedOnUnbiasedSNREstimator(float number_of_independent_voxels[], float fsc_threshold[]) {

    const float nearly_one = 0.998; //0.995

    // Initialize
    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        fsc_threshold[shell_counter] = 0.0;
    }

    wxPrintf("\n\nFSC threshold\n");

    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        if ( use_fixed_fsc_threshold ) {
            fsc_threshold[shell_counter] = fixed_fsc_threshold;
        }
        else {

            fsc_threshold[shell_counter] = RhoThreshold(threshold_snr, threshold_confidence_n_sigma, number_of_independent_voxels[shell_counter]);
        }
        if ( fsc_threshold[shell_counter] > nearly_one )
            fsc_threshold[shell_counter] = nearly_one;
        wxPrintf("%i %f %f\n", shell_counter, number_of_independent_voxels[shell_counter], fsc_threshold[shell_counter]);
    }
    wxPrintf("\n\n");
}

void LocalResolutionEstimator::ComputeLocalFSCAndCompareToThreshold(float fsc_threshold[], Image* local_resolution_volume, Image box_mask) {

    MyDebugAssertTrue(input_volume_one->is_in_real_space, "Volume one is not in real space");
    MyDebugAssertTrue(input_volume_two->is_in_real_space, "Volume two is not in real space");

    // Local vars
    int i, j, k, shell_counter;
    int center_of_first_box;
    int center_of_last_box;
    center_of_first_box = box_size / 2;
    center_of_last_box  = input_volume_one->logical_x_dimension - (box_size / 2) - 1;
    float       computed_fsc[number_of_fsc_shells];
    const float resolution_value_between_estimation_points = 0.0;

    // Work arrays needed for computing the FSCs fast
    Image  work_box_one(box_one);
    Image  work_box_two(box_two);
    Image  work_box_cross(box_one);
    double work_sum_of_squares[number_of_fsc_shells];
    double work_sum_of_other_squares[number_of_fsc_shells];
    double work_sum_of_cross_products[number_of_fsc_shells];

    // Debug
#ifdef DEBUG
    const int dbg_i = 96;
    const int dbg_j = 96;
    const int dbg_k = 96;
#endif

    // Initialisation
    for ( int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        computed_fsc[shell_counter] = 0.0;
    }

    // Whiten the input volumes
    if ( whiten_half_maps ) {

        input_volume_one->ForwardFFT( );
        input_volume_two->ForwardFFT( );
        input_volume_one->Whiten(0.5);
        input_volume_two->Whiten(0.5);
        input_volume_one->BackwardFFT( );
        input_volume_two->BackwardFFT( );
    }

#ifdef DEBUG
    input_volume_one->QuickAndDirtyWriteSlices("dbg_1.mrc", 1, input_volume_one->logical_z_dimension);
    input_volume_two->QuickAndDirtyWriteSlices("dbg_2.mrc", 1, input_volume_one->logical_z_dimension);
#endif

    // Sliding window over the volumes
    long total_number_of_boxes   = std::min(input_volume_one->logical_z_dimension - box_size, last_slice - first_slice + 1) * (input_volume_one->logical_y_dimension - box_size) * (input_volume_one->logical_x_dimension - box_size) / pow(sampling_step, 3);
    long pixel_counter           = 0;
    int  counter_number_of_boxes = 0;
    MyPrintfGreen("Total number of boxes = %li\n", total_number_of_boxes);
    float        current_resolution;
    float        previous_resolution;
    float		 inverse_resolution;
    ProgressBar* my_progress_bar;
    my_progress_bar = new ProgressBar(total_number_of_boxes);
    bool       below_threshold, just_a_glitch;
    const bool stringent_mode = false;
    const bool allow_glitches = false;

    /// SET TIM THRESHOLD! TODO: Sort this out properly
   /* for ( int shell_counter = 1; shell_counter < number_of_fsc_shells; shell_counter++ ) {
        current_resolution = pixel_size_in_Angstroms * 2.0 * float(number_of_fsc_shells - 1) / float(shell_counter);

        if ( current_resolution > 12 && fsc_threshold[shell_counter] >= 0.95f )
            fsc_threshold[shell_counter] = 0.95f;
        else if ( current_resolution > 10 && fsc_threshold[shell_counter] >= 0.93f )
            fsc_threshold[shell_counter] = 0.93f;
        else if ( current_resolution > 8 && fsc_threshold[shell_counter] >= 0.9f )
            fsc_threshold[shell_counter] = 0.9f;
        else if ( current_resolution > 6 && fsc_threshold[shell_counter] >= 0.75f )
            fsc_threshold[shell_counter] = 0.75f;
        else if ( current_resolution > 4 && fsc_threshold[shell_counter] >= 0.5f )
            fsc_threshold[shell_counter] = 0.5f;
        else
            fsc_threshold[shell_counter] = 0.2f;

        wxPrintf("Setting Threshold to %.2f for %.2f A\n", fsc_threshold[shell_counter], current_resolution);
    }
	*/
#ifdef DEBUG
    bool on_dbg_point;
#endif

    Image FSC_Filtered_Volume; //making the new volume the size of the original and setting to 0, doing this before so it doesn't keep setting to constant
    FSC_Filtered_Volume.Allocate(input_volume_one->logical_x_dimension, input_volume_one->logical_y_dimension, input_volume_one->logical_z_dimension);
    FSC_Filtered_Volume.SetToConstant(0.0f);

    Image combined_half_maps;
    combined_half_maps.Allocate(input_volume_one->logical_x_dimension, input_volume_one->logical_y_dimension, input_volume_one->logical_z_dimension);
    combined_half_maps.SetToConstant(0.0f);
    combined_half_maps.CopyFrom(input_volume_one);
    combined_half_maps.AddImage(input_volume_two);

    Curve FSC_filter;

    for ( k = 0; k < input_volume_one->logical_z_dimension; k++ ) {
        if ( k >= first_slice - 1 && k <= last_slice - 1 && k % sampling_step == 0 ) {
            for ( j = 0; j < input_volume_one->logical_y_dimension; j++ ) {
                if ( j % sampling_step == 0 ) {
                    for ( i = 0; i < input_volume_one->logical_x_dimension; i++ ) {
#ifdef DEBUG
                        on_dbg_point = i == dbg_i && j == dbg_j && k == dbg_k;
                        if ( on_dbg_point )
                            wxPrintf("On the debug point\n");
#endif

                        if ( i % sampling_step == 0 ) {
                            counter_number_of_boxes++;
                            my_progress_bar->Update(counter_number_of_boxes);

                            if ( input_volume_mask->real_values[pixel_counter] == 0.0 || i < center_of_first_box || i > center_of_last_box || j < center_of_first_box || j > center_of_last_box || k < center_of_first_box || k > center_of_last_box ) {
#ifdef DEBUG
                                if ( i == dbg_i && j == dbg_j && k == dbg_k )
                                    wxPrintf("At debug point, but mask was 0.0\n");
#endif
                                local_resolution_volume->real_values[pixel_counter] = resolution_value_where_wont_estimate;

                             }

                            else {
                                //TODO: if performance is an issue, write a dedicated method to copy voxels over from large volume into small one, but with padding already done (i.e. not to the edges of small volume)

                                // Get voxels from input volumes into smaller boxes, mask them
                                box_one_no_padding.is_in_real_space = true;
                                box_two_no_padding.is_in_real_space = true;
                                box_original_no_padding.is_in_real_space = true;
                                box_combined_half_maps_no_padding.is_in_real_space = true;
                                input_volume_one->ClipInto(&box_one_no_padding, 0.0, false, 0.0, i - input_volume_one->physical_address_of_box_center_x, j - input_volume_one->physical_address_of_box_center_y, k - input_volume_one->physical_address_of_box_center_z);
                                box_one_no_padding.MultiplyPixelWise(box_mask);
                                input_volume_two->ClipInto(&box_two_no_padding, 0.0, false, 0.0, i - input_volume_one->physical_address_of_box_center_x, j - input_volume_one->physical_address_of_box_center_y, k - input_volume_one->physical_address_of_box_center_z);
                                box_two_no_padding.MultiplyPixelWise(box_mask);
                                // added original volume to split as well
                                input_original_volume->ClipInto(&box_original_no_padding, 0.0, false, 0.0, i - input_volume_one->physical_address_of_box_center_x, j - input_volume_one->physical_address_of_box_center_y, k - input_volume_one->physical_address_of_box_center_z);
                                box_original_no_padding.MultiplyPixelWise(box_mask);
                                // trying to use the combined half maps instead of the original volume
                                combined_half_maps.ClipInto(&box_combined_half_maps_no_padding, 0.0, false, 0.0, i - input_volume_one->physical_address_of_box_center_x, j - input_volume_one->physical_address_of_box_center_y, k - input_volume_one->physical_address_of_box_center_z);
                                box_combined_half_maps_no_padding.MultiplyPixelWise(box_mask);

                                // Pad the small boxes in real space
                                box_one.is_in_real_space = true;
                                box_two.is_in_real_space = true;
                                box_original.is_in_real_space = true;
                                box_combined_half_maps.is_in_real_space = true;
                                box_one_no_padding.ClipInto(&box_one, 0.0);
                                box_two_no_padding.ClipInto(&box_two, 0.0);
                                box_original_no_padding.ClipInto(&box_original, 0.0);
                                box_combined_half_maps_no_padding.ClipInto(&box_combined_half_maps, 0.0);

#ifdef DEBUG
                                if ( on_dbg_point ) {
                                    box_one.WriteSlicesAndFillHeader("dbg_vol1.mrc", pixel_size_in_Angstroms);
                                    box_two.WriteSlicesAndFillHeader("dbg_vol2.mrc", pixel_size_in_Angstroms);
                                }
#endif

                                box_one.ForwardFFT(false);
                                box_two.ForwardFFT(false);

                                // FSC
                                box_one.ComputeFSCVectorized(&box_two, &work_box_one, &work_box_two, &work_box_cross, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);
                                //box_one.ComputeFSC(&box_two, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);

                                // FSC CURVE FILTERING HACK
                                FSC_filter.AddPoint(0,1);

                                for ( int shell_counter = 1; shell_counter < number_of_fsc_shells; shell_counter++ ) {
                                	inverse_resolution = float (shell_counter) / (pixel_size_in_Angstroms * 2.0 * float(number_of_fsc_shells - 1));
                                	if (computed_fsc[shell_counter] < 0) {
                                		computed_fsc[shell_counter] = 0;
                                	}
                                	FSC_filter.AddPoint(inverse_resolution , computed_fsc[shell_counter]);
                                }

                                float resolution_limit = inverse_resolution;


                             // Filter using FSC on original volume boxes
/*
                                		box_original.WriteSlicesAndFillHeader("dbg_original_box_unfiltered.mrc", pixel_size_in_Angstroms);

                                        box_original.ForwardFFT();
                                        box_original.ApplyCurveFilter(&FSC_filter, resolution_limit);
                                        box_original.BackwardFFT();

                                        box_original.WriteSlicesAndFillHeader("dbg_original_box.mrc", pixel_size_in_Angstroms);

                                long physical_coord = input_volume_one->ReturnReal1DAddressFromPhysicalCoord(i, j, k);
                                local_resolution_volume->real_values[physical_coord] = box_original.ReturnRealPixelFromPhysicalCoord(box_original.physical_address_of_box_center_x, box_original.physical_address_of_box_center_y, box_original.physical_address_of_box_center_z);
                                local_resolution_volume->real_values[pixel_counter] = local_resolution_volume->real_values[physical_coord];
*/
                             // Filter using FSC on combined half maps volume boxes

                                        box_combined_half_maps.WriteSlicesAndFillHeader("/tmp/dbg_combinedhm_box_unfiltered.mrc", pixel_size_in_Angstroms);

                                        box_combined_half_maps.ForwardFFT();
                                        box_combined_half_maps.ApplyCurveFilter(&FSC_filter, resolution_limit);
                                        box_combined_half_maps.BackwardFFT();

                                        box_combined_half_maps.WriteSlicesAndFillHeader("/tmp/dbg_combinedhm_box.mrc", pixel_size_in_Angstroms);

                                 long physical_coord = input_volume_one->ReturnReal1DAddressFromPhysicalCoord(i, j, k);
                                 local_resolution_volume->real_values[physical_coord] = box_combined_half_maps.ReturnRealPixelFromPhysicalCoord(box_combined_half_maps.physical_address_of_box_center_x, box_combined_half_maps.physical_address_of_box_center_y, box_combined_half_maps.physical_address_of_box_center_z);
                                 local_resolution_volume->real_values[pixel_counter] = local_resolution_volume->real_values[physical_coord];


#ifdef DEBUG
                                // Debug: printout FSC curve

                                if ( on_dbg_point ) {
                                	std::ofstream fsc_curves;
                                	fsc_curves.open ("/tmp/fsc_curves.txt", std::ios_base::app);
                                	fsc_curves << "0 0.0 1.0 \n";
                                    for ( int shell_counter = 1; shell_counter < number_of_fsc_shells; shell_counter++ ) {
                                    	inverse_resolution = float (shell_counter) / (pixel_size_in_Angstroms * 2.0 * float(number_of_fsc_shells - 1));
                                    	wxPrintf("%i %.2f %.4f\n", shell_counter, inverse_resolution, computed_fsc[shell_counter]);
                                        fsc_curves << shell_counter << " " << inverse_resolution << " " << computed_fsc[shell_counter] << "\n";
                                    }
                                    wxPrintf("\n\n");

                                }

#endif


                            }
                        }

                        pixel_counter++;
                    }
                }

                pixel_counter += input_volume_one->padding_jump_value;
            }

        }

        else {
            // We are not estimating this slice
            int ii, jj;
            for ( jj = 0; jj < input_volume_one->logical_y_dimension; jj++ ) {
                for ( ii = 0; ii < input_volume_one->logical_x_dimension; ii++ ) {
                    // set the local resolution to indicate we didn't measure it
                    local_resolution_volume->real_values[pixel_counter] = resolution_value_between_estimation_points;

                    pixel_counter++;
                }
                pixel_counter += input_volume_one->padding_jump_value;
            }
#ifdef DEBUG
            if ( on_dbg_point )
                wxPrintf("At debug point, but not estimating this slice\n");
#endif
        }


    }
    local_resolution_volume->QuickAndDirtyWriteSlices("/tmp/FSC_Local_Filtered_Test.mrc", 1, input_volume_one->logical_z_dimension, true);
    combined_half_maps.QuickAndDirtyWriteSlices("/tmp/combined_half_maps.mrc", 1, input_volume_one->logical_z_dimension, true);
    delete my_progress_bar;
}


float LocalResolutionEstimator::ReturnResolutionOfIntersectionBetweenFSCAndThreshold(float resolution_one, float resolution_two, float fsc_one, float fsc_two, float threshold_one, float threshold_two) {
    MyDebugAssertTrue(resolution_one > 0.0, "Resolution one must be greater than zero");
    MyDebugAssertTrue(resolution_two > 0.0, "Resolution two must be greater than zero");
    MyDebugAssertTrue(resolution_one > resolution_two, "Resolution one should be larger than resolution two");
    MyDebugAssertTrue(fsc_one >= fsc_two, "FSC curve should not be rising in the interval");
    MyDebugAssertTrue(threshold_one >= threshold_two, "Threshold curve should not be rising in the interval");

    float sf_one, sf_two;
    float sf_of_intersection;
    sf_one = 1.0 / resolution_one;
    sf_two = 1.0 / resolution_two;

    // TODO: THIS OVERIDES ALEXIS' STUFF - CHECK IT LATER
  //  return 1.0f / ((sf_one + sf_two) / 2);
    sf_of_intersection = sf_one + (sf_two - sf_one) * (threshold_one - fsc_one) / ((fsc_two - fsc_one) - (threshold_two - threshold_one));

    MyDebugAssertTrue(sf_of_intersection >= sf_one && sf_of_intersection <= sf_two, "Sanity check failed. Spatial frequency of interesection (%.3f) should be between two input points (%.3f and %.3f)\nThresh1 = %.2f, Thresh2 = %.2f", sf_of_intersection, sf_one, sf_two, threshold_one, threshold_two);

    return 1.0 / sf_of_intersection;
}

float LocalResolutionEstimator::SigmaZSquaredAuto(float n) {
    float r; // correlation
    float s; // sigma_z_squared
    if ( n < 10.0 ) {
        r = 0.9;
        s = (1 / (n - 1)) * (1 + (4 - powf(r, 2)) / (2 * (n - 1)) + (176 - 21 * powf(r, 2) - 21 * r) / (48 * powf((n - 1), 2)));
    }
    else {
        s = 1 / (n - 3);
    }
    return s;

    MyDebugAssertTrue(s > 0.0, "Oops. Sigma Z squared (auto) should be positive: %f", s);
}

// Equation 8 in Bershad & Rockmore
float LocalResolutionEstimator::SigmaZSquared(float r, float n) {

    MyDebugAssertTrue(n > 1.0, "n must be greater than 1.0: %f", n);

    float s; //sigma_z_squared

    s = (1.0 / (n - 1.0)) * (1 + (4.0 - powf(r, 2)) / (2.0 * (n - 1.0)) + (176.0 - 21.0 * powf(r, 2) - 21.0 * r) / (48.0 * powf((n - 1.0), 2)));

    MyDebugAssertTrue(s > 0.0, "Oops. Sigma Z squared should be positive: n = %f, r = %f, s = %f", n, r, s);

    return s;
}

float LocalResolutionEstimator::RhoThreshold(float alpha_t, float n_sigmas, float n_voxels, int n_iterations) {
    MyDebugAssertTrue(alpha_t > 0.0, "Threshold SNR must be positive: %f", alpha_t);
    MyDebugAssertTrue(n_sigmas > 0.0, "Confidence level must be positive: %f", n_sigmas);
    MyDebugAssertTrue(n_voxels >= 0.0, "Number of voxels must be positive: %f", n_voxels);
    MyDebugAssertTrue(n_iterations >= 0, "Number of iterations can't be negative");

    float rho_t;

    if ( n_voxels < 3.0 ) {
        rho_t = 1.0;
    }
    else {
        // we don't yet know what rho_t will be, even approximately, so let's use
        // an inaccurate approximation to sigma_z
        rho_t = 1.0 - 2.0 / ((2.0 * alpha_t + 1.0 + 2.0 * n_sigmas * sqrt(exp(4 * SigmaZSquaredAuto(n_voxels)) - 1) * (alpha_t + 0.5)) * exp(2 * SigmaZSquaredAuto(n_voxels)) + 1.0);

        for ( int foo = 0; foo < n_iterations; foo++ ) {
            // now that we have a good guess, let's use the more accurate sigma_z
            rho_t = 1.0 - 2.0 / ((2.0 * alpha_t + 1.0 + 2.0 * n_sigmas * sqrt(exp(4 * SigmaZSquared(rho_t, n_voxels)) - 1) * (alpha_t + 0.5)) * exp(2 * SigmaZSquared(rho_t, n_voxels)) + 1.0);
        }
    }

    MyDebugAssertFalse(std::isnan(rho_t), "Oops. rho_t NaN\n");
    MyDebugAssertTrue(rho_t >= 0.0, "Oops. negative rho_t: %f\n", rho_t);
    MyDebugAssertTrue(rho_t <= 1.0, "Oops. rho_t too big: %f\n", rho_t);

    return rho_t;
}
