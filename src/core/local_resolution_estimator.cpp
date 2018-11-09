#include "core_headers.h"

// Use this define to dump intermediate files
//#define dump_intermediate_files


LocalResolutionEstimator::LocalResolutionEstimator()
{
	input_volume_one							=	NULL;
	input_volume_two							=	NULL;
	input_volume_mask							=	NULL;
	box_size									=	0;
	number_of_fsc_shells						=	0;
	first_slice									=	0;
	last_slice									=	0;
	sampling_step								=	0;
	number_of_fsc_shells						=	0;
	pixel_size_in_Angstroms						=	0.0;
	highest_resolution_expected_in_Angstroms	=	0.0;
	maximum_radius_in_Angstroms					=	0.0;
	shell_number_lut_is_allocated				=	false;
	fixed_fsc_threshold							=	0.0;
	use_fixed_fsc_threshold						=	false;
	threshold_snr								=	0.0;
	threshold_confidence_n_sigma				=	0.0;
	resolution_value_before_first_shell			=	0.0;
	resolution_value_where_wont_estimate		=	0.0;
	symmetry_redundancy							=	0;
	whiten_half_maps							=	true;
	shell_number_lut							=	NULL;
	padding_factor								=	1;
}

LocalResolutionEstimator::~LocalResolutionEstimator()
{
	DeallocateShellNumberLUT();
	box_one.Deallocate();
	box_two.Deallocate();
	box_one_no_padding.Deallocate();
	box_two_no_padding.Deallocate();

}

void LocalResolutionEstimator::SetAllUserParameters(Image *wanted_input_volume_one, Image *wanted_input_volume_two, Image *wanted_mask_volume, int wanted_first_slice, int wanted_last_slice, int wanted_sampling_step, float input_pixel_size_in_Angstroms, int wanted_box_size, float wanted_threshold_snr, float wanted_threshold_confidence_n_sigma, bool wanted_use_fixed_fsc_threshold, float wanted_fixed_fsc_threshold, wxString wanted_symmetry_symbol, bool wanted_whiten_half_maps, int wanted_padding_factor)
{
	MyDebugAssertTrue(IsEven(box_size),"Box size should be even");
	SetInputVolumes(wanted_input_volume_one,wanted_input_volume_two,wanted_mask_volume);
	first_slice = wanted_first_slice;
	last_slice = wanted_last_slice;
	sampling_step = wanted_sampling_step;
	SetPixelSize(input_pixel_size_in_Angstroms);
	box_size = wanted_box_size;
	threshold_snr = wanted_threshold_snr;
	threshold_confidence_n_sigma = wanted_threshold_confidence_n_sigma;
	use_fixed_fsc_threshold = wanted_use_fixed_fsc_threshold;
	fixed_fsc_threshold = wanted_fixed_fsc_threshold;
	SymmetryMatrix sym_matrix;
	sym_matrix.Init(wanted_symmetry_symbol);
	symmetry_redundancy = sym_matrix.number_of_matrices;
	whiten_half_maps = wanted_whiten_half_maps;
	padding_factor = wanted_padding_factor;
	number_of_fsc_shells = box_size*padding_factor/2;

	// Update based on user-supplied parameter values
	resolution_value_before_first_shell = pixel_size_in_Angstroms * box_size * padding_factor ; // TODO: check / think about whether the resolution of a shell is computed properly. On average, the frequencies contributing to a shell are not the frequency of the average radius of the shell... Especially relevant in first couple of shells!
	resolution_value_where_wont_estimate = resolution_value_before_first_shell;
}

void LocalResolutionEstimator::EstimateLocalResolution(Image *local_resolution_volume)
{
	MyDebugAssertTrue(input_volume_one->IsCubic(),"This method assumes the input volumes are cubic");
	MyDebugAssertTrue(local_resolution_volume->HasSameDimensionsAs(input_volume_one),"Local res volume does not have expected dimensions");


	AllocateLocalVolumes();

	/*
	 * Setup a couple of arrays on the stack.
	 * TODO: consider whether doing this on the stack is really helping performance. If not, probably better to make these arrays members of the class (which I think means they have to go on the heap)
	 */
	float number_of_independent_voxels[number_of_fsc_shells];
	float fsc_threshold[number_of_fsc_shells];
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		number_of_independent_voxels[shell_counter] = 0.0;
		fsc_threshold[shell_counter] = 0.0;
	}

	// Precompute a mask for the small boxes
	Image box_mask(box_one_no_padding);
	box_mask.SetToConstant(1.0);
	box_mask.CosineMask(box_size / 4, box_size / 2, false, true, 0.0);


    // Precompute, for each Fourier voxel, the shell it will contribute to.
	AllocateShellNumberLUT();
	PopulateShellNumberLUT();

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
	mask_factor = 1.0/(PI * float(padding_factor*padding_factor));
	symmetry_factor = 1.0 / float(symmetry_redundancy);
	MyDebugPrint("Scaling factor from mask = %f\nScaling factor from symmetry = %f\n",mask_factor,symmetry_factor);
	CountIndependentVoxelsPerShell(number_of_independent_voxels,mask_factor * symmetry_factor);


	// Compute the FSC threshold
	ComputeFSCThresholdBasedOnUnbiasedSNREstimator(number_of_independent_voxels,fsc_threshold);

	// Set all voxels
	//local_resolution_volume->SetToConstant(resolution_value_where_wont_estimate);

	// Compute FSC locally and compare to the threshold
	ComputeLocalFSCAndCompareToThreshold(fsc_threshold, local_resolution_volume, box_mask);

	// Cleanup
	DeallocateShellNumberLUT();


}

void LocalResolutionEstimator::AllocateLocalVolumes()
{
	MyDebugAssertTrue(box_size > 0,"Box size was not set");
	box_one_no_padding.Allocate(box_size,box_size,box_size);
	box_two_no_padding.Allocate(box_size,box_size,box_size);
	box_one.Allocate(box_size*padding_factor,box_size*padding_factor,box_size*padding_factor);
	box_two.Allocate(box_size*padding_factor,box_size*padding_factor,box_size*padding_factor);
	// This below to calm valgrind down
	box_one.SetToConstant(0.0);
	box_two.SetToConstant(0.0);
	box_one_no_padding.SetToConstant(0.0);
	box_two_no_padding.SetToConstant(0.0);
}

void LocalResolutionEstimator::SetInputVolumes(Image *wanted_input_volume_one, Image *wanted_input_volume_two, Image *wanted_mask_volume)
{
	MyDebugAssertTrue(wanted_input_volume_one->is_in_memory,"Input volume one is not in memory");
	MyDebugAssertTrue(wanted_input_volume_two->is_in_memory,"Input volume two is not in memory");
	MyDebugAssertTrue(wanted_mask_volume->is_in_memory,"Mask volume two is not in memory");
	MyDebugAssertTrue(wanted_input_volume_one->HasSameDimensionsAs(wanted_input_volume_two),"The input volumes do not have the same dimensions");
	MyDebugAssertTrue(wanted_mask_volume->HasSameDimensionsAs(wanted_input_volume_one),"The mask volume does not have the same dimensions as the input volumes");
	input_volume_one = wanted_input_volume_one;
	input_volume_two = wanted_input_volume_two;
	input_volume_mask = wanted_mask_volume;
}

void LocalResolutionEstimator::AllocateShellNumberLUT()
{
	if (! shell_number_lut_is_allocated )
	{
		MyDebugAssertTrue(box_one.is_in_memory,"Local volume is not allocated - cannot allocate shell number LUT");
		shell_number_lut = new int[box_one.real_memory_allocated/2];
	}
}

void LocalResolutionEstimator::DeallocateShellNumberLUT()
{
	if (shell_number_lut_is_allocated) delete [] shell_number_lut;
}

/*
 * Precompute, for each Fourier voxel, the shell it will contribute to.
 * Note that we will not compute the FSC into the corners (past 0.5/pixel).
 * This method will use box_one as a scratch array and overwrite it. Beware.
 */
void LocalResolutionEstimator::PopulateShellNumberLUT()
{
	box_one.ComputeSpatialFrequencyAtEveryVoxel();
	for (long pixel_counter = 0; pixel_counter < box_one.real_memory_allocated/2; pixel_counter++)
	{
		shell_number_lut[pixel_counter] = myroundint(real(box_one.complex_values[pixel_counter]) * 2.0 * float(number_of_fsc_shells-1));
		if (shell_number_lut[pixel_counter] >= number_of_fsc_shells) shell_number_lut[pixel_counter] = 0; // we basically ignore the corners of the volume (beyond 0.5 freq)

		// Debugging
		MyDebugAssertTrue(shell_number_lut[pixel_counter] >= 0, "Oops bad shell number");
		MyDebugAssertTrue(shell_number_lut[pixel_counter] < number_of_fsc_shells, "Oops, crazy shell number (%i) at pixel number %li",shell_number_lut[pixel_counter], pixel_counter);
		if (real(box_one.complex_values[pixel_counter]) <= 0.5) MyDebugAssertTrue(shell_number_lut[pixel_counter] < number_of_fsc_shells,"Oops bad calculation: %f, %i",real(box_one.complex_values[pixel_counter]),shell_number_lut[pixel_counter] );
	}
}

void LocalResolutionEstimator::CountIndependentVoxelsPerShell(float number_of_independent_voxels[],float wanted_scaling_factor)
{
	// Initialize
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		number_of_independent_voxels[shell_counter] = 0.0;
	}

	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++)
	{
		number_of_independent_voxels[shell_counter] = 0.0;
	}
	for (long pixel_counter = 0; pixel_counter < box_one.real_memory_allocated/2; pixel_counter++)
	{
		// Keep track of number of voxels contributing to each shell
		number_of_independent_voxels[shell_number_lut[pixel_counter]] += 1.0;
	}

	// It could be that e.g. a real-space mask was applied, meaning that not all Fourier voxels are independent from each other
	MyDebugAssertTrue(wanted_scaling_factor >= 0.0 && wanted_scaling_factor <= 1.0, "Bad value of scaling factor for independent voxels");
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter++)
	{
		number_of_independent_voxels[shell_counter] *= wanted_scaling_factor;
	}

	number_of_independent_voxels[0] = 0.0;
}

void LocalResolutionEstimator::ComputeFSCThresholdBasedOnUnbiasedSNREstimator(float number_of_independent_voxels[], float fsc_threshold[])
{

	const float nearly_one = 0.998; //0.995

	// Initialize
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		fsc_threshold[shell_counter] = 0.0;
	}

	wxPrintf("\n\nFSC threshold\n");

	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++ )
	{
		if (use_fixed_fsc_threshold)
		{
			fsc_threshold[shell_counter] = fixed_fsc_threshold;
		}
		else
		{
			fsc_threshold[shell_counter] = RhoThreshold(threshold_snr, threshold_confidence_n_sigma, number_of_independent_voxels[shell_counter]);
		}
		if (fsc_threshold[shell_counter] > nearly_one) fsc_threshold[shell_counter] = nearly_one;
		wxPrintf("%i %f %f\n",shell_counter,number_of_independent_voxels[shell_counter],fsc_threshold[shell_counter]);
	}
	wxPrintf("\n\n");

}

void LocalResolutionEstimator::ComputeLocalFSCAndCompareToThreshold(float fsc_threshold[], Image *local_resolution_volume, Image box_mask)
{

	MyDebugAssertTrue(input_volume_one->is_in_real_space,"Volume one is not in real space");
	MyDebugAssertTrue(input_volume_two->is_in_real_space,"Volume two is not in real space");

	// Local vars
	int i,j,k, shell_counter;
	int center_of_first_box;
	int center_of_last_box;
	center_of_first_box = box_size / 2;
	center_of_last_box = input_volume_one->logical_x_dimension - (box_size / 2) - 1;
	float computed_fsc[number_of_fsc_shells];
	const float resolution_value_between_estimation_points = 0.0;

	// Work arrays needed for computing the FSCs fast
	Image work_box_one(box_one);
	Image work_box_two(box_two);
	Image work_box_cross(box_one);
	double work_sum_of_squares[number_of_fsc_shells];
	double work_sum_of_other_squares[number_of_fsc_shells];
	double work_sum_of_cross_products[number_of_fsc_shells];

	// Debug
#ifdef DEBUG
	const int dbg_i = 166;
	const int dbg_j = 132;
	const int dbg_k = 128;
#endif

	// Initialisation
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		computed_fsc[shell_counter] = 0.0;
	}

	// Whiten the input volumes
	if (whiten_half_maps)
	{

		input_volume_one->ForwardFFT();
		input_volume_two->ForwardFFT();
		input_volume_one->Whiten(0.5);
		input_volume_two->Whiten(0.5);
		input_volume_one->BackwardFFT();
		input_volume_two->BackwardFFT();
	}

#ifdef DEBUG
	input_volume_one->QuickAndDirtyWriteSlices("dbg_1.mrc", 1, input_volume_one->logical_z_dimension);
	input_volume_two->QuickAndDirtyWriteSlices("dbg_2.mrc", 1, input_volume_one->logical_z_dimension);
#endif

	// Sliding window over the volumes
	long total_number_of_boxes = std::min(input_volume_one->logical_z_dimension - box_size, last_slice-first_slice+1) * (input_volume_one->logical_y_dimension - box_size) * (input_volume_one->logical_x_dimension - box_size) / pow(sampling_step,3);
	long pixel_counter = 0;
	int counter_number_of_boxes = 0;
	MyPrintfGreen("Total number of boxes = %li\n",total_number_of_boxes);
	float current_resolution;
	float previous_resolution;
	ProgressBar *my_progress_bar;
	my_progress_bar = new ProgressBar(total_number_of_boxes);
	bool below_threshold, just_a_glitch;
	const bool stringent_mode = true;
	const bool allow_glitches = true;
#ifdef DEBUG
	bool on_dbg_point;
#endif

	for (k = 0; k < input_volume_one->logical_z_dimension; k ++ )
	{
		if (k >= first_slice-1 && k <= last_slice-1 && k%sampling_step == 0)
		{
			for (j = 0; j < input_volume_one->logical_y_dimension; j ++ )
			{
				if (j%sampling_step == 0)
				{
					for (i = 0; i < input_volume_one->logical_x_dimension; i ++ )
					{
#ifdef DEBUG
						on_dbg_point = i == dbg_i && j == dbg_j && k == dbg_k;
						if (on_dbg_point) wxPrintf("On the debug point\n");
#endif

						if (i%sampling_step == 0)
						{
							counter_number_of_boxes++;
							my_progress_bar->Update(counter_number_of_boxes);

							if (input_volume_mask->real_values[pixel_counter] == 0.0 || i < center_of_first_box || i > center_of_last_box || j < center_of_first_box || j > center_of_last_box || k < center_of_first_box || k > center_of_last_box)
							{
#ifdef DEBUG
								if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but mask was 0.0\n");
#endif
								local_resolution_volume->real_values[pixel_counter] = resolution_value_where_wont_estimate;
							}
							else
							{
								//TODO: if performance is an issue, write a dedicated method to copy voxels over from large volume into small one, but with padding already done (i.e. not to the edges of small volume)

								// Get voxels from input volumes into smaller boxes, mask them
								box_one_no_padding.is_in_real_space = true;
								box_two_no_padding.is_in_real_space = true;
								input_volume_one->ClipInto(&box_one_no_padding,0.0,false,0.0,i - input_volume_one->physical_address_of_box_center_x,j - input_volume_one->physical_address_of_box_center_y,k - input_volume_one->physical_address_of_box_center_z);
								box_one_no_padding.MultiplyPixelWise(box_mask);
								input_volume_two->ClipInto(&box_two_no_padding,0.0,false,0.0,i - input_volume_one->physical_address_of_box_center_x,j - input_volume_one->physical_address_of_box_center_y,k - input_volume_one->physical_address_of_box_center_z);
								box_two_no_padding.MultiplyPixelWise(box_mask);

								// Pad the small boxes in real space
								box_one.is_in_real_space = true;
								box_two.is_in_real_space = true;
								box_one_no_padding.ClipInto(&box_one,0.0);
								box_two_no_padding.ClipInto(&box_two,0.0);

#ifdef DEBUG
								if (on_dbg_point)
								{
									box_one.WriteSlicesAndFillHeader("dbg_vol1.mrc", pixel_size_in_Angstroms);
									box_two.WriteSlicesAndFillHeader("dbg_vol2.mrc", pixel_size_in_Angstroms);
								}
#endif

								box_one.ForwardFFT(false);
								box_two.ForwardFFT(false);

								// FSC
								box_one.ComputeFSCVectorized(&box_two, &work_box_one, &work_box_two, &work_box_cross, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);
								//box_one.ComputeFSC(&box_two, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);

#ifdef DEBUG
								// Debug: printout FSC curve
								if (on_dbg_point)
								{
									wxPrintf("\n\n");
									for (int shell_counter = 1; shell_counter < number_of_fsc_shells; shell_counter ++ )
									{
										current_resolution = pixel_size_in_Angstroms * 2.0 * float(number_of_fsc_shells-1) / float(shell_counter);
										wxPrintf("%i %.2f %.4f %.4f\n",shell_counter,current_resolution,fsc_threshold[shell_counter],computed_fsc[shell_counter]);
									}
									wxPrintf("\n\n");
								}
#endif


								// Walk the FSC curve and check when it crosses our threshold
								previous_resolution = 0.0;
								for (shell_counter = 1; shell_counter < number_of_fsc_shells; shell_counter ++ )
								{
									current_resolution = pixel_size_in_Angstroms * 2.0 * float(number_of_fsc_shells-1) / float(shell_counter);

									// Test: are we below the threshold?
									below_threshold = computed_fsc[shell_counter] < fsc_threshold[shell_counter];

									/*
									 * Maybe it was just a numerical glitch and in fact the FSC curve is staying above threshold.
									 * This is particularly liable to happen in the very first shells, where N is very low,
									 * FSC is very high and small numerical errors can have an outsize effect.
									 * So if we're below threshold now but the next shells are back up above threshold,
									 * and if we're up above 0.9 anyway, let's assume it's just a glitch in the matrix.
									 */
									just_a_glitch = false;
									if (below_threshold && computed_fsc[shell_counter] > 0.9)
									{
										if (stringent_mode)
										{
											if (shell_counter < number_of_fsc_shells - 2) just_a_glitch = computed_fsc[shell_counter+1] > fsc_threshold[shell_counter+1] && computed_fsc[shell_counter+2] > fsc_threshold[shell_counter+2];
										}
										else
										{
											if (shell_counter < number_of_fsc_shells - 4) just_a_glitch = computed_fsc[shell_counter+1] > fsc_threshold[shell_counter+1] || (computed_fsc[shell_counter+2] > fsc_threshold[shell_counter+2] && computed_fsc[shell_counter+3] > fsc_threshold[shell_counter+3]);
										}
									}

									if (! allow_glitches) just_a_glitch = false;

									if (below_threshold && !just_a_glitch)
									{

										if (previous_resolution == 0.0)
										{
											local_resolution_volume->real_values[pixel_counter] = resolution_value_before_first_shell;
										}
										else
										{
											MyDebugAssertTrue(computed_fsc[shell_counter] <= computed_fsc[shell_counter-1],"Oops, FSC is not dropping");
											local_resolution_volume->real_values[pixel_counter] = ReturnResolutionOfIntersectionBetweenFSCAndThreshold(previous_resolution, current_resolution, computed_fsc[shell_counter-1], computed_fsc[shell_counter], fsc_threshold[shell_counter-1], fsc_threshold[shell_counter]);
										}
#ifdef DEBUG
										if (on_dbg_point) { wxPrintf("Estimated local resolution: %f Å. Previous resolution: %f Å. Current_resolution: %f Å\n",local_resolution_volume->real_values[pixel_counter], previous_resolution, current_resolution); }
#endif
										break;

									}
									else if (shell_counter == number_of_fsc_shells - 1)
									{
										local_resolution_volume->real_values[pixel_counter] = current_resolution;
#ifdef DEBUG
										if (on_dbg_point) wxPrintf("At debug point, but got to last shell\n");
#endif
									}
									previous_resolution = current_resolution;
								}
#ifdef DEBUG
								if (on_dbg_point) { wxPrintf("Estimated local resolution: %f Å\n",local_resolution_volume->real_values[pixel_counter]); }
#endif

							}

						}
						else
						{
							// set the local resolution to indicate we didn't measure it
							local_resolution_volume->real_values[pixel_counter] = resolution_value_between_estimation_points;
#ifdef DEBUG
							if (on_dbg_point) wxPrintf("At debug point, but between estimation points\n");
#endif
						}

						pixel_counter++;
					}
				}
				else
				{
#ifdef DEBUG
					if (on_dbg_point) wxPrintf("At debug point, but not estimating this line\n");
#endif
					// We are not estimating this line
					for (int i = 0; i < input_volume_one->logical_x_dimension; i ++ )
					{
						// set the local resolution to indicate we didn't measure it
						local_resolution_volume->real_values[pixel_counter] = resolution_value_between_estimation_points;

						pixel_counter++;
					}
				}

				pixel_counter += input_volume_one->padding_jump_value;
			}
		}
		else
		{
			// We are not estimating this slice
			int ii,jj;
			for (jj = 0; jj < input_volume_one->logical_y_dimension; jj ++ )
			{
				for (ii = 0; ii < input_volume_one->logical_x_dimension; ii ++ )
				{
					// set the local resolution to indicate we didn't measure it
					local_resolution_volume->real_values[pixel_counter] = resolution_value_between_estimation_points;

					pixel_counter++;
				}
				pixel_counter += input_volume_one->padding_jump_value;
			}
#ifdef DEBUG
			if (on_dbg_point) wxPrintf("At debug point, but not estimating this slice\n");
#endif
		}
	}
	delete my_progress_bar;
}

float LocalResolutionEstimator::ReturnResolutionOfIntersectionBetweenFSCAndThreshold(float resolution_one, float resolution_two, float fsc_one, float fsc_two, float threshold_one, float threshold_two)
{
	MyDebugAssertTrue(resolution_one > 0.0, "Resolution one must be greater than zero");
	MyDebugAssertTrue(resolution_two > 0.0, "Resolution two must be greater than zero");
	MyDebugAssertTrue(resolution_one > resolution_two,"Resolution one should be larger than resolution two");
	MyDebugAssertTrue(fsc_one >= fsc_two, "FSC curve should not be rising in the interval");
	MyDebugAssertTrue(threshold_one >= threshold_two, "Threshold curve should not be rising in the interval");

	float sf_one, sf_two;
	float sf_of_intersection;
	sf_one = 1.0 / resolution_one;
	sf_two = 1.0 / resolution_two;

	sf_of_intersection = sf_one + (sf_two - sf_one) * (threshold_one - fsc_one) / ((fsc_two-fsc_one)-(threshold_two-threshold_one));

	MyDebugAssertTrue(sf_of_intersection >= sf_one && sf_of_intersection <= sf_two,"Sanity check failed. Spatial frequency of interesection should be between two input points");

	return 1.0 / sf_of_intersection;

}

float 	LocalResolutionEstimator::SigmaZSquaredAuto(float n)
{
	float r; // correlation
	float s; // sigma_z_squared
	if (n < 10.0)
	{
	    r = 0.9;
	    s = (1/(n-1))*(1+(4-powf(r,2))/(2*(n-1))+(176-21*powf(r,2)-21*r)/(48*powf((n-1),2)));
	}
	else
	{
	    s = 1/(n-3);
	}
	return s;

	MyDebugAssertTrue(s > 0.0, "Oops. Sigma Z squared (auto) should be positive: %f",s);
}

// Equation 8 in Bershad & Rockmore
float 	LocalResolutionEstimator::SigmaZSquared(float r, float n)
{

	MyDebugAssertTrue(n > 1.0,"n must be greater than 1.0: %f",n);

	float s; //sigma_z_squared

	s = (1.0/(n-1.0))*(1+(4.0-powf(r,2))/(2.0*(n-1.0))+(176.0-21.0*powf(r,2)-21.0*r)/(48.0*powf((n-1.0),2)));

	MyDebugAssertTrue(s > 0.0, "Oops. Sigma Z squared should be positive: n = %f, r = %f, s = %f",n,r,s);

	return s;
}

float	LocalResolutionEstimator::RhoThreshold(float alpha_t, float n_sigmas, float n_voxels, int n_iterations)
{
	MyDebugAssertTrue(alpha_t > 0.0,"Threshold SNR must be positive: %f", alpha_t);
	MyDebugAssertTrue(n_sigmas > 0.0, "Confidence level must be positive: %f", n_sigmas);
	MyDebugAssertTrue(n_voxels >= 0.0, "Number of voxels must be positive: %f",n_voxels);
	MyDebugAssertTrue(n_iterations >= 0, "Number of iterations can't be negative");

	float rho_t;

	if (n_voxels < 3.0)
	{
	    rho_t = 1.0;
	}
	else
	{
	    // we don't yet know what rho_t will be, even approximately, so let's use
	    // an inaccurate approximation to sigma_z
	    rho_t = 1.0 - 2.0/( (2.0*alpha_t+1.0+2.0*n_sigmas*sqrt(exp(4*SigmaZSquaredAuto(n_voxels))-1)*(alpha_t+0.5) ) *exp(2*SigmaZSquaredAuto(n_voxels)) + 1.0);

	    for (int foo = 0; foo < n_iterations; foo++)
		{
	        // now that we have a good guess, let's use the more accurate sigma_z
	        rho_t = 1.0 - 2.0/( (2.0*alpha_t+1.0+2.0*n_sigmas*sqrt(exp(4*SigmaZSquared(rho_t,n_voxels))-1)*(alpha_t+0.5) ) *exp(2*SigmaZSquared(rho_t,n_voxels)) + 1.0);
		}
	}

	MyDebugAssertFalse(std::isnan(rho_t),"Oops. rho_t NaN\n");
	MyDebugAssertTrue(rho_t >= 0.0,"Oops. negative rho_t: %f\n",rho_t);
	MyDebugAssertTrue(rho_t <= 1.0,"Oops. rho_t too big: %f\n",rho_t);

	return rho_t;
}

