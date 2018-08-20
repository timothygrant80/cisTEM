#include "core_headers.h"

// Use this define to dump intermediate files
//#define dump_intermediate_files


LocalResolutionEstimator::LocalResolutionEstimator()
{
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
	threshold_snr								=	0.0;
	threshold_confidence_n_sigma				=	0.0;
	resolution_value_before_first_shell			=	0.0;
	resolution_value_where_wont_estimate		=	0.0;

}

LocalResolutionEstimator::~LocalResolutionEstimator()
{
	DeallocateShellNumberLUT();
	box_one.Deallocate();
	box_two.Deallocate();
}

void LocalResolutionEstimator::SetAllUserParameters(Image *wanted_input_volume_one, Image *wanted_input_volume_two, Image *wanted_mask_volume, int wanted_first_slice, int wanted_last_slice, int wanted_sampling_step, float input_pixel_size_in_Angstroms, int wanted_box_size, float wanted_threshold_snr, float wanted_threshold_confidence_n_sigma, bool wanted_use_fixed_fsc_threshold, float wanted_fixed_fsc_threshold)
{
	MyDebugAssertTrue(IsEven(box_size),"Box size should be even");
	SetInputVolumes(wanted_input_volume_one,wanted_input_volume_two,wanted_mask_volume);
	first_slice = wanted_first_slice;
	last_slice = wanted_last_slice;
	sampling_step = wanted_sampling_step;
	SetPixelSize(input_pixel_size_in_Angstroms);
	box_size = wanted_box_size;
	number_of_fsc_shells = box_size/2;
	threshold_snr = wanted_threshold_snr;
	threshold_confidence_n_sigma = wanted_threshold_confidence_n_sigma;
	use_fixed_fsc_threshold = wanted_use_fixed_fsc_threshold;
	fixed_fsc_threshold = wanted_fixed_fsc_threshold;

	// Update based on user-supplied parameter values
	resolution_value_before_first_shell = pixel_size_in_Angstroms * box_size ; //40.0;// // TODO: check / think about whether the resolution of a shell is computed properly. On average, the frequencies contributing to a shell are not the frequency of the average radius of the shell... Especially relevant in first couple of shells!
	resolution_value_where_wont_estimate = resolution_value_before_first_shell;
}

void LocalResolutionEstimator::EstimateLocalResolution(Image *local_resolution_volume)
{
	MyDebugAssertTrue(input_volume_one->IsCubic(),"This method assumes the input volumes are cubic");
	MyDebugAssertTrue(local_resolution_volume->HasSameDimensionsAs(input_volume_one),"Local res volume does not have expected dimensions");


	AllocateLocalVolumes();

	//int number_of_fsc_shells = box_size / 2;

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
	Image box_mask(box_one);
	box_mask.SetToConstant(1.0);
	box_mask.CosineMask(box_size / 4, box_size / 2, false, true, 0.0);



    // Precompute, for each Fourier voxel, the shell it will contribute to.
	AllocateShellNumberLUT();
	PopulateShellNumberLUT();

	// Estimate the number of independent voxels per shell, taking into account real-space apodization
	CountIndependentVoxelsPerShell(number_of_independent_voxels,box_mask.ReturnAverageOfRealValues(0.0,false));


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
	box_one.Allocate(box_size,box_size,box_size);
	box_two.Allocate(box_size,box_size,box_size);
	// This below to calm valgrind down
	box_one.SetToConstant(0.0);
	box_two.SetToConstant(0.0);
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

	const float nearly_one = 0.995;

	// Initialize
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		fsc_threshold[shell_counter] = 0.0;
	}

	wxPrintf("\n\nFSC threshold\n");

	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++ )
	{
		if (number_of_independent_voxels[shell_counter] <= 3.0)
		{
			fsc_threshold[shell_counter] = 1.0;
		}
		else
		{
			fsc_threshold[shell_counter] = 1.0 - 4.0 / ( (1.0 + threshold_confidence_n_sigma * sqrt(exp(4.0/(number_of_independent_voxels[shell_counter] - 3.0))-1.0) ) * (2.0 * threshold_snr + 1.0) * exp(2.0/(number_of_independent_voxels[shell_counter]-3.0)) + 3.0 );
		}
		if (use_fixed_fsc_threshold)
		{
			// Overwrite
			fsc_threshold[shell_counter] = fixed_fsc_threshold;
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
	const int dbg_i = 130;
	const int dbg_j = 108;
	const int dbg_k = 110;
#endif

	// Initialisation
	for (int shell_counter = 0; shell_counter < number_of_fsc_shells; shell_counter ++)
	{
		computed_fsc[shell_counter] = 0.0;
	}

	// Sliding window over the volumes
	long total_number_of_boxes = std::min(input_volume_one->logical_z_dimension - box_size, last_slice-first_slice+1) * (input_volume_one->logical_y_dimension - box_size) * (input_volume_one->logical_x_dimension - box_size) / pow(sampling_step,3);
	long pixel_counter = 0;
	int counter_number_of_boxes = 0;
	MyPrintfGreen("Total number of boxes = %li\n",total_number_of_boxes);
	float current_resolution;
	float previous_resolution;
	ProgressBar *my_progress_bar;
	my_progress_bar = new ProgressBar(total_number_of_boxes);
	bool just_a_glitch;

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
						if (i%sampling_step == 0)
						{
							counter_number_of_boxes++;
							my_progress_bar->Update(counter_number_of_boxes);

#ifdef DEBUG
							if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("On the debug point\n");
#endif

							if (input_volume_mask->real_values[pixel_counter] == 0.0 || i < center_of_first_box || i > center_of_last_box || j < center_of_first_box || j > center_of_last_box || k < center_of_first_box || k > center_of_last_box)
							{
#ifdef DEBUG
								if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but mask was 0.0\n");
#endif
								local_resolution_volume->real_values[pixel_counter] = resolution_value_where_wont_estimate;
							}
							else
							{
								box_one.is_in_real_space = true;
								box_two.is_in_real_space = true;
								input_volume_one->ClipInto(&box_one,0.0,false,0.0,i - input_volume_one->physical_address_of_box_center_x,j - input_volume_one->physical_address_of_box_center_y,k - input_volume_one->physical_address_of_box_center_z);
								box_one.MultiplyPixelWise(box_mask);
								box_one.ForwardFFT(false);
								input_volume_two->ClipInto(&box_two,0.0,false,0.0,i - input_volume_one->physical_address_of_box_center_x,j - input_volume_one->physical_address_of_box_center_y,k - input_volume_one->physical_address_of_box_center_z);
								box_two.MultiplyPixelWise(box_mask);
#ifdef DEBUG
								if (i == dbg_i && j == dbg_j && k == dbg_k)
								{
									box_one.QuickAndDirtyWriteSlices("dbg_vol1.mrc", 1, box_one.logical_z_dimension);
									box_two.QuickAndDirtyWriteSlices("dbg_vol2.mrc", 1, box_two.logical_z_dimension);
								}
#endif
								box_two.ForwardFFT(false);

								box_one.ComputeFSCVectorized(&box_two, &work_box_one, &work_box_two, &work_box_cross, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);
								//box_one.ComputeFSC(&box_two, number_of_fsc_shells, shell_number_lut, computed_fsc, work_sum_of_squares, work_sum_of_other_squares, work_sum_of_cross_products);

#ifdef DEBUG
								// Debug: printout FSC curve
								if (i == dbg_i && j == dbg_j && k == dbg_k)
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

									if (computed_fsc[shell_counter] < fsc_threshold[shell_counter])
									{

										/*
										 * Maybe it was just a numerical glitch and in fact the FSC curve is staying above threshold.
										 * This is particularly liable to happen in the very first shells, where N is very low,
										 * FSC is very high and small numerical errors can have an outsize effect.
										 * So if we're below threshold now but the next shells are back up above threshold,
										 * and if we're up above 0.9 anyway, let's assume it's just a glitch in the matrix.
										 */
										just_a_glitch = false;
										if (shell_counter < number_of_fsc_shells - 2) just_a_glitch = computed_fsc[shell_counter] > 0.9 && (computed_fsc[shell_counter+1] > fsc_threshold[shell_counter+1] || computed_fsc[shell_counter+2] > fsc_threshold[shell_counter+2]);

										if (!just_a_glitch)
										{

											if (previous_resolution == 0.0)
											{
												local_resolution_volume->real_values[pixel_counter] = resolution_value_before_first_shell;
											}
											else
											{
												local_resolution_volume->real_values[pixel_counter] = 1.0 / ( (1.0/previous_resolution) + ((1.0/current_resolution)-(1.0/previous_resolution)) * (fsc_threshold[shell_counter]-computed_fsc[shell_counter-1])/(computed_fsc[shell_counter]-computed_fsc[shell_counter-1])) ;
											}
#ifdef DEBUG
											if (i == dbg_i && j == dbg_j && k == dbg_k) { wxPrintf("Estimated local resolution: %f Å\n",local_resolution_volume->real_values[pixel_counter]); }
#endif
											break;
										}
									}
									else if (shell_counter == number_of_fsc_shells - 1)
									{
										local_resolution_volume->real_values[pixel_counter] = current_resolution;
#ifdef DEBUG
										if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but got to last shell\n");
#endif
									}
									previous_resolution = current_resolution;
								}
#ifdef DEBUG
								if (i == dbg_i && j == dbg_j && k == dbg_k) { wxPrintf("Estimated local resolution: %f Å\n",local_resolution_volume->real_values[pixel_counter]); }
#endif

							}

						}
						else
						{
							// set the local resolution to indicate we didn't measure it
							local_resolution_volume->real_values[pixel_counter] = resolution_value_between_estimation_points;
#ifdef DEBUG
							if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but between estimation points\n");
#endif
						}

						pixel_counter++;
					}
				}
				else
				{
#ifdef DEBUG
					if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but not estimating this line\n");
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
			if (i == dbg_i && j == dbg_j && k == dbg_k) wxPrintf("At debug point, but not estimating this slice\n");
#endif
		}
	}
	delete my_progress_bar;
}
