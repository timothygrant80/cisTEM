#include "core_headers.h"

int ReturnSpectrumBinNumber(int number_of_bins, float number_of_extrema_profile[], Image* number_of_extrema, long address, Image* ctf_values, float ctf_values_profile[]) {
    int   current_bin;
    float diff_number_of_extrema;
    float diff_number_of_extrema_previous;
    float diff_number_of_extrema_next;
    float ctf_diff_from_current_bin;
    float ctf_diff_from_current_bin_old;
    int   chosen_bin;
    //
    //MyDebugPrint("address: %li - number of extrema: %f - ctf_value: %f\n", address, number_of_extrema->real_values[address], ctf_values->real_values[address]);
    MyDebugAssertTrue(address < number_of_extrema->real_memory_allocated, "Oops, bad address: %li\n", address);
    // Let's find the bin which has the same number of preceding extrema and the most similar ctf value
    ctf_diff_from_current_bin = std::numeric_limits<float>::max( );
    chosen_bin                = -1;
    for ( current_bin = 0; current_bin < number_of_bins; current_bin++ ) {
        diff_number_of_extrema = fabs(number_of_extrema->real_values[address] - number_of_extrema_profile[current_bin]);
        if ( current_bin > 0 ) {
            diff_number_of_extrema_previous = fabs(number_of_extrema->real_values[address] - number_of_extrema_profile[current_bin - 1]);
        }
        else {
            diff_number_of_extrema_previous = std::numeric_limits<float>::max( );
        }
        if ( current_bin < number_of_bins - 1 ) {
            diff_number_of_extrema_next = fabs(number_of_extrema->real_values[address] - number_of_extrema_profile[current_bin + 1]);
        }
        else {
            diff_number_of_extrema_next = std::numeric_limits<float>::max( );
        }
        //
        if ( number_of_extrema->real_values[address] > number_of_extrema_profile[number_of_bins - 1] ) {
            chosen_bin = number_of_bins - 1;
        }
        else {
            if ( diff_number_of_extrema <= 0.01 || (diff_number_of_extrema < diff_number_of_extrema_previous &&
                                                    diff_number_of_extrema <= diff_number_of_extrema_next &&
                                                    number_of_extrema_profile[std::max(current_bin - 1, 0)] != number_of_extrema_profile[std::min(current_bin + 1, number_of_bins - 1)]) ) {
                // We're nearly there
                // Let's look for the position for the nearest CTF value
                ctf_diff_from_current_bin_old = ctf_diff_from_current_bin;
                ctf_diff_from_current_bin     = fabs(ctf_values->real_values[address] - ctf_values_profile[current_bin]);
                if ( ctf_diff_from_current_bin < ctf_diff_from_current_bin_old ) {
                    //MyDebugPrint("new chosen bin: %i\n",current_bin);
                    chosen_bin = current_bin;
                }
            }
        }
    }
    if ( chosen_bin == -1 ) {
        //TODO: return false
#ifdef DEBUG
        MyPrintfRed("Could not find bin\n");
        DEBUG_ABORT;
#endif
    }
    else {
        //MyDebugAssertTrue(chosen_bin > 0 && chosen_bin < number_of_bins,"Oops, bad chosen bin number: %i (number of bins = %i)\n",chosen_bin,number_of_bins);
        //MyDebugPrint("final chosen bin = %i\n", chosen_bin);
        return chosen_bin;
    }

    return -1;
}

// Align rotationally a (stack) of image(s) against another image. Return the rotation angle that gives the best normalised cross-correlation.
float SpectrumImage::FindRotationalAlignmentBetweenTwoStacksOfImages(Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius) {
    MyDebugAssertTrue(this[0].is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(this[0].is_in_real_space, "Not in real space");
    MyDebugAssertTrue(this[0].logical_z_dimension == 1, "Meant for images, not volumes");
    MyDebugAssertTrue(other_image[0].is_in_memory, "Memory not allocated - other_image");
    MyDebugAssertTrue(other_image[0].is_in_real_space, "Not in real space - other_image");
    MyDebugAssertTrue(other_image[0].logical_z_dimension == 1, "Meant for images, not volumes - other_image");
    MyDebugAssertTrue(this[0].HasSameDimensionsAs(&other_image[0]), "Images and reference images do not have same dimensions.");

    // Local variables
    const float           minimum_radius_sq           = pow(minimum_radius, 2);
    const float           maximum_radius_sq           = pow(maximum_radius, 2);
    const float           inverse_logical_x_dimension = 1.0 / float(this[0].logical_x_dimension);
    const float           inverse_logical_y_dimension = 1.0 / float(this[0].logical_y_dimension);
    float                 best_cc                     = -std::numeric_limits<float>::max( );
    float                 best_rotation               = -std::numeric_limits<float>::max( );
    float                 current_rotation            = -search_half_range;
    float                 current_rotation_rad;
    EmpiricalDistribution cc_numerator_dist;
    EmpiricalDistribution cc_denom_self_dist;
    EmpiricalDistribution cc_denom_other_dist;
    int                   current_image;
    int                   i, i_logi;
    float                 i_logi_frac, ii_phys;
    int                   j, j_logi;
    float                 j_logi_frac, jj_phys;
    float                 current_interpolated_value;
    long                  address_in_other_image;
    float                 current_cc;

    // Loop over possible rotations
    while ( current_rotation < search_half_range + search_step_size ) {

        current_rotation_rad = current_rotation / 180.0 * PIf;
        cc_numerator_dist.Reset( );
        cc_denom_self_dist.Reset( );
        cc_denom_other_dist.Reset( );
        // Loop over the array of images
        for ( current_image = 0; current_image < number_of_images; current_image++ ) {
            // Loop over the other (reference) image
            address_in_other_image = 0;
            for ( j = 0; j < other_image[0].logical_y_dimension; j++ ) {
                j_logi      = j - other_image[0].physical_address_of_box_center_y;
                j_logi_frac = pow(j_logi * inverse_logical_y_dimension, 2);
                for ( i = 0; i < other_image[0].logical_x_dimension; i++ ) {
                    i_logi      = i - other_image[0].physical_address_of_box_center_x;
                    i_logi_frac = pow(i_logi * inverse_logical_x_dimension, 2) + j_logi_frac;

                    if ( i_logi_frac >= minimum_radius_sq && i_logi_frac <= maximum_radius_sq ) {
                        // We do ccw rotation to go from other_image (reference) to self (input image)
                        ii_phys = i_logi * cos(current_rotation_rad) - j_logi * sin(current_rotation_rad) + this[0].physical_address_of_box_center_x;
                        jj_phys = i_logi * sin(current_rotation_rad) + j_logi * cos(current_rotation_rad) + this[0].physical_address_of_box_center_y;
                        //
                        if ( int(ii_phys) > 0 && int(ii_phys) + 1 < this[0].logical_x_dimension && int(jj_phys) > 0 && int(jj_phys) + 1 < this[0].logical_y_dimension ) // potential optimization: we have to compute the floor and ceiling in the interpolation routine. Is it not worth doing the bounds checking in the interpolation routine somehow?
                        {
                            this[0].GetRealValueByLinearInterpolationNoBoundsCheckImage(ii_phys, jj_phys, current_interpolated_value);
                            //MyDebugPrint("%g %g\n",current_interpolated_value,other_image[0].real_values[address_in_other_image]);
                            cc_numerator_dist.AddSampleValue(current_interpolated_value * other_image[current_image].real_values[address_in_other_image]);
                            cc_denom_other_dist.AddSampleValue(pow(other_image[0].real_values[address_in_other_image], 2)); // potential optimization: since other_image is not being rotated, we should only need to compute this quantity once, not for every potential rotation
                            cc_denom_self_dist.AddSampleValue(pow(current_interpolated_value, 2));
                        }
                    }
                    address_in_other_image++;
                } // i
                address_in_other_image += other_image[0].padding_jump_value;
            } // end of loop over other (reference) image
        } // end of loop over array of images

        current_cc = cc_numerator_dist.GetSampleSum( ) / sqrt(cc_denom_other_dist.GetSampleSum( ) * cc_denom_self_dist.GetSampleSum( ));

        if ( current_cc > best_cc ) {
            best_cc       = current_cc;
            best_rotation = current_rotation;
        }

        // Increment the rotation
        current_rotation += search_step_size;

    } // end of loop over rotations

    return best_rotation;
}

// This function assumes the image is in real space and generates the power spectrum
void SpectrumImage::GeneratePowerspectrum(CTF ctf_to_apply) {
    MyDebugAssertTrue(is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "image not in real space");
    MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

    int         i, j;
    float       j_logi, j_logi_sq, i_logi, i_logi_sq;
    float       current_spatial_frequency_squared;
    long        address                     = 0;
    const float inverse_logical_x_dimension = 1.0 / float(logical_x_dimension);
    const float inverse_logical_y_dimension = 1.0 / float(logical_y_dimension);
    float       current_azimuth;
    float       current_ctf_value;

    for ( j = 0; j < logical_y_dimension; j++ ) {
        address   = j * (padding_jump_value + 2 * physical_address_of_box_center_x);
        j_logi    = float(j - physical_address_of_box_center_y) * inverse_logical_y_dimension;
        j_logi_sq = powf(j_logi, 2);
        for ( i = 0; i < logical_x_dimension; i++ ) {
            i_logi    = float(i - physical_address_of_box_center_x) * inverse_logical_x_dimension;
            i_logi_sq = powf(i_logi, 2);

            // Where are we?
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

            current_azimuth   = atan2f(j_logi, i_logi);
            current_ctf_value = ctf_to_apply.EvaluatePowerspectrumWithThickness(current_spatial_frequency_squared, current_azimuth);
            // Apply powespectrum
            real_values[address + i] = current_ctf_value;
        }
    }
}

// Overlays an rotationally averaged sector and an model based sector on the
// power spectrum
void SpectrumImage::OverlayCTF(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* epa_pre_max, Curve* epa_post_max) {
    MyDebugAssertTrue(this->is_in_memory, "Spectrum memory not allocated");

    //
    EmpiricalDistribution values_in_rings;
    EmpiricalDistribution values_in_fitting_range;
    int                   i;
    int                   j;
    long                  address;
    float                 i_logi, i_logi_sq;
    float                 j_logi, j_logi_sq;
    float                 current_spatial_frequency_squared;
    float                 current_azimuth;
    float                 current_defocus;
    float                 current_phase_aberration;
    float                 sq_sf_of_phase_aberration_maximum;
    const float           lowest_freq  = pow(ctf->GetLowestFrequencyForFitting( ), 2);
    const float           highest_freq = pow(ctf->GetHighestFrequencyForFitting( ), 2);
    float                 current_ctf_value;
    float                 target_sigma;
    int                   chosen_bin;

    //this->QuickAndDirtyWriteSlice("dbg_spec_overlay_entry.mrc",1);

    //
    address = 0;
    for ( j = 0; j < this->logical_y_dimension; j++ ) {
        j_logi    = float(j - this->physical_address_of_box_center_y) * this->fourier_voxel_size_y;
        j_logi_sq = powf(j_logi, 2);
        for ( i = 0; i < this->logical_x_dimension; i++ ) {
            i_logi    = float(i - this->physical_address_of_box_center_x) * this->fourier_voxel_size_x;
            i_logi_sq = powf(i_logi, 2);
            //
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
            current_azimuth                   = atan2(j_logi, i_logi);
            current_defocus                   = ctf->DefocusGivenAzimuth(current_azimuth);
            current_phase_aberration          = ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_spatial_frequency_squared, current_defocus);
            //
            sq_sf_of_phase_aberration_maximum = ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(current_defocus);

            if ( j < this->physical_address_of_box_center_y && i >= this->physical_address_of_box_center_x ) {
                // Experimental 1D average
#ifdef use_epa_rather_than_zero_counting
                if ( current_spatial_frequency_squared <= sq_sf_of_phase_aberration_maximum ) {
                    this->real_values[address] = epa_pre_max->ReturnLinearInterpolationFromX(current_phase_aberration);
                }
                else {
                    this->real_values[address] = epa_post_max->ReturnLinearInterpolationFromX(current_phase_aberration);
                }
#else
                // Work out which bin in the astig rot average this pixel corresponds to
                chosen_bin                 = ReturnSpectrumBinNumber(number_of_bins_in_1d_spectra, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                this->real_values[address] = rotational_average_astig[chosen_bin];
#endif
            }
            //
            if ( current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared <= highest_freq ) {
                current_azimuth   = atan2(j_logi, i_logi);
                current_ctf_value = fabs(ctf->Evaluate(current_spatial_frequency_squared, current_azimuth));
                if ( current_ctf_value > 0.5 )
                    values_in_rings.AddSampleValue(this->real_values[address]);
                values_in_fitting_range.AddSampleValue(this->real_values[address]);
                //if (current_azimuth <= ctf->GetAstigmatismAzimuth()  && current_azimuth >= ctf->GetAstigmatismAzimuth() - 3.1415*0.5) this->real_values[address] = current_ctf_value;
                if ( j < this->physical_address_of_box_center_y && i < this->physical_address_of_box_center_x )
                    this->real_values[address] = current_ctf_value;
            }
            if ( current_spatial_frequency_squared <= lowest_freq ) {
                this->real_values[address] = 0.0;
            }
            //
            address++;
        }
        address += this->padding_jump_value;
    }

    //this->QuickAndDirtyWriteSlice("dbg_spec_overlay_1.mrc",1);

    /*

	// We will renormalize the experimental part of the diagnostic image
	target_sigma = sqrtf(values_in_rings.GetSampleVariance()) ;


	if (target_sigma > 0.0)
	{
		address = 0;
		for (j=0;j < this->logical_y_dimension;j++)
		{
			j_logi = float(j-this->physical_address_of_box_center_y) * this->fourier_voxel_size_y;
			j_logi_sq = powf(j_logi,2);
			for (i=0 ;i < this->logical_x_dimension; i++)
			{
				i_logi = float(i-this->physical_address_of_box_center_x) * this->fourier_voxel_size_x;
				i_logi_sq = powf(i_logi,2);
				//
				current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
				// Normalize the experimental part of the diagnostic image
				if (i > this->physical_address_of_box_center_x || j > this->physical_address_of_box_center_y)
				{
					this->real_values[address] /= target_sigma;
				}
				else
				{
					// Normalize the outside of the theoretical part of the diagnostic image
					if (current_spatial_frequency_squared > highest_freq) this->real_values[address] /= target_sigma;
				}

				address++;
			}
			address += this->padding_jump_value;
		}
	}
	*/

    //this->QuickAndDirtyWriteSlice("dbg_spec_overlay_final.mrc",1);
}

void SpectrumImage::ComputeRotationalAverageOfPowerSpectrum(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_rank[], float number_of_extrema_profile[], float ctf_values_profile[]) {
    MyDebugAssertTrue(this->is_in_memory, "Spectrum memory not allocated");
    MyDebugAssertTrue(number_of_extrema->is_in_memory, "Number of extrema image not allocated");
    MyDebugAssertTrue(ctf_values->is_in_memory, "CTF values image not allocated");
    MyDebugAssertTrue(this->HasSameDimensionsAs(number_of_extrema), "Spectrum and number of extrema images do not have same dimensions");
    MyDebugAssertTrue(this->HasSameDimensionsAs(ctf_values), "Spectrum and CTF values images do not have same dimensions");
    //
    const bool spectrum_is_blank = this->IsConstant( );
    int        counter;
    float      azimuth_of_mid_defocus;
    float      current_spatial_frequency_squared;
    int        number_of_values[number_of_bins];
    int        i, j;
    long       address;
    float      ctf_diff_from_current_bin;
    int        chosen_bin;

    // Initialise the output arrays
    for ( counter = 0; counter < number_of_bins; counter++ ) {
        average[counter]            = 0.0;
        average_fit[counter]        = 0.0;
        average_rank[counter]       = 0.0;
        ctf_values_profile[counter] = 0.0;
        number_of_values[counter]   = 0;
    }

    //
    if ( ! spectrum_is_blank ) {
        // For each bin of our 1D profile we compute the CTF at a chosen defocus
        azimuth_of_mid_defocus = ctf->ReturnAzimuthToUseFor1DPlots( );

        // Now that we've chosen an azimuth, we can compute the CTF for each bin of our 1D profile
        for ( counter = 0; counter < number_of_bins; counter++ ) {
            current_spatial_frequency_squared  = powf(float(counter) * this->fourier_voxel_size_y, 2);
            spatial_frequency[counter]         = sqrt(current_spatial_frequency_squared);
            ctf_values_profile[counter]        = ctf->Evaluate(current_spatial_frequency_squared, azimuth_of_mid_defocus);
            number_of_extrema_profile[counter] = ctf->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared, azimuth_of_mid_defocus);
            //wxPrintf("bin %i: phase shift= %f, number of extrema = %f\n",counter,ctf->PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(current_spatial_frequency_squared,azimuth_of_mid_defocus),number_of_extrema_profile[counter]);
        }

        // Now we can loop over the spectrum again and decide to which bin to add each component
        address = 0;
        for ( j = 0; j < this->logical_y_dimension; j++ ) {
            for ( i = 0; i < this->logical_x_dimension; i++ ) {
                ctf_diff_from_current_bin = std::numeric_limits<float>::max( );
                chosen_bin                = ReturnSpectrumBinNumber(number_of_bins, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                if ( chosen_bin >= 0 ) {
                    average[chosen_bin] += this->real_values[address];
                    number_of_values[chosen_bin]++;
                }
                else {
                    //TODO: return false
                }
                //
                address++;
            }
            address += this->padding_jump_value;
        }

        // Do the actual averaging
        for ( counter = 0; counter < number_of_bins; counter++ ) {
            if ( number_of_values[counter] > 0 ) {
                average[counter] = average[counter] / float(number_of_values[counter]);
                MyDebugAssertFalse(std::isnan(average[counter]), "Average is NaN for bin %i\n", counter);
            }
            else {
                average[counter] = 0.0;
            }
            average_fit[counter] = fabs(ctf_values_profile[counter]);
        }
    }

    // Compute the rank version of the rotational average
    for ( counter = 0; counter < number_of_bins; counter++ ) {
        average_rank[counter] = average[counter];
    }
    Renormalize1DSpectrumForFRC(number_of_bins, average_rank, average_fit, number_of_extrema_profile);
    for ( counter = 0; counter < number_of_bins; counter++ ) {
        MyDebugAssertFalse(std::isnan(average[counter]), "Average is NaN for bin %i\n", counter);
        MyDebugAssertFalse(std::isnan(average_rank[counter]), "AverageRank is NaN for bin %i\n", counter);
    }
}
