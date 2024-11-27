#include "core_headers.h"

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
    const float                   minimum_radius_sq           = pow(minimum_radius, 2);
    const float                   maximum_radius_sq           = pow(maximum_radius, 2);
    const float                   inverse_logical_x_dimension = 1.0 / float(this[0].logical_x_dimension);
    const float                   inverse_logical_y_dimension = 1.0 / float(this[0].logical_y_dimension);
    float                         best_cc                     = -std::numeric_limits<float>::max( );
    float                         best_rotation               = -std::numeric_limits<float>::max( );
    float                         current_rotation            = -search_half_range;
    float                         current_rotation_rad;
    EmpiricalDistribution<double> cc_numerator_dist;
    EmpiricalDistribution<double> cc_denom_self_dist;
    EmpiricalDistribution<double> cc_denom_other_dist;
    int                           current_image;
    int                           i, i_logi;
    float                         i_logi_frac, ii_phys;
    int                           j, j_logi;
    float                         j_logi_frac, jj_phys;
    float                         current_interpolated_value;
    long                          address_in_other_image;
    float                         current_cc;

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
void SpectrumImage::OverlayCTF(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* epa_pre_max, Curve* epa_post_max, bool fit_nodes) {
    MyDebugAssertTrue(this->is_in_memory, "Spectrum memory not allocated");

    //
    EmpiricalDistribution<double> values_in_rings;
    EmpiricalDistribution<double> values_in_fitting_range;
    int                           i;
    int                           j;
    long                          address;
    float                         i_logi, i_logi_sq;
    float                         j_logi, j_logi_sq;
    float                         current_spatial_frequency_squared;
    float                         current_azimuth;
    float                         current_defocus;
    float                         current_phase_aberration;
    float                         sq_sf_of_phase_aberration_maximum;
    const float                   lowest_freq  = pow(ctf->GetLowestFrequencyForFitting( ), 2);
    const float                   highest_freq = pow(ctf->GetHighestFrequencyForFitting( ), 2);
    float                         current_ctf_value;
    float                         target_sigma;
    int                           chosen_bin;

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
                if ( fit_nodes ) {
                    current_ctf_value = ctf->EvaluatePowerspectrumWithThickness(current_spatial_frequency_squared, current_azimuth);
                }
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

// Rescale the spectrum and its 1D rotational avereage so that the peaks and troughs are at 0.0 and 1.0. The location of peaks and troughs are worked out
// by parsing the suppilied 1D average_fit array
void SpectrumImage::RescaleSpectrumAndRotationalAverage(Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit) {
    MyDebugAssertTrue(this->is_in_memory, "Spectrum memory not allocated");
    MyDebugAssertTrue(number_of_bins > 1, "Bad number of bins: %i\n", number_of_bins);

    //
    const bool spectrum_is_blank               = this->IsConstant( );
    const int  rescale_based_on_maximum_number = 2; // This peak will be used as a renormalization.
    const int  sg_width                        = 7;
    const int  sg_order                        = 2;
    const bool rescale_peaks                   = false; // if this is false, only the background will be subtracted, the Thon rings "heights" will be unaffected
    float      background[number_of_bins];
    float      peak[number_of_bins];
    int        bin_counter;
    bool       at_a_maximum, at_a_minimum, maximum_at_previous_bin, minimum_at_previous_bin;
    int        location_of_previous_maximum, location_of_previous_minimum;
    int        current_maximum_number   = 0;
    int        normalisation_bin_number = 0;
    int        i;
    int        j;
    bool       actually_do_rescaling;
    int        chosen_bin;
    long       address;
    int        last_bin_to_rescale;
    float      min_scale_factor;
    float      scale_factor;
    float      rescale_peaks_to;

    Curve* minima_curve = new Curve;
    Curve* maxima_curve = new Curve;

    // Initialise arrays and variables
    for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {
        background[bin_counter] = 0.0;
        peak[bin_counter]       = 0.0;
    }
    location_of_previous_maximum = 0;
    location_of_previous_minimum = 0;
    current_maximum_number       = 0;
    at_a_maximum                 = false;
    at_a_minimum                 = true; // Note, this may not be true if we have the perfect phase plate

    //
    if ( ! spectrum_is_blank ) {
        for ( bin_counter = 1; bin_counter < number_of_bins - 1; bin_counter++ ) {
            // Remember where we were before - minimum, maximum or neither
            maximum_at_previous_bin = at_a_maximum;
            minimum_at_previous_bin = at_a_minimum;
            // Are we at a CTF min or max?
            at_a_minimum = (average_fit[bin_counter] <= average_fit[bin_counter - 1]) && (average_fit[bin_counter] <= average_fit[bin_counter + 1]);
            at_a_maximum = (average_fit[bin_counter] >= average_fit[bin_counter - 1]) && (average_fit[bin_counter] >= average_fit[bin_counter + 1]);
            // It could be that the CTF is constant in this region, in which case we stay at a minimum if we were there
            if ( at_a_maximum && at_a_minimum ) {
                at_a_minimum = minimum_at_previous_bin;
                at_a_maximum = maximum_at_previous_bin;
            }
            // Fill in values for the background or peak by linear interpolation
            if ( at_a_minimum ) {
                for ( i = location_of_previous_minimum + 1; i <= bin_counter; i++ ) {
                    // Linear interpolation of average values at the peaks and troughs of the CTF
                    background[i] = average[location_of_previous_minimum] * float(bin_counter - i) / float(bin_counter - location_of_previous_minimum) + average[bin_counter] * float(i - location_of_previous_minimum) / float(bin_counter - location_of_previous_minimum);
                }
                location_of_previous_minimum = bin_counter;
                minima_curve->AddPoint(spatial_frequency[bin_counter], average[bin_counter]);
            }
            if ( at_a_maximum ) {
                if ( (! maximum_at_previous_bin) && (average_fit[bin_counter] > 0.7) )
                    current_maximum_number = current_maximum_number + 1;
                for ( i = location_of_previous_maximum + 1; i <= bin_counter; i++ ) {
                    // Linear interpolation of average values at the peaks and troughs of the CTF
                    peak[i] = average[location_of_previous_maximum] * float(bin_counter - i) / float(bin_counter - location_of_previous_maximum) + average[bin_counter] * float(i - location_of_previous_maximum) / float(bin_counter - location_of_previous_maximum);
                    //
                    if ( current_maximum_number == rescale_based_on_maximum_number )
                        normalisation_bin_number = bin_counter;
                }
                location_of_previous_maximum = bin_counter;
                maxima_curve->AddPoint(spatial_frequency[bin_counter], average[bin_counter]);
            }
            if ( at_a_maximum && at_a_minimum ) {
                MyPrintfRed("Rescale spectrum: Error. At a minimum and a maximum simultaneously.");
                //TODO: return false instead
                DEBUG_ABORT;
            }
        }

        // Fit the minima and maximum curves using Savitzky-Golay smoothing
        if ( maxima_curve->NumberOfPoints( ) > sg_width )
            maxima_curve->FitSavitzkyGolayToData(sg_width, sg_order);
        if ( minima_curve->NumberOfPoints( ) > sg_width )
            minima_curve->FitSavitzkyGolayToData(sg_width, sg_order);

        // Replace the background and peak envelopes with the smooth min/max curves
        for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {
            if ( minima_curve->NumberOfPoints( ) > sg_width )
                background[bin_counter] = minima_curve->ReturnSavitzkyGolayInterpolationFromX(spatial_frequency[bin_counter]);
            if ( maxima_curve->NumberOfPoints( ) > sg_width )
                peak[bin_counter] = maxima_curve->ReturnSavitzkyGolayInterpolationFromX(spatial_frequency[bin_counter]);
        }

        // Now that we have worked out a background and a peak envelope, let's do the actual rescaling
        actually_do_rescaling = (peak[normalisation_bin_number] - background[normalisation_bin_number]) > 0.0;
        if ( last_bin_without_aliasing != 0 ) {
            last_bin_to_rescale = std::min(last_bin_with_good_fit, last_bin_without_aliasing);
        }
        else {
            last_bin_to_rescale = last_bin_with_good_fit;
        }
        if ( actually_do_rescaling ) {
            min_scale_factor = 0.2;
            rescale_peaks_to = 0.75;
            address          = 0;
            for ( j = 0; j < this->logical_y_dimension; j++ ) {
                for ( i = 0; i < this->logical_x_dimension; i++ ) {
                    chosen_bin = ReturnSpectrumBinNumber(number_of_bins, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                    if ( chosen_bin >= 0 ) {
                        if ( chosen_bin <= last_bin_to_rescale ) {
                            this->real_values[address] -= background[chosen_bin]; // This alone makes the spectrum look very nice already
                            if ( rescale_peaks )
                                this->real_values[address] /= std::min(1.0f, std::max(min_scale_factor, peak[chosen_bin] - background[chosen_bin])) / rescale_peaks_to; // This is supposed to help "boost" weak Thon rings
                        }
                        else {
                            this->real_values[address] -= background[last_bin_to_rescale];
                            if ( rescale_peaks )
                                this->real_values[address] /= std::min(1.0f, std::max(min_scale_factor, peak[last_bin_to_rescale] - background[last_bin_to_rescale])) / rescale_peaks_to;
                        }
                    }
                    else {
                        //TODO: return false
                    }
                    //
                    address++;
                }
                address += this->padding_jump_value;
            }
        }
        else {
            MyDebugPrint("(RescaleSpectrumAndRotationalAverage) Warning: bad peak/background detection");
        }

        // Rescale the 1D average
        if ( peak[normalisation_bin_number] > background[normalisation_bin_number] ) {
            for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {

                average[bin_counter] = (average[bin_counter] - background[bin_counter]) / (peak[normalisation_bin_number] - background[normalisation_bin_number]) * 0.95;
                // We want peaks to reach at least 0.1
                if ( ((peak[bin_counter] - background[bin_counter]) < 0.1) && (fabs(peak[bin_counter] - background[bin_counter]) > 0.000001) && bin_counter <= last_bin_without_aliasing ) {
                    average[bin_counter] = average[bin_counter] / (peak[bin_counter] - background[bin_counter]) * (peak[normalisation_bin_number] - background[normalisation_bin_number]) * 0.1;
                }
            }
        }
        else {
            MyDebugPrint("(RescaleSpectrumAndRotationalAverage): unable to rescale 1D average experimental spectrum\n");
        }

    } // end of test of spectrum_is_blank

    // Cleanup
    delete minima_curve;
    delete maxima_curve;
}

/*
 * Stretch or shrink the powerspectrum to maintain the same bix size, but represent a new physical pixel size (target_pixel_size_after_resampling).
 * Returns the actual new pixel size, which might be sligthly different.
 */
float SpectrumImage::DilatePowerspectrumToNewPixelSize(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                                                       int box_size, Image* resampled_power_spectrum, bool do_resampling, float stretch_factor) {
    int   temporary_box_size;
    int   stretched_dimension;
    float pixel_size_for_fitting;
    bool  resampling_is_necessary;

    Image temp_image;

    // Resample the amplitude spectrum
    if ( resample_if_pixel_too_small && pixel_size_of_input_image < target_pixel_size_after_resampling ) {
        // The input pixel was too small, so let's resample the amplitude spectrum into a large temporary box, before clipping the center out for fitting
        temporary_box_size = round(float(box_size) / pixel_size_of_input_image * target_pixel_size_after_resampling);
        if ( IsOdd(temporary_box_size) )
            temporary_box_size++;
        resampling_is_necessary = this->logical_x_dimension != box_size || this->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(temporary_box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - temporary_box_size * stretch_factor) > fabsf(stretched_dimension - 2 - temporary_box_size * stretch_factor) )
                    stretched_dimension -= 2;

                this->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                this->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(this);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image * float(temporary_box_size) / float(box_size);
    }
    else {
        // The regular way (the input pixel size was large enough)
        resampling_is_necessary = this->logical_x_dimension != box_size || this->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - box_size * stretch_factor) > fabsf(stretched_dimension - 2 - box_size * stretch_factor) )
                    stretched_dimension -= 2;

                this->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                this->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(this);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image;
    }

    return pixel_size_for_fitting;
}

/*
 * Compute average value in power spectrum as a function of wave function aberration. This allows for averaging even when
 * there is significant astigmatism.
 * This should be nicer than counting zeros and looking for nearest CTF value as described in the original ctffind4 manuscript.
 * Inspired by gctf and others, but I think more robust because it takes into account that the aberration decreases again at
 * very high spatial frequencies, when Cs takes over from defocus.
 */
void SpectrumImage::ComputeEquiPhaseAverageOfPowerSpectrum(CTF* ctf, Curve* epa_pre_max, Curve* epa_post_max) {
    MyDebugAssertTrue(this->is_in_memory, "Spectrum memory not allocated");

    const bool spectrum_is_blank = this->IsConstant( );

    const int  curve_oversampling_factor = 3;
    const bool curve_x_is_linear         = true;

    /*
	 * Initialize the curve objects. One keeps track of EPA pre phase aberration maximum (before Cs term takes over), the other post.
	 * In the case where we are overfocus (negative defocus value), the phase aberration starts at 0.0 at the origin
	 * and just gets more and more negative
	 *
	 * This is one of the messiest parts of the code. I really need to come up with a cleaner way to decide how many points
	 * to give each curve. This is a goldilocks problem: too few or too many both give worse curves and FRCs.
	 */
    if ( curve_x_is_linear ) {
        float maximum_aberration_in_ctf            = ctf->ReturnPhaseAberrationMaximum( );
        float maximum_sq_freq_in_spectrum          = powf(this->fourier_voxel_size_x * this->logical_lower_bound_complex_x, 2) + powf(this->fourier_voxel_size_y * this->logical_lower_bound_complex_y, 2);
        float lowest_sq_freq_of_ctf_aberration_max = std::min(fabs(ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(ctf->GetDefocus1( ))),
                                                              fabs(ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(ctf->GetDefocus2( ))));

        float maximum_abs_aberration_in_spectrum = std::max(fabs(ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(maximum_sq_freq_in_spectrum, ctf->GetDefocus1( ))),
                                                            fabs(ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(maximum_sq_freq_in_spectrum, ctf->GetDefocus2( ))));

        /*
		 * Minimum phase aberration might be 0.0 + additional_phase_shift (at the origin), or if the phase aberration function
		 * peaks before Nyquist, it might be at the edge of the spectrum
		 */
        float minimum_aberration_in_ctf_at_edges = std::min(ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(maximum_sq_freq_in_spectrum, ctf->GetDefocus1( )),
                                                            ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(maximum_sq_freq_in_spectrum, ctf->GetDefocus2( )));

        // Watch out: messy heuristics
        int number_of_points_pre_max  = std::max(2, myroundint(this->ReturnMaximumDiagonalRadius( ) * curve_oversampling_factor * maximum_aberration_in_ctf / maximum_abs_aberration_in_spectrum));
        int number_of_points_post_max = std::max(2, myroundint(this->ReturnMaximumDiagonalRadius( ) * curve_oversampling_factor));

        epa_pre_max->SetupXAxis(ctf->GetAdditionalPhaseShift( ), maximum_aberration_in_ctf, number_of_points_pre_max);
        epa_post_max->SetupXAxis(std::min(maximum_aberration_in_ctf, minimum_aberration_in_ctf_at_edges - 0.5f * fabsf(minimum_aberration_in_ctf_at_edges)), maximum_aberration_in_ctf, number_of_points_post_max);
    }
    else {
        MyDebugAssertTrue(false, "Not implemented");
    }
    epa_pre_max->SetYToConstant(0.0);
    epa_post_max->SetYToConstant(0.0);

    /*
	 * We'll also need to keep track of the number of values
	 */
    Curve* count_pre_max  = new Curve;
    Curve* count_post_max = new Curve;
    count_pre_max->CopyFrom(epa_pre_max);
    count_post_max->CopyFrom(epa_post_max);

    if ( ! spectrum_is_blank ) {
        long  address = 0;
        int   i, j;
        float i_logi, j_logi;
        float i_logi_sq, j_logi_sq;
        float current_spatial_frequency_squared;
        float current_azimuth;
        float current_phase_aberration;
        float sq_sf_of_phase_aberration_maximum;
        float current_defocus;
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
                //
                if ( current_spatial_frequency_squared <= sq_sf_of_phase_aberration_maximum ) {
                    // Add to pre-max
                    epa_pre_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, this->real_values[address], curve_x_is_linear);
                    count_pre_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, 1.0, curve_x_is_linear);
                }
                else {
                    /*
					 * We are after the maximum phase aberration (i.e. the Cs term has taken over, phase aberration is decreasing as a function of sf)
					 */
                    // Add to post-max
                    epa_post_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, this->real_values[address], curve_x_is_linear);
                    count_post_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, 1.0, curve_x_is_linear);
                }
                //
                address++;
            }
            address += this->padding_jump_value;
        }

        /*
		 * Do the averaging
		 */
        epa_pre_max->DivideBy(count_pre_max);
        epa_post_max->DivideBy(count_post_max);
    }

    delete count_pre_max;
    delete count_post_max;
}
