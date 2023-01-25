#include "../../core/core_headers.h"
#include "./ctffind.h"

ImageCTFComparison::ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep) {
    MyDebugAssertTrue(wanted_number_of_images > 0, "Bad wanted number of images: %i\n", wanted_number_of_images);
    number_of_images = wanted_number_of_images;
    img              = new Image[wanted_number_of_images];

    ctf                       = wanted_ctf;
    pixel_size                = wanted_pixel_size;
    find_phase_shift          = should_find_phase_shift;
    astigmatism_is_known      = wanted_astigmatism_is_known;
    known_astigmatism         = wanted_known_astigmatism;
    known_astigmatism_angle   = wanted_known_astigmatism_angle;
    fit_defocus_sweep         = should_fit_defocus_sweep;
    fit_with_thickness_nodes  = false;
    azimuths                  = NULL;
    spatial_frequency_squared = NULL;
    addresses                 = NULL;
    number_to_correlate       = 0;
    image_mean                = 0.0;
    norm_image                = 0.0;
}

ImageCTFComparison::~ImageCTFComparison( ) {
    wxPrintf("Deleting ImageCTFComparison\n");
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        img[image_counter].Deallocate( );
    }
    delete[] img;
    wxPrintf("Deleting ImageCTFComparison2\n");
    delete[] azimuths;
    delete[] spatial_frequency_squared;
    wxPrintf("Deleting ImageCTFComparison3\n");
    delete[] addresses;
    number_to_correlate = 0;
    wxPrintf("Deleting ImageCTFComparison4\n");
}

void ImageCTFComparison::SetImage(int wanted_image_number, Image* new_image) {
    MyDebugAssertTrue(wanted_image_number >= 0 && wanted_image_number < number_of_images, "Wanted image number (%i) is out of bounds", wanted_image_number);
    img[wanted_image_number].CopyFrom(new_image);
}

void ImageCTFComparison::SetCTF(CTF new_ctf) {
    ctf = new_ctf;
}

void ImageCTFComparison::SetFitWithThicknessNodes(bool wanted_fit_with_thickness_nodes) {
    fit_with_thickness_nodes = wanted_fit_with_thickness_nodes;
}

void ImageCTFComparison::SetupQuickCorrelation( ) {
    img[0].SetupQuickCorrelationWithCTF(ctf, number_to_correlate, norm_image, image_mean, NULL, NULL, NULL);
    azimuths                  = new float[number_to_correlate];
    spatial_frequency_squared = new float[number_to_correlate];
    addresses                 = new int[number_to_correlate];
    img[0].SetupQuickCorrelationWithCTF(ctf, number_to_correlate, norm_image, image_mean, addresses, spatial_frequency_squared, azimuths);
}

CTF ImageCTFComparison::ReturnCTF( ) { return ctf; }

bool ImageCTFComparison::AstigmatismIsKnown( ) { return astigmatism_is_known; }

float ImageCTFComparison::ReturnKnownAstigmatism( ) { return known_astigmatism; }

float ImageCTFComparison::ReturnKnownAstigmatismAngle( ) { return known_astigmatism_angle; }

bool ImageCTFComparison::FindPhaseShift( ) { return find_phase_shift; }

// This is the function which will be minimised
float CtffindObjectiveFunction(void* scoring_parameters, float array_of_values[]) {
    ImageCTFComparison* comparison_object = reinterpret_cast<ImageCTFComparison*>(scoring_parameters);

    MyDebugAssertFalse(std::isnan(array_of_values[0]), "DF1 is NaN!");
    MyDebugAssertFalse(std::isnan(array_of_values[1]), "DF2 is NaN!");

    CTF my_ctf = comparison_object->ReturnCTF( );
    if ( comparison_object->AstigmatismIsKnown( ) ) {
        MyDebugAssertTrue(comparison_object->ReturnKnownAstigmatism( ) >= 0.0, "Known asitgmatism must be >= 0.0");
        my_ctf.SetDefocus(array_of_values[0], array_of_values[0] - comparison_object->ReturnKnownAstigmatism( ), comparison_object->ReturnKnownAstigmatismAngle( ));
    }
    else {
        my_ctf.SetDefocus(array_of_values[0], array_of_values[1], array_of_values[2]);
    }
    if ( comparison_object->fit_with_thickness_nodes ) {
        my_ctf.SetSampleThickness(array_of_values[3]);
    }
    if ( comparison_object->FindPhaseShift( ) && ! comparison_object->fit_with_thickness_nodes ) {
        if ( comparison_object->AstigmatismIsKnown( ) ) {
            my_ctf.SetAdditionalPhaseShift(array_of_values[1]);
        }
        else {
            my_ctf.SetAdditionalPhaseShift(array_of_values[3]);
        }
    }

    // Evaluate the function
    float score;
    if ( my_ctf.GetDefocus1( ) == 0.0f && my_ctf.GetDefocus2( ) == 0.0f && my_ctf.GetSphericalAberration( ) == 0.0f ) {
        // When defocus = 0.0 and cs = 0.0, CTF is constant and the scoring function breaks down
        score = 0.0;
    }
    else {
        if ( comparison_object->number_to_correlate ) {
            score = -comparison_object->img[0].QuickCorrelationWithCTF(my_ctf, comparison_object->number_to_correlate, comparison_object->norm_image, comparison_object->image_mean, comparison_object->addresses, comparison_object->spatial_frequency_squared, comparison_object->azimuths);
        }
        else {
            score = -comparison_object->img[0].GetCorrelationWithCTF(my_ctf);
        }
    }

    //MyDebugPrint("(CtffindObjectiveFunction) D1 = %6.2f pxl D2 = %6.2f pxl, PhaseShift = %6.3f rad, Ast = %5.2f rad, Low freq = %f 1/pxl, High freq = %f 1/pxl, Score = %g\n",my_ctf.GetDefocus1(),my_ctf.GetDefocus2(),my_ctf.GetAdditionalPhaseShift(), my_ctf.GetAstigmatismAzimuth(),my_ctf.GetLowestFrequencyForFitting(),my_ctf.GetHighestFrequencyForFitting(),score);
    MyDebugAssertFalse(std::isnan(score), "Score is NaN!");
    return score;
}

//#pragma GCC push_options
//#pragma GCC optimize ("O0")

// This is the function which will be minimised when dealing with 1D fitting
float CtffindCurveObjectiveFunction(void* scoring_parameters, float array_of_values[]) {
    CurveCTFComparison* comparison_object = reinterpret_cast<CurveCTFComparison*>(scoring_parameters);

    CTF my_ctf = comparison_object->ctf;
    if ( comparison_object->find_thickness_nodes ) {
        my_ctf.SetSampleThickness(array_of_values[0]);
    }
    else {
        my_ctf.SetDefocus(array_of_values[0], array_of_values[0], 0.0);
        if ( comparison_object->find_phase_shift ) {
            my_ctf.SetAdditionalPhaseShift(array_of_values[1]);
        }
    }

    // Compute the cross-correlation
    double      cross_product    = 0.0;
    double      norm_curve       = 0.0;
    double      norm_ctf         = 0.0;
    int         number_of_values = 0;
    int         bin_counter;
    float       current_spatial_frequency_squared;
    const float lowest_freq  = pow(my_ctf.GetLowestFrequencyForFitting( ), 2);
    const float highest_freq = pow(my_ctf.GetHighestFrequencyForFitting( ), 2);
    float       current_ctf_value;

    for ( bin_counter = 0; bin_counter < comparison_object->number_of_bins; bin_counter++ ) {
        current_spatial_frequency_squared = pow(float(bin_counter) * comparison_object->reciprocal_pixel_size, 2);
        if ( current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared < highest_freq ) {
            current_ctf_value = fabsf(my_ctf.Evaluate(current_spatial_frequency_squared, 0.0));
            if ( comparison_object->find_thickness_nodes ) {
                current_ctf_value = my_ctf.EvaluatePowerspectrumWithThickness(current_spatial_frequency_squared, 0.0);
            }
            MyDebugAssertTrue(current_ctf_value >= -1.0 && current_ctf_value <= 1.0, "Bad ctf value: %f", current_ctf_value);
            number_of_values++;
            cross_product += comparison_object->curve[bin_counter] * current_ctf_value;
            norm_curve += pow(comparison_object->curve[bin_counter], 2);
            norm_ctf += pow(current_ctf_value, 2);
        }
    }

    MyDebugAssertTrue(norm_ctf > 0.0, "Bad norm_ctf: %f\n", norm_ctf);
    MyDebugAssertTrue(norm_curve > 0.0, "Bad norm_curve: %f\n", norm_curve);

    //MyDebugPrint("(CtffindCurveObjectiveFunction) D1 = %6.2f , PhaseShift = %6.3f , Low freq = %f /pxl, High freq = %f/pxl Score = %g\n",array_of_values[0], array_of_values[1], my_ctf.GetLowestFrequencyForFitting(),my_ctf.GetHighestFrequencyForFitting(), - cross_product / sqrtf(norm_ctf * norm_curve));

    // Note, we are not properly normalizing the cross correlation coefficient. For our
    // purposes this should be OK, since the average power of the theoretical CTF should not
    // change much with defocus. At least I hope so.
    //	if (! std::isfinite(- cross_product / sqrtf(norm_ctf * norm_curve)))
    //	{
    //		wxPrintf("param 1, 2, v1, v2, v3 = %g %g %g %g %g\n", array_of_values[0], array_of_values[1], cross_product, norm_ctf, norm_curve);
    //		for ( bin_counter = 0 ; bin_counter < comparison_object->number_of_bins; bin_counter ++ )
    //		{
    //			wxPrintf("bin, val = %i, %g\n", bin_counter, comparison_object->curve[bin_counter]);
    //		}
    //		exit(0);
    //	}
    return -cross_product / sqrtf(norm_ctf * norm_curve);
}

/*
 * Go from an experimental radial average with decaying Thon rings to a function between 0.0 and 1.0 for every oscillation.
 * This is done by treating each interval between a zero and an extremum of the CTF separately, and for each of them,
 * sorting and ranking the values in the radial average.
 * Each value is then replaced by its rank, modified to make it looks like a |CTF| signal.
 * This makes sense as a preparation for evaluating the quality of fit of a CTF when we want to ignore the amplitude of the Thon
 * rings and just focus on whether the fit agrees in terms of the positions of the zeros and extrema.
 * Without this, a very good fit doesn't always have a great FRC for regions where the experimental radial average is decaying rapidly.
 */
void Renormalize1DSpectrumForFRC(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[]) {
    int                 bin_counter;
    int                 bin_of_previous_extremum;
    int                 bin_of_current_extremum;
    int                 i;
    int                 bin_of_zero;
    std::vector<float>  temp_vector;
    std::vector<size_t> temp_ranks;
    float               number_of_extrema_delta;
    //
    bin_of_previous_extremum = 0;
    bin_of_current_extremum  = 0;
    for ( bin_counter = 1; bin_counter < number_of_bins; bin_counter++ ) {
        number_of_extrema_delta = number_of_extrema_profile[bin_counter] - number_of_extrema_profile[bin_counter - 1];
        if ( number_of_extrema_delta >= 0.9 && number_of_extrema_delta <= 1.9 ) // if the CTF is oscillating too quickly, let's not do anything
        {
            // We just passed an extremum, at bin_counter-1
            // (number_of_extrema_profile keeps track of the count of extrema before the spatial frequency corresponding to this bin)
            bin_of_current_extremum = bin_counter - 1;
            if ( bin_of_previous_extremum > 0 ) {
                if ( (bin_of_current_extremum - bin_of_previous_extremum >= 4 && false) || (number_of_extrema_profile[bin_counter] < 7) ) {
                    // Loop from the previous extremum to the one we just found
                    // (there is a zero in between, let's find it)
                    // TODO: redefine the zero as the lowest point between the two extrema?
                    bin_of_zero = (bin_of_current_extremum - bin_of_previous_extremum) / 2 + bin_of_previous_extremum;
                    for ( i = bin_of_previous_extremum; i < bin_of_current_extremum; i++ ) {
                        if ( fit[i] < fit[i - 1] && fit[i] < fit[i + 1] )
                            bin_of_zero = i;
                    }
                    //wxPrintf("bin zero = %i\n",bin_of_zero);

                    // Now we can rank before the zero (the downslope)
                    //wxPrintf("downslope (including zero)...\n");
                    temp_vector.clear( );
                    for ( i = bin_of_previous_extremum; i <= bin_of_zero; i++ ) {
                        //wxPrintf("about to push back %f\n",float(average[i]));
                        temp_vector.push_back(float(average[i]));
                    }
                    temp_ranks = rankSort(temp_vector);
                    for ( i = bin_of_previous_extremum; i <= bin_of_zero; i++ ) {
                        //wxPrintf("replaced %f",average[i]);
                        average[i] = double(float(temp_ranks.at(i - bin_of_previous_extremum)) / float(temp_vector.size( ) - 1));
                        average[i] = sin(average[i] * PI * 0.5);
                        //wxPrintf(" with %f\n",average[i]);
                    }

                    // Now we can rank after the zero (upslope)
                    //wxPrintf("upslope...\n");
                    temp_vector.clear( );
                    for ( i = bin_of_zero + 1; i < bin_of_current_extremum; i++ ) {
                        //wxPrintf("about to push back %f\n",float(average[i]));
                        temp_vector.push_back(float(average[i]));
                    }
                    temp_ranks = rankSort(temp_vector);
                    for ( i = bin_of_zero + 1; i < bin_of_current_extremum; i++ ) {
                        //wxPrintf("[rank]bin %i: replaced %f",i,average[i]);
                        average[i] = double(float(temp_ranks.at(i - bin_of_zero - 1) + 1) / float(temp_vector.size( ) + 1));
                        average[i] = sin(average[i] * PI * 0.5);
                        //wxPrintf(" with %f\n",average[i]);
                    }
                    //MyDebugAssertTrue(abs(average[bin_of_zero]) < 0.01,"Zero bin (%i) isn't set to zero: %f\n", bin_of_zero, average[bin_of_zero]);
                }
                else {
                    // A simpler way, without ranking, is just normalize
                    // between 0.0 and 1.0 (this usually works quite well when Thon rings are on a flat background anyway)
                    float min_value = 1.0;
                    float max_value = 0.0;
                    for ( i = bin_of_previous_extremum; i < bin_of_current_extremum; i++ ) {
                        if ( average[i] > max_value )
                            max_value = average[i];
                        if ( average[i] < min_value )
                            min_value = average[i];
                    }
                    for ( i = bin_of_previous_extremum; i < bin_of_current_extremum; i++ ) {
                        //wxPrintf("bin %i: replaced %f",i,average[i]);
                        average[i] -= min_value;
                        if ( max_value - min_value > 0.0001 )
                            average[i] /= (max_value - min_value);
                        //wxPrintf(" with %f\n",average[i]);
                    }
                }
            }
            bin_of_previous_extremum = bin_of_current_extremum;
        }
        MyDebugAssertFalse(std::isnan(average[bin_counter]), "Average is NaN for bin %i\n", bin_counter);
    }
}

//
void ComputeFRCBetween1DSpectrumAndFit(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[], int first_fit_bin) {

    MyDebugAssertTrue(first_fit_bin >= 0, "Bad first fit bin on entry: %i", first_fit_bin);

    int    bin_counter;
    int    half_window_width[number_of_bins];
    int    bin_of_previous_extremum;
    int    i;
    int    first_bin, last_bin;
    double spectrum_mean, fit_mean;
    double spectrum_sigma, fit_sigma;
    double cross_product;
    float  number_of_bins_in_window;

    const int minimum_window_half_width = number_of_bins / 40;

    // First, work out the size of the window over which we'll compute the FRC value
    bin_of_previous_extremum = 0;
    for ( bin_counter = 1; bin_counter < number_of_bins; bin_counter++ ) {
        if ( number_of_extrema_profile[bin_counter] != number_of_extrema_profile[bin_counter - 1] ) {
            for ( i = bin_of_previous_extremum; i < bin_counter; i++ ) {
                half_window_width[i] = std::max(minimum_window_half_width, int((1.0 + 0.1 * float(number_of_extrema_profile[bin_counter])) * float(bin_counter - bin_of_previous_extremum + 1)));
                half_window_width[i] = std::min(half_window_width[i], number_of_bins / 2 - 1);
                MyDebugAssertTrue(half_window_width[i] < number_of_bins / 2, "Bad half window width: %i. Number of bins: %i\n", half_window_width[i], number_of_bins);
            }
            bin_of_previous_extremum = bin_counter;
        }
    }
    half_window_width[0] = half_window_width[1];
    for ( bin_counter = bin_of_previous_extremum; bin_counter < number_of_bins; bin_counter++ ) {
        half_window_width[bin_counter] = half_window_width[bin_of_previous_extremum - 1];
    }

    // Now compute the FRC for each bin
    for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {
        if ( bin_counter < first_fit_bin ) {
            frc[bin_counter]       = 1.0;
            frc_sigma[bin_counter] = 0.0;
        }
        else {
            spectrum_mean  = 0.0;
            fit_mean       = 0.0;
            spectrum_sigma = 0.0;
            fit_sigma      = 0.0;
            cross_product  = 0.0;
            // Work out the boundaries
            first_bin = bin_counter - half_window_width[bin_counter];
            last_bin  = bin_counter + half_window_width[bin_counter];
            if ( first_bin < first_fit_bin ) {
                first_bin = first_fit_bin;
                last_bin  = first_bin + 2 * half_window_width[bin_counter] + 1;
            }
            if ( last_bin >= number_of_bins ) {
                last_bin  = number_of_bins - 1;
                first_bin = last_bin - 2 * half_window_width[bin_counter] - 1;
            }
            MyDebugAssertTrue(first_bin >= 0 && first_bin < number_of_bins, "Bad first_bin: %i", first_bin);
            MyDebugAssertTrue(last_bin >= 0 && last_bin < number_of_bins, "Bad last_bin: %i", last_bin);
            // First pass
            for ( i = first_bin; i <= last_bin; i++ ) {
                spectrum_mean += average[i];
                fit_mean += fit[i];
            }
            number_of_bins_in_window = float(2 * half_window_width[bin_counter] + 1);
            //wxPrintf("bin %03i, number of extrema: %f, number of bins in window: %f , spectrum_sum = %f\n", bin_counter, number_of_extrema_profile[bin_counter], number_of_bins_in_window,spectrum_mean);
            spectrum_mean /= number_of_bins_in_window;
            fit_mean /= number_of_bins_in_window;
            // Second pass
            for ( i = first_bin; i <= last_bin; i++ ) {
                cross_product += (average[i] - spectrum_mean) * (fit[i] - fit_mean);
                spectrum_sigma += pow(average[i] - spectrum_mean, 2);
                fit_sigma += pow(fit[i] - fit_mean, 2);
            }
            //MyDebugAssertTrue(spectrum_sigma > 0.0 && spectrum_sigma < 10000.0, "Bad spectrum_sigma: %f\n", spectrum_sigma);
            //MyDebugAssertTrue(fit_sigma > 0.0 && fit_sigma < 10000.0, "Bad fit sigma: %f\n", fit_sigma);
            if ( spectrum_sigma > 0.0 && fit_sigma > 0.0 ) {
                frc[bin_counter] = cross_product / (sqrtf(spectrum_sigma / number_of_bins_in_window) * sqrtf(fit_sigma / number_of_bins_in_window)) / number_of_bins_in_window;
            }
            else {
                frc[bin_counter] = 0.0;
            }
            frc_sigma[bin_counter] = 2.0 / sqrtf(number_of_bins_in_window);
        }
        //wxPrintf("First fit bin: %i\n", first_fit_bin);
        MyDebugAssertTrue(frc[bin_counter] > -1.01 && frc[bin_counter] < 1.01, "Bad FRC value: %f", frc[bin_counter]);
    }
}

//
void OverlayCTF(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* epa_pre_max, Curve* epa_post_max, bool fit_nodes) {
    MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");

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

    //spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_entry.mrc",1);

    //
    address = 0;
    for ( j = 0; j < spectrum->logical_y_dimension; j++ ) {
        j_logi    = float(j - spectrum->physical_address_of_box_center_y) * spectrum->fourier_voxel_size_y;
        j_logi_sq = powf(j_logi, 2);
        for ( i = 0; i < spectrum->logical_x_dimension; i++ ) {
            i_logi    = float(i - spectrum->physical_address_of_box_center_x) * spectrum->fourier_voxel_size_x;
            i_logi_sq = powf(i_logi, 2);
            //
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
            current_azimuth                   = atan2(j_logi, i_logi);
            current_defocus                   = ctf->DefocusGivenAzimuth(current_azimuth);
            current_phase_aberration          = ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_spatial_frequency_squared, current_defocus);
            //
            sq_sf_of_phase_aberration_maximum = ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(current_defocus);

            if ( j < spectrum->physical_address_of_box_center_y && i >= spectrum->physical_address_of_box_center_x ) {
                // Experimental 1D average
#ifdef use_epa_rather_than_zero_counting
                if ( current_spatial_frequency_squared <= sq_sf_of_phase_aberration_maximum ) {
                    spectrum->real_values[address] = epa_pre_max->ReturnLinearInterpolationFromX(current_phase_aberration);
                }
                else {
                    spectrum->real_values[address] = epa_post_max->ReturnLinearInterpolationFromX(current_phase_aberration);
                }
#else
                // Work out which bin in the astig rot average this pixel corresponds to
                chosen_bin                     = ReturnSpectrumBinNumber(number_of_bins_in_1d_spectra, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                spectrum->real_values[address] = rotational_average_astig[chosen_bin];
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
                    values_in_rings.AddSampleValue(spectrum->real_values[address]);
                values_in_fitting_range.AddSampleValue(spectrum->real_values[address]);
                //if (current_azimuth <= ctf->GetAstigmatismAzimuth()  && current_azimuth >= ctf->GetAstigmatismAzimuth() - 3.1415*0.5) spectrum->real_values[address] = current_ctf_value;
                if ( j < spectrum->physical_address_of_box_center_y && i < spectrum->physical_address_of_box_center_x )
                    spectrum->real_values[address] = current_ctf_value;
            }
            if ( current_spatial_frequency_squared <= lowest_freq ) {
                spectrum->real_values[address] = 0.0;
            }
            //
            address++;
        }
        address += spectrum->padding_jump_value;
    }

    //spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_1.mrc",1);

    /*

	// We will renormalize the experimental part of the diagnostic image
	target_sigma = sqrtf(values_in_rings.GetSampleVariance()) ;


	if (target_sigma > 0.0)
	{
		address = 0;
		for (j=0;j < spectrum->logical_y_dimension;j++)
		{
			j_logi = float(j-spectrum->physical_address_of_box_center_y) * spectrum->fourier_voxel_size_y;
			j_logi_sq = powf(j_logi,2);
			for (i=0 ;i < spectrum->logical_x_dimension; i++)
			{
				i_logi = float(i-spectrum->physical_address_of_box_center_x) * spectrum->fourier_voxel_size_x;
				i_logi_sq = powf(i_logi,2);
				//
				current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
				// Normalize the experimental part of the diagnostic image
				if (i > spectrum->physical_address_of_box_center_x || j > spectrum->physical_address_of_box_center_y)
				{
					spectrum->real_values[address] /= target_sigma;
				}
				else
				{
					// Normalize the outside of the theoretical part of the diagnostic image
					if (current_spatial_frequency_squared > highest_freq) spectrum->real_values[address] /= target_sigma;
				}

				address++;
			}
			address += spectrum->padding_jump_value;
		}
	}
	*/

    //spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_final.mrc",1);
}

// Rescale the spectrum and its 1D rotational avereage so that the peaks and troughs are at 0.0 and 1.0. The location of peaks and troughs are worked out
// by parsing the suppilied 1D average_fit array
void RescaleSpectrumAndRotationalAverage(Image* spectrum, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit) {
    MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");
    MyDebugAssertTrue(number_of_bins > 1, "Bad number of bins: %i\n", number_of_bins);

    //
    const bool spectrum_is_blank               = spectrum->IsConstant( );
    const int  rescale_based_on_maximum_number = 2; // This peak will be used as a renormalization.
    const int  sg_width                        = 7;
    const int  sg_order                        = 2;
    const bool rescale_peaks                   = false; // if this is false, only the background will be subtracted, the Thon rings "heights" will be unaffected
    float      background[number_of_bins];
    float      peak[number_of_bins];
    int        bin_counter;
    bool       at_a_maximum, at_a_minimum, maximum_at_previous_bin, minimum_at_previous_bin;
    int        location_of_previous_maximum, location_of_previous_minimum;
    int        current_maximum_number = 0;
    int        normalisation_bin_number;
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
        if ( maxima_curve->number_of_points > sg_width )
            maxima_curve->FitSavitzkyGolayToData(sg_width, sg_order);
        if ( minima_curve->number_of_points > sg_width )
            minima_curve->FitSavitzkyGolayToData(sg_width, sg_order);

        // Replace the background and peak envelopes with the smooth min/max curves
        for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {
            if ( minima_curve->number_of_points > sg_width )
                background[bin_counter] = minima_curve->ReturnSavitzkyGolayInterpolationFromX(spatial_frequency[bin_counter]);
            if ( maxima_curve->number_of_points > sg_width )
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
            for ( j = 0; j < spectrum->logical_y_dimension; j++ ) {
                for ( i = 0; i < spectrum->logical_x_dimension; i++ ) {
                    chosen_bin = ReturnSpectrumBinNumber(number_of_bins, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                    if ( chosen_bin >= 0 ) {
                        if ( chosen_bin <= last_bin_to_rescale ) {
                            spectrum->real_values[address] -= background[chosen_bin]; // This alone makes the spectrum look very nice already
                            if ( rescale_peaks )
                                spectrum->real_values[address] /= std::min(1.0f, std::max(min_scale_factor, peak[chosen_bin] - background[chosen_bin])) / rescale_peaks_to; // This is supposed to help "boost" weak Thon rings
                        }
                        else {
                            spectrum->real_values[address] -= background[last_bin_to_rescale];
                            if ( rescale_peaks )
                                spectrum->real_values[address] /= std::min(1.0f, std::max(min_scale_factor, peak[last_bin_to_rescale] - background[last_bin_to_rescale])) / rescale_peaks_to;
                        }
                    }
                    else {
                        //TODO: return false
                    }
                    //
                    address++;
                }
                address += spectrum->padding_jump_value;
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
 * Compute average value in power spectrum as a function of wave function aberration. This allows for averaging even when
 * there is significant astigmatism.
 * This should be nicer than counting zeros and looking for nearest CTF value as described in the original ctffind4 manuscript.
 * Inspired by gctf and others, but I think more robust because it takes into account that the aberration decreases again at
 * very high spatial frequencies, when Cs takes over from defocus.
 */
void ComputeEquiPhaseAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Curve* epa_pre_max, Curve* epa_post_max) {
    MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");

    const bool spectrum_is_blank = spectrum->IsConstant( );

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
        float maximum_sq_freq_in_spectrum          = powf(spectrum->fourier_voxel_size_x * spectrum->logical_lower_bound_complex_x, 2) + powf(spectrum->fourier_voxel_size_y * spectrum->logical_lower_bound_complex_y, 2);
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
        int number_of_points_pre_max  = std::max(2, myroundint(spectrum->ReturnMaximumDiagonalRadius( ) * curve_oversampling_factor * maximum_aberration_in_ctf / maximum_abs_aberration_in_spectrum));
        int number_of_points_post_max = std::max(2, myroundint(spectrum->ReturnMaximumDiagonalRadius( ) * curve_oversampling_factor));

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
        for ( j = 0; j < spectrum->logical_y_dimension; j++ ) {
            j_logi    = float(j - spectrum->physical_address_of_box_center_y) * spectrum->fourier_voxel_size_y;
            j_logi_sq = powf(j_logi, 2);
            for ( i = 0; i < spectrum->logical_x_dimension; i++ ) {
                i_logi    = float(i - spectrum->physical_address_of_box_center_x) * spectrum->fourier_voxel_size_x;
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
                    epa_pre_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, spectrum->real_values[address], curve_x_is_linear);
                    count_pre_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, 1.0, curve_x_is_linear);
                }
                else {
                    /*
					 * We are after the maximum phase aberration (i.e. the Cs term has taken over, phase aberration is decreasing as a function of sf)
					 */
                    // Add to post-max
                    epa_post_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, spectrum->real_values[address], curve_x_is_linear);
                    count_post_max->AddValueAtXUsingLinearInterpolation(current_phase_aberration, 1.0, curve_x_is_linear);
                }
                //
                address++;
            }
            address += spectrum->padding_jump_value;
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

float ReturnAzimuthToUseFor1DPlots(CTF* ctf) {
    const float min_angular_distances_from_axes_radians = 10.0 / 180.0 * PIf;
    float       azimuth_of_mid_defocus;
    float       angular_distance_from_axes;

    // We choose the azimuth to be mid way between the two defoci of the astigmatic CTF
    azimuth_of_mid_defocus = ctf->GetAstigmatismAzimuth( ) + PIf * 0.25f;
    // We don't want the azimuth too close to the axes, which may have been blanked by the central-cross-artefact-suppression-system (tm)
    angular_distance_from_axes = fmod(azimuth_of_mid_defocus, PIf * 0.5f);
    if ( fabs(angular_distance_from_axes) < min_angular_distances_from_axes_radians ) {
        if ( angular_distance_from_axes > 0.0f ) {
            azimuth_of_mid_defocus = min_angular_distances_from_axes_radians;
        }
        else {
            azimuth_of_mid_defocus = -min_angular_distances_from_axes_radians;
        }
    }
    if ( fabs(angular_distance_from_axes) > 0.5f * PIf - min_angular_distances_from_axes_radians ) {
        if ( angular_distance_from_axes > 0.0 ) {
            azimuth_of_mid_defocus = PIf * 0.5f - min_angular_distances_from_axes_radians;
        }
        else {
            azimuth_of_mid_defocus = -PIf * 0.5f + min_angular_distances_from_axes_radians;
        }
    }

    return azimuth_of_mid_defocus;
}

//
void ComputeRotationalAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_rank[], float number_of_extrema_profile[], float ctf_values_profile[]) {
    MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");
    MyDebugAssertTrue(number_of_extrema->is_in_memory, "Number of extrema image not allocated");
    MyDebugAssertTrue(ctf_values->is_in_memory, "CTF values image not allocated");
    MyDebugAssertTrue(spectrum->HasSameDimensionsAs(number_of_extrema), "Spectrum and number of extrema images do not have same dimensions");
    MyDebugAssertTrue(spectrum->HasSameDimensionsAs(ctf_values), "Spectrum and CTF values images do not have same dimensions");
    //
    const bool spectrum_is_blank = spectrum->IsConstant( );
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
        azimuth_of_mid_defocus = ReturnAzimuthToUseFor1DPlots(ctf);

        // Now that we've chosen an azimuth, we can compute the CTF for each bin of our 1D profile
        for ( counter = 0; counter < number_of_bins; counter++ ) {
            current_spatial_frequency_squared  = powf(float(counter) * spectrum->fourier_voxel_size_y, 2);
            spatial_frequency[counter]         = sqrt(current_spatial_frequency_squared);
            ctf_values_profile[counter]        = ctf->Evaluate(current_spatial_frequency_squared, azimuth_of_mid_defocus);
            number_of_extrema_profile[counter] = ctf->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared, azimuth_of_mid_defocus);
            //wxPrintf("bin %i: phase shift= %f, number of extrema = %f\n",counter,ctf->PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(current_spatial_frequency_squared,azimuth_of_mid_defocus),number_of_extrema_profile[counter]);
        }

        // Now we can loop over the spectrum again and decide to which bin to add each component
        address = 0;
        for ( j = 0; j < spectrum->logical_y_dimension; j++ ) {
            for ( i = 0; i < spectrum->logical_x_dimension; i++ ) {
                ctf_diff_from_current_bin = std::numeric_limits<float>::max( );
                chosen_bin                = ReturnSpectrumBinNumber(number_of_bins, number_of_extrema_profile, number_of_extrema, address, ctf_values, ctf_values_profile);
                if ( chosen_bin >= 0 ) {
                    average[chosen_bin] += spectrum->real_values[address];
                    number_of_values[chosen_bin]++;
                }
                else {
                    //TODO: return false
                }
                //
                address++;
            }
            address += spectrum->padding_jump_value;
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

/*
integer function ComputePowerSpectrumBinNumber(number_of_bins,number_of_extrema_profile,number_of_extrema, &
                                                i,j,ctf_values,ctf_values_profile) result(chosen_bin)
    integer,        intent(in)  ::  number_of_bins
    real,           intent(in)  ::  number_of_extrema_profile(:)
    type(Image),    intent(in)  ::  number_of_extrema
    integer,        intent(in)  ::  i,j                         !<  Physical memory address
    type(Image),    intent(in)  ::  ctf_values
    real,           intent(in)  ::  ctf_values_profile(:)
    ! private variables
    integer     ::  current_bin
    real        ::  diff_number_of_extrema, diff_number_of_extrema_previous, diff_number_of_extrema_next
    real        ::  ctf_diff_from_current_bin
    real        ::  ctf_diff_from_current_bin_old
    ! Let's find the bin which has the same number of preceding extrema and the most similar ctf value
    ctf_diff_from_current_bin = huge(1.0e0)
    chosen_bin = 0
    do current_bin=1,number_of_bins
        diff_number_of_extrema  = abs(number_of_extrema%real_values(i,j,1) - number_of_extrema_profile(current_bin))
        if (current_bin .gt. 1) then
            diff_number_of_extrema_previous = abs(number_of_extrema%real_values(i,j,1) &
                                                - number_of_extrema_profile(current_bin-1))
        else
            diff_number_of_extrema_previous = huge(1.0e0)
        endif
        if (current_bin .lt. number_of_bins) then
            diff_number_of_extrema_next     = abs(number_of_extrema%real_values(i,j,1) &
                                                - number_of_extrema_profile(current_bin+1))
        else
            diff_number_of_extrema_next = huge(1.0e0)
        endif
        if (number_of_extrema%real_values(i,j,1) .gt. number_of_extrema_profile(number_of_bins)) then
            chosen_bin = number_of_bins
        else
            if (        diff_number_of_extrema .le. 0.01 &
                .or.    (     diff_number_of_extrema .lt. diff_number_of_extrema_previous &
                        .and. diff_number_of_extrema .le. diff_number_of_extrema_next &
                        .and. number_of_extrema_profile(max(current_bin-1,1)) &
                            .ne. number_of_extrema_profile(min(current_bin+1,number_of_bins))) &
                ) then
                ! We're nearly there
                ! Let's look for the position of the nearest CTF value
                ctf_diff_from_current_bin_old = ctf_diff_from_current_bin
                ctf_diff_from_current_bin = abs(ctf_values%real_values(i,j,1) - ctf_values_profile(current_bin))
                if (ctf_diff_from_current_bin .lt. ctf_diff_from_current_bin_old) then
                    chosen_bin = current_bin
                endif
            endif
        endif
    enddo
    if (chosen_bin .eq. 0) then
        print *, number_of_extrema_profile
        print *, i, j, number_of_extrema%real_values(i,j,1), ctf_values%real_values(i,j,1)
        print *, diff_number_of_extrema, diff_number_of_extrema_previous, diff_number_of_extrema_next
        call this_program%TerminateWithFatalError('ComputeRotationalAverageOfPowerSpectrum','Could not find bin')
    endif
end function ComputePowerSpectrumBinNumber
*/

// Compute an image where each pixel stores the number of preceding CTF extrema. This is described as image "E" in Rohou & Grigorieff 2015 (see Fig 3)
void ComputeImagesWithNumberOfExtremaAndCTFValues(CTF* ctf, Image* number_of_extrema, Image* ctf_values) {
    MyDebugAssertTrue(number_of_extrema->is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(ctf_values->is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(ctf_values->HasSameDimensionsAs(number_of_extrema), "Images do not have same dimensions");

    int   i, j;
    float i_logi, i_logi_sq;
    float j_logi, j_logi_sq;
    float current_spatial_frequency_squared;
    float current_azimuth;
    long  address;

    address = 0;
    for ( j = 0; j < number_of_extrema->logical_y_dimension; j++ ) {
        j_logi    = float(j - number_of_extrema->physical_address_of_box_center_y) * number_of_extrema->fourier_voxel_size_y;
        j_logi_sq = pow(j_logi, 2);
        for ( i = 0; i < number_of_extrema->logical_x_dimension; i++ ) {
            i_logi    = float(i - number_of_extrema->physical_address_of_box_center_x) * number_of_extrema->fourier_voxel_size_x;
            i_logi_sq = pow(i_logi, 2);
            // Where are we?
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
            if ( current_spatial_frequency_squared > 0.0 ) {
                current_azimuth = atan2(j_logi, i_logi);
            }
            else {
                current_azimuth = 0.0;
            }
            //
            ctf_values->real_values[address]        = ctf->Evaluate(current_spatial_frequency_squared, current_azimuth);
            number_of_extrema->real_values[address] = ctf->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared, current_azimuth);
            //
            address++;
        }
        address += number_of_extrema->padding_jump_value;
    }

    number_of_extrema->is_in_real_space = true;
    ctf_values->is_in_real_space        = true;
}

// Align rotationally a (stack) of image(s) against another image. Return the rotation angle that gives the best normalised cross-correlation.
float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* self, Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius) {
    MyDebugAssertTrue(self[0].is_in_memory, "Memory not allocated");
    MyDebugAssertTrue(self[0].is_in_real_space, "Not in real space");
    MyDebugAssertTrue(self[0].logical_z_dimension == 1, "Meant for images, not volumes");
    MyDebugAssertTrue(other_image[0].is_in_memory, "Memory not allocated - other_image");
    MyDebugAssertTrue(other_image[0].is_in_real_space, "Not in real space - other_image");
    MyDebugAssertTrue(other_image[0].logical_z_dimension == 1, "Meant for images, not volumes - other_image");
    MyDebugAssertTrue(self[0].HasSameDimensionsAs(&other_image[0]), "Images and reference images do not have same dimensions.");

    // Local variables
    const float           minimum_radius_sq           = pow(minimum_radius, 2);
    const float           maximum_radius_sq           = pow(maximum_radius, 2);
    const float           inverse_logical_x_dimension = 1.0 / float(self[0].logical_x_dimension);
    const float           inverse_logical_y_dimension = 1.0 / float(self[0].logical_y_dimension);
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
                        ii_phys = i_logi * cos(current_rotation_rad) - j_logi * sin(current_rotation_rad) + self[0].physical_address_of_box_center_x;
                        jj_phys = i_logi * sin(current_rotation_rad) + j_logi * cos(current_rotation_rad) + self[0].physical_address_of_box_center_y;
                        //
                        if ( int(ii_phys) > 0 && int(ii_phys) + 1 < self[0].logical_x_dimension && int(jj_phys) > 0 && int(jj_phys) + 1 < self[0].logical_y_dimension ) // potential optimization: we have to compute the floor and ceiling in the interpolation routine. Is it not worth doing the bounds checking in the interpolation routine somehow?
                        {
                            self[0].GetRealValueByLinearInterpolationNoBoundsCheckImage(ii_phys, jj_phys, current_interpolated_value);
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
