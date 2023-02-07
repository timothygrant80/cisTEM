#include "../../core/core_headers.h"
#include "./ctffind.h"

// Helper function to write the current spectrum and fit into the debug json object
void write_fit_result_JSON_debug(wxJSONValue& debug_json_output, char* name, int number_of_bins_in_1d_spectra, double* rotational_average_astig, double* rotational_average_astig_fit, double* spatial_frequency = nullptr) {
    if ( spatial_frequency != nullptr ) {
        debug_json_output["spatial_frequency"] = wxJSONValue(wxJSONTYPE_ARRAY);
        for ( int counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
            debug_json_output["spatial_frequency"].Append(spatial_frequency[counter]);
        }
    }

    debug_json_output[name]                             = wxJSONValue(wxJSONTYPE_OBJECT);
    debug_json_output[name]["rotational_average_astig"] = wxJSONValue(wxJSONTYPE_ARRAY);
    for ( int counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
        debug_json_output[name]["rotational_average_astig"].Append(rotational_average_astig[counter]);
    }
    debug_json_output[name]["rotational_average_astig_fit"] = wxJSONValue(wxJSONTYPE_ARRAY);
    for ( int counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
        debug_json_output[name]["rotational_average_astig_fit"].Append(rotational_average_astig_fit[counter]);
    }
}

// This is a copy of the objective function of Ctffind with some adaptions for the node fitting
float CtffindNodesObjectiveFunction(void* scoring_parameters, float array_of_values[]) {
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
    if ( array_of_values[3] < 500.0f ) {
        array_of_values[3] = 500.0f;
    }
    if ( array_of_values[3] > 50000.0f ) {
        array_of_values[3] = 50000.0f;
    }

    my_ctf.SetSampleThickness(array_of_values[3]);

    // Evaluate the function
    float score;
    if ( my_ctf.GetDefocus1( ) == 0.0f && my_ctf.GetDefocus2( ) == 0.0f && my_ctf.GetSphericalAberration( ) == 0.0f ) {
        // When defocus = 0.0 and cs = 0.0, CTF is constant and the scoring function breaks down
        score = 0.0;
    }
    else {
        float weights[comparison_object->number_to_correlate];
        int   adress = 0;
        float squared_spatial_frequency;
        for ( int i = 0; i < comparison_object->number_to_correlate; i++ ) {
            weights[i] = 1.0f;
            if ( comparison_object->fit_nodes_downweight_nodes )
                weights[i] = fabs(my_ctf.IntegratedDefocusModulationRoundedSquare(comparison_object->spatial_frequency_squared[i])) > 0.99 ? 1.0f : 0.0f;
        }

        if ( comparison_object->number_to_correlate ) {
            score = -comparison_object->img[0].QuickCorrelationWithCTFThickness(my_ctf, comparison_object->number_to_correlate, comparison_object->norm_image, comparison_object->image_mean, comparison_object->addresses, comparison_object->spatial_frequency_squared, comparison_object->azimuths, weights, comparison_object->fit_nodes_rounded_square);
        }
        else {
            score = -comparison_object->img[0].GetCorrelationWithCTF(my_ctf);
        }
    }

    MyDebugPrint("(CtffindObjectiveFunction) D1 = %6.2f pxl D2 = %6.2f pxl, PhaseShift = %6.3f rad, Ast = %5.2f rad, Low freq = %f 1/pxl, High freq = %f 1/pxl, Thickness = %f pxl,  Score = %g\n", my_ctf.GetDefocus1( ), my_ctf.GetDefocus2( ), my_ctf.GetAdditionalPhaseShift( ), my_ctf.GetAstigmatismAzimuth( ), my_ctf.GetLowestFrequencyForFitting( ), my_ctf.GetHighestFrequencyForFitting( ), my_ctf.GetSampleThickness( ), score);
    MyDebugAssertFalse(std::isnan(score), "Score is NaN!");
    return score;
}

void do_1D_bruteforce(CTFNodeFitInput* input, wxJSONValue& debug_json_output) {
    int   counter;
    float current_sq_sf;
    float azimuth_for_1d_plots = ReturnAzimuthToUseFor1DPlots(input->current_ctf);
    CTF   my_ctf;
    my_ctf.CopyFrom(*(input->current_ctf));
    input->comparison_object_1D->curve = new float[input->number_of_bins_in_1d_spectra];
    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        input->comparison_object_1D->curve[counter] = input->rotational_average_astig[counter];
    }
    input->comparison_object_1D->find_phase_shift         = false;
    input->comparison_object_1D->find_thickness_nodes     = true;
    input->comparison_object_1D->ctf                      = my_ctf;
    input->comparison_object_1D->fit_nodes_rounded_square = input->use_rounded_square;

    // We can now look for the defocus value
    float bf_halfrange[2] = {1750 / input->pixel_size_for_fitting, 1000 / input->pixel_size_for_fitting};

    float bf_midpoint[2] = {500 / input->pixel_size_for_fitting + bf_halfrange[0], my_ctf.DefocusGivenAzimuth(azimuth_for_1d_plots)};

    float bf_stepsize[2] = {10 / input->pixel_size_for_fitting, 10 / input->pixel_size_for_fitting};

    int number_of_search_dimensions = 2;

    // Actually run the BF search
    BruteForceSearch* brute_force_search = new BruteForceSearch( );
    brute_force_search->Init(&CtffindCurveObjectiveFunction, input->comparison_object_1D, number_of_search_dimensions, bf_midpoint, bf_halfrange, bf_stepsize, false, false, 1);
    float* all_values = nullptr;
    float* all_scores = nullptr;
    int    num_values;
    int    num_scores;
    brute_force_search->Run(&all_values, &all_scores, &num_values, &num_scores);

    debug_json_output["1D_brute_force_search"]               = wxJSONValue(wxJSONTYPE_OBJECT);
    debug_json_output["1D_brute_force_search"]["all_values"] = wxJSONValue(wxJSONTYPE_ARRAY);
    for ( int counter = 0; counter < num_values; counter++ ) {
        debug_json_output["1D_brute_force_search"]["all_values"].Append(all_values[counter] * input->pixel_size_for_fitting);
    }
    debug_json_output["1D_brute_force_search"]["all_scores"] = wxJSONValue(wxJSONTYPE_ARRAY);
    for ( int counter = 0; counter < num_scores; counter++ ) {
        debug_json_output["1D_brute_force_search"]["all_scores"].Append(all_scores[counter]);
    }
    //delete[] all_values;
    //delete[] all_scores;

    input->current_ctf->SetSampleThickness(brute_force_search->GetBestValue(0));
    input->current_ctf->SetDefocus(brute_force_search->GetBestValue(1), brute_force_search->GetBestValue(1), input->current_ctf->GetAstigmatismAzimuth( ));
    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        current_sq_sf                                = powf(input->spatial_frequency[counter], 2);
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots, input->use_rounded_square);
    }
    if ( input->debug ) {
        write_fit_result_JSON_debug(debug_json_output, "after_1D_brute_force", input->number_of_bins_in_1d_spectra, input->rotational_average_astig, input->rotational_average_astig_fit, nullptr);
        debug_json_output["thickness_estimates"]["after_1D_brute_force"] = brute_force_search->GetBestValue(0) * input->pixel_size_for_fitting;
    }
    delete brute_force_search;
    delete input->comparison_object_1D->curve;
}

void do_2D_refinement(CTFNodeFitInput* input, wxJSONValue& debug_json_output) {
    int   counter;
    float current_sq_sf;
    float azimuth_for_1d_plots = ReturnAzimuthToUseFor1DPlots(input->current_ctf);

    int   number_of_search_dimensions = 4;
    float cg_accuracy[4]              = {100.0, 100.0, 0.05, 10.0}; //TODO: try defocus_search_step  / pix_size_for_fitting / 10.0
    float cg_starting_point[4]        = {input->current_ctf->GetDefocus1( ), input->current_ctf->GetDefocus2( ), input->current_ctf->GetAstigmatismAzimuth( ), input->current_ctf->GetSampleThickness( )};

    input->comparison_object_2D->SetCTF(*(input->current_ctf));
    input->comparison_object_2D->SetFitWithThicknessNodes(true);
    input->comparison_object_2D->SetupQuickCorrelation( );
    input->comparison_object_2D->fit_nodes_downweight_nodes = input->downweight_nodes;
    input->comparison_object_2D->fit_nodes_rounded_square   = input->use_rounded_square;
    MyDebugPrint("fit_nodes_rounded_square = %i", input->comparison_object_2D->fit_nodes_rounded_square);
    ConjugateGradient* conjugate_gradient_minimizer = new ConjugateGradient( );
    conjugate_gradient_minimizer->Init(&CtffindNodesObjectiveFunction, input->comparison_object_2D, number_of_search_dimensions, cg_starting_point, cg_accuracy);

    conjugate_gradient_minimizer->Run( );

    // Remember the results of the refinement
    for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
        cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
    }

    input->current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[1], cg_starting_point[2]);
    input->current_ctf->SetSampleThickness(cg_starting_point[3]);
    // Replace rotational_average_astig_fit with thickness model

    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        current_sq_sf                                = powf(input->spatial_frequency[counter], 2);
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots, input->use_rounded_square);
    }
    if ( input->debug ) {
        write_fit_result_JSON_debug(debug_json_output, "after_2D_refine", input->number_of_bins_in_1d_spectra, input->rotational_average_astig, input->rotational_average_astig_fit, nullptr);
        debug_json_output["thickness_estimates"]["after_2D_refine"] = cg_starting_point[3] * input->pixel_size_for_fitting;
    }
    delete conjugate_gradient_minimizer;
}

void recalculate_1D_spectra(CTFNodeFitInput* input, double* rotational_average_astig_renormalized, float* number_of_extrema_profile, wxJSONValue debug_json_output) {
    ComputeEquiPhaseAverageOfPowerSpectrum(input->average_spectrum, input->current_ctf, &(input->equiphase_average_pre_max), &(input->equiphase_average_post_max));
    // Replace the old curve with EPA values
    //double* rotational_average_astig_renormalized = new double[input->number_of_bins_in_1d_spectra];
    {
        int   counter;
        float current_sq_sf;
        float azimuth_for_1d_plots         = ReturnAzimuthToUseFor1DPlots(input->current_ctf);
        float defocus_for_1d_plots         = input->current_ctf->DefocusGivenAzimuth(azimuth_for_1d_plots);
        float sq_sf_of_phase_shift_maximum = input->current_ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(defocus_for_1d_plots);
        //float* number_of_extrema_profile    = new float[input->number_of_bins_in_1d_spectra];
        int number_of_extrema = 0;
        int prev_slope        = 0;
        int slope;
        for ( counter = 1; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            current_sq_sf = powf(input->spatial_frequency[counter], 2);
            if ( current_sq_sf <= sq_sf_of_phase_shift_maximum ) {
                input->rotational_average_astig[counter] = input->equiphase_average_pre_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
            }
            else {
                input->rotational_average_astig[counter] = input->equiphase_average_post_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
            }
            rotational_average_astig_renormalized[counter] = input->rotational_average_astig[counter];
            slope                                          = (input->rotational_average_astig_fit[counter] > input->rotational_average_astig_fit[counter - 1]) ? 1 : -1;
            // Add extrema when rising slope changes to falling slope
            if ( slope - prev_slope == -2 ) {
                number_of_extrema++;
            }
            prev_slope                         = slope;
            number_of_extrema_profile[counter] = number_of_extrema;
        }
        number_of_extrema_profile[0] = 0;
        Renormalize1DSpectrumForFRC(input->number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, input->rotational_average_astig_fit, number_of_extrema_profile);
        // I should sqrt rotational_average_astig_renormalized to amke it look
        // like the abs and not the power of 2
        if ( input->use_rounded_square ) {
            for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
                input->rotational_average_astig_fit[counter] = sqrt(input->rotational_average_astig_fit[counter]);
            }
        }
    }
}

void ComputeFRCBetween1DSpectrumAndFitNodes(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[], int first_fit_bin) {

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
    // Dirty hack. We just increase the window where FRC gets computed to make
    // it more robust in the nodes where noise domintaes
    for ( bin_counter = 0; bin_counter < number_of_bins; bin_counter++ ) {
        half_window_width[bin_counter] *= 1.5;
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

int calculate_new_frc(CTFNodeFitInput* input, double* rotational_average_astig_renormalized, float* number_of_extrema_profile, wxJSONValue debug_json_output) {
    int          last_bin_with_good_fit;
    static float low_threshold              = 0.1;
    static float frc_significance_threshold = 0.5; // In analogy to the usual criterion when comparing experimental results to the atomic model
    static float high_threshold             = 0.66;
    bool         at_last_bin_with_good_fit;
    int          number_of_bins_above_low_threshold          = 0;
    int          number_of_bins_above_significance_threshold = 0;
    int          number_of_bins_above_high_threshold         = 0;
    int          first_bin_to_check                          = 0.1 * input->number_of_bins_in_1d_spectra;
    int          counter;
    float        sq_spatial_frequency;

    MyDebugPrint("Startign FRC calc");

    int first_fit_bin = 0;
    for ( int bin_counter = input->number_of_bins_in_1d_spectra - 1; bin_counter >= 0; bin_counter-- ) {
        if ( input->spatial_frequency[bin_counter] >= input->current_ctf->GetLowestFrequencyForFitting( ) )
            first_fit_bin = bin_counter;
    }
    ComputeFRCBetween1DSpectrumAndFitNodes(input->number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, input->rotational_average_astig_fit, number_of_extrema_profile, input->fit_frc, input->fit_frc_sigma, first_fit_bin);
    MyDebugAssertTrue(first_bin_to_check >= 0 && first_bin_to_check < input->number_of_bins_in_1d_spectra, "Bad first bin to check\n");
    //wxPrintf("Will only check from bin %i of %i onwards\n", first_bin_to_check, number_of_bins_in_1d_spectra);
    last_bin_with_good_fit = -1;
    // Set FRC to 1.0 in the nodes where there is not modulation
    double prev_value;
    double after_value;
    bool   in_node     = false;
    bool   out_of_node = false;
    if ( false ) {
        for ( counter = first_bin_to_check; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            sq_spatial_frequency = powf(input->spatial_frequency[counter], 2);
            if ( fabsf(input->current_ctf->IntegratedDefocusModulation(sq_spatial_frequency)) < 0.1 ) {
                prev_value = input->fit_frc[counter - 1];
                in_node    = true;
                // Go ahead until we are out of the node to figure out the value after the node
                out_of_node     = false;
                int start_count = counter;
                int end_count;
                for ( int node_counter = counter; node_counter < input->number_of_bins_in_1d_spectra; node_counter++ ) {
                    sq_spatial_frequency = powf(input->spatial_frequency[node_counter], 2);
                    if ( fabsf(input->current_ctf->IntegratedDefocusModulation(sq_spatial_frequency)) >= 0.1 ) {
                        after_value = input->fit_frc[node_counter];
                        out_of_node = true;
                        end_count   = counter - start_count;
                        break;
                    }
                }
                // Now interpolate between the values before and after the node or
                // set to 0 if we are at the end of the spectrum

                for ( ; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
                    sq_spatial_frequency = powf(input->spatial_frequency[counter], 2);
                    if ( ! out_of_node ) {
                        input->fit_frc[counter] = 0.0;
                    }
                    else {
                        input->fit_frc[counter] = prev_value + (after_value - prev_value) * (counter - start_count) / (end_count);
                    }
                    if ( fabsf(input->current_ctf->IntegratedDefocusModulation(sq_spatial_frequency)) >= 0.1 ) {

                        break;
                    }
                }

                input->fit_frc[counter] = 1.0;
            }
        }
    }

    for ( counter = first_bin_to_check; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        //wxPrintf("On bin %i, fit_frc = %f, rot averate astig = %f\n", counter, fit_frc[counter], rotational_average_astig[counter]);
        at_last_bin_with_good_fit = ((number_of_bins_above_low_threshold > 3) && (input->fit_frc[counter] < low_threshold)) ||
                                    ((number_of_bins_above_high_threshold > 3) && (input->fit_frc[counter] < frc_significance_threshold));
        if ( at_last_bin_with_good_fit ) {
            last_bin_with_good_fit = counter;
            break;
        }
        // Count number of bins above given thresholds
        if ( input->fit_frc[counter] > low_threshold )
            number_of_bins_above_low_threshold++;
        if ( input->fit_frc[counter] > frc_significance_threshold )
            number_of_bins_above_significance_threshold++;
        if ( input->fit_frc[counter] > high_threshold )
            number_of_bins_above_high_threshold++;
    }
    //wxPrintf("%i bins out of %i checked were above significance threshold\n",number_of_bins_above_significance_threshold,number_of_bins_in_1d_spectra-first_bin_to_check);
    if ( number_of_bins_above_significance_threshold == input->number_of_bins_in_1d_spectra - first_bin_to_check )
        last_bin_with_good_fit = input->number_of_bins_in_1d_spectra - 1;
    if ( number_of_bins_above_significance_threshold == 0 )
        last_bin_with_good_fit = 1;
    last_bin_with_good_fit = std::min(last_bin_with_good_fit, input->number_of_bins_in_1d_spectra);
    MyDebugPrint(" Done!\n");
    return last_bin_with_good_fit;
}

CTFNodeFitOuput fit_thickness_nodes(CTFNodeFitInput* input) {
    // Store debug output in a JSON object
    wxJSONValue debug_json_output;

    float first_thickness_estimate = input->current_ctf->ThicknessWhereIntegrateDefocusModulationIsZero(powf(input->spatial_frequency[input->last_bin_with_good_fit], 2.0));
    if ( input->debug ) {
        debug_json_output["thickness_estimates"]            = wxJSONValue(wxJSONTYPE_OBJECT);
        debug_json_output["thickness_estimates"]["initial"] = first_thickness_estimate * input->pixel_size_for_fitting;
        write_fit_result_JSON_debug(debug_json_output, "initial_fit", input->number_of_bins_in_1d_spectra, input->rotational_average_astig, input->rotational_average_astig_fit, input->spatial_frequency);
    }
    input->current_ctf->SetSampleThickness(first_thickness_estimate);
    // Recalculate spectra and fit using the initial estimate
    int   counter;
    float current_sq_sf;
    float azimuth_for_1d_plots         = ReturnAzimuthToUseFor1DPlots(input->current_ctf);
    float defocus_for_1d_plots         = input->current_ctf->DefocusGivenAzimuth(azimuth_for_1d_plots);
    float sq_sf_of_phase_shift_maximum = input->current_ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(defocus_for_1d_plots);
    Curve rotational_average_astig_curve;
    for ( counter = 1; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        current_sq_sf = powf(input->spatial_frequency[counter], 2);
        if ( current_sq_sf <= sq_sf_of_phase_shift_maximum ) {
            input->rotational_average_astig[counter] = input->equiphase_average_pre_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
        }
        else {
            input->rotational_average_astig[counter] = input->equiphase_average_post_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
        }
        rotational_average_astig_curve.AddPoint(input->spatial_frequency[counter], input->rotational_average_astig[counter]);
    }
    rotational_average_astig_curve.FitPolynomialToData(3);
    // Subtract polynomial fit from rotational_average_astig and add 0.5
    // This is done to line the spectra up better with the CTF model including
    // the thickness nodes
    for ( counter = 1; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        input->rotational_average_astig[counter] -= rotational_average_astig_curve.polynomial_fit[counter];
        input->rotational_average_astig[counter] += 0.5;
    }
    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        current_sq_sf                                = powf(input->spatial_frequency[counter], 2);
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots, input->use_rounded_square);
    }

    // Write the initial fit to the debug JSON object
    if ( input->debug )
        write_fit_result_JSON_debug(debug_json_output, "after_first_estimate", input->number_of_bins_in_1d_spectra, input->rotational_average_astig, input->rotational_average_astig_fit, nullptr);

    input->current_ctf->SetLowestFrequencyForFitting(1.0 / input->low_resolution_limit * input->pixel_size_for_fitting);
    input->current_ctf->SetHighestFrequencyForFitting(1.0 / input->high_resolution_limit * input->pixel_size_for_fitting);
    if ( input->bruteforce_1D ) {
        do_1D_bruteforce(input, debug_json_output);
    }
    if ( input->refine_2D ) {
        do_2D_refinement(input, debug_json_output);
    }
    double* rotational_average_astig_renormalized = new double[input->number_of_bins_in_1d_spectra];
    float*  number_of_extrema_profile             = new float[input->number_of_bins_in_1d_spectra];
    recalculate_1D_spectra(input, rotational_average_astig_renormalized, number_of_extrema_profile, debug_json_output);
    if ( input->debug ) {
        debug_json_output["frc"]                          = wxJSONValue(wxJSONTYPE_OBJECT);
        debug_json_output["frc"]["renormalized_spectrum"] = wxJSONValue(wxJSONTYPE_ARRAY);
        for ( int counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            debug_json_output["frc"]["renormalized_spectrum"].Append(rotational_average_astig_renormalized[counter]);
        }
        for ( int counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            debug_json_output["frc"]["fit"].Append(input->rotational_average_astig_fit[counter]);
        }
        debug_json_output["frc"]["number_of_extrema_profile"] = wxJSONValue(wxJSONTYPE_ARRAY);
        for ( int counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            debug_json_output["frc"]["number_of_extrema_profile"].Append(number_of_extrema_profile[counter]);
        }
    }
    int last_bin_with_good_fit = calculate_new_frc(input, rotational_average_astig_renormalized, number_of_extrema_profile, debug_json_output);
    MyDebugPrint("Offesting spectrum\n");
    for ( counter = 1; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        input->rotational_average_astig[counter] -= rotational_average_astig_curve.polynomial_fit[counter];
        input->rotational_average_astig[counter] += 0.5;
    }

    if ( input->debug ) {
        // Write out the json debug file
        MyDebugPrint("Print out debug");
        wxJSONWriter writer;
        wxString     json_string;
        writer.Write(debug_json_output, json_string);
        wxFile debug_file;
        debug_file.Open(input->debug_filename + "thickness.json", wxFile::write);
        debug_file.Write(json_string);
        debug_file.Close( );
    }
    CTFNodeFitOuput output = {last_bin_with_good_fit};
    MyDebugPrint("Cleaning up\n");
    delete[] rotational_average_astig_renormalized;
    MyDebugPrint("Cleaning up2\n");
    delete[] number_of_extrema_profile;
    MyDebugPrint("Cleaning up3\n");
    return output;
}