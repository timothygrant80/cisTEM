#include "../../core/core_headers.h"
#include "./ctffind.h"

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

void do_1D_bruteforce(CTFNodeFitInput* input, wxJSONValue& debug_json_output) {
    int   counter;
    float current_sq_sf;
    float azimuth_for_1d_plots = ReturnAzimuthToUseFor1DPlots(input->current_ctf);
    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        input->comparison_object_1D->curve[counter] = input->rotational_average_astig[counter];
    }
    input->comparison_object_1D->find_phase_shift     = false;
    input->comparison_object_1D->find_thickness_nodes = true;
    input->comparison_object_1D->ctf                  = *(input->current_ctf);

    // We can now look for the defocus value
    float bf_halfrange[1] = {1750 / input->pixel_size_for_fitting};

    float bf_midpoint[1] = {500 / input->pixel_size_for_fitting + bf_halfrange[0]};

    float bf_stepsize[1] = {50 / input->pixel_size_for_fitting};

    int number_of_search_dimensions = 1;

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
    delete[] all_values;
    delete[] all_scores;

    input->current_ctf->SetSampleThickness(brute_force_search->GetBestValue(0));
    for ( counter = 0; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
        current_sq_sf                                = powf(input->spatial_frequency[counter], 2);
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots);
    }
    if ( input->debug ) {
        write_fit_result_JSON_debug(debug_json_output, "after_1D_brute_force", input->number_of_bins_in_1d_spectra, input->rotational_average_astig, input->rotational_average_astig_fit, nullptr);
        debug_json_output["thickness_estimates"]["after_1D_brute_force"] = brute_force_search->GetBestValue(0) * input->pixel_size_for_fitting;
    }
    delete brute_force_search;
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
    ConjugateGradient* conjugate_gradient_minimizer = new ConjugateGradient( );
    conjugate_gradient_minimizer->Init(&CtffindObjectiveFunction, input->comparison_object_2D, number_of_search_dimensions, cg_starting_point, cg_accuracy);

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
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots);
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

        for ( counter = 1; counter < input->number_of_bins_in_1d_spectra; counter++ ) {
            current_sq_sf = powf(input->spatial_frequency[counter], 2);
            if ( current_sq_sf <= sq_sf_of_phase_shift_maximum ) {
                input->rotational_average_astig[counter] = input->equiphase_average_pre_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
            }
            else {
                input->rotational_average_astig[counter] = input->equiphase_average_post_max.ReturnLinearInterpolationFromX(input->current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
            }
            rotational_average_astig_renormalized[counter] = input->rotational_average_astig[counter];
        }
        Renormalize1DSpectrumForFRC(input->number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, input->rotational_average_astig_fit, number_of_extrema_profile);
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
    ComputeFRCBetween1DSpectrumAndFit(input->number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, input->rotational_average_astig_fit, number_of_extrema_profile, input->fit_frc, input->fit_frc_sigma, first_fit_bin);
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
        input->rotational_average_astig_fit[counter] = input->current_ctf->EvaluatePowerspectrumWithThickness(current_sq_sf, azimuth_for_1d_plots);
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