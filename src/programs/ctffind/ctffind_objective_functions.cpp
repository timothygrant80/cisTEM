#include "../../core/core_headers.h"
#include "./ctffind.h"

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

// This is the function which will be minimised when dealing with 1D fitting
float CtffindCurveObjectiveFunction(void* scoring_parameters, float array_of_values[]) {
    CurveCTFComparison* comparison_object = reinterpret_cast<CurveCTFComparison*>(scoring_parameters);

    CTF my_ctf = comparison_object->ctf;
    if ( comparison_object->find_thickness_nodes ) {
        my_ctf.SetSampleThickness(array_of_values[0]);
        my_ctf.SetDefocus(array_of_values[1], array_of_values[1], 0.0);
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
                current_ctf_value = my_ctf.EvaluatePowerspectrumWithThickness(current_spatial_frequency_squared, 0.0, comparison_object->fit_nodes_rounded_square);
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
