#include "../../core/core_headers.h"
#include "./ctffind.h"

void CTFTilt::Init(ImageFile& wanted_input_file, float wanted_high_res_limit_ctf_fit, float wanted_high_res_limit_tilt_fit, float wanted_minimum_defocus, float wanted_maximum_defocus,
                   float wanted_pixel_size, float wanted_acceleration_voltage_in_kV, float wanted_spherical_aberration_in_mm, float wanted_amplitude_contrast, float wanted_additional_phase_shift_in_radians,
                   bool wanted_debug, std::string wanted_debug_json_output_filename) {
    box_size  = 128;
    tile_size = 128;

    // check below for n_sections
    n_steps                        = 2;
    rough_defocus_determined       = false;
    defocus_astigmatism_determined = false;
    power_spectra_calculated       = false;

    // Must be odd
    box_convolution = 55;

    micrograph_square_dimension = std::max(wanted_input_file.ReturnXSize( ), wanted_input_file.ReturnYSize( ));
    if ( IsOdd((micrograph_square_dimension)) )
        micrograph_square_dimension++;

    int image_x_dim_forsubsection = wanted_input_file.ReturnXSize( );
    int image_y_dim_forsubsection = wanted_input_file.ReturnYSize( );
    input_image_buffer            = new Image[wanted_input_file.ReturnZSize( )];
    image_buffer_counter          = 0;

    original_pixel_size               = wanted_pixel_size;
    low_res_limit                     = 40.0f;
    refine_mode                       = 0;
    defocus_1                         = 0.0f;
    defocus_2                         = 0.0f;
    astigmatic_angle                  = 0.0f;
    acceleration_voltage_in_kV        = wanted_acceleration_voltage_in_kV;
    spherical_aberration_in_mm        = wanted_spherical_aberration_in_mm;
    amplitude_contrast                = wanted_amplitude_contrast;
    additional_phase_shift_in_radians = wanted_additional_phase_shift_in_radians;
    best_tilt_axis                    = 0.0f;
    best_tilt_angle                   = 0.0f;
    high_res_limit_ctf_fit            = wanted_high_res_limit_ctf_fit;
    high_res_limit_tilt_fit           = wanted_high_res_limit_tilt_fit;
    minimum_defocus                   = wanted_minimum_defocus;
    maximum_defocus                   = wanted_maximum_defocus;

    float binning_factor                = 0.5f * high_res_limit_ctf_fit / original_pixel_size;
    micrograph_binned_dimension_for_ctf = ReturnClosestFactorizedUpper(myroundint(micrograph_square_dimension / binning_factor), 5, true);
    binning_factor                      = float(micrograph_square_dimension) / float(micrograph_binned_dimension_for_ctf);
    ctf_fit_pixel_size                  = original_pixel_size * binning_factor;

    resampled_power_spectrum.Allocate(box_size, box_size, 1);
    resampled_power_spectrum_binned_image.Allocate(box_size, box_size, 1);
    resampled_power_spectrum_binned_image.SetToConstant(0.0f);
    average_spectrum.Allocate(box_size, box_size, 1);
    ctf_transform.Allocate(box_size, box_size, 1);
    ctf_image.Allocate(box_size, box_size, 1);

    micrograph_subregion_dimension = 2000.0f / ctf_fit_pixel_size;
    MyDebugPrint("micrograph_subregion_dimension: %d\n", micrograph_subregion_dimension);
    micrograph_subregion_dimension = ReturnClosestFactorizedLower(micrograph_subregion_dimension, 5);
    MyDebugPrint("micrograph_subregion_dimension factorized: %d\n", micrograph_subregion_dimension);
    if ( micrograph_subregion_dimension < micrograph_binned_dimension_for_ctf )
        power_spectrum_binned_image.Allocate(micrograph_subregion_dimension, micrograph_subregion_dimension, 1);
    else
        power_spectrum_binned_image.Allocate(micrograph_binned_dimension_for_ctf, micrograph_binned_dimension_for_ctf, 1);

    tilt_binning_factor = 0.5f * high_res_limit_tilt_fit / original_pixel_size;
    MyDebugPrint("Tilt binning factor = %f", tilt_binning_factor);
    tilt_fit_pixel_size = original_pixel_size * tilt_binning_factor;
    n_sections_x        = int(image_x_dim_forsubsection / tilt_binning_factor / tile_size);
    n_sections_y        = int(image_y_dim_forsubsection / tilt_binning_factor / tile_size);
    MyDebugPrint("number of sections along x and y dimensions: %d, %d\n", n_sections_x, n_sections_y);

    int ix, iy;
    int sub_section_dimension;
    int section_counter     = 0;
    sub_section_dimension_x = myroundint(image_x_dim_forsubsection / tilt_binning_factor) / n_sections_x;
    if ( IsOdd(sub_section_dimension_x) )
        sub_section_dimension_x--;
    sub_section_dimension_y = myroundint(image_y_dim_forsubsection / tilt_binning_factor) / n_sections_y;
    if ( IsOdd(sub_section_dimension_y) )
        sub_section_dimension_y--;
    sub_section_dimension = std::min(sub_section_dimension_x, sub_section_dimension_y);
    MyDebugPrint("subsection dim, subsection_dim_x, subsection_dim_y: %d, %d, %d\n", sub_section_dimension, sub_section_dimension_x, sub_section_dimension_y);
    tile_size = std::min(sub_section_dimension, tile_size);
    MyDebugPrint("tile size: %d\n", tile_size);
    sub_section.Allocate(tile_size, tile_size, 1, true);
    power_spectrum_sub_section.Allocate(tile_size, tile_size, 1);

    resampled_power_spectra = new Image[((n_sections_x - 1) * n_steps + 1) * ((n_sections_y - 1) * n_steps + 1)];
    invalid_powerspectrum   = new bool[((n_sections_x - 1) * n_steps + 1) * ((n_sections_y - 1) * n_steps + 1)];

    int tile_num = ((n_sections_x - 1) * n_steps + 1) * ((n_sections_y - 1) * n_steps + 1);
    MyDebugPrint("total tiles: %d\n", tile_num);
    int secxsecy = n_sections_x * n_sections_y;
    MyDebugPrint("section_x*section_y %d\n", secxsecy);

    // for ( iy = -(n_sections_y - 1) * n_steps / 2; iy <= (n_sections_y - 1) * n_steps / 2; iy++ ) {
    //     for ( ix = -(n_sections_x - 1) * n_steps / 2; ix <= (n_sections_x - 1) * n_steps / 2; ix++ ) {
    for ( int i = 0; i < tile_num; i++ ) {
        resampled_power_spectra[section_counter].Allocate(box_size, box_size, 1, true);
        resampled_power_spectra[section_counter].SetToConstant(0.0f);
        invalid_powerspectrum[section_counter] = false;
        section_counter++;
    }
    //     }
    // }

    debug                      = wanted_debug;
    debug_json_output_filename = wanted_debug_json_output_filename;
    //	CalculatePowerSpectra(true);
}

CTFTilt::~CTFTilt( ) {
    // Write out the json file
    if ( debug ) {
        wxJSONWriter writer;
        wxString     json_string;
        writer.Write(debug_json_output, json_string);
        wxFile debug_file;
        debug_file.Open(debug_json_output_filename, wxFile::write);
        debug_file.Write(json_string);
        debug_file.Close( );
    }
    delete[] input_image_buffer;
    delete[] resampled_power_spectra;
    delete[] invalid_powerspectrum;
}

void CTFTilt::CalculatePowerSpectra(bool subtract_average) {
    MyDebugAssertTrue(! power_spectra_calculated, "Error: Power spectra already calculated\n");

    int i;
    int section_counter = 0;
    int ix, iy;

    Image temp_image;

    // Power spectrum for rough CTF fit
    temp_image.Allocate(resampled_power_spectrum_binned_image.logical_x_dimension, resampled_power_spectrum_binned_image.logical_y_dimension, false);
    input_image_binned.Allocate(micrograph_square_dimension, micrograph_square_dimension, true);
    input_image.ClipInto(&input_image_binned);
    input_image_binned.ForwardFFT( );
    input_image_binned.Resize(micrograph_binned_dimension_for_ctf, micrograph_binned_dimension_for_ctf, 1);
    input_image_binned.complex_values[0] = 0.0f + I * 0.0f;
    input_image_binned.BackwardFFT( );
    input_image_binned.CosineRectangularMask(0.9f * input_image_binned.physical_address_of_box_center_x, 0.9f * input_image_binned.physical_address_of_box_center_y, 0.0f, 0.1f * input_image_binned.logical_x_dimension);

    if ( micrograph_subregion_dimension < input_image_binned.logical_x_dimension )
        input_image_binned.Resize(micrograph_subregion_dimension, micrograph_subregion_dimension, 1);
    input_image_binned.ForwardFFT( );
    input_image_binned.ComputeAmplitudeSpectrumFull2D(&power_spectrum_binned_image);
    //	for (int i = 0; i < power_spectrum_binned_image.real_memory_allocated; i++) power_spectrum_binned_image.real_values[i] = powf(power_spectrum_binned_image.real_values[i], 2);
    power_spectrum_binned_image.ForwardFFT( );
    power_spectrum_binned_image.ClipInto(&temp_image);
    temp_image.BackwardFFT( );
    resampled_power_spectrum_binned_image.AddImage(&temp_image);
    //	resampled_power_spectrum_binned_image.CosineMask(resampled_power_spectrum.logical_x_dimension * 0.1f, resampled_power_spectrum.logical_x_dimension * 0.2f, true);
    if ( subtract_average ) {
        resampled_power_spectrum_binned_image.SpectrumBoxConvolution(&average_spectrum, box_convolution, float(resampled_power_spectrum_binned_image.logical_x_dimension) * ctf_fit_pixel_size / low_res_limit);
        resampled_power_spectrum_binned_image.SubtractImage(&average_spectrum);
    }
    resampled_power_spectrum_binned_image.CosineMask(resampled_power_spectrum_binned_image.logical_x_dimension * 0.3f, resampled_power_spectrum_binned_image.logical_x_dimension * 0.4f);
    //	resampled_power_spectrum_binned_image.AddMultiplyConstant(-resampled_power_spectrum_binned_image.ReturnAverageOfRealValues(), sqrtf(1.0f / resampled_power_spectrum_binned_image.ReturnVarianceOfRealValues()));
    // Scaling to yield correlation coefficient when evaluating scores
    //	resampled_power_spectrum_binned_image.MultiplyByConstant(2.0f / resampled_power_spectrum_binned_image.logical_x_dimension / resampled_power_spectrum_binned_image.logical_y_dimension);

    // Power spectra for tilted CTF fit
    temp_image.Allocate(resampled_power_spectra[0].logical_x_dimension, resampled_power_spectra[0].logical_y_dimension, false);
    input_image.ForwardFFT( );
    input_image.Resize(myroundint(input_image.logical_x_dimension / tilt_binning_factor), myroundint(input_image.logical_y_dimension / tilt_binning_factor), 1);
    input_image.complex_values[0] = 0.0f + I * 0.0f;
    input_image.BackwardFFT( );
    if ( debug && subtract_average ) {
        debug_json_output["search_tiles"] = wxJSONValue(wxJSONTYPE_ARRAY);
    }
    // for ( iy = -(n_sections_y - 1) * n_steps / 2; iy <= (n_sections_y - 1) * n_steps / 2; iy++ ) {
    //     for ( ix = -(n_sections_x - 1) * n_steps / 2; ix <= (n_sections_x - 1) * n_steps / 2; ix++ ) {
    // int tmp_count = 0;
    for ( iy = -(n_sections_y - 1) * n_steps; iy <= (n_sections_y - 1) * n_steps; iy += 2 ) {
        for ( ix = -(n_sections_x - 1) * n_steps; ix <= (n_sections_x - 1) * n_steps; ix += 2 ) {
            //			pointer_to_original_image->QuickAndDirtyWriteSlice("binned_input_image.mrc", 1);
            // tmp_count++;
            input_image.ClipInto(&sub_section, 0.0f, false, 0.0f, float(ix) / 2.0 * sub_section_dimension_x / float(n_steps), float(iy) / 2.0 * sub_section_dimension_y / float(n_steps), 0);
            if ( debug && subtract_average ) {
                debug_json_output["search_tiles"].Append(wxJSONValue(wxJSONTYPE_OBJECT));
                debug_json_output["search_tiles"][debug_json_output["search_tiles"].Size( ) - 1]["x"]      = float(ix) / 2.0 * sub_section_dimension_x / float(n_steps);
                debug_json_output["search_tiles"][debug_json_output["search_tiles"].Size( ) - 1]["y"]      = float(iy) / 2.0 * sub_section_dimension_y / float(n_steps);
                debug_json_output["search_tiles"][debug_json_output["search_tiles"].Size( ) - 1]["width"]  = sub_section_dimension_x;
                debug_json_output["search_tiles"][debug_json_output["search_tiles"].Size( ) - 1]["height"] = sub_section_dimension_y;
            }
            if ( sub_section.ReturnAverageOfRealValues( ) < 0.0f ) {
                invalid_powerspectrum[section_counter] = true;
                section_counter++;
                continue;
            }
            sub_section.CosineRectangularMask(0.9f * sub_section.physical_address_of_box_center_x, 0.9f * sub_section.physical_address_of_box_center_y, 0.0f, 0.1f * sub_section.logical_x_dimension);
            if ( debug ) {
                // sub_section.QuickAndDirtyWriteSlice("sub_section.mrc", 1 + section_counter);
            }
            sub_section.MultiplyByConstant(sqrtf(1.0f / sub_section.ReturnVarianceOfRealValues( )));
            sub_section.ForwardFFT( );
            sub_section.ComputeAmplitudeSpectrumFull2D(&power_spectrum_sub_section);
            for ( i = 0; i < power_spectrum_sub_section.real_memory_allocated; i++ )
                power_spectrum_sub_section.real_values[i] = powf(power_spectrum_sub_section.real_values[i], 2);
            power_spectrum_sub_section.ForwardFFT( );
            power_spectrum_sub_section.ClipInto(&temp_image);
            //			resampled_power_spectra[section_counter].CosineMask(0.45f, 0.1f);
            temp_image.BackwardFFT( );
            resampled_power_spectra[section_counter].AddImage(&temp_image);
            if ( subtract_average ) {
                resampled_power_spectra[section_counter].SpectrumBoxConvolution(&average_spectrum, box_convolution, float(resampled_power_spectra[0].logical_x_dimension) * tilt_fit_pixel_size / low_res_limit);
                resampled_power_spectra[section_counter].SubtractImage(&average_spectrum);
            }
            //			resampled_power_spectrum.CosineMask(resampled_power_spectrum.logical_x_dimension * 0.45f, resampled_power_spectrum.logical_x_dimension * 0.1f);
            //			resampled_power_spectra[section_counter].QuickAndDirtyWriteSlice("power_spectra.mrc", 1 + section_counter);
            section_counter++;
        }
    }
    // wxPrintf("-------------------------------------------------------------------");
    // wxPrintf("the tiles real boxed: %d\n", tmp_count);

    power_spectra_calculated = true;
}

void CTFTilt::UpdateInputImage(Image* wanted_input_image) {
    // MyDebugAssertTrue(input_image_x_dimension == wanted_input_image->logical_x_dimension && input_image_y_dimension == wanted_input_image->logical_y_dimension, "Error: Image dimensions do not match\n");
    // input_image_buffer[image_buffer_counter].Allocate(wanted_input_image->logical_x_dimension, wanted_input_image->logical_y_dimension, 1, true);
    input_image_buffer[image_buffer_counter].CopyFrom(wanted_input_image);
    image_buffer_counter++;
    input_image.CopyFrom(wanted_input_image);
    power_spectra_calculated = false;
}

float CTFTilt::FindRoughDefocus( ) {
    float  variance;
    float  variance_max = -FLT_MAX;
    float  defocus;
    float  average_defocus;
    double start_values[4];

    refine_mode = 0;

    for ( defocus = minimum_defocus; defocus < maximum_defocus; defocus += 100.0f ) {
        start_values[1] = defocus;
        start_values[2] = defocus;
        start_values[3] = 0.0f;
        variance        = -ScoreValues(start_values);

        if ( variance > variance_max ) {
            variance_max    = variance;
            average_defocus = defocus;
        }
    }

    defocus_1                = average_defocus;
    defocus_2                = average_defocus;
    rough_defocus_determined = true;

    //	wxPrintf("defocus, var = %g %g\n\n", average_defocus, variance_max);
    return variance_max;
}

float CTFTilt::FindDefocusAstigmatism( ) {
    MyDebugAssertTrue(rough_defocus_determined, "Error: Rough defocus not yet determined\n");

    DownhillSimplex simplex_minimzer(3);

    float average_defocus = (defocus_1 + defocus_2) / 2.0f;

    double ranges[4];
    double start_values[4];
    double min_values[4];

    refine_mode = 0;

    ranges[0] = 0.0f;
    ranges[1] = 1000.0f;
    ranges[2] = 1000.0f;
    ranges[3] = 180.0f;

    start_values[0] = 0.0f;
    start_values[1] = average_defocus;
    start_values[2] = average_defocus;
    start_values[3] = 0.0f;

    simplex_minimzer.SetIinitalValues(start_values, ranges);

    simplex_minimzer.initial_values[1][1] = start_values[1] * simplex_minimzer.value_scalers[1] + ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(8.0f / 9.0f);
    simplex_minimzer.initial_values[1][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[1][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[2][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[2][2] = start_values[2] * simplex_minimzer.value_scalers[2] + ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[2][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[3][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[3][2] = start_values[2] * simplex_minimzer.value_scalers[2] - ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[3][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[4][1] = start_values[1] * simplex_minimzer.value_scalers[1];
    simplex_minimzer.initial_values[4][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[4][3] = start_values[3] * simplex_minimzer.value_scalers[3] + ranges[3] * simplex_minimzer.value_scalers[3];
    MyDebugPrint("scalers %f, %f, %f\n", simplex_minimzer.value_scalers[1], simplex_minimzer.value_scalers[2], simplex_minimzer.value_scalers[3]);
    simplex_minimzer.MinimizeFunction(this, SampleTiltScoreFunctionForSimplex);
    simplex_minimzer.GetMinimizedValues(min_values);

    defocus_1                      = min_values[1];
    defocus_2                      = min_values[2];
    astigmatic_angle               = min_values[3];
    defocus_astigmatism_determined = true;

    //	wxPrintf("defocus_1, defocus_2, astigmatic_angle = %g %g %g\n\n", defocus_1, defocus_2, astigmatic_angle);
    return -ScoreValues(min_values);
}

float CTFTilt::SearchTiltAxisAndAngle( ) {
    MyDebugAssertTrue(power_spectra_calculated, "Error: Power spectra not calculated\n");
    MyDebugAssertTrue(defocus_astigmatism_determined, "Error: Defocus astigmatism not yet determined\n");

    float  variance;
    float  variance_max = -FLT_MAX;
    float  axis_step    = 20.0f;
    float  angle_step   = 10.0f;
    float  tilt_angle;
    float  tilt_axis;
    float  average_defocus = (defocus_1 + defocus_2) / 2.0f;
    double start_values[4];

    //	if (! power_spectra_calculated) CalculatePowerSpectra();

    refine_mode = 1;
    if ( debug ) {
        debug_json_output["tilt_axis_and_angle_search"] = wxJSONValue(wxJSONTYPE_ARRAY);
    }
    for ( tilt_angle = 0.0f; tilt_angle <= 80.0f; tilt_angle += angle_step ) {
        for ( tilt_axis = 0.0f; tilt_axis < 360.0f; tilt_axis += axis_step ) {
            start_values[1] = tilt_axis;
            start_values[2] = tilt_angle;
            start_values[3] = average_defocus;
            variance        = -ScoreValues(start_values);
            if ( debug ) {
                debug_json_output["tilt_axis_and_angle_search"].Append(wxJSONValue(wxJSONTYPE_ARRAY));
                debug_json_output["tilt_axis_and_angle_search"][debug_json_output["tilt_axis_and_angle_search"].Size( ) - 1].Append(tilt_axis);
                debug_json_output["tilt_axis_and_angle_search"][debug_json_output["tilt_axis_and_angle_search"].Size( ) - 1].Append(tilt_angle);
                debug_json_output["tilt_axis_and_angle_search"][debug_json_output["tilt_axis_and_angle_search"].Size( ) - 1].Append(variance);
            }
            if ( variance > variance_max ) {
                variance_max    = variance;
                best_tilt_axis  = tilt_axis;
                best_tilt_angle = tilt_angle;
                MyDebugPrint("tilt axis, angle, var = %g %g %g\n", best_tilt_axis, best_tilt_angle, variance_max);
            }
        }
    }
    return variance_max;
}

// // just refine the tilt and axis direction
// float CTFTilt::RefineTiltAxisAndAngle( ) {
//     MyDebugAssertTrue(power_spectra_calculated, "Error: Power spectra not calculated\n");
//     MyDebugAssertTrue(defocus_astigmatism_determined, "Error: Defocus astigmatism not yet determined\n");

//     DownhillSimplex simplex_minimzer(2);

//     double ranges[3];
//     double start_values[3];
//     double min_values[3];

//     //	if (! power_spectra_calculated) CalculatePowerSpectra();

//     // refine_mode = 1;

//     ranges[0] = 0.0f;
//     // ranges[1] = 40.0f;
//     // ranges[2] = 20.0f;
//     ranges[1] = 20.0f;
//     ranges[2] = 10.0f;

//     start_values[0] = 0.0f;
//     start_values[1] = best_tilt_axis;
//     start_values[2] = best_tilt_angle;

//     // wxPrintf("scalers1 %f, %f, %f\n", simplex_minimzer.value_scalers[1], simplex_minimzer.value_scalers[2], simplex_minimzer.value_scalers[3]);

//     simplex_minimzer.SetIinitalValues(start_values, ranges);
//     // wxPrintf("scalers1 %f, %f, %f\n", simplex_minimzer.value_scalers[1], simplex_minimzer.value_scalers[2], simplex_minimzer.value_scalers[3]);

//     simplex_minimzer.initial_values[1][1] = start_values[1] * simplex_minimzer.value_scalers[1] + ranges[1] * simplex_minimzer.value_scalers[1];
//     simplex_minimzer.initial_values[1][2] = start_values[2] * simplex_minimzer.value_scalers[2];

//     simplex_minimzer.initial_values[2][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] / 2.0f;
//     simplex_minimzer.initial_values[2][2] = start_values[2] * simplex_minimzer.value_scalers[2] + ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(3.0f / 4.0f);

//     simplex_minimzer.initial_values[3][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] / 2.0f;
//     simplex_minimzer.initial_values[3][2] = start_values[2] * simplex_minimzer.value_scalers[2] - ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(3.0f / 4.0f);

//     wxPrintf("initial values %f, %f, %f\n", simplex_minimzer.initial_values[4][1], simplex_minimzer.initial_values[4][2], simplex_minimzer.initial_values[4][3]);

//     simplex_minimzer.MinimizeFunction(this, SampleTiltScoreFunctionForSimplexTiltAxis);
//     simplex_minimzer.GetMinimizedValues(min_values);
//     // defocus_1       = defocus_1 + min_values[3] - average_defocus;
//     // defocus_2       = defocus_2 + min_values[3] - average_defocus;
//     best_tilt_axis  = min_values[1];
//     best_tilt_angle = min_values[2];

//     return -ScoreValues(min_values);
// }

// refine the tilt, tilt axis, and defocus
float CTFTilt::RefineTiltAxisAndAngle( ) {
    MyDebugAssertTrue(power_spectra_calculated, "Error: Power spectra not calculated\n");
    MyDebugAssertTrue(defocus_astigmatism_determined, "Error: Defocus astigmatism not yet determined\n");

    DownhillSimplex simplex_minimzer(3);
    // defocus_1              = 20884.36;
    // defocus_2              = 24401.83;
    float  average_defocus = (defocus_1 + defocus_2) / 2.0f;
    double ranges[4];
    double start_values[4];
    double min_values[4];

    //	if (! power_spectra_calculated) CalculatePowerSpectra();

    refine_mode = 1;

    ranges[0] = 0.0f;
    // ranges[1] = 40.0f;
    // ranges[2] = 20.0f;
    ranges[1] = 20.0f;
    ranges[2] = 10.0f;
    ranges[3] = 1000.0f;
    // ranges[3] = 0.0f;

    start_values[0] = 0.0f;
    start_values[1] = best_tilt_axis;
    start_values[2] = best_tilt_angle;
    start_values[3] = average_defocus;

    simplex_minimzer.SetIinitalValues(start_values, ranges);
    MyDebugPrint("scalers1 %f, %f, %f\n", simplex_minimzer.value_scalers[1], simplex_minimzer.value_scalers[2], simplex_minimzer.value_scalers[3]);

    simplex_minimzer.initial_values[1][1] = start_values[1] * simplex_minimzer.value_scalers[1] + ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(8.0f / 9.0f);
    simplex_minimzer.initial_values[1][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[1][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[2][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[2][2] = start_values[2] * simplex_minimzer.value_scalers[2] + ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[2][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[3][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[3][2] = start_values[2] * simplex_minimzer.value_scalers[2] - ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[3][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[4][1] = start_values[1] * simplex_minimzer.value_scalers[1];
    simplex_minimzer.initial_values[4][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[4][3] = start_values[3] * simplex_minimzer.value_scalers[3] + ranges[3] * simplex_minimzer.value_scalers[3];

    simplex_minimzer.MinimizeFunction(this, SampleTiltScoreFunctionForSimplex);
    simplex_minimzer.GetMinimizedValues(min_values);

    defocus_1       = defocus_1 + min_values[3] - average_defocus;
    defocus_2       = defocus_2 + min_values[3] - average_defocus;
    best_tilt_axis  = min_values[1];
    best_tilt_angle = min_values[2];

    return -ScoreValues(min_values);
}

float CTFTilt::CalculateTiltCorrectedSpectra(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                                             int box_size, Image* resampled_spectrum) {
    //	MyDebugAssertTrue(power_spectra_calculated,"Error: Power spectra not calculated\n");
    //	MyDebugAssertTrue(image_to_correct.is_in_real_space,"Error: Input image not in real space\n");
    MyDebugAssertTrue(resampled_spectrum->logical_x_dimension == resampled_spectrum->logical_y_dimension, "Error: Output spectrum not square\n");

    int   i;
    int   sub_section_x;
    int   sub_section_y;
    int   sub_section_dimension;
    int   stretched_dimension;
    int   n_sec;
    int   n_stp = 2;
    int   ix, iy;
    int   image_counter;
    float height;
    float x_coordinate_2d;
    float y_coordinate_2d;
    float x_rotated;
    float y_rotated;
    float average_defocus = (defocus_1 + defocus_2) / 2.0f;
    float stretch_factor;
    float padding_value;
    float pixel_size_for_fitting;
    //	float sigma;
    //	float offset_x;
    //	float offset_y;

    AnglesAndShifts rotation_angle;
    Image           section;
    Image           stretched_section;
    Image           power_spectrum;
    Image           resampled_power_spectrum;
    Image           counts_per_pixel;

    pixel_size_for_fitting = PixelSizeForFitting(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, &power_spectrum, &resampled_power_spectrum, false);
    sub_section_dimension  = resampled_spectrum->logical_x_dimension * (pixel_size_for_fitting / pixel_size_of_input_image);
    if ( IsOdd(sub_section_dimension) )
        sub_section_dimension--;
    n_sec = std::max(input_image_buffer[0].logical_x_dimension / sub_section_dimension, input_image_buffer[0].logical_y_dimension / sub_section_dimension);
    if ( IsEven(n_sec) )
        n_sec++;
    //	wxPrintf("n_sec, n_stp = %i %i\n", n_sec, n_stp);

    sub_section_x = input_image_buffer[0].logical_x_dimension / (n_sec + 1);
    if ( IsOdd(sub_section_x) )
        sub_section_x--;
    sub_section_y = input_image_buffer[0].logical_y_dimension / (n_sec + 1);
    if ( IsOdd(sub_section_y) )
        sub_section_y--;
    //	sub_section_dimension = std::min(sub_section_x, sub_section_y);

    //	offset_x = float(n_sec) / 2.0f * sub_section_x;
    //	offset_y = float(n_sec) / 2.0f * sub_section_y;

    section.Allocate(sub_section_dimension, sub_section_dimension, 1, true);
    power_spectrum.Allocate(sub_section_dimension, sub_section_dimension, 1, true);
    //	resampled_power_spectrum.Allocate(resampled_spectrum.logical_x_dimension, resampled_spectrum.logical_y_dimension, false);
    resampled_spectrum->SetToConstant(0.0f);
    resampled_spectrum->is_in_real_space = true;

    rotation_angle.GenerateRotationMatrix2D(best_tilt_axis);
    counts_per_pixel.Allocate(resampled_spectrum->logical_x_dimension, resampled_spectrum->logical_y_dimension, true);
    counts_per_pixel.SetToConstant(0.0f);

    //	int section_counter = 0;
    MyDebugPrint("Calculating tilt corrected spectra with image_buffer_counter = %i, n_sec = %i, n_stp = %i\n", image_buffer_counter, n_sec, n_stp);
    for ( image_counter = 0; image_counter < image_buffer_counter; image_counter++ ) {
        //		wxPrintf("working on frame %i\n", image_counter);
        for ( iy = -(n_sec - 1) * n_stp / 2; iy <= (n_sec - 1) * n_stp / 2; iy++ ) {
            for ( ix = -(n_sec - 1) * n_stp / 2; ix <= (n_sec - 1) * n_stp / 2; ix++ ) {
                x_coordinate_2d = float(ix) * sub_section_x / float(n_stp) * pixel_size_of_input_image;
                y_coordinate_2d = float(iy) * sub_section_y / float(n_stp) * pixel_size_of_input_image;
                rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);
                height = y_rotated * tanf(deg_2_rad(best_tilt_angle));
                if ( average_defocus > 100.0f )
                    stretch_factor = sqrtf(fabsf(average_defocus + height) / average_defocus);
                else
                    stretch_factor = 1.0f;
                //				wxPrintf("x, y, x_coordinate_2d, y_coordinate_2d, stretch_factor = %i %i %g %g %g\n", ix, iy, x_coordinate_2d, y_coordinate_2d, stretch_factor);

                input_image_buffer[image_counter].ClipInto(&section, 0.0f, false, 0.0f, myroundint(float(ix) * sub_section_x / float(n_stp)), myroundint(float(iy) * sub_section_y / float(n_stp)), 0);
                //wxPrintf("section size: %d %d\n", section.logical_x_dimension, section.logical_y_dimension);
                padding_value = section.ReturnAverageOfRealValues( );
                section.CosineRectangularMask(0.9f * section.physical_address_of_box_center_x, 0.9f * section.physical_address_of_box_center_y,
                                              0.0f, 0.1f * section.logical_x_dimension, false, true, padding_value);
                //				sigma = sqrtf(section.ReturnVarianceOfRealValues());
                //				if (sigma > 0.0f) section.MultiplyByConstant(1.0f / sigma);
                //				section.QuickAndDirtyWriteSlice("sections.mrc", section_counter + 1);
                section.ForwardFFT( );
                section.complex_values[0] = 0.0f + I * 0.0f;
                section.ComputeAmplitudeSpectrumFull2D(&power_spectrum);
                //wxPrintf("power_spectrum size: %d %d\n", power_spectrum.logical_x_dimension, power_spectrum.logical_y_dimension);
                //wxPrintf("resampled_spectrum size: %d %d\n", resampled_spectrum->logical_x_dimension, resampled_spectrum->logical_y_dimension);
                //				power_spectrum.QuickAndDirtyWriteSlice("spectrum.mrc", section_counter + 1);
                //				section_counter++;
                //				for (i = 0; i < power_spectrum_sub_section.real_memory_allocated; i++) power_spectrum_sub_section.real_values[i] = powf(power_spectrum_sub_section.real_values[i], 2);
                //wxPrintf("%d %f %f %f", resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size);
                PixelSizeForFitting(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, &power_spectrum, &resampled_power_spectrum, true, stretch_factor);

                //				power_spectrum.ForwardFFT();
                //				stretched_dimension = myroundint(resampled_spectrum->logical_x_dimension * stretch_factor);
                //				if (IsOdd(stretched_dimension)) stretched_dimension++;
                //				if (fabsf(stretched_dimension - resampled_spectrum->logical_x_dimension * stretch_factor) > fabsf(stretched_dimension - 2 - resampled_spectrum->logical_x_dimension * stretch_factor)) stretched_dimension -= 2;
                //				resampled_power_spectrum.Allocate(stretched_dimension, stretched_dimension, false);
                //				power_spectrum.ClipInto(&resampled_power_spectrum);
                //				resampled_power_spectrum.BackwardFFT();
                //				resampled_power_spectrum.Resize(resampled_spectrum->logical_x_dimension, resampled_spectrum->logical_y_dimension, 1, 0.0f);
                resampled_spectrum->AddImage(&resampled_power_spectrum);
                for ( i = 0; i < resampled_spectrum->real_memory_allocated; i++ )
                    if ( resampled_power_spectrum.real_values[i] != 0.0f )
                        counts_per_pixel.real_values[i] += 1.0f;
            }
        }
    }
    resampled_spectrum->DividePixelWise(counts_per_pixel);
    //	resampled_spectrum.SpectrumBoxConvolution(&average_spectrum, box_convolution, float(resampled_power_spectra[0].logical_x_dimension) * tilt_fit_pixel_size / low_res_limit);
    //	resampled_spectrum.SubtractImage(&average_spectrum);

    return pixel_size_for_fitting;
}

double CTFTilt::ScoreValues(double input_values[]) {
    // 0 = ignore, stupid code conversion

    // refine_mode = 0
    // 1 = defocus_1
    // 2 = defocus_2
    // 3 = astigmatic_angle

    // refine_mode = 1
    // 1 = tilt_axis
    // 2 = tilt_angle
    // 3 = average_defocus

    int    i, j;
    int    ix, iy;
    long   pointer;
    long   counter         = 0;
    int    section_counter = 0;
    float  fraction_of_nonzero_pixels;
    float  minimum_radius_sq;
    float  radius_sq;
    float  average_defocus;
    float  ddefocus;
    float  height;
    float  variance;
    float  x_coordinate_2d;
    float  y_coordinate_2d;
    float  x_rotated;
    float  y_rotated;
    double sum_ctf           = 0.0;
    double sum_power         = 0.0;
    double sum_ctf_squared   = 0.0;
    double sum_power_squared = 0.0;
    double sum_ctf_power     = 0.0;
    double correlation_coefficient;

    AnglesAndShifts rotation_angle;

    CTF input_ctf;

    switch ( refine_mode ) {
        // Fitting defocus 1, defocus 2, astigmatic angle
        case 0:
            minimum_radius_sq = powf(float(resampled_power_spectrum.logical_x_dimension) * ctf_fit_pixel_size / low_res_limit, 2);
            average_defocus   = (input_values[1] + input_values[2]) / 2.0f;
            input_ctf.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast, input_values[1], input_values[2], input_values[3], ctf_fit_pixel_size, additional_phase_shift_in_radians);
            ctf_transform.CalculateCTFImage(input_ctf);
            ctf_transform.ComputeAmplitudeSpectrumFull2D(&ctf_image);
            //			ctf_image.CosineMask(ctf_image.logical_x_dimension * 0.4f, ctf_image.logical_x_dimension * 0.2f);
            for ( i = 0; i < ctf_image.real_memory_allocated; i++ )
                ctf_image.real_values[i] = powf(ctf_image.real_values[i], 4);
            resampled_power_spectrum.CopyFrom(&resampled_power_spectrum_binned_image);
            //			ctf_image.QuickAndDirtyWriteSlice("junk1.mrc", 1);
            //			resampled_power_spectrum.QuickAndDirtyWriteSlice("junk2.mrc", 1);

            pointer = 0;
            for ( j = 0; j < ctf_image.logical_y_dimension; j++ ) {
                for ( i = 0; i < ctf_image.logical_x_dimension; i++ ) {
                    radius_sq = float((i - ctf_image.physical_address_of_box_center_x) * (i - ctf_image.physical_address_of_box_center_x) + (j - ctf_image.physical_address_of_box_center_y) * (j - ctf_image.physical_address_of_box_center_y));
                    if ( radius_sq > minimum_radius_sq ) {
                        sum_ctf += ctf_image.real_values[pointer];
                        sum_ctf_squared += pow(ctf_image.real_values[pointer], 2);
                        sum_power += resampled_power_spectrum.real_values[pointer];
                        sum_power_squared += pow(resampled_power_spectrum.real_values[pointer], 2);
                        sum_ctf_power += ctf_image.real_values[pointer] * resampled_power_spectrum.real_values[pointer];
                        counter++;
                    }
                    pointer++;
                }
                pointer += ctf_image.padding_jump_value;
            }

            sum_ctf /= counter;
            sum_ctf_squared /= counter;
            sum_power /= counter;
            sum_power_squared /= counter;
            sum_ctf_power /= counter;
            correlation_coefficient = (sum_ctf_power - sum_ctf * sum_power) / sqrt(sum_ctf_squared - pow(sum_ctf, 2)) / sqrt(sum_power_squared - pow(sum_power, 2));

            break;

        // Fitting tilt axis, tilt angle, average defocus
        case 1:
            MyDebugAssertTrue(sub_section_dimension_x > 0 && sub_section_dimension_y > 0, "Error: sub_section_dimensions not set\n");

            minimum_radius_sq = powf(float(resampled_power_spectrum.logical_x_dimension) * tilt_fit_pixel_size / low_res_limit, 2);
            rotation_angle.GenerateRotationMatrix2D(input_values[1]);

            for ( iy = -(n_sections_y - 1) * n_steps; iy <= (n_sections_y - 1) * n_steps; iy += 2 ) {
                for ( ix = -(n_sections_x - 1) * n_steps; ix <= (n_sections_x - 1) * n_steps; ix += 2 ) {
                    if ( invalid_powerspectrum[section_counter] ) {
                        section_counter++;
                        continue;
                    }
                    x_coordinate_2d = float(ix) / 2.0 * sub_section_dimension_x / float(n_steps) * tilt_fit_pixel_size;
                    y_coordinate_2d = float(iy) / 2.0 * sub_section_dimension_y / float(n_steps) * tilt_fit_pixel_size;
                    rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);
                    height = y_rotated * tanf(deg_2_rad(input_values[2]));
                    ;

                    average_defocus = input_values[3] + height;
                    ddefocus        = average_defocus - (defocus_1 + defocus_2) / 2.0f;
                    input_ctf.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast, defocus_1 + ddefocus, defocus_2 + ddefocus, astigmatic_angle, tilt_fit_pixel_size, additional_phase_shift_in_radians);
                    ctf_transform.CalculateCTFImage(input_ctf);
                    ctf_transform.ComputeAmplitudeSpectrumFull2D(&ctf_image);
                    //					ctf_image.QuickAndDirtyWriteSlice("ctf_images.mrc", 1 + section_counter);
                    for ( i = 0; i < ctf_image.real_memory_allocated; i++ )
                        ctf_image.real_values[i] = powf(ctf_image.real_values[i], 2);

                    pointer = 0;
                    for ( j = 0; j < ctf_image.logical_y_dimension; j++ ) {
                        for ( i = 0; i < ctf_image.logical_x_dimension; i++ ) {
                            radius_sq = float((i - ctf_image.physical_address_of_box_center_x) * (i - ctf_image.physical_address_of_box_center_x) + (j - ctf_image.physical_address_of_box_center_y) * (j - ctf_image.physical_address_of_box_center_y));
                            if ( radius_sq > minimum_radius_sq ) {
                                sum_ctf += ctf_image.real_values[pointer];
                                sum_ctf_squared += pow(ctf_image.real_values[pointer], 2);
                                sum_power += resampled_power_spectra[section_counter].real_values[pointer];
                                sum_power_squared += pow(resampled_power_spectra[section_counter].real_values[pointer], 2);
                                sum_ctf_power += ctf_image.real_values[pointer] * resampled_power_spectra[section_counter].real_values[pointer];
                                counter++;
                            }
                            pointer++;
                        }
                        pointer += ctf_image.padding_jump_value;
                    }
                    section_counter++;
                }
            }

            sum_ctf /= counter;
            sum_ctf_squared /= counter;
            sum_power /= counter;
            sum_power_squared /= counter;
            sum_ctf_power /= counter;
            correlation_coefficient = (sum_ctf_power - sum_ctf * sum_power) / sqrt(sum_ctf_squared - pow(sum_ctf, 2)) / sqrt(sum_power_squared - pow(sum_power, 2));
            // Penalize tilt angles larger than 65 deg
            // if ( fabsf(float(input_values[2])) > 65.0f )
            //     correlation_coefficient -= (fabsf(float(input_values[2])) - 65.0f) / 5.0f;
            if ( fabsf(float(input_values[2])) > 85.0f )
                correlation_coefficient -= (fabsf(float(input_values[2])) - 85.0f) / 5.0f;

            break;
    }

    return -correlation_coefficient;
}

double CTFTilt::ScoreValuesFixedDefocus(double input_values[]) {
    // 0 = ignore, stupid code conversion
    // 1 = tilt_axis
    // 2 = tilt_angle
    // 3 = average_defocus
    // wxPrintf("defocus1 defocus2 %f %f\n", defocus_1, defocus_2);

    int    i, j;
    int    ix, iy;
    long   pointer;
    long   counter         = 0;
    int    section_counter = 0;
    float  fraction_of_nonzero_pixels;
    float  minimum_radius_sq;
    float  radius_sq;
    float  average_defocus;
    float  original_average_defocus = (defocus_1 + defocus_2) / 2.0f;
    float  ddefocus;
    float  height;
    float  variance;
    float  x_coordinate_2d;
    float  y_coordinate_2d;
    float  x_rotated;
    float  y_rotated;
    double sum_ctf           = 0.0;
    double sum_power         = 0.0;
    double sum_ctf_squared   = 0.0;
    double sum_power_squared = 0.0;
    double sum_ctf_power     = 0.0;
    double correlation_coefficient;

    AnglesAndShifts rotation_angle;

    CTF input_ctf;

    MyDebugAssertTrue(sub_section_dimension_x > 0 && sub_section_dimension_y > 0, "Error: sub_section_dimensions not set\n");

    minimum_radius_sq = powf(float(resampled_power_spectrum.logical_x_dimension) * tilt_fit_pixel_size / low_res_limit, 2);
    rotation_angle.GenerateRotationMatrix2D(input_values[1]);

    for ( iy = -(n_sections_y - 1) * n_steps / 2; iy <= (n_sections_y - 1) * n_steps / 2; iy++ ) {
        for ( ix = -(n_sections_x - 1) * n_steps / 2; ix <= (n_sections_x - 1) * n_steps / 2; ix++ ) {
            if ( invalid_powerspectrum[section_counter] ) {
                section_counter++;
                continue;
            }
            x_coordinate_2d = float(ix) * sub_section_dimension_x / float(n_steps) * tilt_fit_pixel_size;
            y_coordinate_2d = float(iy) * sub_section_dimension_y / float(n_steps) * tilt_fit_pixel_size;
            rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_rotated, y_rotated);
            height = y_rotated * tanf(deg_2_rad(input_values[2]));
            ;

            average_defocus = original_average_defocus + height;
            ddefocus        = average_defocus - (defocus_1 + defocus_2) / 2.0f;
            input_ctf.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast, defocus_1 + ddefocus, defocus_2 + ddefocus, astigmatic_angle, tilt_fit_pixel_size, additional_phase_shift_in_radians);
            ctf_transform.CalculateCTFImage(input_ctf);
            ctf_transform.ComputeAmplitudeSpectrumFull2D(&ctf_image);
            //					ctf_image.QuickAndDirtyWriteSlice("ctf_images.mrc", 1 + section_counter);
            for ( i = 0; i < ctf_image.real_memory_allocated; i++ )
                ctf_image.real_values[i] = powf(ctf_image.real_values[i], 2);

            pointer = 0;
            for ( j = 0; j < ctf_image.logical_y_dimension; j++ ) {
                for ( i = 0; i < ctf_image.logical_x_dimension; i++ ) {
                    radius_sq = float((i - ctf_image.physical_address_of_box_center_x) * (i - ctf_image.physical_address_of_box_center_x) + (j - ctf_image.physical_address_of_box_center_y) * (j - ctf_image.physical_address_of_box_center_y));
                    if ( radius_sq > minimum_radius_sq ) {
                        sum_ctf += ctf_image.real_values[pointer];
                        sum_ctf_squared += pow(ctf_image.real_values[pointer], 2);
                        sum_power += resampled_power_spectra[section_counter].real_values[pointer];
                        sum_power_squared += pow(resampled_power_spectra[section_counter].real_values[pointer], 2);
                        sum_ctf_power += ctf_image.real_values[pointer] * resampled_power_spectra[section_counter].real_values[pointer];
                        counter++;
                    }
                    pointer++;
                }
                pointer += ctf_image.padding_jump_value;
            }
            section_counter++;
        }
    }

    sum_ctf /= counter;
    sum_ctf_squared /= counter;
    sum_power /= counter;
    sum_power_squared /= counter;
    sum_ctf_power /= counter;
    correlation_coefficient = (sum_ctf_power - sum_ctf * sum_power) / sqrt(sum_ctf_squared - pow(sum_ctf, 2)) / sqrt(sum_power_squared - pow(sum_power, 2));
    // Penalize tilt angles larger than 65 deg
    // if ( fabsf(float(input_values[2])) > 65.0f )
    //     correlation_coefficient -= (fabsf(float(input_values[2])) - 65.0f) / 5.0f;
    if ( fabsf(float(input_values[2])) > 85.0f )
        correlation_coefficient -= (fabsf(float(input_values[2])) - 85.0f) / 5.0f;

    // break;
    // }

    return -correlation_coefficient;
}

double SampleTiltScoreFunctionForSimplex(void* pt2Object, double values[]) {
    CTFTilt* scorer_to_use = reinterpret_cast<CTFTilt*>(pt2Object);
    float    score         = scorer_to_use->ScoreValues(values);
    MyDebugPrint("%f, %f, %f, %f = %f\n", values[0], values[1], values[2], values[3], score);
    return score;
}

double SampleTiltScoreFunctionForSimplexTiltAxis(void* pt2Object, double values[]) {
    CTFTilt* scorer_to_use = reinterpret_cast<CTFTilt*>(pt2Object);
    // wxPrintf("average_defocus: %f\n",average_defocus);
    float score = scorer_to_use->ScoreValuesFixedDefocus(values);
    MyDebugPrint("%f, %f, %f = %f\n", values[0], values[1], values[2], score);
    return score;
}
