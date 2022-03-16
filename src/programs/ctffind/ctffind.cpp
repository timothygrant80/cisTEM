#include "../../core/core_headers.h"

//#define threshold_spectrum
#define use_epa_rather_than_zero_counting

// The timing that ctffind originally tracks is always on, by direct reference to cistem_timer::StopWatch
// The profiling for development is under conrtol of --enable-profiling.
#ifdef PROFILING
using namespace cistem_timer;
#else
using namespace cistem_timer_noop;
#endif

const std::string ctffind_version = "4.1.14";

/*
 * Changelog
 * - 4.1.14
 * -- bug fixes (memory, equiphase averaging)
 * -- bug fixes from David Mastronarde (fixed/known phase shift)
 * - 4.1.13
 * -- EPA bug fixed (affected 4.1.12 only)
 * - 4.1.12
 * -- diagnostic image includes radial "equi-phase" average of experimental spectrum in bottom right quadrant
 * -- new "equi-phase averaging" code will probably replace zero-counting eventually
 * -- bug fix (affected diagnostics for all 4.x): at very high resolution, Cs dominates and the phase aberration decreases
 * -- bug fix (affected 4.1.11): fitting was only done up to 5Å
 * -- slow, exhaustive search is no longer the default (since astigmatism-related bugs appear fixed)
 * -- Number of OpenMP threads defaults to 1 and can be set by:
 * --- using interactive user input (under expert options)
 * --- using the -j command-line option (overrides interactive user input)
 * -- printout timing information
 * - 4.1.11
 * -- speed-ups from David Mastronarde, including OpenMP threading of the exhaustive search
 * -- score is now a normalized cross-correlation coefficient (David Mastronarde)
 * - 4.1.10
 * -- astimatism-related bug fixes from David Mastronarde
 * - 4.1.9
 * -- FRC is between a re-normalized version of the amplitude spectrum, to emphasize phase of the Thon rings over their relative amplitudes
 * -- tweaked criteria for "fit resolution"
 * -- tweaked FRC computation
 * -- astigmatism restraint is off by default
 * -- fixed bug affecting astigmatism 
 * -- tweaked background subtraction (thanks Niko!) - helps with noisy VPP spectra
 */

class
        CtffindApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void AddCommandLineOptions( );

  private:
};

double SampleTiltScoreFunctionForSimplex(void* pt2Object, double values[]);
//TODO: gives this function a good name that describes what it actually does (it actually rescales the power spectrum!)
float PixelSizeForFitting(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                          int box_size, Image* current_power_spectrum, Image* resampled_power_spectrum, bool do_resampling = true, float stretch_factor = 1.0f);

class CTFTilt {
    int   refine_mode;
    int   micrograph_subregion_dimension;
    int   micrograph_binned_dimension_for_ctf;
    int   micrograph_square_dimension;
    int   sub_section_dimension_x;
    int   sub_section_dimension_y;
    int   box_size;
    int   n_sections;
    int   n_steps;
    int   box_convolution;
    int   input_image_x_dimension;
    int   input_image_y_dimension;
    int   image_buffer_counter;
    float tilt_binning_factor;
    float original_pixel_size;
    float ctf_fit_pixel_size;
    float tilt_fit_pixel_size;
    float low_res_limit;
    float acceleration_voltage_in_kV;
    float spherical_aberration_in_mm;
    float amplitude_contrast;
    float additional_phase_shift_in_radians;
    float high_res_limit_ctf_fit;
    float high_res_limit_tilt_fit;
    float minimum_defocus;
    float maximum_defocus;

    Image  input_image;
    Image  input_image_binned;
    Image  average_spectrum;
    Image  power_spectrum_binned_image;
    Image  power_spectrum_sub_section;
    Image  resampled_power_spectrum;
    Image  resampled_power_spectrum_binned_image;
    Image  average_power_spectrum;
    Image  ctf_transform;
    Image  ctf_image;
    Image  sub_section;
    Image* resampled_power_spectra;
    Image* input_image_buffer;

    bool rough_defocus_determined;
    bool defocus_astigmatism_determined;
    bool power_spectra_calculated;

  public:
    CTFTilt(ImageFile& wanted_input_file, float wanted_high_res_limit_ctf_fit, float wanted_high_res_limit_tilt_fit, float wanted_minimum_defocus, float wanted_maximum_defocus,
            float wanted_pixel_size, float wanted_acceleration_voltage_in_kV, float wanted_spherical_aberration_in_mm, float wanted_amplitude_contrast, float wanted_additional_phase_shift_in_radians);
    ~CTFTilt( );
    void   CalculatePowerSpectra(bool subtract_average = false);
    void   UpdateInputImage(Image* wanted_input_image);
    float  FindRoughDefocus( );
    float  FindDefocusAstigmatism( );
    float  SearchTiltAxisAndAngle( );
    float  RefineTiltAxisAndAngle( );
    float  CalculateTiltCorrectedSpectra(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                                         int box_size, Image* resampled_spectrum);
    double ScoreValues(double[]);

    float defocus_1;
    float defocus_2;
    float astigmatic_angle;
    float best_tilt_axis;
    float best_tilt_angle;
};

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    void AddCommandLineOptions( );

  private:
};

CTFTilt::CTFTilt(ImageFile& wanted_input_file, float wanted_high_res_limit_ctf_fit, float wanted_high_res_limit_tilt_fit, float wanted_minimum_defocus, float wanted_maximum_defocus,
                 float wanted_pixel_size, float wanted_acceleration_voltage_in_kV, float wanted_spherical_aberration_in_mm, float wanted_amplitude_contrast, float wanted_additional_phase_shift_in_radians) {
    box_size                       = 128;
    n_sections                     = 3;
    n_steps                        = 4;
    rough_defocus_determined       = false;
    defocus_astigmatism_determined = false;
    power_spectra_calculated       = false;

    // Must be odd
    box_convolution = 55;

    // Set later
    sub_section_dimension_x = 0;
    sub_section_dimension_y = 0;

    micrograph_square_dimension = std::max(wanted_input_file.ReturnXSize( ), wanted_input_file.ReturnYSize( ));
    if ( IsOdd((micrograph_square_dimension)) )
        micrograph_square_dimension++;
    input_image_x_dimension = micrograph_square_dimension;
    input_image_y_dimension = micrograph_square_dimension;
    input_image_buffer      = new Image[wanted_input_file.ReturnZSize( )];
    image_buffer_counter    = 0;

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
    micrograph_subregion_dimension = ReturnClosestFactorizedLower(micrograph_subregion_dimension, 5);
    if ( micrograph_subregion_dimension < micrograph_binned_dimension_for_ctf )
        power_spectrum_binned_image.Allocate(micrograph_subregion_dimension, micrograph_subregion_dimension, 1);
    else
        power_spectrum_binned_image.Allocate(micrograph_binned_dimension_for_ctf, micrograph_binned_dimension_for_ctf, 1);

    resampled_power_spectra = new Image[((n_sections - 1) * n_steps + 1) * ((n_sections - 1) * n_steps + 1)];
    tilt_binning_factor     = 0.5f * high_res_limit_tilt_fit / original_pixel_size;
    tilt_fit_pixel_size     = original_pixel_size * tilt_binning_factor;

    int ix, iy;
    int sub_section_dimension;
    int section_counter     = 0;
    sub_section_dimension_x = myroundint(input_image_x_dimension / tilt_binning_factor) / n_sections;
    if ( IsOdd(sub_section_dimension_x) )
        sub_section_dimension_x--;
    sub_section_dimension_y = myroundint(input_image_y_dimension / tilt_binning_factor) / n_sections;
    if ( IsOdd(sub_section_dimension_y) )
        sub_section_dimension_y--;
    sub_section_dimension = std::min(sub_section_dimension_x, sub_section_dimension_y);
    sub_section.Allocate(sub_section_dimension, sub_section_dimension, 1, true);
    power_spectrum_sub_section.Allocate(sub_section_dimension, sub_section_dimension, 1);

    for ( iy = -(n_sections - 1) * n_steps / 2; iy <= (n_sections - 1) * n_steps / 2; iy++ ) {
        for ( ix = -(n_sections - 1) * n_steps / 2; ix <= (n_sections - 1) * n_steps / 2; ix++ ) {
            resampled_power_spectra[section_counter].Allocate(box_size, box_size, 1, true);
            resampled_power_spectra[section_counter].SetToConstant(0.0f);
            section_counter++;
        }
    }
    //	CalculatePowerSpectra(true);
}

CTFTilt::~CTFTilt( ) {
    delete[] input_image_buffer;
    delete[] resampled_power_spectra;
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

    for ( iy = -(n_sections - 1) * n_steps / 2; iy <= (n_sections - 1) * n_steps / 2; iy++ ) {
        for ( ix = -(n_sections - 1) * n_steps / 2; ix <= (n_sections - 1) * n_steps / 2; ix++ ) {
            //			pointer_to_original_image->QuickAndDirtyWriteSlice("binned_input_image.mrc", 1);
            input_image.ClipInto(&sub_section, 0.0f, false, 0.0f, float(ix) * sub_section_dimension_x / float(n_steps), float(iy) * sub_section_dimension_y / float(n_steps), 0);
            sub_section.CosineRectangularMask(0.9f * sub_section.physical_address_of_box_center_x, 0.9f * sub_section.physical_address_of_box_center_y, 0.0f, 0.1f * sub_section.logical_x_dimension);
            //			sub_section.QuickAndDirtyWriteSlice("sub_sections.mrc", 1 + section_counter);
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
    power_spectra_calculated = true;
}

void CTFTilt::UpdateInputImage(Image* wanted_input_image) {
    MyDebugAssertTrue(input_image_x_dimension == wanted_input_image->logical_x_dimension && input_image_y_dimension == wanted_input_image->logical_y_dimension, "Error: Image dimensions do not match\n");

    input_image_buffer[image_buffer_counter].Allocate(wanted_input_image->logical_x_dimension, wanted_input_image->logical_y_dimension, 1, true);
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

    float  average_defocus = (defocus_1 + defocus_2) / 2.0f;
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
    float  axis_step    = 10.0f;
    float  angle_step   = 5.0f;
    float  tilt_angle;
    float  tilt_axis;
    float  average_defocus = (defocus_1 + defocus_2) / 2.0f;
    double start_values[4];

    //	if (! power_spectra_calculated) CalculatePowerSpectra();

    refine_mode = 1;

    for ( tilt_angle = 0.0f; tilt_angle <= 60.0f; tilt_angle += angle_step ) {
        for ( tilt_axis = 0.0f; tilt_axis < 360.0f; tilt_axis += axis_step ) {
            start_values[1] = tilt_axis;
            start_values[2] = tilt_angle;
            start_values[3] = average_defocus;
            variance        = -ScoreValues(start_values);

            if ( variance > variance_max ) {
                variance_max    = variance;
                best_tilt_axis  = tilt_axis;
                best_tilt_angle = tilt_angle;
                //				wxPrintf("tilt axis, angle, var = %g %g %g\n", best_tilt_axis, best_tilt_angle, variance_max);
            }
        }
    }
    return variance_max;
}

float CTFTilt::RefineTiltAxisAndAngle( ) {
    MyDebugAssertTrue(power_spectra_calculated, "Error: Power spectra not calculated\n");
    MyDebugAssertTrue(defocus_astigmatism_determined, "Error: Defocus astigmatism not yet determined\n");

    DownhillSimplex simplex_minimzer(3);

    float  average_defocus = (defocus_1 + defocus_2) / 2.0f;
    double ranges[4];
    double start_values[4];
    double min_values[4];

    //	if (! power_spectra_calculated) CalculatePowerSpectra();

    refine_mode = 1;

    ranges[0] = 0.0f;
    ranges[1] = 40.0f;
    ranges[2] = 20.0f;
    ranges[3] = 1000.0f;

    start_values[0] = 0.0f;
    start_values[1] = best_tilt_axis;
    start_values[2] = best_tilt_angle;
    start_values[3] = average_defocus;

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

    n_sec = std::max(input_image_buffer[0].logical_x_dimension / resampled_spectrum->logical_x_dimension, input_image_buffer[0].logical_y_dimension / resampled_spectrum->logical_y_dimension);
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
    sub_section_dimension = resampled_spectrum->logical_x_dimension;
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
                padding_value = section.ReturnAverageOfRealValues( );
                section.CosineRectangularMask(0.9f * section.physical_address_of_box_center_x, 0.9f * section.physical_address_of_box_center_y,
                                              0.0f, 0.1f * section.logical_x_dimension, false, true, padding_value);
                //				sigma = sqrtf(section.ReturnVarianceOfRealValues());
                //				if (sigma > 0.0f) section.MultiplyByConstant(1.0f / sigma);
                //				section.QuickAndDirtyWriteSlice("sections.mrc", section_counter + 1);
                section.ForwardFFT( );
                section.complex_values[0] = 0.0f + I * 0.0f;
                section.ComputeAmplitudeSpectrumFull2D(&power_spectrum);
                //				power_spectrum.QuickAndDirtyWriteSlice("spectrum.mrc", section_counter + 1);
                //				section_counter++;
                //				for (i = 0; i < power_spectrum_sub_section.real_memory_allocated; i++) power_spectrum_sub_section.real_values[i] = powf(power_spectrum_sub_section.real_values[i], 2);

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

            for ( iy = -(n_sections - 1) * n_steps / 2; iy <= (n_sections - 1) * n_steps / 2; iy++ ) {
                for ( ix = -(n_sections - 1) * n_steps / 2; ix <= (n_sections - 1) * n_steps / 2; ix++ ) {
                    x_coordinate_2d = float(ix) * sub_section_dimension_x / float(n_steps) * tilt_fit_pixel_size;
                    y_coordinate_2d = float(iy) * sub_section_dimension_y / float(n_steps) * tilt_fit_pixel_size;
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
            if ( fabsf(float(input_values[2])) > 65.0f )
                correlation_coefficient -= (fabsf(float(input_values[2])) - 65.0f) / 5.0f;

            break;
    }

    return -correlation_coefficient;
}

double SampleTiltScoreFunctionForSimplex(void* pt2Object, double values[]) {
    CTFTilt* scorer_to_use = reinterpret_cast<CTFTilt*>(pt2Object);
    //	float score = scorer_to_use->ScoreValues(values);
    //	wxPrintf("%f, %f, %f, %f = %f\n", values[1], values[2], values[3], values[4], score);
    return scorer_to_use->ScoreValues(values);
}

float PixelSizeForFitting(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                          int box_size, Image* current_power_spectrum, Image* resampled_power_spectrum, bool do_resampling, float stretch_factor) {
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
        resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(temporary_box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - temporary_box_size * stretch_factor) > fabsf(stretched_dimension - 2 - temporary_box_size * stretch_factor) )
                    stretched_dimension -= 2;

                current_power_spectrum->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                current_power_spectrum->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(current_power_spectrum);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image * float(temporary_box_size) / float(box_size);
    }
    else {
        // The regular way (the input pixel size was large enough)
        resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - box_size * stretch_factor) > fabsf(stretched_dimension - 2 - box_size * stretch_factor) )
                    stretched_dimension -= 2;

                current_power_spectrum->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                current_power_spectrum->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(current_power_spectrum);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image;
    }

    return pixel_size_for_fitting;
}

class ImageCTFComparison {
  public:
    ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep);
    ~ImageCTFComparison( );
    void  SetImage(int wanted_image_number, Image* new_image);
    void  SetCTF(CTF new_ctf);
    CTF   ReturnCTF( );
    bool  AstigmatismIsKnown( );
    float ReturnKnownAstigmatism( );
    float ReturnKnownAstigmatismAngle( );
    bool  FindPhaseShift( );
    void  SetupQuickCorrelation( );

    int    number_of_images;
    Image* img; // Usually an amplitude spectrum, or an array of amplitude spectra
    int    number_to_correlate;
    double norm_image;
    double image_mean;
    float* azimuths;
    float* spatial_frequency_squared;
    int*   addresses;

  private:
    CTF   ctf;
    float pixel_size;
    bool  find_phase_shift;
    bool  astigmatism_is_known;
    float known_astigmatism;
    float known_astigmatism_angle;
    bool  fit_defocus_sweep;
};

class CurveCTFComparison {
  public:
    float* curve; // Usually the 1D rotational average of the amplitude spectrum of an image
    int    number_of_bins;
    float  reciprocal_pixel_size; // In reciprocal pixels
    CTF    ctf;
    bool   find_phase_shift;
};

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
    azimuths                  = NULL;
    spatial_frequency_squared = NULL;
    addresses                 = NULL;
    number_to_correlate       = 0;
    image_mean                = 0.0;
    norm_image                = 0.0;
}

ImageCTFComparison::~ImageCTFComparison( ) {
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        img[image_counter].Deallocate( );
    }
    delete[] img;
    delete[] azimuths;
    delete[] spatial_frequency_squared;
    delete[] addresses;
    number_to_correlate = 0;
}

void ImageCTFComparison::SetImage(int wanted_image_number, Image* new_image) {
    MyDebugAssertTrue(wanted_image_number >= 0 && wanted_image_number < number_of_images, "Wanted image number (%i) is out of bounds", wanted_image_number);
    img[wanted_image_number].CopyFrom(new_image);
}

void ImageCTFComparison::SetCTF(CTF new_ctf) {
    ctf = new_ctf;
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
    if ( comparison_object->FindPhaseShift( ) ) {
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
    my_ctf.SetDefocus(array_of_values[0], array_of_values[0], 0.0);
    if ( comparison_object->find_phase_shift ) {
        my_ctf.SetAdditionalPhaseShift(array_of_values[1]);
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

//#pragma GCC pop_options

float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* self, Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
void  ComputeImagesWithNumberOfExtremaAndCTFValues(CTF* ctf, Image* number_of_extrema, Image* ctf_values);
int   ReturnSpectrumBinNumber(int number_of_bins, float number_of_extrema_profile[], Image* number_of_extrema, long address, Image* ctf_values, float ctf_values_profile[]);
void  ComputeRotationalAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_renormalized[], float number_of_extrema_profile[], float ctf_values_profile[]);
void  ComputeEquiPhaseAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Curve* epa_pre_max, Curve* epa_post_max);
void  OverlayCTF(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* equiphase_average_pre_max, Curve* equiphase_average_post_max);
void  ComputeFRCBetween1DSpectrumAndFit(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[], int first_fit_bin);
void  RescaleSpectrumAndRotationalAverage(Image* spectrum, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit);
void  Renormalize1DSpectrumForFRC(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[]);
float ReturnAzimuthToUseFor1DPlots(CTF* ctf);

IMPLEMENT_APP(CtffindApp)

// override the DoInteractiveUserInput

void CtffindApp::DoInteractiveUserInput( ) {

    float lowest_allowed_minimum_resolution = 50.0;

    std::string input_filename                     = "/dev/null";
    bool        input_is_a_movie                   = false;
    int         number_of_frames_to_average        = 0;
    std::string output_diagnostic_filename         = "/dev/null";
    float       pixel_size                         = 0.0;
    float       acceleration_voltage               = 0.0;
    float       spherical_aberration               = 0.0;
    float       amplitude_contrast                 = 0.0;
    int         box_size                           = 0;
    float       minimum_resolution                 = 0.0;
    float       maximum_resolution                 = 0.0;
    float       minimum_defocus                    = 0.0;
    float       maximum_defocus                    = 0.0;
    float       defocus_search_step                = 0.0;
    bool        astigmatism_is_known               = false;
    float       known_astigmatism                  = 0.0;
    float       known_astigmatism_angle            = 0.0;
    bool        slower_search                      = false;
    bool        should_restrain_astigmatism        = false;
    float       astigmatism_tolerance              = 0.0;
    bool        find_additional_phase_shift        = false;
    bool        determine_tilt                     = false;
    float       minimum_additional_phase_shift     = 0.0;
    float       maximum_additional_phase_shift     = 0.0;
    float       additional_phase_shift_search_step = 0.0;
    bool        give_expert_options                = false;
    bool        resample_if_pixel_too_small        = false;
    bool        movie_is_gain_corrected            = false;
    bool        movie_is_dark_corrected;
    wxString    dark_filename;
    wxString    gain_filename                    = "/dev/null";
    bool        correct_movie_mag_distortion     = false;
    float       movie_mag_distortion_angle       = 0.0;
    float       movie_mag_distortion_major_scale = 1.0;
    float       movie_mag_distortion_minor_scale = 1.0;
    bool        defocus_is_known                 = false;
    float       known_defocus_1                  = 0.0;
    float       known_defocus_2                  = 0.0;
    float       known_phase_shift                = 0.0;
    int         desired_number_of_threads        = 1;
    int         eer_frames_per_image             = 0;
    int         eer_super_res_factor             = 1;

    // Things we need for old school input
    double     temp_double               = -1.0;
    long       temp_long                 = -1;
    float      xmag                      = -1;
    float      dstep                     = -1.0;
    int        token_counter             = -1;
    const bool old_school_input          = command_line_parser.FoundSwitch("old-school-input");
    const bool old_school_input_ctffind4 = command_line_parser.FoundSwitch("old-school-input-ctffind4");

    if ( old_school_input || old_school_input_ctffind4 ) {

        astigmatism_is_known        = false;
        known_astigmatism           = 0.0;
        known_astigmatism_angle     = 0.0;
        resample_if_pixel_too_small = true;
        movie_is_gain_corrected     = true;
        gain_filename               = "";
        movie_is_dark_corrected     = true;
        dark_filename               = "";

        char     buf[4096];
        wxString my_string;

        // Line 1
        std::cin.getline(buf, 4096);
        input_filename = buf;

        // Line 2
        std::cin.getline(buf, 4096);
        output_diagnostic_filename = buf;

        // Line 3
        std::cin.getline(buf, 4096);
        my_string = buf;
        wxStringTokenizer tokenizer(my_string, ",");
        if ( tokenizer.CountTokens( ) != 5 ) {
            SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 3 of input\n", tokenizer.CountTokens( ), 5));
            exit(-1);
        }
        token_counter = -1;
        while ( tokenizer.HasMoreTokens( ) ) {
            token_counter++;
            tokenizer.GetNextToken( ).ToDouble(&temp_double);
            switch ( token_counter ) {
                case 0:
                    spherical_aberration = float(temp_double);
                    break;
                case 1:
                    acceleration_voltage = float(temp_double);
                    break;
                case 2:
                    amplitude_contrast = float(temp_double);
                    break;
                case 3:
                    xmag = float(temp_double);
                    break;
                case 4:
                    dstep = float(temp_double);
                    break;
                default:
                    wxPrintf("Ooops - bad token number: %li\n", tokenizer.GetPosition( ));
                    MyDebugAssertTrue(false, "oops\n");
            }
        }
        pixel_size = dstep * 10000.0 / xmag;

        // Line 4
        std::cin.getline(buf, 4096);
        my_string = buf;
        tokenizer.SetString(my_string, ",");
        if ( tokenizer.CountTokens( ) != 7 ) {
            SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 4 of input\n", tokenizer.CountTokens( ), 7));
            exit(-1);
        }
        token_counter = -1;
        while ( tokenizer.HasMoreTokens( ) ) {
            token_counter++;
            switch ( token_counter ) {
                case 0:
                    tokenizer.GetNextToken( ).ToLong(&temp_long);
                    box_size = int(temp_long);
                    break;
                case 1:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    minimum_resolution = float(temp_double);
                    break;
                case 2:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    maximum_resolution = float(temp_double);
                    break;
                case 3:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    minimum_defocus = float(temp_double);
                    break;
                case 4:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    maximum_defocus = float(temp_double);
                    break;
                case 5:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    defocus_search_step = float(temp_double);
                    break;
                case 6:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    astigmatism_tolerance = float(temp_double);
                    break;
            }
        }
        // If we are getting dAst = 0.0, which is the default in Relion, the user probably
        // expects the ctffind3 behaviour, which is no restraint on astigmatism
        if ( astigmatism_tolerance == 0.0 )
            astigmatism_tolerance = -100.0;

        // Output for old-school users
        if ( is_running_locally ) {
            wxPrintf("\n CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]\n");
            wxPrintf("%5.1f%9.1f%8.2f%10.1f%9.3f\n\n", spherical_aberration, acceleration_voltage, amplitude_contrast, xmag, dstep);
        }

        // Extra lines of input
        if ( old_school_input_ctffind4 ) {
            // Line 5
            std::cin.getline(buf, 4096);
            my_string = buf;
            tokenizer.SetString(my_string, ",");
            if ( tokenizer.CountTokens( ) != 2 ) {
                SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 5 of input\n", tokenizer.CountTokens( ), 2));
                exit(-1);
            }
            while ( tokenizer.HasMoreTokens( ) ) {
                switch ( tokenizer.GetPosition( ) ) {
                    case 0:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        if ( int(temp_double) != 0 ) {
                            input_is_a_movie = true;
                        }
                        else {
                            input_is_a_movie = false;
                        }
                        break;
                    case 1:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        number_of_frames_to_average = 1;
                        if ( input_is_a_movie ) {
                            number_of_frames_to_average = int(temp_double);
                        }
                        break;
                }
            }

            // Line 6
            std::cin.getline(buf, 4096);
            my_string = buf;
            tokenizer.SetString(my_string, ",");
            if ( tokenizer.CountTokens( ) != 4 ) {
                SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 6 of input\n", tokenizer.CountTokens( ), 4));
                exit(-1);
            }
            while ( tokenizer.HasMoreTokens( ) ) {
                switch ( tokenizer.GetPosition( ) ) {
                    case 0:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        if ( int(temp_double) != 0 ) {
                            find_additional_phase_shift = true;
                        }
                        else {
                            find_additional_phase_shift = false;
                        }
                        break;
                    case 1:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        minimum_additional_phase_shift = 0.0;
                        if ( find_additional_phase_shift ) {
                            minimum_additional_phase_shift = float(temp_double);
                        }
                        break;
                    case 2:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        maximum_additional_phase_shift = 0.0;
                        if ( find_additional_phase_shift ) {
                            maximum_additional_phase_shift = float(temp_double);
                        }
                        break;
                    case 3:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        additional_phase_shift_search_step = 0.0;
                        if ( find_additional_phase_shift ) {
                            additional_phase_shift_search_step = float(temp_double);
                        }
                        break;
                }
            }
        } // end of old school ctffind4 input
        else {
            input_is_a_movie                   = false;
            find_additional_phase_shift        = false;
            minimum_additional_phase_shift     = 0.0;
            maximum_additional_phase_shift     = 0.0;
            additional_phase_shift_search_step = 0.0;
            number_of_frames_to_average        = 1;
        }

        // Do some argument checking on movie processing option
        MRCFile input_file(input_filename, false);
        if ( input_is_a_movie ) {
            if ( input_file.ReturnZSize( ) < number_of_frames_to_average ) {
                SendError(wxString::Format("Input stack has %i images, so you cannot average %i frames together\n", input_file.ReturnZSize( ), number_of_frames_to_average));
                ExitMainLoop( );
            }
        }
        else {
            // We're not doing movie processing
            if ( input_file.ReturnZSize( ) > 1 ) {
                SendError("Input stacks are only supported --old-school-input-ctffind4 if doing movie processing\n");
                ExitMainLoop( );
            }
        }

        if ( find_additional_phase_shift ) {
            if ( minimum_additional_phase_shift > maximum_additional_phase_shift ) {
                SendError(wxString::Format("Minimum phase shift (%f) cannot be greater than maximum phase shift (%f)\n", minimum_additional_phase_shift, maximum_additional_phase_shift));
                ExitMainLoop( );
            }
        }

        desired_number_of_threads = 1;

    } // end of test for old-school-input or old-school-input-ctffind4
    else {

        UserInput* my_input = new UserInput("Ctffind", ctffind_version);

        input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);

        ImageFile input_file;
        if ( FilenameExtensionMatches(input_filename, "eer") ) {
            // If the input file is EER format, we will find out how many EER frames to average per image only later, so for now we have no way of knowing the logical Z dimension, but we know it's a movie
            input_is_a_movie = true;
        }
        else {
            input_file.OpenFile(input_filename, false);
            if ( input_file.ReturnZSize( ) > 1 ) {
                input_is_a_movie = my_input->GetYesNoFromUser("Input is a movie (stack of frames)", "Answer yes if the input file is a stack of frames from a dose-fractionated movie. If not, each image will be processed separately", "No");
            }
            else {
                input_is_a_movie = false;
            }
        }

        if ( input_is_a_movie ) {
            number_of_frames_to_average = my_input->GetIntFromUser("Number of frames to average together", "If the number of electrons per frame is too low, there may be strong artefacts in the estimated power spectrum. This can be alleviated by averaging frames with each other in real space before computing their Fourier transforms", "1");
        }
        else {
            number_of_frames_to_average = 1;
        }

        output_diagnostic_filename = my_input->GetFilenameFromUser("Output diagnostic image file name", "Will contain the experimental power spectrum and the best CTF fit", "diagnostic_output.mrc", false);
        pixel_size                 = my_input->GetFloatFromUser("Pixel size", "In Angstroms", "1.0", 0.0);
        acceleration_voltage       = my_input->GetFloatFromUser("Acceleration voltage", "in kV", "300.0", 0.0);
        spherical_aberration       = my_input->GetFloatFromUser("Spherical aberration", "in mm", "2.70", 0.0);
        amplitude_contrast         = my_input->GetFloatFromUser("Amplitude contrast", "Fraction of amplitude contrast", "0.07", 0.0, 1.0);
        box_size                   = my_input->GetIntFromUser("Size of amplitude spectrum to compute", "in pixels", "512", 128);
        minimum_resolution         = my_input->GetFloatFromUser("Minimum resolution", "Lowest resolution used for fitting CTF (Angstroms)", "30.0", 0.0, lowest_allowed_minimum_resolution);
        maximum_resolution         = my_input->GetFloatFromUser("Maximum resolution", "Highest resolution used for fitting CTF (Angstroms)", "5.0", 0.0, minimum_resolution);
        minimum_defocus            = my_input->GetFloatFromUser("Minimum defocus", "Positive values for underfocus. Lowest value to search over (Angstroms)", "5000.0");
        maximum_defocus            = my_input->GetFloatFromUser("Maximum defocus", "Positive values for underfocus. Highest value to search over (Angstroms)", "50000.0", minimum_defocus);
        defocus_search_step        = my_input->GetFloatFromUser("Defocus search step", "Step size for defocus search (Angstroms)", "100.0", 1.0);
        astigmatism_is_known       = my_input->GetYesNoFromUser("Do you know what astigmatism is present?", "Answer yes if you already know how much astigmatism was present. If you answer no, the program will search for the astigmatism and astigmatism angle", "No");
        if ( astigmatism_is_known ) {
            slower_search = my_input->GetYesNoFromUser("Slower, more exhaustive search?", "Answer yes to use a slower exhaustive search against 2D spectra (rather than 1D radial averages) for the initial search", "No");
            ;
            should_restrain_astigmatism = false;
            astigmatism_tolerance       = -100.0;
            known_astigmatism           = my_input->GetFloatFromUser("Known astigmatism", "In Angstroms, the amount of astigmatism, defined as the difference between the defocus along the major and minor axes", "0.0", 0.0);
            known_astigmatism_angle     = my_input->GetFloatFromUser("Known astigmatism angle", "In degrees, the angle of astigmatism", "0.0");
        }
        else {
            slower_search               = my_input->GetYesNoFromUser("Slower, more exhaustive search?", "Answer yes if you expect very high astigmatism (say, greater than 1000A) or in tricky cases. In that case, a slower exhaustive search against 2D spectra (rather than 1D radial averages) will be used for the initial search", "No");
            should_restrain_astigmatism = my_input->GetYesNoFromUser("Use a restraint on astigmatism?", "If you answer yes, the CTF parameter search and refinement will penalise large astigmatism. You will specify the astigmatism tolerance in the next question. If you answer no, no such restraint will apply", "No");
            if ( should_restrain_astigmatism ) {
                astigmatism_tolerance = my_input->GetFloatFromUser("Expected (tolerated) astigmatism", "Astigmatism values much larger than this will be penalised (Angstroms). Give a negative value to turn off this restraint.", "200.0");
            }
            else {
                astigmatism_tolerance = -100.0; // a negative value here signals that we don't want any restraint on astigmatism
            }
        }

        find_additional_phase_shift = my_input->GetYesNoFromUser("Find additional phase shift?", "Input micrograph was recorded using a phase plate with variable phase shift, which you want to find", "No");

        if ( find_additional_phase_shift ) {
            minimum_additional_phase_shift     = my_input->GetFloatFromUser("Minimum phase shift (rad)", "Lower bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians", "0.0", -3.15, 3.15);
            maximum_additional_phase_shift     = my_input->GetFloatFromUser("Maximum phase shift (rad)", "Upper bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians", "3.15", minimum_additional_phase_shift, 3.15);
            additional_phase_shift_search_step = my_input->GetFloatFromUser("Phase shift search step", "Step size for phase shift search (radians)", "0.5", 0.0, maximum_additional_phase_shift - minimum_additional_phase_shift);
        }
        else {
            minimum_additional_phase_shift     = 0.0;
            maximum_additional_phase_shift     = 0.0;
            additional_phase_shift_search_step = 0.0;
        }

        // Currently, tilt determination only works when there is no additional phase shift
        if ( ! find_additional_phase_shift )
            determine_tilt = my_input->GetYesNoFromUser("Determine sample tilt?", "Answer yes if you tilted the sample and need to determine tilt axis and angle", "No");

        give_expert_options = my_input->GetYesNoFromUser("Do you want to set expert options?", "There are options which normally not changed, but can be accessed by answering yes here", "No");
        if ( give_expert_options ) {
            resample_if_pixel_too_small = my_input->GetYesNoFromUser("Resample micrograph if pixel size too small?", "When the pixel is too small, Thon rings appear very thin and near the origin of the spectrum, which can lead to suboptimal fitting. This options resamples micrographs to a more reasonable pixel size if needed", "Yes");
            if ( input_is_a_movie ) {
                movie_is_dark_corrected = my_input->GetYesNoFromUser("Movie is dark-subtracted?", "If the movie is not dark-subtracted you will need to provide a dark reference image", "Yes");
                if ( movie_is_dark_corrected ) {
                    dark_filename = "";
                }
                else {
                    dark_filename = my_input->GetFilenameFromUser("Dark image filename", "The filename of the dark reference image for the detector/camera", "dark.dm4", true);
                }

                movie_is_gain_corrected = my_input->GetYesNoFromUser("Movie is gain-corrected?", "If the movie is not gain-corrected, you will need to provide a gain reference image", "Yes");
                if ( movie_is_gain_corrected ) {
                    gain_filename = "";
                }
                else {
                    gain_filename = my_input->GetFilenameFromUser("Gain image filename", "The filename of the gain reference image for the detector/camera", "gain.dm4", true);
                }

                correct_movie_mag_distortion = my_input->GetYesNoFromUser("Correct Movie Mag. Distortion?", "If the movie has a mag distortion you can specify the parameters to correct it prior to estimation", "No");

                if ( correct_movie_mag_distortion == true ) {
                    movie_mag_distortion_angle       = my_input->GetFloatFromUser("Mag. distortion angle", "The angle of the distortion", "0.0");
                    movie_mag_distortion_major_scale = my_input->GetFloatFromUser("Mag. distortion major scale", "The scale factor along the major axis", "1.0");
                    movie_mag_distortion_minor_scale = my_input->GetFloatFromUser("Mag. distortion minor scale", "The scale factor along the minor axis", "1.0");
                    ;
                }
                else {
                    movie_mag_distortion_angle       = 0.0;
                    movie_mag_distortion_major_scale = 1.0;
                    movie_mag_distortion_minor_scale = 1.0;
                }

                if ( FilenameExtensionMatches(input_filename, "eer") ) {
                    eer_frames_per_image = my_input->GetIntFromUser("Number of EER frames per image", "If the input movie is in EER format, we will average EER frames together so that each frame image for alignment has a reasonable exposure", "25", 1);
                    eer_super_res_factor = my_input->GetIntFromUser("EER super resolution factor", "Choose between 1 (no supersampling), 2, or 4 (image pixel size will be 4 times smaller that the camera phyiscal pixel)", "1", 1, 4);
                }
                else {
                    eer_frames_per_image = 0;
                    eer_super_res_factor = 1;
                }
            }
            else {
                movie_is_gain_corrected = true;
                movie_is_dark_corrected = true;
                gain_filename           = "";
                dark_filename           = "";
                eer_frames_per_image    = 0;
                eer_super_res_factor    = 1;
            }
            defocus_is_known = my_input->GetYesNoFromUser("Do you already know the defocus?", "Answer yes if you already know the defocus and you just want to know the score or fit resolution. If you answer yes, the known astigmatism parameter specified eariler will be ignored", "No");
            if ( defocus_is_known ) {
                /*
				 * Right now, we don't support phase plate data for this. The proper way to do this would be to also ask whether phase shift is known.
				 * Another acceptable solution might be to say that if you know the defocus you must also know the phase shift (in other words, this would
				 * only be used to test the ctffind scoring function / diagnostics using given defocus parameters). Neither are implemented right now,
				 * because I don't need either.
				 */
                known_defocus_1         = my_input->GetFloatFromUser("Known defocus 1", "In Angstroms, the defocus along the first axis", "0.0");
                known_defocus_2         = my_input->GetFloatFromUser("Known defocus 2", "In Angstroms, the defocus along the second axis", "0.0");
                known_astigmatism_angle = my_input->GetFloatFromUser("Known astigmatism angle", "In degrees, the angle of astigmatism", "0.0");
                if ( find_additional_phase_shift ) {
                    known_phase_shift = my_input->GetFloatFromUser("Known phase shift (radians)", "In radians, the phase shift (from a phase plate presumably)", "0.0");
                }
            }
            else {
                known_defocus_1         = 0.0;
                known_defocus_2         = 0.0;
                known_astigmatism_angle = 0.0;
                known_phase_shift       = 0.0;
            }
            desired_number_of_threads = my_input->GetIntFromUser("Desired number of parallel threads", "The command-line option -j will override this", "1", 1);
        }
        else // expert options not supplied by user
        {
            resample_if_pixel_too_small = true;
            movie_is_gain_corrected     = true;
            movie_is_dark_corrected     = true;
            gain_filename               = "";
            dark_filename               = "";
            defocus_is_known            = false;
            desired_number_of_threads   = 1;
            eer_frames_per_image        = 0;
            eer_super_res_factor        = 1;
        }

        delete my_input;
    }

    //	my_current_job.Reset(39);
    my_current_job.ManualSetArguments("tbitffffifffffbfbfffbffbbsbsbfffbfffbiii", input_filename.c_str( ), //1
                                      input_is_a_movie,
                                      number_of_frames_to_average,
                                      output_diagnostic_filename.c_str( ),
                                      pixel_size,
                                      acceleration_voltage,
                                      spherical_aberration,
                                      amplitude_contrast,
                                      box_size,
                                      minimum_resolution, //10
                                      maximum_resolution,
                                      minimum_defocus,
                                      maximum_defocus,
                                      defocus_search_step,
                                      slower_search,
                                      astigmatism_tolerance,
                                      find_additional_phase_shift,
                                      minimum_additional_phase_shift,
                                      maximum_additional_phase_shift,
                                      additional_phase_shift_search_step, //20
                                      astigmatism_is_known,
                                      known_astigmatism,
                                      known_astigmatism_angle,
                                      resample_if_pixel_too_small,
                                      movie_is_gain_corrected,
                                      gain_filename.ToStdString( ).c_str( ),
                                      movie_is_dark_corrected,
                                      dark_filename.ToStdString( ).c_str( ),
                                      correct_movie_mag_distortion,
                                      movie_mag_distortion_angle, //30
                                      movie_mag_distortion_major_scale,
                                      movie_mag_distortion_minor_scale,
                                      defocus_is_known,
                                      known_defocus_1,
                                      known_defocus_2,
                                      known_phase_shift,
                                      determine_tilt,
                                      desired_number_of_threads,
                                      eer_frames_per_image,
                                      eer_super_res_factor);
}

// Optional command-line stuff
void CtffindApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("old-school-input", "Pretend this is ctffind3 (for compatibility with old scripts and programs)");
    command_line_parser.AddLongSwitch("old-school-input-ctffind4", "Accept parameters from stdin, like ctffind3, but with extra lines for ctffind4-specific options (movie processing and phase shift estimation");
    command_line_parser.AddLongSwitch("amplitude-spectrum-input", "The input image is an amplitude spectrum, not a real-space image");
    command_line_parser.AddLongSwitch("filtered-amplitude-spectrum-input", "The input image is filtered (background-subtracted) amplitude spectrum");
    command_line_parser.AddLongSwitch("fast", "Skip computation of fit statistics as well as spectrum contrast enhancement");
    command_line_parser.AddOption("j", "", "Desired number of threads. Overrides interactive user input. Is overriden by env var OMP_NUM_THREADS", wxCMD_LINE_VAL_NUMBER);
}

// override the do calculation method which will be what is actually run..

bool CtffindApp::DoCalculation( ) {

    cistem_timer::StopWatch ctffind_timing;
    ctffind_timing.start("Initialization");
    StopWatch profile_timing;
    // Arguments for this job

    const std::string input_filename                     = my_current_job.arguments[0].ReturnStringArgument( );
    const bool        input_is_a_movie                   = my_current_job.arguments[1].ReturnBoolArgument( );
    const int         number_of_frames_to_average        = my_current_job.arguments[2].ReturnIntegerArgument( );
    const std::string output_diagnostic_filename         = my_current_job.arguments[3].ReturnStringArgument( );
    float             pixel_size_of_input_image          = my_current_job.arguments[4].ReturnFloatArgument( ); // no longer const, as the mag distortion can change it.
    const float       acceleration_voltage               = my_current_job.arguments[5].ReturnFloatArgument( );
    const float       spherical_aberration               = my_current_job.arguments[6].ReturnFloatArgument( );
    const float       amplitude_contrast                 = my_current_job.arguments[7].ReturnFloatArgument( );
    const int         box_size                           = my_current_job.arguments[8].ReturnIntegerArgument( );
    const float       minimum_resolution                 = my_current_job.arguments[9].ReturnFloatArgument( );
    const float       maximum_resolution                 = my_current_job.arguments[10].ReturnFloatArgument( );
    const float       minimum_defocus                    = my_current_job.arguments[11].ReturnFloatArgument( );
    const float       maximum_defocus                    = my_current_job.arguments[12].ReturnFloatArgument( );
    const float       defocus_search_step                = my_current_job.arguments[13].ReturnFloatArgument( );
    const bool        slower_search                      = my_current_job.arguments[14].ReturnBoolArgument( );
    const float       astigmatism_tolerance              = my_current_job.arguments[15].ReturnFloatArgument( );
    const bool        find_additional_phase_shift        = my_current_job.arguments[16].ReturnBoolArgument( );
    const float       minimum_additional_phase_shift     = my_current_job.arguments[17].ReturnFloatArgument( );
    const float       maximum_additional_phase_shift     = my_current_job.arguments[18].ReturnFloatArgument( );
    const float       additional_phase_shift_search_step = my_current_job.arguments[19].ReturnFloatArgument( );
    const bool        astigmatism_is_known               = my_current_job.arguments[20].ReturnBoolArgument( );
    const float       known_astigmatism                  = my_current_job.arguments[21].ReturnFloatArgument( );
    const float       known_astigmatism_angle            = my_current_job.arguments[22].ReturnFloatArgument( );
    const bool        resample_if_pixel_too_small        = my_current_job.arguments[23].ReturnBoolArgument( );
    const bool        movie_is_gain_corrected            = my_current_job.arguments[24].ReturnBoolArgument( );
    const wxString    gain_filename                      = my_current_job.arguments[25].ReturnStringArgument( );
    const bool        movie_is_dark_corrected            = my_current_job.arguments[26].ReturnBoolArgument( );
    const wxString    dark_filename                      = my_current_job.arguments[27].ReturnStringArgument( );
    const bool        correct_movie_mag_distortion       = my_current_job.arguments[28].ReturnBoolArgument( );
    const float       movie_mag_distortion_angle         = my_current_job.arguments[29].ReturnFloatArgument( );
    const float       movie_mag_distortion_major_scale   = my_current_job.arguments[30].ReturnFloatArgument( );
    const float       movie_mag_distortion_minor_scale   = my_current_job.arguments[31].ReturnFloatArgument( );
    const bool        defocus_is_known                   = my_current_job.arguments[32].ReturnBoolArgument( );
    const float       known_defocus_1                    = my_current_job.arguments[33].ReturnFloatArgument( );
    const float       known_defocus_2                    = my_current_job.arguments[34].ReturnFloatArgument( );
    const float       known_phase_shift                  = my_current_job.arguments[35].ReturnFloatArgument( );
    const bool        determine_tilt                     = my_current_job.arguments[36].ReturnBoolArgument( );
    int               desired_number_of_threads          = my_current_job.arguments[37].ReturnIntegerArgument( );
    int               eer_frames_per_image               = my_current_job.arguments[38].ReturnIntegerArgument( );
    int               eer_super_res_factor               = my_current_job.arguments[39].ReturnIntegerArgument( );

    // if we are applying a mag distortion, it can change the pixel size, so do that here to make sure it is used forever onwards..

    if ( input_is_a_movie && correct_movie_mag_distortion ) {
        pixel_size_of_input_image = ReturnMagDistortionCorrectedPixelSize(pixel_size_of_input_image, movie_mag_distortion_major_scale, movie_mag_distortion_minor_scale);
    }

    // These variables will be set by command-line options
    const bool old_school_input                  = command_line_parser.FoundSwitch("old-school-input") || command_line_parser.FoundSwitch("old-school-input-ctffind4");
    const bool amplitude_spectrum_input          = command_line_parser.FoundSwitch("amplitude-spectrum-input");
    const bool filtered_amplitude_spectrum_input = command_line_parser.FoundSwitch("filtered-amplitude-spectrum-input");
    const bool compute_extra_stats               = ! command_line_parser.FoundSwitch("fast");
    const bool boost_ring_contrast               = ! command_line_parser.FoundSwitch("fast");
    long       command_line_desired_number_of_threads;
    if ( command_line_parser.Found("j", &command_line_desired_number_of_threads) ) {
        // Command-line argument overrides
        desired_number_of_threads = command_line_desired_number_of_threads;
    }

    // Resampling of input images to ensure that the pixel size isn't too small
    const float target_nyquist_after_resampling    = 2.8; // Angstroms
    const float target_pixel_size_after_resampling = 0.5 * target_nyquist_after_resampling;
    float       pixel_size_for_fitting             = pixel_size_of_input_image;
    int         temporary_box_size;

    // Maybe the user wants to hold the phase shift value (which they can do by giving the same value for min and max)
    const bool fixed_additional_phase_shift = fabs(maximum_additional_phase_shift - minimum_additional_phase_shift) < 0.01;

    // This could become a user-supplied parameter later - for now only for developers / expert users
    const bool follow_1d_search_with_local_2D_brute_force = false;

    // Initial search should be done only using up to that resolution, to improve radius of convergence
    const float maximum_resolution_for_initial_search = 5.0;

    // Debugging
    const bool dump_debug_files = false;

    /*
	 *  Scoring function
	 */
    float MyFunction(float[]);

    // Other variables
    int                 number_of_movie_frames;
    int                 number_of_micrographs;
    ImageFile           input_file;
    Image*              average_spectrum        = new Image( );
    Image*              average_spectrum_masked = new Image( );
    wxString            output_text_fn;
    ProgressBar*        my_progress_bar;
    NumericTextFile*    output_text;
    NumericTextFile*    output_text_avrot;
    int                 current_micrograph_number;
    int                 number_of_tiles_used;
    Image*              current_power_spectrum = new Image( );
    int                 current_first_frame_within_average;
    int                 current_frame_within_average;
    int                 current_input_location;
    Image*              current_input_image        = new Image( );
    Image*              current_input_image_square = new Image( );
    int                 micrograph_square_dimension;
    Image*              temp_image               = new Image( );
    Image*              sum_image                = new Image( );
    Image*              resampled_power_spectrum = new Image( );
    bool                resampling_is_necessary;
    CTF*                current_ctf = new CTF( );
    float               average, sigma;
    int                 convolution_box_size;
    ImageCTFComparison* comparison_object_2D;
    CurveCTFComparison  comparison_object_1D;
    float               estimated_astigmatism_angle;
    float               bf_halfrange[4];
    float               bf_midpoint[4];
    float               bf_stepsize[4];
    float               cg_starting_point[4];
    float               cg_accuracy[4];
    int                 number_of_search_dimensions;
    BruteForceSearch*   brute_force_search;
    int                 counter;
    ConjugateGradient*  conjugate_gradient_minimizer;
    int                 current_output_location;
    int                 number_of_bins_in_1d_spectra;
    Curve*              number_of_averaged_pixels                 = new Curve( );
    Curve*              rotational_average                        = new Curve( );
    Image*              number_of_extrema_image                   = new Image( );
    Image*              ctf_values_image                          = new Image( );
    double*             rotational_average_astig                  = NULL;
    double*             rotational_average_astig_renormalized     = NULL;
    double*             spatial_frequency                         = NULL;
    double*             spatial_frequency_in_reciprocal_angstroms = NULL;
    double*             rotational_average_astig_fit              = NULL;
    float*              number_of_extrema_profile                 = NULL;
    float*              ctf_values_profile                        = NULL;
    double*             fit_frc                                   = NULL;
    double*             fit_frc_sigma                             = NULL;
    MRCFile             output_diagnostic_file(output_diagnostic_filename, true);
    int                 last_bin_with_good_fit;
    double*             values_to_write_out = new double[7];
    float               best_score_after_initial_phase;
    int                 last_bin_without_aliasing;
    ImageFile           gain_file;
    Image*              gain = new Image( );
    ImageFile           dark_file;
    Image*              dark = new Image( );
    float               final_score;
    float               tilt_axis;
    float               tilt_angle;

    // Open the input file
    bool input_file_is_valid = input_file.OpenFile(input_filename, false, false, false, eer_super_res_factor, eer_frames_per_image);
    if ( ! input_file_is_valid ) {
        SendInfo(wxString::Format("Input movie %s seems to be corrupt. Ctffind results may not be meaningful.\n", input_filename));
    }
    else {
        wxPrintf("Input file looks OK, proceeding\n");
    }

    // Some argument checking
    if ( determine_tilt && find_additional_phase_shift ) {
        SendError(wxString::Format("Error: Finding additional phase shift and determining sample tilt cannot be active at the same time. Terminating."));
        ExitMainLoop( );
    }
    if ( determine_tilt && (amplitude_spectrum_input || filtered_amplitude_spectrum_input) ) {
        SendError(wxString::Format("Error: Determining sample tilt cannot be run with either amplitude-spectrum-input or filtered-amplitude-spectrum-input. Terminating."));
        DEBUG_ABORT; // THis applies to CLI not GUI
    }
    if ( minimum_resolution < maximum_resolution ) {
        SendError(wxString::Format("Error: Minimum resolution (%f) higher than maximum resolution (%f). Terminating.", minimum_resolution, maximum_resolution));
        ExitMainLoop( );
    }
    if ( minimum_defocus > maximum_defocus ) {
        SendError(wxString::Format("Minimum defocus must be less than maximum defocus. Terminating."));
        ExitMainLoop( );
    }

    // How many micrographs are we dealing with
    if ( input_is_a_movie ) {
        // We only support 1 movie per file
        number_of_movie_frames = input_file.ReturnZSize( );
        number_of_micrographs  = 1;
    }
    else {
        number_of_movie_frames = 1;
        number_of_micrographs  = input_file.ReturnZSize( );
    }

    if ( is_running_locally ) {
        // Print out information about input file
        input_file.PrintInfo( );
    }

    // Prepare the output text file
    output_text_fn = FilenameReplaceExtension(output_diagnostic_filename, "txt");

    if ( is_running_locally ) {
        output_text = new NumericTextFile(output_text_fn, OPEN_TO_WRITE, 7);

        // Print header to the output text file
        output_text->WriteCommentLine("# Output from CTFFind version %s, run on %s\n", ctffind_version.c_str( ), wxDateTime::Now( ).FormatISOCombined(' ').ToStdString( ).c_str( ));
        output_text->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n", input_filename.c_str( ), number_of_micrographs);
        output_text->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.2f mm ; amplitude contrast: %0.2f\n", pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast);
        output_text->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um\n", box_size, minimum_resolution, maximum_resolution, minimum_defocus, maximum_defocus);
        output_text->WriteCommentLine("# Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully\n");
    }

    // Prepare a text file with 1D rotational average spectra
    output_text_fn = FilenameAddSuffix(output_text_fn.ToStdString( ), "_avrot");

    if ( ! old_school_input && number_of_micrographs > 1 && is_running_locally ) {
        wxPrintf("Will estimate the CTF parameters for %i micrographs.\n", number_of_micrographs);
        wxPrintf("Results will be written to this file: %s\n", output_text->ReturnFilename( ));
        wxPrintf("\nEstimating CTF parameters...\n\n");
        my_progress_bar = new ProgressBar(number_of_micrographs);
    }

    ctffind_timing.lap("Initialization");
    ctffind_timing.start("Spectrum computation");

    // Prepare the dark/gain_reference
    if ( input_is_a_movie && ! movie_is_gain_corrected ) {
        profile_timing.start("Read gain reference");
        gain_file.OpenFile(gain_filename.ToStdString( ), false);
        gain->ReadSlice(&gain_file, 1);
        profile_timing.lap("Read gain reference");
    }

    if ( input_is_a_movie && ! movie_is_dark_corrected ) {
        profile_timing.start("Read dark reference");
        dark_file.OpenFile(dark_filename.ToStdString( ), false);
        dark->ReadSlice(&dark_file, 1);
        profile_timing.lap("Read dark reference");
    }

    // Prepare the average spectrum image
    average_spectrum->Allocate(box_size, box_size, true);

    // Loop over micrographs
    for ( current_micrograph_number = 1; current_micrograph_number <= number_of_micrographs; current_micrograph_number++ ) {
        if ( is_running_locally && (old_school_input || number_of_micrographs == 1) )
            wxPrintf("Working on micrograph %i of %i\n", current_micrograph_number, number_of_micrographs);

        number_of_tiles_used = 0;
        average_spectrum->SetToConstant(0.0);
        average_spectrum->is_in_real_space = true;

        if ( amplitude_spectrum_input || filtered_amplitude_spectrum_input ) {
            current_power_spectrum->ReadSlice(&input_file, current_micrograph_number);
            current_power_spectrum->ForwardFFT( );
            average_spectrum->Allocate(box_size, box_size, 1, false);
            current_power_spectrum->ClipInto(average_spectrum);
            average_spectrum->BackwardFFT( );
            average_spectrum_masked->CopyFrom(average_spectrum);
        }
        else {
            CTFTilt tilt_scorer(input_file, 5.0f, 10.0f, minimum_defocus, maximum_defocus, pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast, 0.0f);

            for ( current_first_frame_within_average = 1; current_first_frame_within_average <= number_of_movie_frames; current_first_frame_within_average += number_of_frames_to_average ) {
                for ( current_frame_within_average = 1; current_frame_within_average <= number_of_frames_to_average; current_frame_within_average++ ) {
                    current_input_location = current_first_frame_within_average + number_of_movie_frames * (current_micrograph_number - 1) + (current_frame_within_average - 1);
                    if ( current_input_location > number_of_movie_frames * current_micrograph_number )
                        continue;
                    // Read the image in
                    profile_timing.start("Read and check image");
                    current_input_image->ReadSlice(&input_file, current_input_location);
                    if ( current_input_image->IsConstant( ) ) {

                        if ( is_running_locally == false ) {
                            // don't crash, as this will lead to the gui job never finishing, instead send a blank result..
                            SendError(wxString::Format("Error: location %i of input file %s is blank, defocus parameters will be set to 0", current_input_location, input_filename));

                            float results_array[10];
                            results_array[0] = 0.0; // Defocus 1 (Angstroms)
                            results_array[1] = 0.0; // Defocus 2 (Angstroms)
                            results_array[2] = 0.0; // Astigmatism angle (degrees)
                            results_array[3] = 0.0; // Additional phase shift (e.g. from phase plate) (radians)
                            results_array[4] = 0.0; // CTFFIND score
                            results_array[5] = 0.0;
                            results_array[6] = 0.0;
                            results_array[7] = 0.0;
                            results_array[8] = 0.0;
                            results_array[9] = 0.0;

                            my_result.SetResult(10, results_array);

                            delete average_spectrum;
                            delete average_spectrum_masked;
                            delete current_power_spectrum;
                            delete current_input_image;
                            delete current_input_image_square;
                            delete temp_image;
                            delete sum_image;
                            delete resampled_power_spectrum;
                            delete number_of_extrema_image;
                            delete ctf_values_image;
                            delete gain;
                            delete dark;
                            delete[] values_to_write_out;

                            return true;
                        }
                        else {
                            SendError(wxString::Format("Error: location %i of input file %s is blank", current_input_location, input_filename));
                            ExitMainLoop( );
                        }
                    }
                    profile_timing.lap("Read and check image");

                    // Apply dark reference
                    if ( input_is_a_movie && ! movie_is_dark_corrected ) {
                        profile_timing.start("Apply dark");
                        if ( ! current_input_image->HasSameDimensionsAs(dark) ) {
                            SendError(wxString::Format("Error: location %i of input file %s does not have same dimensions as the dark image", current_input_location, input_filename));
                            ExitMainLoop( );
                        }

                        current_input_image->SubtractImage(dark);
                        profile_timing.lap("Apply dark");
                    }

                    // Apply gain reference
                    if ( input_is_a_movie && ! movie_is_gain_corrected ) {
                        profile_timing.start("Apply gain");
                        if ( ! current_input_image->HasSameDimensionsAs(gain) ) {
                            SendError(wxString::Format("Error: location %i of input file %s does not have same dimensions as the gain image", current_input_location, input_filename));
                            ExitMainLoop( );
                        }
                        current_input_image->MultiplyPixelWise(*gain);
                        profile_timing.lap("Apply gain");
                    }
                    // correct for mag distortion
                    if ( input_is_a_movie && correct_movie_mag_distortion ) {
                        profile_timing.start("Correct mag distortion");
                        current_input_image->CorrectMagnificationDistortion(movie_mag_distortion_angle, movie_mag_distortion_major_scale, movie_mag_distortion_minor_scale);
                        profile_timing.lap("Correct mag distortion");
                    }
                    // Make the image square
                    profile_timing.start("Crop image to shortest dimension");
                    micrograph_square_dimension = std::max(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension);
                    if ( IsOdd((micrograph_square_dimension)) )
                        micrograph_square_dimension++;
                    if ( current_input_image->logical_x_dimension != micrograph_square_dimension || current_input_image->logical_y_dimension != micrograph_square_dimension ) {
                        current_input_image_square->Allocate(micrograph_square_dimension, micrograph_square_dimension, true);
                        //current_input_image->ClipInto(current_input_image_square,current_input_image->ReturnAverageOfRealValues());
                        current_input_image->ClipIntoLargerRealSpace2D(current_input_image_square, current_input_image->ReturnAverageOfRealValues( ));
                        current_input_image->Consume(current_input_image_square);
                    }
                    profile_timing.lap("Crop image to shortest dimension");
                    //
                    profile_timing.start("Average frames");
                    if ( current_frame_within_average == 1 ) {
                        sum_image->Allocate(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension, true);
                        sum_image->SetToConstant(0.0);
                    }
                    sum_image->AddImage(current_input_image);
                    profile_timing.lap("Average frames");
                } // end of loop over frames to average together
                current_input_image->Consume(sum_image);

                // Taper the edges of the micrograph in real space, to lessen Gibbs artefacts
                // Introduces an artefact of its own, so it's not clear on balance whether tapering helps, especially with modern micrographs from good detectors
                //current_input_image->TaperEdges();

                number_of_tiles_used++;
                if ( determine_tilt ) {
                    //					wxPrintf("Read frame = %i\n", number_of_tiles_used);
                    tilt_scorer.UpdateInputImage(current_input_image);
                    if ( current_first_frame_within_average + number_of_frames_to_average > number_of_movie_frames )
                        tilt_scorer.CalculatePowerSpectra(true);
                    else
                        tilt_scorer.CalculatePowerSpectra( );
                }
                else {
                    profile_timing.start("Compute amplitude spectrum");
                    // Compute the amplitude spectrum
                    current_power_spectrum->Allocate(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension, true);
                    current_input_image->ForwardFFT(false);
                    current_input_image->ComputeAmplitudeSpectrumFull2D(current_power_spectrum);

                    //current_power_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_resampling.mrc",1);

                    // Set origin of amplitude spectrum to 0.0
                    current_power_spectrum->real_values[current_power_spectrum->ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum->physical_address_of_box_center_x, current_power_spectrum->physical_address_of_box_center_y, current_power_spectrum->physical_address_of_box_center_z)] = 0.0;

                    // Resample the amplitude spectrum
                    pixel_size_for_fitting = PixelSizeForFitting(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, current_power_spectrum, resampled_power_spectrum);

                    average_spectrum->AddImage(resampled_power_spectrum);
                    profile_timing.lap("Compute amplitude spectrum");
                }

            } // end of loop over movie frames

            if ( determine_tilt ) {
                // Find rough defocus
                tilt_scorer.FindRoughDefocus( );
                //	wxPrintf("\nFindRoughDefocus values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Find astigmatism
                tilt_scorer.FindDefocusAstigmatism( );
                //	wxPrintf("\nFindDefocusAstigmatism values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Search tilt axis and angle
                tilt_scorer.SearchTiltAxisAndAngle( );
                //	wxPrintf("\nSearchTiltAxisAndAngle values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Refine tilt axis and angle
                tilt_scorer.RefineTiltAxisAndAngle( );
                //	wxPrintf("\nRefineTiltAxisAndAngle values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                tilt_axis  = tilt_scorer.best_tilt_axis;
                tilt_angle = tilt_scorer.best_tilt_angle;
                if ( tilt_angle < 0.0f ) {
                    tilt_axis += 180.0f;
                    tilt_angle = -tilt_angle;
                }
                if ( tilt_axis > 360.0f ) {
                    tilt_axis -= 360.0f;
                }
                //				wxPrintf("Final values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g || %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                pixel_size_for_fitting = tilt_scorer.CalculateTiltCorrectedSpectra(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, average_spectrum);
                average_spectrum->MultiplyByConstant(float(number_of_tiles_used));
            }
            else {
                tilt_angle = 0.0f;
                tilt_axis  = 0.0f;
            }

            // We need to take care of the scaling of the FFTs, as well as the averaging of tiles
            if ( resampling_is_necessary ) {
                average_spectrum->MultiplyByConstant(1.0 / (float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension * current_power_spectrum->logical_x_dimension * current_power_spectrum->logical_y_dimension));
            }
            else {
                average_spectrum->MultiplyByConstant(1.0 / (float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension));
            }

        } // end of test of whether we were given amplitude spectra on input

        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_bg_sub.mrc",1);

        // Filter the amplitude spectrum, remove background
        if ( ! filtered_amplitude_spectrum_input ) {
            profile_timing.start("Filter spectrum");
            average_spectrum->ComputeFilteredAmplitudeSpectrumFull2D(average_spectrum_masked, current_power_spectrum, average, sigma, minimum_resolution, maximum_resolution, pixel_size_for_fitting);
            profile_timing.lap("Filter spectrum");
        }

        /*
		 *
		 *
		 * We now have a spectrum which we can use to fit CTFs
		 *
		 *
		 */
        ctffind_timing.lap("Spectrum computation");
        ctffind_timing.start("Parameter search");

        if ( dump_debug_files )
            average_spectrum->WriteSlicesAndFillHeader("dbg_spectrum_for_fitting.mrc", pixel_size_for_fitting);

#ifdef threshold_spectrum
        wxPrintf("DEBUG: thresholding spectrum\n");
        for ( counter = 0; counter < average_spectrum->real_memory_allocated; counter++ ) {
            average_spectrum->real_values[counter] = std::max(average_spectrum->real_values[counter], -0.0f);
            average_spectrum->real_values[counter] = std::min(average_spectrum->real_values[counter], 1.0f);
        }
        average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_thr.mrc", 1);
#endif

        // Set up the CTF object, initially limited to at most maximum_resolution_for_initial_search
        current_ctf->Init(acceleration_voltage, spherical_aberration, amplitude_contrast, minimum_defocus, minimum_defocus, 0.0, 1.0 / minimum_resolution, 1.0 / std::max(maximum_resolution, maximum_resolution_for_initial_search), astigmatism_tolerance, pixel_size_for_fitting, minimum_additional_phase_shift);
        current_ctf->SetDefocus(minimum_defocus / pixel_size_for_fitting, minimum_defocus / pixel_size_for_fitting, 0.0);
        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);

        // Set up the comparison object
        comparison_object_2D = new ImageCTFComparison(1, *current_ctf, pixel_size_for_fitting, find_additional_phase_shift && ! fixed_additional_phase_shift, astigmatism_is_known, known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf, false);
        comparison_object_2D->SetImage(0, average_spectrum_masked);
        comparison_object_2D->SetupQuickCorrelation( );

        if ( defocus_is_known ) {
            profile_timing.start("Calculate Score for known defocus");
            current_ctf->SetDefocus(known_defocus_1 / pixel_size_for_fitting, known_defocus_2 / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
            current_ctf->SetAdditionalPhaseShift(known_phase_shift);
            current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
            comparison_object_2D->SetCTF(*current_ctf);
            comparison_object_2D->SetupQuickCorrelation( );
            final_score = 0.0;
            final_score = comparison_object_2D->img[0].QuickCorrelationWithCTF(*current_ctf, comparison_object_2D->number_to_correlate, comparison_object_2D->norm_image, comparison_object_2D->image_mean, comparison_object_2D->addresses,
                                                                               comparison_object_2D->spatial_frequency_squared, comparison_object_2D->azimuths);
            profile_timing.lap("Calculate Score for known defocus");
        }
        else {

            if ( is_running_locally && old_school_input ) {
                wxPrintf("\nSEARCHING CTF PARAMETERS...\n");
            }

            // Let's look for the astigmatism angle first
            if ( astigmatism_is_known ) {
                estimated_astigmatism_angle = known_astigmatism_angle;
            }
            else {
                profile_timing.start("Estimate initial astigmatism");
                temp_image->CopyFrom(average_spectrum);
                temp_image->ApplyMirrorAlongY( );
                //temp_image.QuickAndDirtyWriteSlice("dbg_spec_y.mrc",1);
                estimated_astigmatism_angle = 0.5 * FindRotationalAlignmentBetweenTwoStacksOfImages(average_spectrum, temp_image, 1, 90.0, 5.0, pixel_size_for_fitting / minimum_resolution, pixel_size_for_fitting / std::max(maximum_resolution, maximum_resolution_for_initial_search));
                profile_timing.lap("Estimate initial astigmatism");
            }

            //MyDebugPrint ("Estimated astigmatism angle = %f degrees\n", estimated_astigmatism_angle);

            /*
			 * Initial brute-force search, in 1D (fast, but not as accurate)
			 */
            if ( ! slower_search ) {
                profile_timing.start("Setup 1D search");
                // 1D rotational average
                number_of_bins_in_1d_spectra = int(ceil(average_spectrum_masked->ReturnMaximumDiagonalRadius( )));
                rotational_average->SetupXAxis(0.0, float(number_of_bins_in_1d_spectra) * average_spectrum_masked->fourier_voxel_size_x, number_of_bins_in_1d_spectra);
                number_of_averaged_pixels->CopyFrom(rotational_average);
                average_spectrum_masked->Compute1DRotationalAverage(*rotational_average, *number_of_averaged_pixels, true);

                comparison_object_1D.ctf   = *current_ctf;
                comparison_object_1D.curve = new float[number_of_bins_in_1d_spectra];
                for ( counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
                    comparison_object_1D.curve[counter] = rotational_average->data_y[counter];
                }
                comparison_object_1D.find_phase_shift      = find_additional_phase_shift && ! fixed_additional_phase_shift;
                comparison_object_1D.number_of_bins        = number_of_bins_in_1d_spectra;
                comparison_object_1D.reciprocal_pixel_size = average_spectrum_masked->fourier_voxel_size_x;

                // We can now look for the defocus value
                bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[1];

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = additional_phase_shift_search_step;

                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 2;
                }
                else {
                    number_of_search_dimensions = 1;
                }

                // DNM: Do one-time set of phase shift for fixed value
                if ( find_additional_phase_shift && fixed_additional_phase_shift ) {
                    current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                }
                profile_timing.lap("Setup 1D search");
                profile_timing.start("Perform 1D search");
                // Actually run the BF search
                brute_force_search = new BruteForceSearch( );
                brute_force_search->Init(&CtffindCurveObjectiveFunction, &comparison_object_1D, number_of_search_dimensions, bf_midpoint, bf_halfrange, bf_stepsize, false, false, desired_number_of_threads);
                brute_force_search->Run( );

                /*
				wxPrintf("After 1D brute\n");
				wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
				wxPrintf("%12.2f%12.2f%12.2f%12.5f\n",brute_force_search->GetBestValue(0),brute_force_search->GetBestValue(0),0.0,brute_force_search->GetBestScore());
				wxPrintf("%12.2f%12.2f%12.2f%12.5f\n",brute_force_search->GetBestValue(0)*pixel_size_for_fitting,brute_force_search->GetBestValue(0)*pixel_size_for_fitting,0.0,brute_force_search->GetBestScore());
				*/

                /*
				 * We can now do a local optimization.
				 * The end point of the BF search is the beginning of the CG search, but we will want to use
				 * the full resolution range
				 */
                profile_timing.lap("Perform 1D search");
                profile_timing.start("Optimize 1D search");
                current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
                comparison_object_1D.ctf = *current_ctf;
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
                }
                cg_accuracy[0]               = 100.0;
                cg_accuracy[1]               = 0.05;
                conjugate_gradient_minimizer = new ConjugateGradient( );
                conjugate_gradient_minimizer->Init(&CtffindCurveObjectiveFunction, &comparison_object_1D, number_of_search_dimensions, cg_starting_point, cg_accuracy);
                conjugate_gradient_minimizer->Run( );
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
                }
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0], estimated_astigmatism_angle / 180.0 * PIf);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                    }
                }

                // Remember the best score so far
                best_score_after_initial_phase = -conjugate_gradient_minimizer->GetBestScore( );

                // Cleanup
                delete conjugate_gradient_minimizer;
                delete brute_force_search;
                delete[] comparison_object_1D.curve;
                profile_timing.lap("Optimize 1D search");
            } // end of the fast search over the 1D function

            /*
			 * Brute-force search over the 2D scoring function.
			 * This is either the first search we are doing, or just a refinement
			 * starting from the result of the 1D search
			 */
            if ( slower_search || (! slower_search && follow_1d_search_with_local_2D_brute_force) ) {
                // Setup the parameters for the brute force search
                profile_timing.start("Setup 2D search");
                if ( slower_search ) // This is the first search we are doing - scan the entire range the user specified
                {

                    if ( astigmatism_is_known ) {
                        bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                        bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                        bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                        bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = additional_phase_shift_search_step;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 2;
                        }
                        else {
                            number_of_search_dimensions = 1;
                        }
                    }
                    else {
                        bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                        bf_halfrange[1] = bf_halfrange[0];
                        bf_halfrange[2] = 0.0;
                        bf_halfrange[3] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                        bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                        bf_midpoint[1] = bf_midpoint[0];
                        bf_midpoint[2] = estimated_astigmatism_angle / 180.0 * PIf;
                        bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = bf_stepsize[0];
                        bf_stepsize[2] = 0.0;
                        bf_stepsize[3] = additional_phase_shift_search_step;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 4;
                        }
                        else {
                            number_of_search_dimensions = 3;
                        }
                    }
                }
                else // we will do a brute-force search near the result of the search over the 1D objective function
                {
                    if ( astigmatism_is_known ) {

                        bf_midpoint[0] = current_ctf->GetDefocus1( );
                        bf_midpoint[1] = current_ctf->GetAdditionalPhaseShift( );

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = additional_phase_shift_search_step;

                        bf_halfrange[0] = 2.0 * defocus_search_step / pixel_size_for_fitting + 0.1;
                        bf_halfrange[1] = 2.0 * additional_phase_shift_search_step + 0.01;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 2;
                        }
                        else {
                            number_of_search_dimensions = 1;
                        }
                    }
                    else {

                        bf_midpoint[0] = current_ctf->GetDefocus1( );
                        bf_midpoint[1] = current_ctf->GetDefocus2( );
                        bf_midpoint[2] = current_ctf->GetAstigmatismAzimuth( );
                        bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = bf_stepsize[0];
                        bf_stepsize[2] = 0.0;
                        bf_stepsize[3] = additional_phase_shift_search_step;

                        if ( astigmatism_tolerance > 0 ) {
                            bf_halfrange[0] = 2.0 * astigmatism_tolerance / pixel_size_for_fitting + 0.1;
                        }
                        else {
                            bf_halfrange[0] = 2.0 * defocus_search_step / pixel_size_for_fitting + 0.1;
                        }
                        bf_halfrange[1] = bf_halfrange[0];
                        bf_halfrange[2] = 0.0;
                        bf_halfrange[3] = 2.0 * additional_phase_shift_search_step + 0.01;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 4;
                        }
                        else {
                            number_of_search_dimensions = 3;
                        }
                    }
                }
                profile_timing.lap("Setup 2D search");
                profile_timing.start("Perform 2D search");
                // Actually run the BF search (we run a local minimizer at every grid point only if this is a refinement search following 1D search (otherwise the full brute-force search would get too long)
                brute_force_search = new BruteForceSearch( );
                brute_force_search->Init(&CtffindObjectiveFunction, comparison_object_2D, number_of_search_dimensions, bf_midpoint, bf_halfrange, bf_stepsize, ! slower_search, is_running_locally, desired_number_of_threads);
                brute_force_search->Run( );

                profile_timing.lap("Perform 2D search");
                profile_timing.start("Setup 2D search optimization");
                // If we did the slower exhaustive search as our first search, we want to update the max fit resolution from maximum_resolution_for_initial_search -> maximum_resolution.
                if ( slower_search ) {
                    current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
                }

                // The end point of the BF search is the beginning of the CG search
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
                }

                //
                if ( astigmatism_is_known ) {
                    current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
                    if ( find_additional_phase_shift ) {
                        if ( fixed_additional_phase_shift ) {
                            current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                        }
                        else {
                            current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                        }
                    }
                }
                else {
                    current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[1], cg_starting_point[2]);
                    if ( find_additional_phase_shift ) {
                        if ( fixed_additional_phase_shift ) {
                            current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                        }
                        else {
                            current_ctf->SetAdditionalPhaseShift(cg_starting_point[3]);
                        }
                    }
                }
                current_ctf->EnforceConvention( );

                // Remember the best score so far
                best_score_after_initial_phase = -brute_force_search->GetBestScore( );

                delete brute_force_search;
                profile_timing.lap("Setup 2D search optimization");
            }

            // Print out the results of brute force search
            //if (is_running_locally && old_school_input)
            {
                wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
                wxPrintf("%12.2f%12.2f%12.2f%12.5f\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf, best_score_after_initial_phase);
            }

            // Now we refine in the neighbourhood by using Powell's conjugate gradient algorithm
            if ( is_running_locally && old_school_input ) {
                wxPrintf("\nREFINING CTF PARAMETERS...\n");
                wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
            }

            profile_timing.start("Setup 2D search optimization");
            /*
			 * Set up the conjugate gradient minimization of the 2D scoring function
			 */
            if ( astigmatism_is_known ) {
                cg_starting_point[0] = current_ctf->GetDefocus1( );
                if ( find_additional_phase_shift )
                    cg_starting_point[1] = current_ctf->GetAdditionalPhaseShift( );
                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 2;
                }
                else {
                    number_of_search_dimensions = 1;
                }
                cg_accuracy[0] = 100.0;
                cg_accuracy[1] = 0.05;
            }
            else {
                cg_accuracy[0]       = 100.0; //TODO: try defocus_search_step  / pix_size_for_fitting / 10.0
                cg_accuracy[1]       = 100.0;
                cg_accuracy[2]       = 0.025;
                cg_accuracy[3]       = 0.05;
                cg_starting_point[0] = current_ctf->GetDefocus1( );
                cg_starting_point[1] = current_ctf->GetDefocus2( );
                if ( slower_search || (! slower_search && follow_1d_search_with_local_2D_brute_force) ) {
                    // we did a search against the 2D power spectrum so we have a better estimate
                    // of the astigmatism angle in the CTF object
                    cg_starting_point[2] = current_ctf->GetAstigmatismAzimuth( );
                }
                else {
                    // all we have right now is the guessed astigmatism angle from the mirror
                    // trick before any CTF fitting was even tried
                    cg_starting_point[2] = estimated_astigmatism_angle / 180.0 * PIf;
                }

                if ( find_additional_phase_shift )
                    cg_starting_point[3] = current_ctf->GetAdditionalPhaseShift( );
                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 4;
                }
                else {
                    number_of_search_dimensions = 3;
                }
            }
            profile_timing.lap("Setup 2D search optimization");
            // CG minimization
            profile_timing.start("Peform 2D search optimization");
            comparison_object_2D->SetCTF(*current_ctf);
            conjugate_gradient_minimizer = new ConjugateGradient( );
            conjugate_gradient_minimizer->Init(&CtffindObjectiveFunction, comparison_object_2D, number_of_search_dimensions, cg_starting_point, cg_accuracy);
            conjugate_gradient_minimizer->Run( );
            profile_timing.lap("Peform 2D search optimization");
            // Remember the results of the refinement
            for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
            }
            if ( astigmatism_is_known ) {
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                    }
                }
            }
            else {
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[1], cg_starting_point[2]);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[3]);
                    }
                }
            }
            current_ctf->EnforceConvention( );

            // Print results to the terminal
            if ( is_running_locally && old_school_input ) {
                wxPrintf("%12.2f%12.2f%12.2f%12.5f   Final Values\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf, -conjugate_gradient_minimizer->GetBestScore( ));
                if ( find_additional_phase_shift ) {
                    wxPrintf("Final phase shift = %0.3f radians\n", current_ctf->GetAdditionalPhaseShift( ));
                }
            }

            final_score = -conjugate_gradient_minimizer->GetBestScore( );
        } // End of test for defocus_is_known

        /*
		 * We're all done with our search & refinement of defocus and phase shift parameter values.
		 * Now onto diagnostics.
		 */
        ctffind_timing.lap("Parameter search");
        ctffind_timing.start("Diagnostics");
        profile_timing.start("Renormalize spectrum");
        // Generate diagnostic image
        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_diag_start.mrc", 1);
        current_output_location = current_micrograph_number;
        average_spectrum->AddConstant(-1.0 * average_spectrum->ReturnAverageOfRealValuesOnEdges( ));

        /*
		 *  Attempt some renormalisations - we want to do this over a range not affected by the central peak or strong Thon rings,
		 *  so as to emphasize the "regular" Thon rings
		 */
        float start_zero               = sqrtf(current_ctf->ReturnSquaredSpatialFrequencyOfAZero(3, current_ctf->GetAstigmatismAzimuth( ), true));
        float finish_zero              = sqrtf(current_ctf->ReturnSquaredSpatialFrequencyOfAZero(4, current_ctf->GetAstigmatismAzimuth( ), true));
        float normalization_radius_min = start_zero * average_spectrum->logical_x_dimension;
        float normalization_radius_max = finish_zero * average_spectrum->logical_x_dimension;

        if ( start_zero > current_ctf->GetHighestFrequencyForFitting( ) || start_zero < current_ctf->GetLowestFrequencyForFitting( ) || finish_zero > current_ctf->GetHighestFrequencyForFitting( ) || finish_zero < current_ctf->GetLowestFrequencyForFitting( ) ) {
            normalization_radius_max = current_ctf->GetHighestFrequencyForFitting( ) * average_spectrum->logical_x_dimension;
            normalization_radius_min = std::max(0.5f * normalization_radius_max, current_ctf->GetLowestFrequencyForFitting( ) * average_spectrum->logical_x_dimension);
        }

        MyDebugAssertTrue(normalization_radius_max > normalization_radius_min, "Bad values for min (%f) and max (%f) normalization radii\n");

        if ( normalization_radius_max - normalization_radius_min > 2.0 ) {
            average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                       normalization_radius_max,
                                                                       average, sigma);
            average_spectrum->CircleMask(5.0, true);
            average_spectrum->SetMaximumValueOnCentralCross(average);
            average_spectrum->SetMinimumAndMaximumValues(average - 4.0 * sigma, average + 4.0 * sigma);
            average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                       normalization_radius_max,
                                                                       average, sigma);
            average_spectrum->AddConstant(-1.0 * average);
            average_spectrum->MultiplyByConstant(1.0 / sigma);
            average_spectrum->AddConstant(average);
        }

        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_diag_1.mrc", 1);
        profile_timing.lap("Renormalize spectrum");
        profile_timing.start("Compute final 1D spectrum");
        // 1D rotational average
        number_of_bins_in_1d_spectra = int(ceil(average_spectrum->ReturnMaximumDiagonalRadius( )));
        rotational_average->SetupXAxis(0.0, float(number_of_bins_in_1d_spectra) * average_spectrum->fourier_voxel_size_x, number_of_bins_in_1d_spectra);
        rotational_average->ZeroYData( );
        //number_of_averaged_pixels.ZeroYData();
        number_of_averaged_pixels->CopyFrom(rotational_average);
        average_spectrum->Compute1DRotationalAverage(*rotational_average, *number_of_averaged_pixels, true);
        profile_timing.lap("Compute final 1D spectrum");
        // Rotational average, taking astigmatism into account
        Curve equiphase_average_pre_max;
        Curve equiphase_average_post_max;
        if ( compute_extra_stats ) {
            profile_timing.start("Compute EPA");
            number_of_extrema_image->Allocate(average_spectrum->logical_x_dimension, average_spectrum->logical_y_dimension, true);
            ctf_values_image->Allocate(average_spectrum->logical_x_dimension, average_spectrum->logical_y_dimension, true);
            spatial_frequency                     = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig              = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig_renormalized = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig_fit          = new double[number_of_bins_in_1d_spectra];
            number_of_extrema_profile             = new float[number_of_bins_in_1d_spectra];
            ctf_values_profile                    = new float[number_of_bins_in_1d_spectra];
            fit_frc                               = new double[number_of_bins_in_1d_spectra];
            fit_frc_sigma                         = new double[number_of_bins_in_1d_spectra];
            ComputeImagesWithNumberOfExtremaAndCTFValues(current_ctf, number_of_extrema_image, ctf_values_image);
            //ctf_values_image.QuickAndDirtyWriteSlice("dbg_ctf_values.mrc",1);
            if ( dump_debug_files ) {
                average_spectrum->QuickAndDirtyWriteSlice("dbg_spectrum_before_1dave.mrc", 1);
                number_of_extrema_image->QuickAndDirtyWriteSlice("dbg_num_extrema.mrc", 1);
            }
            ComputeRotationalAverageOfPowerSpectrum(average_spectrum, current_ctf, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, rotational_average_astig_fit, rotational_average_astig_renormalized, number_of_extrema_profile, ctf_values_profile);
#ifdef use_epa_rather_than_zero_counting
            ComputeEquiPhaseAverageOfPowerSpectrum(average_spectrum, current_ctf, &equiphase_average_pre_max, &equiphase_average_post_max);
            // Replace the old curve with EPA values
            {
                float current_sq_sf;
                float azimuth_for_1d_plots         = ReturnAzimuthToUseFor1DPlots(current_ctf);
                float defocus_for_1d_plots         = current_ctf->DefocusGivenAzimuth(azimuth_for_1d_plots);
                float sq_sf_of_phase_shift_maximum = current_ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(defocus_for_1d_plots);
                for ( counter = 1; counter < number_of_bins_in_1d_spectra; counter++ ) {
                    current_sq_sf = powf(spatial_frequency[counter], 2);
                    if ( current_sq_sf <= sq_sf_of_phase_shift_maximum ) {
                        rotational_average_astig[counter] = equiphase_average_pre_max.ReturnLinearInterpolationFromX(current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
                    }
                    else {
                        rotational_average_astig[counter] = equiphase_average_post_max.ReturnLinearInterpolationFromX(current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
                    }
                    rotational_average_astig_renormalized[counter] = rotational_average_astig[counter];
                }
                Renormalize1DSpectrumForFRC(number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, rotational_average_astig_fit, number_of_extrema_profile);
            }
#endif
            // Here, do FRC
            int first_fit_bin = 0;
            for ( int bin_counter = number_of_bins_in_1d_spectra - 1; bin_counter >= 0; bin_counter-- ) {
                if ( spatial_frequency[bin_counter] >= current_ctf->GetLowestFrequencyForFitting( ) )
                    first_fit_bin = bin_counter;
            }
            ComputeFRCBetween1DSpectrumAndFit(number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, rotational_average_astig_fit, number_of_extrema_profile, fit_frc, fit_frc_sigma, first_fit_bin);
            profile_timing.lap("Compute EPA");
            profile_timing.start("Detect antialiasing");
            // At what bin does CTF aliasing become problematic?
            last_bin_without_aliasing         = 0;
            int location_of_previous_extremum = 0;
            for ( counter = 1; counter < number_of_bins_in_1d_spectra; counter++ ) {
                if ( number_of_extrema_profile[counter] - number_of_extrema_profile[counter - 1] >= 0.9 ) {
                    // We just reached a new extremum
                    if ( counter - location_of_previous_extremum < 4 ) {
                        last_bin_without_aliasing = location_of_previous_extremum;
                        break;
                    }
                    location_of_previous_extremum = counter;
                }
            }
            if ( is_running_locally && old_school_input && last_bin_without_aliasing != 0 ) {
                wxPrintf("CTF aliasing apparent from %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
            }
            profile_timing.lap("Detect antialiasing");
        }

        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_diag_2.mrc", 1);

        // Until what frequency were CTF rings detected?
        if ( compute_extra_stats ) {
            profile_timing.start("Compute resolution cutoff");
            static float low_threshold              = 0.1;
            static float frc_significance_threshold = 0.5; // In analogy to the usual criterion when comparing experimental results to the atomic model
            static float high_threshold             = 0.66;
            bool         at_last_bin_with_good_fit;
            int          number_of_bins_above_low_threshold          = 0;
            int          number_of_bins_above_significance_threshold = 0;
            int          number_of_bins_above_high_threshold         = 0;
            int          first_bin_to_check                          = 0.1 * number_of_bins_in_1d_spectra;
            MyDebugAssertTrue(first_bin_to_check >= 0 && first_bin_to_check < number_of_bins_in_1d_spectra, "Bad first bin to check\n");
            //wxPrintf("Will only check from bin %i of %i onwards\n", first_bin_to_check, number_of_bins_in_1d_spectra);
            last_bin_with_good_fit = -1;
            for ( counter = first_bin_to_check; counter < number_of_bins_in_1d_spectra; counter++ ) {
                //wxPrintf("On bin %i, fit_frc = %f, rot averate astig = %f\n", counter, fit_frc[counter], rotational_average_astig[counter]);
                at_last_bin_with_good_fit = ((number_of_bins_above_low_threshold > 3) && (fit_frc[counter] < low_threshold)) ||
                                            ((number_of_bins_above_high_threshold > 3) && (fit_frc[counter] < frc_significance_threshold));
                if ( at_last_bin_with_good_fit ) {
                    last_bin_with_good_fit = counter;
                    break;
                }
                // Count number of bins above given thresholds
                if ( fit_frc[counter] > low_threshold )
                    number_of_bins_above_low_threshold++;
                if ( fit_frc[counter] > frc_significance_threshold )
                    number_of_bins_above_significance_threshold++;
                if ( fit_frc[counter] > high_threshold )
                    number_of_bins_above_high_threshold++;
            }
            //wxPrintf("%i bins out of %i checked were above significance threshold\n",number_of_bins_above_significance_threshold,number_of_bins_in_1d_spectra-first_bin_to_check);
            if ( number_of_bins_above_significance_threshold == number_of_bins_in_1d_spectra - first_bin_to_check )
                last_bin_with_good_fit = number_of_bins_in_1d_spectra - 1;
            if ( number_of_bins_above_significance_threshold == 0 )
                last_bin_with_good_fit = 1;
            last_bin_with_good_fit = std::min(last_bin_with_good_fit, number_of_bins_in_1d_spectra);
            profile_timing.lap("Compute resolution cutoff");
        }
        else {
            last_bin_with_good_fit = 1;
        }
#ifdef DEBUG
        //MyDebugAssertTrue(last_bin_with_good_fit >= 0 && last_bin_with_good_fit < number_of_bins_in_1d_spectra,"Did not find last bin with good fit: %i", last_bin_with_good_fit);
        if ( ! (last_bin_with_good_fit >= 0 && last_bin_with_good_fit < number_of_bins_in_1d_spectra) ) {
            wxPrintf("WARNING: Did not find last bin with good fit: %i\n", last_bin_with_good_fit);
        }
#else
        if ( last_bin_with_good_fit < 1 && last_bin_with_good_fit >= number_of_bins_in_1d_spectra ) {
            last_bin_with_good_fit = 1;
        }
#endif

        // Prepare output diagnostic image
        //average_spectrum->AddConstant(- average_spectrum->ReturnAverageOfRealValuesOnEdges()); // this used to be done in OverlayCTF / CTFOperation in the Fortran code
        //average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_3.mrc",1);
        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_rescaling.mrc", 1);
        profile_timing.start("Write diagnostic image");
        if ( compute_extra_stats ) {
            RescaleSpectrumAndRotationalAverage(average_spectrum, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, rotational_average_astig_fit, number_of_extrema_profile, ctf_values_profile, last_bin_without_aliasing, last_bin_with_good_fit);
        }
        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_thresholding.mrc",1);

        normalization_radius_max = std::max(normalization_radius_max, float(average_spectrum->logical_x_dimension * spatial_frequency[last_bin_with_good_fit]));
        average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                   normalization_radius_max,
                                                                   average, sigma);

        average_spectrum->SetMinimumAndMaximumValues(average - sigma, average + 2.0 * sigma);

        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_overlay.mrc",1);
        OverlayCTF(average_spectrum, current_ctf, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, number_of_extrema_profile, ctf_values_profile, &equiphase_average_pre_max, &equiphase_average_post_max);

        average_spectrum->WriteSlice(&output_diagnostic_file, current_output_location);
        output_diagnostic_file.SetDensityStatistics(average_spectrum->ReturnMinimumValue( ), average_spectrum->ReturnMaximumValue( ), average_spectrum->ReturnAverageOfRealValues( ), 0.1);
        profile_timing.lap("Write diagnostic image");
        // Keep track of time
        ctffind_timing.lap("Diagnostics");

        // Print more detailed results to terminal
        if ( is_running_locally && number_of_micrographs == 1 ) {
            ctffind_timing.print_times( );
            profile_timing.print_times( );

            wxPrintf("\n\nEstimated defocus values        : %0.2f , %0.2f Angstroms\nEstimated azimuth of astigmatism: %0.2f degrees\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) / PIf * 180.0);
            if ( find_additional_phase_shift ) {
                wxPrintf("Additional phase shift          : %0.3f degrees (%0.3f radians) (%0.3f PIf)\n", current_ctf->GetAdditionalPhaseShift( ) / PIf * 180.0, current_ctf->GetAdditionalPhaseShift( ), current_ctf->GetAdditionalPhaseShift( ) / PIf);
            }
            if ( determine_tilt )
                wxPrintf("Tilt_axis, tilt angle           : %0.2f , %0.2f degrees\n", tilt_axis, tilt_angle);
            wxPrintf("Score                           : %0.5f\n", final_score);
            wxPrintf("Pixel size for fitting          : %0.3f Angstroms\n", pixel_size_for_fitting);
            if ( compute_extra_stats ) {
                wxPrintf("Thon rings with good fit up to  : %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit]);
                if ( last_bin_without_aliasing != 0 ) {
                    wxPrintf("CTF aliasing apparent from      : %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
                }
                else {
                    wxPrintf("Did not detect CTF aliasing\n");
                }
            }
        }

        // Warn the user if significant aliasing occurred within the fit range
        if ( compute_extra_stats && last_bin_without_aliasing != 0 && spatial_frequency[last_bin_without_aliasing] < current_ctf->GetHighestFrequencyForFitting( ) ) {
            if ( is_running_locally && number_of_micrographs == 1 ) {
                MyPrintfRed("Warning: CTF aliasing occurred within your CTF fitting range. Consider computing a larger spectrum (current size = %i).\n", box_size);
            }
            else {
                //SendInfo(wxString::Format("Warning: for image %s (location %i of %i), CTF aliasing occurred within the CTF fitting range. Consider computing a larger spectrum (current size = %i)\n",input_filename,current_micrograph_number, number_of_micrographs,box_size));
            }
        }

        if ( is_running_locally ) {
            // Write out results to a summary file
            values_to_write_out[0] = current_micrograph_number;
            values_to_write_out[1] = current_ctf->GetDefocus1( ) * pixel_size_for_fitting;
            values_to_write_out[2] = current_ctf->GetDefocus2( ) * pixel_size_for_fitting;
            values_to_write_out[3] = current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf;
            values_to_write_out[4] = current_ctf->GetAdditionalPhaseShift( );
            values_to_write_out[5] = final_score;
            if ( compute_extra_stats ) {
                values_to_write_out[6] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit];
            }
            else {
                values_to_write_out[6] = 0.0;
            }
            output_text->WriteLine(values_to_write_out);

            if ( (! old_school_input) && number_of_micrographs > 1 && is_running_locally )
                my_progress_bar->Update(current_micrograph_number);
        }

        // Write out avrot
        // TODO: add to the output a line with non-normalized avrot, so that users can check for things like ice crystal reflections
        if ( compute_extra_stats ) {
            if ( current_micrograph_number == 1 ) {
                output_text_avrot = new NumericTextFile(output_text_fn, OPEN_TO_WRITE, number_of_bins_in_1d_spectra);
                output_text_avrot->WriteCommentLine("# Output from CTFFind version %s, run on %s\n", ctffind_version.c_str( ), wxDateTime::Now( ).FormatISOCombined(' ').ToUTF8( ).data( ));
                output_text_avrot->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n", input_filename.c_str( ), number_of_micrographs);
                output_text_avrot->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.2f mm ; amplitude contrast: %0.2f\n", pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast);
                output_text_avrot->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um; num. frames averaged: %i\n", box_size, minimum_resolution, maximum_resolution, minimum_defocus, maximum_defocus, number_of_frames_to_average);
                output_text_avrot->WriteCommentLine("# 6 lines per micrograph: #1 - spatial frequency (1/Angstroms); #2 - 1D rotational average of spectrum (assuming no astigmatism); #3 - 1D rotational average of spectrum; #4 - CTF fit; #5 - cross-correlation between spectrum and CTF fit; #6 - 2sigma of expected cross correlation of noise\n");
            }
            spatial_frequency_in_reciprocal_angstroms = new double[number_of_bins_in_1d_spectra];
            for ( counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
                spatial_frequency_in_reciprocal_angstroms[counter] = spatial_frequency[counter] / pixel_size_for_fitting;
            }
            output_text_avrot->WriteLine(spatial_frequency_in_reciprocal_angstroms);
            output_text_avrot->WriteLine(rotational_average->data_y);
            output_text_avrot->WriteLine(rotational_average_astig);
            output_text_avrot->WriteLine(rotational_average_astig_fit);
            output_text_avrot->WriteLine(fit_frc);
            output_text_avrot->WriteLine(fit_frc_sigma);
            delete[] spatial_frequency_in_reciprocal_angstroms;
        }

        delete comparison_object_2D;

    } // End of loop over micrographs

    if ( is_running_locally && (! old_school_input) && number_of_micrographs > 1 ) {
        delete my_progress_bar;
        wxPrintf("\n");
    }

    // Tell the user where the outputs are
    if ( is_running_locally ) {

        wxPrintf("\n\nSummary of results                          : %s\n", output_text->ReturnFilename( ));
        wxPrintf("Diagnostic images                           : %s\n", output_diagnostic_filename);
        if ( compute_extra_stats ) {
            wxPrintf("Detailed results, including 1D fit profiles : %s\n", output_text_avrot->ReturnFilename( ));
            wxPrintf("Use this command to plot 1D fit profiles    : ctffind_plot_results.sh %s\n", output_text_avrot->ReturnFilename( ));
        }

        wxPrintf("\n\n");
    }

    // Send results back
    float results_array[10];
    results_array[0] = current_ctf->GetDefocus1( ) * pixel_size_for_fitting; // Defocus 1 (Angstroms)
    results_array[1] = current_ctf->GetDefocus2( ) * pixel_size_for_fitting; // Defocus 2 (Angstroms)
    results_array[2] = current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf; // Astigmatism angle (degrees)
    results_array[3] = current_ctf->GetAdditionalPhaseShift( ); // Additional phase shift (e.g. from phase plate) (radians)
    results_array[4] = final_score; // CTFFIND score

    if ( last_bin_with_good_fit == 0 ) {
        results_array[5] = 0.0; //	A value of 0.0 indicates that the calculation to determine the goodness of fit failed for some reason
    }
    else {
        results_array[5] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit]; //	The resolution (Angstroms) up to which Thon rings are well fit by the CTF
    }
    if ( last_bin_without_aliasing == 0 ) {
        results_array[6] = 0.0; // 	A value of 0.0 indicates that no aliasing was detected
    }
    else {
        results_array[6] = pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]; //	The resolution (Angstroms) at which aliasing was just detected
    }

    results_array[7] = average_spectrum->ReturnIcinessOfSpectrum(pixel_size_for_fitting);
    results_array[8] = tilt_angle;
    results_array[9] = tilt_axis;

    my_result.SetResult(10, results_array);

    // Cleanup
    delete current_ctf;
    delete average_spectrum;
    delete average_spectrum_masked;
    delete current_power_spectrum;
    delete current_input_image;
    delete current_input_image_square;
    delete temp_image;
    delete sum_image;
    delete resampled_power_spectrum;
    delete number_of_extrema_image;
    delete ctf_values_image;
    delete gain;
    delete dark;
    delete[] values_to_write_out;
    if ( is_running_locally )
        delete output_text;
    if ( compute_extra_stats ) {
        delete[] spatial_frequency;
        delete[] rotational_average_astig;
        delete[] rotational_average_astig_renormalized;
        delete[] rotational_average_astig_fit;
        delete[] number_of_extrema_profile;
        delete[] ctf_values_profile;
        delete[] fit_frc;
        delete[] fit_frc_sigma;
        delete output_text_avrot;
    }
    if ( ! defocus_is_known )
        delete conjugate_gradient_minimizer;

    delete number_of_averaged_pixels;
    delete rotational_average;

    // Return
    return true;
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
            MyDebugAssertTrue(spectrum_sigma > 0.0 && spectrum_sigma < 10000.0, "Bad spectrum_sigma: %f\n", spectrum_sigma);
            MyDebugAssertTrue(fit_sigma > 0.0 && fit_sigma < 10000.0, "Bad fit sigma: %f\n", fit_sigma);
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
void OverlayCTF(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* epa_pre_max, Curve* epa_post_max) {
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
