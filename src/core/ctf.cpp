#include "core_headers.h"

CTF::CTF( ) {
    spherical_aberration   = 0;
    wavelength             = 0;
    amplitude_contrast     = 0;
    defocus_1              = 0;
    defocus_2              = 0;
    defocus_half_range     = 0;
    astigmatism_azimuth    = 0;
    additional_phase_shift = 0;
    beam_tilt_x            = 0;
    beam_tilt_y            = 0;
    beam_tilt              = 0;
    beam_tilt_azimuth      = 0;
    particle_shift_x       = 0;
    particle_shift_y       = 0;
    particle_shift         = 0;
    sample_thickness       = 0;
    particle_shift_azimuth = 0;
    // Fitting parameters
    lowest_frequency_for_fitting    = 0;
    highest_frequency_for_fitting   = 0;
    astigmatism_tolerance           = 0;
    highest_frequency_with_good_fit = -1.0; // Avoid divide by 0 failure
    //
    precomputed_amplitude_contrast_term = 0;
    squared_wavelength                  = 0;
    cubed_wavelength                    = 0;

    squared_illumination_aperture = -1; // microradian
    squared_energy_half_width     = -1;
    low_resolution_contrast       = 0;
}

CTF::CTF(float wanted_acceleration_voltage, // keV
         float wanted_spherical_aberration, // mm
         float wanted_amplitude_contrast,
         float wanted_defocus_1_in_angstroms, // A
         float wanted_defocus_2_in_angstroms, // A
         float wanted_astigmatism_azimuth, // degrees
         float wanted_lowest_frequency_for_fitting, // 1/A
         float wanted_highest_frequency_for_fitting, // 1/A
         float wanted_astigmatism_tolerance, // A. Set to negative to indicate no restraint on astigmatism.
         float pixel_size, // A
         float wanted_additional_phase_shift_in_radians, // rad
         float wanted_beam_tilt_x_in_radians, // rad
         float wanted_beam_tilt_y_in_radians, // rad
         float wanted_particle_shift_x_in_angstroms, // A
         float wanted_particle_shift_y_in_angstroms, // A
         float wanted_sample_thickness_in_nm) // nm
{
    Init(wanted_acceleration_voltage, wanted_spherical_aberration, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth, wanted_lowest_frequency_for_fitting, wanted_highest_frequency_for_fitting, wanted_astigmatism_tolerance, pixel_size, wanted_additional_phase_shift_in_radians, wanted_beam_tilt_x_in_radians, wanted_beam_tilt_y_in_radians, wanted_particle_shift_x_in_angstroms, wanted_particle_shift_y_in_angstroms, wanted_sample_thickness_in_nm);
}

CTF::CTF(float wanted_acceleration_voltage, // keV
         float wanted_spherical_aberration, // mm
         float wanted_amplitude_contrast,
         float wanted_defocus_1_in_angstroms, // A
         float wanted_defocus_2_in_angstroms, // A
         float wanted_astigmatism_azimuth, // degrees
         float pixel_size, // A
         float wanted_additional_phase_shift_in_radians, // rad
         float wanted_beam_tilt_x_in_radians, // rad
         float wanted_beam_tilt_y_in_radians, // rad
         float wanted_particle_shift_x_in_angstroms, // A
         float wanted_particle_shift_y_in_angstroms, // A
         float wanted_samples_thickness_in_nm) // nm
{
    Init(wanted_acceleration_voltage, wanted_spherical_aberration, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth, 0.0, 1.0 / (2.0 * pixel_size), -10.0, pixel_size, wanted_additional_phase_shift_in_radians, wanted_beam_tilt_x_in_radians, wanted_beam_tilt_y_in_radians, wanted_particle_shift_x_in_angstroms, wanted_particle_shift_y_in_angstroms, wanted_samples_thickness_in_nm);
}

CTF::~CTF( ) {
    // Nothing to do
}

void CTF::Init(float wanted_acceleration_voltage_in_kV, // keV
               float wanted_spherical_aberration_in_mm, // mm
               float wanted_amplitude_contrast,
               float wanted_defocus_1_in_angstroms, // A
               float wanted_defocus_2_in_angstroms, // A
               float wanted_astigmatism_azimuth_in_degrees, // degrees
               float pixel_size_in_angstroms, // A
               float wanted_additional_phase_shift_in_radians, // rad
               float wanted_sample_thickness_in_nm) // nm
{
    Init(wanted_acceleration_voltage_in_kV, wanted_spherical_aberration_in_mm, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth_in_degrees, 0.0, 1.0 / (2.0 * pixel_size_in_angstroms), -10.0, pixel_size_in_angstroms, wanted_additional_phase_shift_in_radians, 0.0f, 0.0f, 0.0f, 0.0f, wanted_sample_thickness_in_nm);
}

// Initialise a CTF object
void CTF::Init(float wanted_acceleration_voltage_in_kV, // keV
               float wanted_spherical_aberration_in_mm, // mm
               float wanted_amplitude_contrast,
               float wanted_defocus_1_in_angstroms, // A
               float wanted_defocus_2_in_angstroms, //A
               float wanted_astigmatism_azimuth_in_degrees, // degrees
               float wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
               float wanted_highest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
               float wanted_astigmatism_tolerance_in_angstroms, // A. Set to negative to indicate no restraint on astigmatism.
               float pixel_size_in_angstroms, // A
               float wanted_additional_phase_shift_in_radians, // rad
               float wanted_beam_tilt_x_in_radians, // rad
               float wanted_beam_tilt_y_in_radians, // rad
               float wanted_particle_shift_x_in_angstroms, // A
               float wanted_particle_shift_y_in_angstroms, // A
               float wanted_sample_thickness_in_nm) // nm
{
    wavelength                    = ReturnWavelenthInAngstroms(wanted_acceleration_voltage_in_kV) / pixel_size_in_angstroms;
    squared_wavelength            = powf(wavelength, 2);
    cubed_wavelength              = powf(wavelength, 3);
    spherical_aberration          = wanted_spherical_aberration_in_mm * 10000000.0 / pixel_size_in_angstroms;
    amplitude_contrast            = wanted_amplitude_contrast;
    defocus_1                     = wanted_defocus_1_in_angstroms / pixel_size_in_angstroms;
    defocus_2                     = wanted_defocus_2_in_angstroms / pixel_size_in_angstroms;
    astigmatism_azimuth           = wanted_astigmatism_azimuth_in_degrees / 180.0 * PI;
    additional_phase_shift        = wanted_additional_phase_shift_in_radians;
    lowest_frequency_for_fitting  = wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
    highest_frequency_for_fitting = wanted_highest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
    astigmatism_tolerance         = wanted_astigmatism_tolerance_in_angstroms / pixel_size_in_angstroms;
    sample_thickness              = wanted_sample_thickness_in_nm * 10.0 / pixel_size_in_angstroms;

    // gcc catches the zero division somehow, but intel returns a nan. Needed for handling the real and complex terms of the CTF separately.
    if ( fabs(amplitude_contrast - 1.0) < 1e-3 )
        precomputed_amplitude_contrast_term = PI / 2.0f;
    else
        precomputed_amplitude_contrast_term = atanf(amplitude_contrast / sqrtf(1.0 - powf(amplitude_contrast, 2)));

    beam_tilt_x = wanted_beam_tilt_x_in_radians;
    beam_tilt_y = wanted_beam_tilt_y_in_radians;
    beam_tilt   = sqrtf(powf(beam_tilt_x, 2) + powf(beam_tilt_y, 2));
    if ( beam_tilt_x == 0.0f && beam_tilt_y == 0.0f ) {
        beam_tilt_azimuth = 0.0f;
    }
    else {
        beam_tilt_azimuth = atan2f(beam_tilt_y, beam_tilt_x);
    }
    particle_shift_x = wanted_particle_shift_x_in_angstroms / pixel_size_in_angstroms;
    particle_shift_y = wanted_particle_shift_y_in_angstroms / pixel_size_in_angstroms;
    particle_shift   = sqrtf(powf(particle_shift_x, 2) + powf(particle_shift_y, 2));
    if ( particle_shift_x == 0.0f && particle_shift_y == 0.0f ) {
        particle_shift_azimuth = 0.0f;
    }
    else {
        particle_shift_azimuth = atan2f(particle_shift_y, particle_shift_x);
    }
}

/*
 * Eqn 11 of Rohou & Grigorieff, modified by corrigendum of October 2018
 */
int CTF::ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(float squared_spatial_frequency, float azimuth) {

    int   number_of_extrema                     = 0;
    int   number_of_extrema_before_chi_extremum = 0;
    float sq_sf_of_chi_extremum                 = ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(azimuth);

    if ( squared_spatial_frequency <= sq_sf_of_chi_extremum ) {
        number_of_extrema = floor(1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth) + 0.5);
    }
    else {
        number_of_extrema_before_chi_extremum = floor(1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(sq_sf_of_chi_extremum, azimuth) + 0.5);
        number_of_extrema                     = floor(1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth) + 0.5);
        number_of_extrema                     = number_of_extrema_before_chi_extremum + abs(number_of_extrema - number_of_extrema_before_chi_extremum);
    }
    number_of_extrema = abs(number_of_extrema);
    MyDebugAssertTrue(number_of_extrema >= 0, "Bad number of extrema: %i (%i rounded from %f, phase shift = %f)\n", number_of_extrema,
                      int(floor(1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth) + 0.5)),
                      1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth) + 0.5,
                      PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth));
    return number_of_extrema;
}

/*
 * Compute the frequency of the Nth zero of the CTF.
 */
float CTF::ReturnSquaredSpatialFrequencyOfAZero(int which_zero, float azimuth, bool inaccurate_is_ok) {
    /*
	 * The method used below (ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth) makes assumptions which mean it will
	 * only return the correct spatial frequency for the CTF zeroes between the origin and the frequency at which
	 * the phase aberration peaks.
	 */
    float phase_shift;
    float defocus = DefocusGivenAzimuth(azimuth);
    if ( defocus > 0.0 ) {
        phase_shift = which_zero * PI;
    }
    else {
        phase_shift = -1.0 * which_zero * PI;
    }

    if ( ! inaccurate_is_ok ) {
        MyDebugAssertTrue(phase_shift <= ReturnPhaseAberrationMaximum( ), "Oops, this method only works for the first few zeroes");
    }
    return ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(phase_shift, azimuth);
}

/*
 * Return the maximum phase aberration anywhere on the spectrum
 */
float CTF::ReturnPhaseAberrationMaximum( ) {
    return std::max(ReturnPhaseAberrationMaximumGivenDefocus(defocus_1), ReturnPhaseAberrationMaximumGivenDefocus(defocus_2));
}

float CTF::ReturnPhaseAberrationMaximumGivenDefocus(float defocus) {
    return PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(defocus), defocus);
}

/*
 * Return the squared spatial frequency at which the phase aberration function reaches its extremum
 */
float CTF::ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(float azimuth) {
    return ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(DefocusGivenAzimuth(azimuth));
}

/*
 * Return the squared spatial frequency at which the phase aberration function reaches its extremum
 */
float CTF::ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(float defocus) {
    if ( defocus <= 0.0 ) {
        /*
		 * When the defocus value is negative (i.e. we are overfocus), the phase aberration
		 * quadratic decreases from sf 0.0 onwards. Mathematically, the parabola peaks in
		 * the negative squared spatial frequency. But there's no such thing as a negative squared
		 * sf.
		 */
        return 0.0;
    }
    else if ( spherical_aberration == 0.0 ) {
        return 9999.999;
    }
    else {
        return defocus / (squared_wavelength * spherical_aberration);
    }
}

/*
 * Return the squared spatial frequency at which a given phase aberration is obtained. Note that this will return only one of the
 * spatial frequencies, whereas the phase aberration function is a quadratic polynomial function of the squared
 * spatial frequency, which means that a given phase shift can be achieved at two spatial frequencies.
 *
 * Also, this function is written assuming underfocus and positive spherical aberration
 *
 * TODO: take into account multiple solutions
 */
float CTF::ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(float phase_shift, float azimuth) {
    const float a   = -0.5 * PI * cubed_wavelength * spherical_aberration;
    const float b   = PI * wavelength * DefocusGivenAzimuth(azimuth);
    const float c   = additional_phase_shift + precomputed_amplitude_contrast_term;
    const float det = powf(b, 2.0) - 4.0 * a * (c - phase_shift);

    if ( spherical_aberration == 0.0 ) {
        return (phase_shift - c) / b;
    }
    else {

        MyDebugAssertTrue(a != 0.0, "Bad values for either cubed wavelength (%f) or spherical aberration (%f) (a = %f)\n", cubed_wavelength, spherical_aberration, a);

        if ( det < 0.0 ) {
            //MyPrintWithDetails("Ooops, negative determinant\n");
            //DEBUG_ABORT;
            return 0.0;
        }
        else {
            const float solution_one = (-b + sqrtf(det)) / (2.0 * a);
            const float solution_two = (-b - sqrtf(det)) / (2.0 * a);

            if ( solution_one > 0 && solution_two > 0 ) {
                if ( solution_one <= solution_two ) {
                    return solution_one;
                }
                else {
                    return solution_two;
                }
            }
            else if ( solution_one > 0 ) {
                return solution_one;
            }
            else if ( solution_two > 0 )
                return solution_two;
            else {
#ifdef DEBUG
                MyPrintWithDetails("Ooops, did not find solutions to the phase aberration equation\n");
                DEBUG_ABORT;
#else
                return 0.0;
#endif
            }
        }
    }
}

// Set the envelope paramters
void CTF::SetEnvelope(float wanted_acceleration_voltage, float wanted_pixel_size_angstrom, float dose_rate) {

    // wanted_acceleration_voltage is in the lab reference frame (eg. 300)
    // dose_rate shold be in elec/Ang^2/s - so at 3EPS with  0.5A pix == 3/4

    float       chromatic_abberation;
    const float energy_spread   = 1.1f; // eV from Wim, for Krios
    const float beam_brightness = 12e7f; // xfeg 10-14e7 A/m^2/sr/Volt from Wim (sfeg 2-4)
    const float spherical_SI    = spherical_aberration / 1e10f * wanted_pixel_size_angstrom;

    squared_illumination_aperture = 16000.0f * dose_rate / (PIf * beam_brightness);
    if ( wanted_acceleration_voltage < 301 && wanted_acceleration_voltage > 299 ) {
        squared_illumination_aperture /= RELATIVISTIC_VOLTAGE_300;
        chromatic_abberation = spherical_SI * LORENTZ_FACTOR_300;
        // Aberration corrector adds 0.4mm @ 300KeV (via Wim)
        if ( spherical_SI < 1e-6 ) {
            chromatic_abberation += 2.52 * LORENTZ_FACTOR_300;
        }
        squared_energy_half_width = (1 + LORENTZ_FACTOR_300) / (1 + 0.5f * LORENTZ_FACTOR_300) * chromatic_abberation * energy_spread / (ELECTRON_REST_MASS + RELATIVISTIC_VOLTAGE_300);
    }
    else if ( wanted_acceleration_voltage < 201 && wanted_acceleration_voltage > 199 ) {
        squared_illumination_aperture /= RELATIVISTIC_VOLTAGE_200;
        chromatic_abberation = spherical_SI * LORENTZ_FACTOR_200;
        if ( spherical_SI < 1e-6 ) {
            chromatic_abberation += 2.52 * LORENTZ_FACTOR_200;
        }
        squared_energy_half_width = (1 + LORENTZ_FACTOR_200) / (1 + 0.5f * LORENTZ_FACTOR_200) * chromatic_abberation * energy_spread / (ELECTRON_REST_MASS + RELATIVISTIC_VOLTAGE_200);
    }
    else if ( wanted_acceleration_voltage < 101 && wanted_acceleration_voltage > 99 ) {
        squared_illumination_aperture /= RELATIVISTIC_VOLTAGE_100;
        chromatic_abberation = spherical_SI * LORENTZ_FACTOR_100;
        if ( spherical_SI < 1e-6 ) {
            chromatic_abberation += 2.52 * LORENTZ_FACTOR_100;
        }
        squared_energy_half_width = (1 + LORENTZ_FACTOR_100) / (1 + 0.5f * LORENTZ_FACTOR_100) * chromatic_abberation * energy_spread / (ELECTRON_REST_MASS + RELATIVISTIC_VOLTAGE_100);
    }
    else {
        wxPrintf("Error: Unsupported voltage (%f)\n\n", wanted_acceleration_voltage);
        DEBUG_ABORT;
    }

    // Everything in this class with units of length should be expressed normalized by the wanted pixel size
    squared_energy_half_width /= wanted_pixel_size_angstrom;
    squared_energy_half_width *= squared_energy_half_width;

    // These are the FWHM values. I want the 1/e half widths / 2sqrt(log(2)) = sqrt(4log(2)) = sqrt(log(2^4))
    squared_energy_half_width /= logf(16.0f);
    squared_illumination_aperture /= logf(16.0f);
}

// Set the defocus and astigmatism angle, given in pixels and radians
void CTF::SetDefocus(float wanted_defocus_1_pixels, float wanted_defocus_2_pixels, float wanted_astigmatism_angle_radians) {
    defocus_1           = wanted_defocus_1_pixels;
    defocus_2           = wanted_defocus_2_pixels;
    astigmatism_azimuth = wanted_astigmatism_angle_radians;
}

// Set the additional phase shift, given in radians
void CTF::SetAdditionalPhaseShift(float wanted_additional_phase_shift_radians) {
    additional_phase_shift = fmodf(wanted_additional_phase_shift_radians, (float)PI);
}

// Set the beam tilt, given in radians
void CTF::SetBeamTilt(float wanted_beam_tilt_x_in_radians, float wanted_beam_tilt_y_in_radians, float wanted_particle_shift_x_in_pixels, float wanted_particle_shift_y_in_pixels) {
    beam_tilt_x = wanted_beam_tilt_x_in_radians;
    beam_tilt_y = wanted_beam_tilt_y_in_radians;
    beam_tilt   = sqrtf(powf(beam_tilt_x, 2) + powf(beam_tilt_y, 2));
    if ( beam_tilt_x == 0.0f && beam_tilt_y == 0.0f ) {
        beam_tilt_azimuth = 0.0f;
    }
    else {
        beam_tilt_azimuth = atan2f(beam_tilt_y, beam_tilt_x);
    }
    particle_shift_x = wanted_particle_shift_x_in_pixels;
    particle_shift_y = wanted_particle_shift_y_in_pixels;
    particle_shift   = sqrtf(powf(particle_shift_x, 2) + powf(particle_shift_y, 2));
    if ( particle_shift_x == 0.0f && particle_shift_y == 0.0f ) {
        particle_shift_azimuth = 0.0f;
    }
    else {
        particle_shift_azimuth = atan2f(particle_shift_y, particle_shift_x);
    }
}

// Set the highest frequency used for fitting
void CTF::SetHighestFrequencyForFitting(float wanted_highest_frequency_in_reciprocal_pixels) {
    highest_frequency_for_fitting = wanted_highest_frequency_in_reciprocal_pixels;
}

void CTF::SetLowResolutionContrast(float wanted_low_resolution_contrast) {
    MyDebugAssertTrue(wanted_low_resolution_contrast >= 0.0f && wanted_low_resolution_contrast <= 1.0f, "Bad low_resolution_contrast: %f", wanted_low_resolution_contrast);
    low_resolution_contrast = asin(wanted_low_resolution_contrast);
}

void CTF::SetSampleThickness(float wanted_sample_thickness_in_pixels) {
    sample_thickness = wanted_sample_thickness_in_pixels;
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
std::complex<float> CTF::EvaluateComplex(float squared_spatial_frequency, float azimuth) {
    float phase_aberration = PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth);
    return -sinf(phase_aberration) - I * cosf(phase_aberration);

    // Frealign code:
    // CTF=CMPLX(WGH1*SCHI-WGH2*CCHI,-WGH1*CCHI-WGH2*SCHI)
    //	wxPrintf("cistem real  = %g\n", -sinf(chi + pre));
    //	wxPrintf("frealign real= %g\n", sqrtf(1.0 - powf(amplitude_contrast, 2)) * sinf(-chi) - amplitude_contrast * cosf(-chi));
    //	wxPrintf("cistem comp  = %g\n", -cosf(chi + pre));
    //	wxPrintf("frealign comp= %g\n", -sqrtf(1.0 - powf(amplitude_contrast, 2)) * cosf(-chi) - amplitude_contrast * sinf(-chi));
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
float CTF::Evaluate(float squared_spatial_frequency, float azimuth) {
    if ( defocus_1 == 0.0f && defocus_2 == 0.0f )
        return -0.7; // for defocus sweep
    else {
        if ( low_resolution_contrast == 0.0f )
            return -sinf(PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth));
        else {
            float low_res_limit = ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(azimuth);
            float phase_shift   = PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth);
            float threshold     = PI / 2.0f;
            if ( phase_shift >= threshold )
                return -sinf(phase_shift);
            else
                return -sinf(phase_shift + low_resolution_contrast * (threshold - phase_shift) / threshold);
        }
    }
}

// Return the value of the powerspectrum at the given squared spatial frequency and azimuth taken into account the sample thickness
// Formulas according to "TEM bright field imaging of thick specimens: nodes in
// Thon ring patterns" by Tichelaar, et.al. who got it from McMullan et al. (2015)

float CTF::EvaluatePowerspectrumWithThickness(float squared_spatial_frequency, float azimuth) {
    float phase_aberration = PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth);
    return 0.5f * (1 - IntegratedDefocusModulation(squared_spatial_frequency) * cosf(2 * phase_aberration));
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
float CTF::EvaluateWithEnvelope(float squared_spatial_frequency, float azimuth) {

    // Check that things are set
    if ( this->squared_energy_half_width == -1 || this->squared_illumination_aperture == -1 ) {
        wxPrintf("\nTo use EvaluateWithEnvelope, call SetEnvelope first\n");
        exit(-1);
    }
    // Don't get hung up on speed here: this can all be cleaned up FIXME
    float spatial_frequency = sqrtf(squared_spatial_frequency);

    float common_term = -1.0f * PISQf * squared_spatial_frequency / (2.0f * (1 + (2 * PISQf * squared_illumination_aperture * squared_spatial_frequency * squared_energy_half_width)));

    float envelope_term = expf(common_term * (squared_wavelength * squared_energy_half_width * squared_spatial_frequency +
                                              2.0f * squared_illumination_aperture *
                                                      powf(spherical_aberration * squared_wavelength * squared_spatial_frequency - 0.5f * (defocus_1 + defocus_2), 2)));

    return -sinf(PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency, azimuth)) * envelope_term;
}

/* returns the argument (radians) to the sine and cosine terms of the ctf
We follow the convention that underfocusing the objective lens
gives rise to a positive phase shift of scattered electrons, whereas the spherical aberration gives a
negative phase shift of scattered electrons.
Note that there is an additional (precomputed) term so that the CTF can then be computed by simply
taking the sine of the returned phase shift.
*/
float CTF::PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(float squared_spatial_frequency, float defocus) {
    return PIf * wavelength * squared_spatial_frequency * (defocus - 0.5f * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
}

float CTF::PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(float squared_spatial_frequency, float azimuth) {
    return PIf * wavelength * squared_spatial_frequency * (DefocusGivenAzimuth(azimuth) - 0.5f * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
}

std::complex<float> CTF::EvaluateBeamTiltPhaseShift(float squared_spatial_frequency, float azimuth) {
    MyDebugAssertTrue(squared_spatial_frequency >= 0.0f, "Bad squared spatial frequency: %f", squared_spatial_frequency);
    // Save some time if no beam tilt
    if ( beam_tilt == 0.0f && particle_shift == 0.0f ) {
        return 1.0f + I * 0.0f;
    }
    else {
        float phase_shift = PhaseShiftGivenBeamTiltAndShift(squared_spatial_frequency, BeamTiltGivenAzimuth(azimuth), ParticleShiftGivenAzimuth(azimuth));
        //		wxPrintf("p1 = %g\n", phase_shift);
        //		phase_shift = fmodf(phase_shift, 2.0f * (float)PI);
        //		if (phase_shift > PI) phase_shift -= 2.0f * PI;
        //		if (phase_shift <= -PI) phase_shift += 2.0f * PI;
        //		wxPrintf("p2 = %g\n", phase_shift);
        //		phase_shift = rad_2_deg(phase_shift);
        //		return phase_shift + I * 0.0f;
        // This should be exp(-I*phaseshift), but because our convention is to compute -1*sin( f - CS ) == sin( CS -f ) the phase shift carries an implicit *-1.0f.
        // This only affects the sin term here so swap (-) for (+)
        //		return cosf( phase_shift ) - I * sinf( phase_shift );
        return cosf(phase_shift) + I * sinf(phase_shift);
    }
}

// Return the phase shift generated by beam tilt
float CTF::PhaseShiftGivenBeamTiltAndShift(float squared_spatial_frequency, float beam_tilt, float particle_shift) {
    float spatial_frequency = sqrtf(squared_spatial_frequency);
    float phase_shift       = 2.0f * PIf * spherical_aberration * squared_wavelength * squared_spatial_frequency * spatial_frequency * beam_tilt;
    phase_shift -= 2.0f * PIf * spatial_frequency * particle_shift;
    return clamp_angular_range_negative_pi_to_pi(phase_shift);
}

// Return the effective defocus at the azimuth of interest
float CTF::DefocusGivenAzimuth(float azimuth) {
    return 0.5f * (defocus_1 + defocus_2 + cosf(2.0f * (azimuth - astigmatism_azimuth)) * (defocus_1 - defocus_2));
}

// Return the effective beam tilt at the azimuth of interest
float CTF::BeamTiltGivenAzimuth(float azimuth) {
    return beam_tilt * cosf(azimuth - beam_tilt_azimuth);
}

// Return the effective beam tilt at the azimuth of interest
float CTF::ParticleShiftGivenAzimuth(float azimuth) {
    return particle_shift * cosf(azimuth - particle_shift_azimuth);
}

// Compare two CTF objects and return true if they are within a specified defocus tolerance
bool CTF::IsAlmostEqualTo(CTF* wanted_ctf, float delta_defocus) {
    float delta;

    if ( fabsf(this->spherical_aberration - wanted_ctf->spherical_aberration) > 0.01 )
        return false;
    if ( fabsf(this->wavelength - wanted_ctf->wavelength) > 0.0001 )
        return false;
    if ( fabsf(this->amplitude_contrast - wanted_ctf->amplitude_contrast) > 0.0001 )
        return false;
    if ( fabsf(this->defocus_1 - wanted_ctf->defocus_1) > delta_defocus )
        return false;
    if ( fabsf(this->defocus_2 - wanted_ctf->defocus_2) > delta_defocus )
        return false;
    if ( fabsf(this->beam_tilt_x - wanted_ctf->beam_tilt_x) > 0.00001 )
        return false;
    if ( fabsf(this->beam_tilt_y - wanted_ctf->beam_tilt_y) > 0.00001 )
        return false;
    if ( fabsf(this->particle_shift_x - wanted_ctf->particle_shift_x) > 0.001 )
        return false;
    if ( fabsf(this->particle_shift_y - wanted_ctf->particle_shift_y) > 0.001 )
        return false;

    delta = fabsf(this->additional_phase_shift - wanted_ctf->additional_phase_shift);
    delta = fmodf(delta, 2.0f * (float)PI);
    // 0.0277 = 5/180 (5 deg tolerance)
    if ( delta > 0.0277 )
        return false;

    delta = fabsf(this->astigmatism_azimuth - wanted_ctf->astigmatism_azimuth);
    delta = fmodf(delta, (float)PI);
    // 0.0277 = 5/180 (5 deg tolerance)
    if ( delta > 0.0277 )
        return false;

    return true;
}

bool CTF::BeamTiltIsAlmostEqualTo(CTF* wanted_ctf, float delta_beam_tilt) {
    if ( fabsf(this->beam_tilt_x - wanted_ctf->beam_tilt_x) > delta_beam_tilt )
        return false;
    if ( fabsf(this->beam_tilt_y - wanted_ctf->beam_tilt_y) > delta_beam_tilt )
        return false;

    return true;
}

void CTF::CopyFrom(CTF other_ctf) {
    spherical_aberration   = other_ctf.spherical_aberration;
    wavelength             = other_ctf.wavelength;
    amplitude_contrast     = other_ctf.amplitude_contrast;
    defocus_1              = other_ctf.defocus_1;
    defocus_2              = other_ctf.defocus_2;
    defocus_half_range     = other_ctf.defocus_half_range;
    astigmatism_azimuth    = other_ctf.astigmatism_azimuth;
    additional_phase_shift = other_ctf.additional_phase_shift;
    // Fitting parameters
    lowest_frequency_for_fitting  = other_ctf.lowest_frequency_for_fitting;
    highest_frequency_for_fitting = other_ctf.highest_frequency_for_fitting;
    astigmatism_tolerance         = other_ctf.astigmatism_tolerance;
    //
    precomputed_amplitude_contrast_term = other_ctf.precomputed_amplitude_contrast_term;
    squared_wavelength                  = other_ctf.squared_wavelength;
    cubed_wavelength                    = other_ctf.cubed_wavelength;
}

// Enforce the convention that df1 > df2 and -90 < angast < 90
void CTF::EnforceConvention( ) {
    float defocus_tmp;

    if ( defocus_1 < defocus_2 ) {
        defocus_tmp = defocus_2;
        defocus_2   = defocus_1;
        defocus_1   = defocus_tmp;
        astigmatism_azimuth += PIf * 0.5f;
    }
    astigmatism_azimuth -= PIf * roundf(astigmatism_azimuth / PIf);
}

void CTF::ChangePixelSize(float old_pixel_size, float new_pixel_size) {
    float scale = old_pixel_size / new_pixel_size;

    spherical_aberration *= scale;
    wavelength *= scale;
    amplitude_contrast *= scale;
    defocus_1 *= scale;
    defocus_2 *= scale;
    defocus_half_range *= scale;

    // Fitting parameters
    lowest_frequency_for_fitting /= scale;
    highest_frequency_for_fitting /= scale;
    astigmatism_tolerance *= scale;
    //
    squared_wavelength = powf(wavelength, 2);
    cubed_wavelength   = powf(wavelength, 3);
}

float CTF::ReturnAzimuthToUseFor1DPlots( ) {
    const float min_angular_distances_from_axes_radians = 10.0 / 180.0 * PIf;
    float       azimuth_of_mid_defocus;
    float       angular_distance_from_axes;

    // We choose the azimuth to be mid way between the two defoci of the astigmatic CTF
    azimuth_of_mid_defocus = this->GetAstigmatismAzimuth( ) + PIf * 0.25f;
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

// Compute an image where each pixel stores the number of preceding CTF extrema. This is described as image "E" in Rohou & Grigorieff 2015 (see Fig 3)
void CTF::ComputeImagesWithNumberOfExtremaAndCTFValues(Image* number_of_extrema, Image* ctf_values) {
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
            ctf_values->real_values[address]        = this->Evaluate(current_spatial_frequency_squared, current_azimuth);
            number_of_extrema->real_values[address] = this->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared, current_azimuth);
            //
            address++;
        }
        address += number_of_extrema->padding_jump_value;
    }

    number_of_extrema->is_in_real_space = true;
    ctf_values->is_in_real_space        = true;
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

    // DNM 3/29/23: Initialize in case there are no extrema and extend to the rest only if an extremum is found
    for ( i = 1; i < number_of_bins; i++ )
        half_window_width[i] = minimum_window_half_width;

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
    if ( bin_of_previous_extremum > 0 ) {
        for ( bin_counter = bin_of_previous_extremum; bin_counter < number_of_bins; bin_counter++ ) {
            half_window_width[bin_counter] = half_window_width[bin_of_previous_extremum - 1];
        }
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