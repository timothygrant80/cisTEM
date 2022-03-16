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
         float wanted_particle_shift_y_in_angstroms) // A
{
    Init(wanted_acceleration_voltage, wanted_spherical_aberration, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth, wanted_lowest_frequency_for_fitting, wanted_highest_frequency_for_fitting, wanted_astigmatism_tolerance, pixel_size, wanted_additional_phase_shift_in_radians, wanted_beam_tilt_x_in_radians, wanted_beam_tilt_y_in_radians, wanted_particle_shift_x_in_angstroms, wanted_particle_shift_y_in_angstroms);
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
         float wanted_particle_shift_y_in_angstroms) // A
{
    Init(wanted_acceleration_voltage, wanted_spherical_aberration, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth, 0.0, 1.0 / (2.0 * pixel_size), -10.0, pixel_size, wanted_additional_phase_shift_in_radians, wanted_beam_tilt_x_in_radians, wanted_beam_tilt_y_in_radians, wanted_particle_shift_x_in_angstroms, wanted_particle_shift_y_in_angstroms);
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
               float wanted_additional_phase_shift_in_radians) // rad
{
    Init(wanted_acceleration_voltage_in_kV, wanted_spherical_aberration_in_mm, wanted_amplitude_contrast, wanted_defocus_1_in_angstroms, wanted_defocus_2_in_angstroms, wanted_astigmatism_azimuth_in_degrees, 0.0, 1.0 / (2.0 * pixel_size_in_angstroms), -10.0, pixel_size_in_angstroms, wanted_additional_phase_shift_in_radians, 0.0f, 0.0f, 0.0f, 0.0f);
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
               float wanted_particle_shift_y_in_angstroms) // A
{
    wavelength                    = WavelengthGivenAccelerationVoltage(wanted_acceleration_voltage_in_kV) / pixel_size_in_angstroms;
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

// Given acceleration voltage in keV, return the electron wavelength in Angstroms
float CTF::WavelengthGivenAccelerationVoltage(float acceleration_voltage) {
    //	return 12.26f / sqrtf(1000.0f * acceleration_voltage + 0.9784f * powf(1000.0f * acceleration_voltage,2)/powf(10.0f,6));
    return 12.2639f / sqrtf(1000.0f * acceleration_voltage + 0.97845e-6 * powf(1000.0f * acceleration_voltage, 2));
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
