#include "core_headers.h"


CTF::CTF()
{
	spherical_aberration = 0;
	wavelength = 0;
	amplitude_contrast = 0;
	defocus_1 = 0;
	defocus_2 = 0;
	defocus_half_range = 0;
	astigmatism_azimuth = 0;
	additional_phase_shift = 0;
	// Fitting parameters
	lowest_frequency_for_fitting = 0;
	highest_frequency_for_fitting = 0;
	astigmatism_tolerance = 0;
	//
	precomputed_amplitude_contrast_term = 0;
	squared_wavelength = 0;
}

CTF::~CTF()
{
	// Nothing to do
}

// Initialise a CTF object
void CTF::Init(	float wanted_acceleration_voltage, // keV
				float wanted_spherical_aberration, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1, // um
				float wanted_defocus_2, //um
				float wanted_astigmatism_azimuth, // degrees
				float wanted_lowest_frequency_for_fitting, // 1/A
				float wanted_highest_frequency_for_fitting, // 1/A
				float wanted_astigmatism_tolerance, // A. Set to negative to indicate no restraint on astigmatism.
				float pixel_size, // A
				float wanted_additional_phase_shift // rad
				)
{
    wavelength = WavelengthGivenAccelerationVoltage(wanted_acceleration_voltage) / pixel_size;
    squared_wavelength = pow(wavelength,2);
    spherical_aberration = wanted_spherical_aberration * 10000000.0 / pixel_size;
    amplitude_contrast = wanted_amplitude_contrast;
    defocus_1 = wanted_defocus_1 * 10000.0 / pixel_size;
    defocus_2 = wanted_defocus_2 * 10000.0 / pixel_size;
    astigmatism_azimuth = wanted_astigmatism_azimuth / 180.0 * PI;
    additional_phase_shift = wanted_additional_phase_shift;
    lowest_frequency_for_fitting = wanted_lowest_frequency_for_fitting * pixel_size;
    highest_frequency_for_fitting = wanted_highest_frequency_for_fitting * pixel_size;
    astigmatism_tolerance = wanted_astigmatism_tolerance / pixel_size;
    precomputed_amplitude_contrast_term = atan(amplitude_contrast/sqrt(1.0 - amplitude_contrast));
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
float CTF::Evaluate(float squared_spatial_frequency, float azimuth)
{
	return -sin( PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) );
}

/* returns the argument (radians) to the sine and cosine terms of the ctf
We follow the convention, like the rest of the cryo-EM/3DEM field, that underfocusing the objective lens
gives rise to a positive phase shift of scattered electrons, whereas the spherical aberration gives a
negative phase shift of scattered electrons.
Note that there is an additional (precomputed) term so that the CTF can then be computed by simply
taking the sine of the returned phase shift.
*/
float CTF::PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(float squared_spatial_frequency, float azimuth)
{
	return PI * wavelength * squared_spatial_frequency * ( DefocusGivenAzimuth(azimuth) - 0.5 * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
}

// Return the effective defocus at the azimuth of interest
float CTF::DefocusGivenAzimuth(float azimuth)
{
	return 0.5 * ( defocus_1 + defocus_2 + cos( 2.0 * (azimuth - astigmatism_azimuth )) * (defocus_1 - defocus_2));
}

// Given acceleration voltage in keV, return the electron wavelength in Angstroms
float CTF::WavelengthGivenAccelerationVoltage( float acceleration_voltage )
{
	return 12.26 / sqrt(1000.0 * acceleration_voltage + 0.9784 * pow(1000.0 * acceleration_voltage,2)/pow(10.0,6));
}
