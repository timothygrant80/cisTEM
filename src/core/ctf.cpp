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
	cubed_wavelength = 0;
}

CTF::CTF(		float wanted_acceleration_voltage, // keV
				float wanted_spherical_aberration, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1, // A
				float wanted_defocus_2, //A
				float wanted_astigmatism_azimuth, // degrees
				float wanted_lowest_frequency_for_fitting, // 1/A
				float wanted_highest_frequency_for_fitting, // 1/A
				float wanted_astigmatism_tolerance, // A. Set to negative to indicate no restraint on astigmatism.
				float pixel_size, // A
				float wanted_additional_phase_shift )// rad
{
	Init(wanted_acceleration_voltage,wanted_spherical_aberration,wanted_amplitude_contrast,wanted_defocus_1,wanted_defocus_2,wanted_astigmatism_azimuth,wanted_lowest_frequency_for_fitting,wanted_highest_frequency_for_fitting,wanted_astigmatism_tolerance,pixel_size,wanted_additional_phase_shift);
}


CTF::~CTF()
{
	// Nothing to do
}

// Initialise a CTF object
void CTF::Init(	float wanted_acceleration_voltage, // keV
				float wanted_spherical_aberration, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1, // A
				float wanted_defocus_2, //A
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
    cubed_wavelength = pow(wavelength,3);
    spherical_aberration = wanted_spherical_aberration * 10000000.0 / pixel_size;
    amplitude_contrast = wanted_amplitude_contrast;
    defocus_1 = wanted_defocus_1 / pixel_size;
    defocus_2 = wanted_defocus_2 / pixel_size;
    astigmatism_azimuth = wanted_astigmatism_azimuth / 180.0 * PI;
    additional_phase_shift = wanted_additional_phase_shift;
    lowest_frequency_for_fitting = wanted_lowest_frequency_for_fitting * pixel_size;
    highest_frequency_for_fitting = wanted_highest_frequency_for_fitting * pixel_size;
    astigmatism_tolerance = wanted_astigmatism_tolerance / pixel_size;
    precomputed_amplitude_contrast_term = atan(amplitude_contrast/sqrt(1.0 - amplitude_contrast));
}

int CTF::ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(float squared_spatial_frequency, float azimuth)
{
	return floor( 1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) + 0.5);
}

// Compute the frequency of the Nth zero of the CTF
float CTF::ReturnSquaredSpatialFrequencyOfAZero(int which_zero, float azimuth)
{
	float phase_shift = which_zero * PI;
	return ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(phase_shift,azimuth);
}

float CTF::ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(float phase_shift, float azimuth)
{
	const float a = -0.5 * PI * cubed_wavelength * spherical_aberration;
	const float b = PI * wavelength * DefocusGivenAzimuth(azimuth);
	const float c = additional_phase_shift + precomputed_amplitude_contrast_term;

	const float solution_one = ( -b + sqrt(pow(b,2) - 4.0 * a * (c-phase_shift))) / (2.0 * a);
	const float solution_two = ( -b - sqrt(pow(b,2) - 4.0 * a * (c-phase_shift))) / (2.0 * a);

	if ( solution_one > 0 && solution_two > 0 )
	{
		MyDebugPrintWithDetails("Oops, I don't know which solution to select");
	}
	else if ( solution_one > 0 )
	{
		return solution_one;
	}
	else
	{
		return solution_two;
	}
}


// Set the defocus and astigmatism angle, given in pixels and radians
void CTF::SetDefocus(float wanted_defocus_1_pixels, float wanted_defocus_2_pixels, float wanted_astigmatism_angle_radians)
{
	defocus_1 = wanted_defocus_1_pixels;
	defocus_2 = wanted_defocus_2_pixels;
	astigmatism_azimuth = wanted_astigmatism_angle_radians;
}

// Set the additional phase shift, given in radians
void CTF::SetAdditionalPhaseShift(float wanted_additional_phase_shift_radians)
{
	additional_phase_shift = wanted_additional_phase_shift_radians;
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

// Enforce the convention that df1 > df2 and -90 < angast < 90
void CTF::EnforceConvention() {
	float defocus_tmp;

	if ( defocus_1 < defocus_2 )
	{
		defocus_tmp = defocus_2;
		defocus_2 = defocus_1;
		defocus_1 = defocus_tmp;
		astigmatism_azimuth += PI*0.5;
	}
	astigmatism_azimuth -= PI * round(astigmatism_azimuth/PI);
}

