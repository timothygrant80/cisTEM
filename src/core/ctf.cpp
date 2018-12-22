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
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth, // degrees
				float wanted_lowest_frequency_for_fitting, // 1/A
				float wanted_highest_frequency_for_fitting, // 1/A
				float wanted_astigmatism_tolerance, // A. Set to negative to indicate no restraint on astigmatism.
				float pixel_size, // A
				float wanted_additional_phase_shift_in_radians )// rad
{
	Init(wanted_acceleration_voltage,wanted_spherical_aberration,wanted_amplitude_contrast,wanted_defocus_1_in_angstroms,wanted_defocus_2_in_angstroms,wanted_astigmatism_azimuth,wanted_lowest_frequency_for_fitting,wanted_highest_frequency_for_fitting,wanted_astigmatism_tolerance,pixel_size,wanted_additional_phase_shift_in_radians);
}

CTF::CTF(		float wanted_acceleration_voltage, // keV
				float wanted_spherical_aberration, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth, // degrees
				float pixel_size, // A
				float wanted_additional_phase_shift_in_radians )// rad
{
	Init(wanted_acceleration_voltage,wanted_spherical_aberration,wanted_amplitude_contrast,wanted_defocus_1_in_angstroms,wanted_defocus_2_in_angstroms,wanted_astigmatism_azimuth,0.0,1.0/(2.0*pixel_size),-10.0,pixel_size,wanted_additional_phase_shift_in_radians);
}



CTF::~CTF()
{
	// Nothing to do
}

void CTF::Init(	float wanted_acceleration_voltage_in_kV, // keV
				float wanted_spherical_aberration_in_mm, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth_in_degrees, // degrees
				float pixel_size_in_angstroms, // A
				float wanted_additional_phase_shift_in_radians // rad
				)
{
	Init(wanted_acceleration_voltage_in_kV,wanted_spherical_aberration_in_mm,wanted_amplitude_contrast,wanted_defocus_1_in_angstroms,wanted_defocus_2_in_angstroms,wanted_astigmatism_azimuth_in_degrees,0.0,1.0/(2.0*pixel_size_in_angstroms),-10.0,pixel_size_in_angstroms,wanted_additional_phase_shift_in_radians);
}

// Initialise a CTF object
void CTF::Init(	float wanted_acceleration_voltage_in_kV, // keV
				float wanted_spherical_aberration_in_mm, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth_in_degrees, // degrees
				float wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
				float wanted_highest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
				float wanted_astigmatism_tolerance_in_angstroms, // A. Set to negative to indicate no restraint on astigmatism.
				float pixel_size_in_angstroms, // A
				float wanted_additional_phase_shift_in_radians // rad
				)
{
    wavelength = WavelengthGivenAccelerationVoltage(wanted_acceleration_voltage_in_kV) / pixel_size_in_angstroms;
    squared_wavelength = powf(wavelength,2);
    cubed_wavelength = powf(wavelength,3);
    spherical_aberration = wanted_spherical_aberration_in_mm * 10000000.0 / pixel_size_in_angstroms;
    amplitude_contrast = wanted_amplitude_contrast;
    defocus_1 = wanted_defocus_1_in_angstroms / pixel_size_in_angstroms;
    defocus_2 = wanted_defocus_2_in_angstroms / pixel_size_in_angstroms;
    astigmatism_azimuth = wanted_astigmatism_azimuth_in_degrees / 180.0 * PI;
    additional_phase_shift = wanted_additional_phase_shift_in_radians;
    lowest_frequency_for_fitting = wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
    highest_frequency_for_fitting = wanted_highest_frequency_for_fitting_in_reciprocal_angstroms * pixel_size_in_angstroms;
    astigmatism_tolerance = wanted_astigmatism_tolerance_in_angstroms / pixel_size_in_angstroms;

    // gcc catches the zero division somehow, but intel returns a nan. Needed for handling the real and complex terms of the CTF separately.
    if (fabs(amplitude_contrast - 1.0) < 1e-3) precomputed_amplitude_contrast_term = PI / 2.0f;
    else precomputed_amplitude_contrast_term = atanf(amplitude_contrast/sqrtf(1.0 - powf(amplitude_contrast, 2)));

}

/*
 * Eqn 11 of Rohou & Grigorieff, modified by corrigendum of October 2018
 */
int CTF::ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(float squared_spatial_frequency, float azimuth)
{

	int number_of_extrema = 0;
	int number_of_extrema_before_chi_extremum = 0;
	float sq_sf_of_chi_extremum = ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(azimuth);

	if (squared_spatial_frequency <= sq_sf_of_chi_extremum)
	{
		number_of_extrema = floor( 1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) + 0.5);
	}
	else
	{
		number_of_extrema_before_chi_extremum = floor( 1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(sq_sf_of_chi_extremum,azimuth) + 0.5);
		number_of_extrema = floor( 1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) + 0.5);
		number_of_extrema = number_of_extrema_before_chi_extremum + abs(number_of_extrema - number_of_extrema_before_chi_extremum);
	}
	MyDebugAssertTrue(number_of_extrema >= 0,"Bad number of extrema: %i (rounded from %f, phase shift = %f)\n",number_of_extrema, 1.0 / PI * PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) + 0.5,PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth));
	return number_of_extrema;
}

/*
 * Compute the frequency of the Nth zero of the CTF.
 */
float CTF::ReturnSquaredSpatialFrequencyOfAZero(int which_zero, float azimuth)
{
	/*
	 * The method used below (ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth) makes assumptions which mean it will
	 * only return the correct spatial frequency for the CTF zeroes between the origin and the frequency at which
	 * the phase aberration peaks.
	 */
	MyDebugAssertTrue(which_zero < ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(azimuth),azimuth),"Oops, this method only works for the first few zeroes");
	float phase_shift = which_zero * PI;
	return ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(phase_shift,azimuth);
}

/*
 * Return the maximum phase aberration
 */
float CTF::ReturnPhaseAberrationMaximum()
{
	// We use the maximum defocus
	float defocus_max = std::max(fabs(defocus_1),fabs(defocus_2));
	float sq_sf_of_phase_aberration_max;
	if (spherical_aberration == 0.0)
	{
		sq_sf_of_phase_aberration_max = 99999.99;
	}
	else
	{
		sq_sf_of_phase_aberration_max= defocus_max / (squared_wavelength * spherical_aberration);
	}
	return PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(sq_sf_of_phase_aberration_max,defocus_max);
}

/*
 * Return the squared spatial frequency at which the phase aberration function reaches its extremum
 */
float CTF::ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenAzimuth(float azimuth)
{
	return ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(DefocusGivenAzimuth(azimuth));
}

/*
 * Return the squared spatial frequency at which the phase aberration function reaches its extremum
 */
float CTF::ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(float defocus)
{
	if (spherical_aberration == 0.0)
	{
		return 9999.999;
	}
	else
	{
		return defocus / (squared_wavelength * spherical_aberration) ;
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
float CTF::ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(float phase_shift, float azimuth)
{
	const float a = -0.5 * PI * cubed_wavelength * spherical_aberration;
	const float b = PI * wavelength * DefocusGivenAzimuth(azimuth);
	const float c = additional_phase_shift + precomputed_amplitude_contrast_term;
	const float det = powf(b,2.0) - 4.0 * a * (c-phase_shift);

	if (spherical_aberration == 0.0)
	{
		return (phase_shift - c) / b;
	}
	else
	{

		MyDebugAssertTrue(a != 0.0,"Bad values for either cubed wavelength (%f) or spherical aberration (%f) (a = %f)\n",cubed_wavelength,spherical_aberration,a);


		if (det < 0.0)
		{
			//MyPrintWithDetails("Ooops, negative determinant\n");
			//DEBUG_ABORT;
			return 0.0;
		}
		else
		{
			const float solution_one = ( -b + sqrtf(det)) / (2.0 * a);
			const float solution_two = ( -b - sqrtf(det)) / (2.0 * a);

			if ( solution_one > 0 && solution_two > 0 )
			{
				if (solution_one <= solution_two)
				{
					return solution_one;
				}
				else
				{
					return solution_two;
				}
			}
			else if ( solution_one > 0 )
			{
				return solution_one;
			}
			else if ( solution_two > 0 )
				return solution_two;
			else
			{
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
//#pragma GCC pop_options


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
	additional_phase_shift = fmodf(wanted_additional_phase_shift_radians,(float)PI);
}

// Set the highest frequency used for fitting
void CTF::SetHighestFrequencyForFitting(float wanted_highest_frequency_in_reciprocal_pixels)
{
	highest_frequency_for_fitting = wanted_highest_frequency_in_reciprocal_pixels;
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
std::complex<float> CTF::EvaluateComplex(float squared_spatial_frequency, float azimuth)
{
	float phase_aberration = PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth);
	return	- sinf( phase_aberration ) - I * cosf( phase_aberration );

// Frealign code:
// CTF=CMPLX(WGH1*SCHI-WGH2*CCHI,-WGH1*CCHI-WGH2*SCHI)
//	wxPrintf("cistem real  = %g\n", -sinf(chi + pre));
//	wxPrintf("frealign real= %g\n", sqrtf(1.0 - powf(amplitude_contrast, 2)) * sinf(-chi) - amplitude_contrast * cosf(-chi));
//	wxPrintf("cistem comp  = %g\n", -cosf(chi + pre));
//	wxPrintf("frealign comp= %g\n", -sqrtf(1.0 - powf(amplitude_contrast, 2)) * cosf(-chi) - amplitude_contrast * sinf(-chi));
}

// Return the value of the CTF at the given squared spatial frequency and azimuth
float CTF::Evaluate(float squared_spatial_frequency, float azimuth)
{
	return -sinf( PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(squared_spatial_frequency,azimuth) );
}

/* returns the argument (radians) to the sine and cosine terms of the ctf
We follow the convention, like the rest of the cryo-EM/3DEM field, that underfocusing the objective lens
gives rise to a positive phase shift of scattered electrons, whereas the spherical aberration gives a
negative phase shift of scattered electrons.
Note that there is an additional (precomputed) term so that the CTF can then be computed by simply
taking the sine of the returned phase shift.
*/
float CTF::PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(float squared_spatial_frequency, float defocus)
{
	return PI * wavelength * squared_spatial_frequency * ( defocus - 0.5 * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
}

float CTF::PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(float squared_spatial_frequency, float azimuth)
{
	return PI * wavelength * squared_spatial_frequency * ( DefocusGivenAzimuth(azimuth) - 0.5 * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
}

// Return the effective defocus at the azimuth of interest
float CTF::DefocusGivenAzimuth(float azimuth)
{
	return 0.5 * ( defocus_1 + defocus_2 + cosf( 2.0 * (azimuth - astigmatism_azimuth )) * (defocus_1 - defocus_2));
}

// Given acceleration voltage in keV, return the electron wavelength in Angstroms
float CTF::WavelengthGivenAccelerationVoltage( float acceleration_voltage )
{
	return 12.26 / sqrtf(1000.0 * acceleration_voltage + 0.9784 * powf(1000.0 * acceleration_voltage,2)/powf(10.0,6));
}


// Compare two CTF objects and return true if they are within a specified defocus tolerance
bool CTF::IsAlmostEqualTo(CTF *wanted_ctf, float delta_defocus)
{
	float delta;

	if (fabsf(this->spherical_aberration - wanted_ctf->spherical_aberration) > 0.01) return false;
	if (fabsf(this->wavelength - wanted_ctf->wavelength) > 0.0001) return false;
	if (fabsf(this->amplitude_contrast - wanted_ctf->amplitude_contrast) > 0.0001) return false;
	if (fabsf(this->defocus_1 - wanted_ctf->defocus_1) > delta_defocus) return false;
	if (fabsf(this->defocus_2 - wanted_ctf->defocus_2) > delta_defocus) return false;

	delta = fabsf(this->additional_phase_shift - wanted_ctf->additional_phase_shift);
	delta = fmodf(delta, 2.0f * (float)PI);
// 0.0277 = 5/180 (5 deg tolerance)
	if (delta > 0.0277) return false;

	delta = fabsf(this->astigmatism_azimuth - wanted_ctf->astigmatism_azimuth);
	delta = fmodf(delta, (float)PI);
// 0.0277 = 5/180 (5 deg tolerance)
	if (delta > 0.0277) return false;

	return true;
}

void CTF::CopyFrom(CTF other_ctf)
{
	spherical_aberration = other_ctf.spherical_aberration;
	wavelength = other_ctf.wavelength;
	amplitude_contrast = other_ctf.amplitude_contrast;
	defocus_1 = other_ctf.defocus_1;
	defocus_2 = other_ctf.defocus_2;
	defocus_half_range = other_ctf.defocus_half_range;
	astigmatism_azimuth = other_ctf.astigmatism_azimuth;
	additional_phase_shift = other_ctf.additional_phase_shift;
	// Fitting parameters
	lowest_frequency_for_fitting = other_ctf.lowest_frequency_for_fitting;
	highest_frequency_for_fitting = other_ctf.highest_frequency_for_fitting;
	astigmatism_tolerance = other_ctf.astigmatism_tolerance;
	//
	precomputed_amplitude_contrast_term = other_ctf.precomputed_amplitude_contrast_term;
	squared_wavelength = other_ctf.squared_wavelength;
	cubed_wavelength = other_ctf.cubed_wavelength;
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
	astigmatism_azimuth -= PI * roundf(astigmatism_azimuth/PI);
}

void CTF::ChangePixelSize(float old_pixel_size, float new_pixel_size)
{
	float scale = old_pixel_size/new_pixel_size;

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
	squared_wavelength = powf(wavelength,2);
	cubed_wavelength = powf(wavelength,3);
}
