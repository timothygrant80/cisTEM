/*
 * wave_function_propagator.h
 *
 *  Created on: Oct 2, 2019
 *      Author: himesb
 */

#ifndef PROGRAMS_SCATTERING_POTENTIAL_WAVE_FUNCTION_PROPAGATOR_H_
#define PROGRAMS_SCATTERING_POTENTIAL_WAVE_FUNCTION_PROPAGATOR_H_

typedef struct _fitParams
{


	float pixel_size;
	float kv;
	float Cs;
	float AmplitudeContrast;
	float Size;
	float min_resolution;
	float max_resolution;
	float min_defocus;
	float max_defocus;
	float nThreads;
	float defocus_step;


} fitParams;
class WaveFunctionPropagator  {
public:
	WaveFunctionPropagator(float set_real_part_wave_function_in,  float wanted_objective_aperture_diameter_micron,
			   	   	   	   float wanted_pixel_size, int wanted_number_threads, float beam_tilt_x, float beam_tilt_y, bool do_beam_tilt_full, float* propagator_distance);
	virtual ~WaveFunctionPropagator();

	float min_resolution_for_fitting = 1/30.0f; // 1/A
	float max_resolution_for_fitting =  -1.0f;//1/5.0f; // 1/A, < 0 for nyquist. The brute force search in ctffind is limited to 5 A so if a match is desired, this should also be at most 5
	bool is_set_ctf;
	bool is_set_fresnel_propagator;
	bool is_set_input_projections;
	bool need_to_allocated_member_images;
	bool do_coherence_envelope;
	bool do_beam_tilt_full;

	int nThreads;
	int nSlabs;
	int unpadded_x_dimension;
	int unpadded_y_dimension;
	float *wave_function_in;
	float requested_kv;
	float expected_dose_per_pixel;

	Image* temp_img;
	Image* t_N;
	Image* wave_function;
	Image* phase_grating;
	Image* amplitude_grating;
	Image* aperture_grating;


	CTF* ctf;
	CTF* fresnel_propagtor;
	CTF* ctf_for_fitting;

	float* propagator_distance;

	const int	copy_from_1[4] 	 = {0,1,1,0};
	const int	copy_from_2[4]	 = {0,1,1,0};
	const int  	mult_by[4]		 = {0,1,0,1};
	const int 	prop_apply_real[4] = {0,0,1,1};
	const int 	prop_apply_imag[4] = {1,1,0,0};
	const int  	ctf_apply[4]	   = {0,1,0,1};

	float beam_tilt_x;
	float beam_tilt_y;
	float beam_tilt_magnitude;
	float beam_tilt_azimuth;
	float beam_tilt_shift_factor_x;
	float beam_tilt_shift_factor_y;

	float pixel_size;
	float pixel_size_squared;
	float objective_aperture_resolution;

	fitParams for_ctffind;
	bool fit_params_are_set = false;

	// Regular
	void SetCTF(float wanted_acceleration_voltage,
				float wanted_spherical_aberration,
				float wanted_defocus_1,
				float wanted_defocus_2,
				float wanted_astigmatism_azimuth,
				float wanted_additional_phase_shift_in_radians,
				float defocus_offset);

	// With coherence envelop
	void SetCTF(float wanted_acceleration_voltage,
				float wanted_spherical_aberration,
				float wanted_defocus_1,
				float wanted_defocus_2,
				float wanted_astigmatism_azimuth,
				float wanted_additional_phase_shift_in_radians,
				float defocus_offset,
				float wanted_dose_rate);

	void SetFresnelPropagator(float wanted_acceleration_voltage, float propagation_distance);
//	void SetInputProjections(Image* scattering_potential, Image* inelastic_potential, int nSlabs);
	void SetInputWaveFunction(int size_x, int size_y);
	void SetFitParams(	float pixel_size, float kv, float Cs, float AmplitudeContrast, float Size, float min_resolution, float max_resolution, float min_defocus, float max_defocus, float nThreads, float defocus_step);


	float DoPropagation(Image* sum_image, Image* scattering_potential, Image* inelastic_potential,
			int tilt_IDX, int nSlabs,
		   float* image_mean, float* inelastic_mean, float* propagator_distance, bool estimate_amplitude_contrast, float tilt_angle);

	void  SetObjectiveAperture(float set_diameter_to) { objective_aperture_diameter = set_diameter_to; }
	float GetObjectiveAperture() { return objective_aperture_diameter; }

	void  SetBandpassFalloff(float set_falloff_to) { mask_falloff = set_falloff_to; }
	float GetBandpassfalloff() { return mask_falloff; }

	void ReturnImageContrast(Image &wave_function_sq_modulus, float* contrast, bool is_phase_contrast_image, float tilt_angle);

	inline float ReturnPlasmonConversionFactor(float squared_spatial_frequency_in_angstrom)
	{
		float conversion_factor = 0.0f;
//		float spatial_frequency_in_angstrom = sqrtf(squared_spatial_frequency_in_angstrom);

//		if (spatial_frequency_in_angstrom < plasmon_switch)
//		{
			for (int i = 0; i < 5; i++)
			{
				// FIXME pre-square the const B vals and do fits to freq^2
				conversion_factor += plasmon_conversion_A[i] * expf(- squared_spatial_frequency_in_angstrom / (plasmon_conversion_B[i]*plasmon_conversion_B[i]));
			}
//		}
//		else
//		{
//
//			for (int i = 0; i < 2; i++)
//			{
//				conversion_factor += plasmon_conversion_C[i] * expf(plasmon_conversion_D[i] * spatial_frequency_in_angstrom);
//			}
//		}

		return conversion_factor;

	}

	float ReturnObjectiveApertureResoution(float wavelength_in_angstrom)
	{
		float res = (wavelength_in_angstrom * objective_lens_focal_length * 1e7) / (GetObjectiveAperture()/2.0f * 1e4);
		return res;
	}
	//	1/(150e-6/(1.97e-12*3.5e-3))


private:

	const float objective_lens_focal_length = 3.5; // millimeter, for krios ctwin
	float objective_aperture_diameter = 2000.0f; // default large enough so it as if there were no aperture
	float mask_falloff = 14.0f; // reciprocal pixels
//	const float plasmon_conversion[6] = { 1.216e-2, -2.181e-4, 1.066e-3, 1.275e-6, 1.223e-3, 1.274e-6};
	const float plasmon_switch = 0.25f; // 4 A spatial freq
	const float plasmon_conversion_A[5] = {0.326, 0.2465, 0.1301, 0.2076, 0.0851};
	const float plasmon_conversion_B[5] = {0.01354, 0.03255, 0.08223 , 0.0053, 0.6945};
	const float plasmon_conversion_C[2] = {7.254e-4, 1.921e-7};
	const float plasmon_conversion_D[2] = {7.288,11.81};




};

#endif /* PROGRAMS_SCATTERING_POTENTIAL_WAVE_FUNCTION_PROPAGATOR_H_ */
