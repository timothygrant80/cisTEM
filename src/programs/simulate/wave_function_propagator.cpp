/*
 * wave_function_propagator.cpp
 *
 *  Created on: Oct 2, 2019
 *      Author: himesb
 */

#include "../../core/core_headers.h"
#include "wave_function_propagator.h"
#include <unistd.h>  // For the ctffind to disk method (uniqe identifier based on pid)

WaveFunctionPropagator::WaveFunctionPropagator(float set_real_part_wave_function_in, float wanted_objective_aperture_diameter_micron,
											   float wanted_pixel_size, int wanted_number_threads, float beam_tilt_x, float beam_tilt_y, bool do_beam_tilt_full, float* propagator_distance)
{
	temp_img = new Image[4];
	t_N = new Image[4];
	wave_function = new Image[2];
	phase_grating = new Image[2];
	amplitude_grating = new Image[2];
	aperture_grating = new Image[2];
	ctf = new CTF[2];
	fresnel_propagtor = new CTF[2];
	ctf_for_fitting = new CTF[1];

	wave_function_in = new float[2];
	wave_function_in[0] = set_real_part_wave_function_in;
	wave_function_in[1] = 0.0f;

	this->propagator_distance = propagator_distance;


	pixel_size = wanted_pixel_size;
	pixel_size_squared = pixel_size * pixel_size;
	expected_dose_per_pixel = wave_function_in[0]*wave_function_in[0];
	nThreads = wanted_number_threads;

	if (max_resolution_for_fitting < 0.0f)
	{
		max_resolution_for_fitting = 0.5f / pixel_size;
	}
	SetObjectiveAperture(wanted_objective_aperture_diameter_micron);

	is_set_ctf = false;
	is_set_fresnel_propagator = false;
	is_set_input_projections = false;
	need_to_allocated_member_images = true;
	do_coherence_envelope = false;
	this->do_beam_tilt_full = do_beam_tilt_full;

	this->beam_tilt_x = beam_tilt_x;
	this->beam_tilt_y = beam_tilt_y;
	beam_tilt_magnitude = sqrtf(beam_tilt_x*beam_tilt_x + beam_tilt_y*beam_tilt_y); // FIXME these should
	beam_tilt_azimuth = atan2f(beam_tilt_y,beam_tilt_x);
	beam_tilt_shift_factor_x = tanf(beam_tilt_magnitude)*cosf(beam_tilt_azimuth);
	beam_tilt_shift_factor_y = tanf(beam_tilt_magnitude)*sinf(beam_tilt_azimuth);

}

WaveFunctionPropagator::~WaveFunctionPropagator()
{

	delete [] temp_img;
	delete [] t_N;
	delete [] wave_function;
	delete [] phase_grating;
	delete [] amplitude_grating;
	delete [] aperture_grating;
	delete [] ctf;
	delete [] fresnel_propagtor;
	delete [] ctf_for_fitting;
}

void WaveFunctionPropagator::SetCTF(float wanted_acceleration_voltage,
									float wanted_spherical_aberration,
									float wanted_defocus_1,
									float wanted_defocus_2,
									float wanted_astigmatism_azimuth,
									float wanted_additional_phase_shift_in_radians,
									float defocus_offset)
{
	// defocus values are assumed to include any absolute offsets relative to a slabs position in the full 3d specimen.
	// The amplitude contrast is forced to 1 or 0 to retrieve just the real/imag portion of the conventional CTF
	//

	SetFresnelPropagator(wanted_acceleration_voltage, 0.0f);

	ctf[0].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				1.0f,
				wanted_defocus_1-defocus_offset,
				wanted_defocus_2-defocus_offset,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians);

	ctf[1].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				0.0f,
				wanted_defocus_1-defocus_offset,
				wanted_defocus_2-defocus_offset,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians);

	ctf_for_fitting->Init(	wanted_acceleration_voltage,
							wanted_spherical_aberration,
							0.0f,
							wanted_defocus_1,
							wanted_defocus_2,
							wanted_astigmatism_azimuth,
							min_resolution_for_fitting,
							max_resolution_for_fitting,
							-1,
							pixel_size,
							wanted_additional_phase_shift_in_radians,
							0.0f,0.0f,0.0f,0.0f);


	requested_kv = wanted_acceleration_voltage;
	is_set_ctf = true;
}

void WaveFunctionPropagator::SetCTF(float wanted_acceleration_voltage,
									float wanted_spherical_aberration,
									float wanted_defocus_1,
									float wanted_defocus_2,
									float wanted_astigmatism_azimuth,
									float wanted_additional_phase_shift_in_radians,
									float defocus_offset,
									float wanted_dose_rate)
{
	// defocus values are assumed to include any absolute offsets relative to a slabs position in the full 3d specimen.
	// The amplitude contrast is forced to 1 or 0 to retrieve just the real/imag portion of the conventional CTF
	//

	SetFresnelPropagator(wanted_acceleration_voltage, 0.0f);

	ctf[0].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				1.0f,
				wanted_defocus_1-defocus_offset,
				wanted_defocus_2-defocus_offset,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians);

	ctf[1].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				0.0,
				wanted_defocus_1-defocus_offset,
				wanted_defocus_2-defocus_offset,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians);

	float local_max_res;


	ctf_for_fitting->Init(	wanted_acceleration_voltage,
							wanted_spherical_aberration,
							0.0f,
							wanted_defocus_1,
							wanted_defocus_2,
							wanted_astigmatism_azimuth,
							min_resolution_for_fitting,
							max_resolution_for_fitting,
							-1,
							pixel_size,
							wanted_additional_phase_shift_in_radians,
							0.0f,0.0f,0.0f,0.0f);

	ctf[0].SetEnvelope(wanted_acceleration_voltage,pixel_size,wanted_dose_rate / (pixel_size_squared));
	ctf[1].SetEnvelope(wanted_acceleration_voltage,pixel_size,wanted_dose_rate / (pixel_size_squared));

	requested_kv = wanted_acceleration_voltage;
	do_coherence_envelope = true;
	is_set_ctf = true;
}

void WaveFunctionPropagator::SetFitParams(	float pixel_size, float kv, float Cs, float AmplitudeContrast, float Size, float min_resolution, float max_resolution, float min_defocus, float max_defocus, float nThreads, float defocus_step)
{
	for_ctffind.pixel_size = pixel_size;
	for_ctffind.kv = kv;
	for_ctffind.Cs = Cs;
	for_ctffind.AmplitudeContrast = AmplitudeContrast;
	for_ctffind.Size = Size;
	for_ctffind.min_resolution = min_resolution;
	for_ctffind.max_resolution = max_resolution;
	for_ctffind.min_defocus = min_defocus;
	for_ctffind.max_defocus = max_defocus;
	for_ctffind.nThreads = nThreads;
	for_ctffind.defocus_step = defocus_step;

	fit_params_are_set = true;

}


void WaveFunctionPropagator::SetFresnelPropagator(float wanted_acceleration_voltage, float propagation_distance)
{


	if (wanted_acceleration_voltage > 0)
	{
		fresnel_propagtor[0].Init(wanted_acceleration_voltage,
						   0.0,
						   1.0,
						   propagation_distance,
						   propagation_distance,
						   0.0,
						   pixel_size,
						   0.0);

		fresnel_propagtor[1].Init(wanted_acceleration_voltage,
						   0.0,
						   0.0,
						   propagation_distance,
						   propagation_distance,
						   0.0,
						   pixel_size,
						   0.0);

		is_set_fresnel_propagator = true;
	}
	else
	{
		fresnel_propagtor[0].SetDefocus(propagation_distance, propagation_distance, 0.0f);
		fresnel_propagtor[1].SetDefocus(propagation_distance, propagation_distance, 0.0f);
	}


}

//void WaveFunctionPropagator::SetInputProjections(Image* scattering_potential, Image* inelastic_potential, int nSlabs)
//{
//	this->nSlabs = nSlabs;
//	this->scattering_potential = scattering_potential;
//	this->inelastic_potential = inelastic_potential;
//
//	// This is set to false after each round of propagation.
//	is_set_input_projections = true;
//}
void WaveFunctionPropagator::SetInputWaveFunction(int size_x, int size_y)
{

//	MyAssertTrue(is_set_input_projections, "The input projections must be set")

	int local_threads = 1;
	if (nThreads > 1) { local_threads = 2; ;}

	if (need_to_allocated_member_images)
		{
		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 2; iPar++)
		{
			wave_function[iPar].Allocate(size_x,size_y,1);
			wave_function[iPar].SetToConstant(wave_function_in[iPar]);
			phase_grating[iPar].Allocate(size_x,size_y,1);
			amplitude_grating[iPar].Allocate(size_x,size_y,1);
			aperture_grating[iPar].Allocate(size_x,size_y,1);
			// We only want the real part to be set to 1
			for (int iPixel = 0; iPixel < aperture_grating[iPar].real_memory_allocated; iPixel+=2)
			{
				aperture_grating[iPar].real_values[iPixel] = 1.0f;
			}

			aperture_grating[iPar].is_in_real_space = false;
		}

		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 4; iPar++)
		{
			t_N[iPar].Allocate(size_x,size_y,1);
			t_N[iPar].SetToConstant(0.0f);
			temp_img[iPar].Allocate(size_x,size_y,1);
		}
		need_to_allocated_member_images = false;
	}
	else
	{

		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 2; iPar++)
		{
			wave_function[iPar].SetToConstant(wave_function_in[iPar]);
			aperture_grating[iPar].SetToConstant(0.0f);
			for (int iPixel = 0; iPixel < aperture_grating[iPar].real_memory_allocated; iPixel+=2)
			{
				aperture_grating[iPar].real_values[iPixel] = 1.0f;
			}
			aperture_grating[iPar].is_in_real_space = false;
		}

		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 4; iPar++)
		{
			t_N[iPar].SetToConstant(0.0f);
		}
	}

}

float WaveFunctionPropagator::DoPropagation(Image* sum_image, Image* scattering_potential, Image* inelastic_potential,
											int tilt_IDX, int nSlabs,
										   float* image_mean, float* inelastic_mean, float* propagator_distance, bool estimate_amplitude_contrast, float tilt_angle)
{
	MyAssertTrue(is_set_ctf && is_set_fresnel_propagator, "Either the ctf or fresnel propagtor are not set")
	MyAssertTrue(fit_params_are_set, "The ctffind fit parameters have not been set, this is essential for getting the defocus and amplitude contrast correct");

	int size_x = sum_image[0].logical_x_dimension;
	int size_y = sum_image[0].logical_y_dimension;

	unpadded_x_dimension = scattering_potential[0].logical_x_dimension;
	unpadded_y_dimension = scattering_potential[0].logical_y_dimension;

	// Propagate twice.
	// First calc the pure phase image.
	// Second calc the real image with aperture and inelastics. Use this to return the amplitude contrast.
	float total_contrast;
	float phase_contrast;
	float mean_value;

	int starting_val;
	if (estimate_amplitude_contrast) starting_val = 0;
	else starting_val = 1;
//	int starting_val = 1; // override for new amp FIXME

	//
	for (int iContrast = starting_val ; iContrast < 2; iContrast++)
	{


	SetInputWaveFunction(size_x, size_y);

	// FIXME function.cpp and CTF.cpp remove

	objective_aperture_resolution = ReturnObjectiveApertureResoution(1226.39 / sqrtf(requested_kv*1000 + 0.97845e-6*powf(requested_kv*1000,2)) * 1e-2);
	aperture_grating[0].ReturnCosineMaskBandpassResolution(pixel_size, objective_aperture_resolution, mask_falloff);
	// Masks for scattering inside (0) and outside (1) the objective aperture.
	aperture_grating[0].CosineRingMask(-1.0f, objective_aperture_resolution ,mask_falloff);
//	aperture_grating[1].SubtractImage(&aperture_grating[0]);


	// assuming the beam tilt should actually be zero if smaller than this.
	const float beam_tilt_epsilon = 1e-10f;
	bool do_beam_tilt = false;

	if (fabsf(beam_tilt_magnitude) > beam_tilt_epsilon) { do_beam_tilt = true; }

	int local_threads = 1;
	if (nThreads > 3) { local_threads = 4; }
	else if (nThreads > 2) { local_threads = 3;}
	else if (nThreads > 1) { local_threads = 2;}

	// This doesn't make sense. It is almost correct for 0.75 pixel Size. TODO test agains 1.25 and think about why the apparent shift is not correct. Also before testing further, switch to an object that is symmetric in Z in case
	// in case this somehow has to do with the mass distribution.
	float total_shift_x = 0.0f;//-2.0f*pixel_size*cosf(beam_tilt_azimuth);//0.0f;
	float total_shift_y = 0.0f;//-2.0f*pixel_size*sinf(beam_tilt_azimuth);//0.0f;


	for (int iSlab = 0; iSlab < nSlabs; iSlab++)
	{


		SetFresnelPropagator(0.0f, propagator_distance[iSlab]);

		phase_grating[0].SetToConstant(0.0f);
//		phase_grating[0].AddGaussianNoise(scattering_potential[iSlab].ReturnSumOfSquares(0.0f));
//		phase_grating[0].AddConstant(scattering_potential[iSlab].ReturnAverageOfRealValues(0.0f));

		scattering_potential[iSlab].ClipInto(&phase_grating[0],scattering_potential[iSlab].ReturnAverageOfRealValuesOnEdges());

		if (do_beam_tilt)
		{
//			wxPrintf("DO BEAM TILT\n\n");
			// For tilted illumination, the scattering plane sees only the z-component of the wave-vector = lower energy = stronger interaction
			// so the interaction constant must be scaled. (Ishizuka 1982 eq 12) cos(B) = K/Kz K = 1/Lambda pointing to the displaced origin of the Ewald Sphere
			phase_grating[0].DivideByConstant(cosf(beam_tilt_magnitude));

			ctf[0].SetBeamTilt(beam_tilt_x, beam_tilt_y, 0.0f, 0.0f);
			ctf[1].SetBeamTilt(beam_tilt_x, beam_tilt_y, 0.0f, 0.0f);
		}




		amplitude_grating[0].SetToConstant(0.0f);
//		amplitude_grating[0].AddGaussianNoise(inelastic_potential[iSlab].ReturnSumOfSquares(0.0f));
//		amplitude_grating[0].AddConstant(inelastic_potential[iSlab].ReturnAverageOfRealValues(0.0f));

		if (iContrast > 0)
		{
			inelastic_potential[iSlab].ClipInto(&amplitude_grating[0],inelastic_potential[iSlab].ReturnAverageOfRealValuesOnEdges());

			if (amplitude_grating[0].ReturnAverageOfRealValues() > 0.001f)
			{


			// FIXME this is just multipied by the average value of C+C+N+0+C -> a projected mass density would be more accurate (26/Z*1.27) - this is also only valid at 300 KeV
//				amplitude_grating[0].MultiplyByConstant(3.2);

			if (requested_kv < 301 && requested_kv > 299)
			{
				amplitude_grating[0].MultiplyByConstant(1.158f);
			}
			else if(requested_kv < 201 && requested_kv > 199)
			{
				amplitude_grating[0].MultiplyByConstant(1.081f);
			}
			else
			{
				// Nothing to rescale at 100 KeV, this is where the inelastic/elastic ratios were empircally determined.
			}
			amplitude_grating[0].ForwardFFT(true);
			int i;
			int j;

			float x;
			float y;

			long pixel_counter = 0;
			float frequency_squared;

			// Bfactor (this is fit for an energy spread based envelope at 300 KeV)
			float energy_spread_bfactor = -2500;

			float inelastic_scalar_a = 0.005f;
			float inelastic_scalar_b = 0.0025f;
			std::complex<float> inelatic_scalar_b_complex;

			float angert_b = 1.0f; // The relative magnitude of the inelastic to elastic is accounted for in the calculation of the potentials, so this is set to 1
			float angert_c = -1.0*powf(22.6f,2);;

			float alt_1 = -200.0f;
			float alt_2 = 0.5f;

			bool apply_filter = true;
			bool apply_angert = false;
			bool apply_lorentzian = true;


			float sum_of_squares;

			if (apply_filter)
			{
				if ( apply_lorentzian )
				{

					Curve whitening_filter;
					Curve number_of_terms;

					whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((amplitude_grating[0].logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
					number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((amplitude_grating[0].logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));


					amplitude_grating[0].Compute1DRotationalAverage(whitening_filter,number_of_terms,true);

					whitening_filter.SquareRoot();
					whitening_filter.Reciprocal();
					whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());


					amplitude_grating[0].ApplyCurveFilter(&whitening_filter);
				}
				else
				{
					// In Angert's model, the Fourier transform of the inelastic potential is B*exp[-(k/c)^2]*(elastic potential)
					// They found for a 15 eV slit, and 300 keV elec, B = 10.4 +/- 5.4 and C = 22.6 +/- 5.7 from 117 images of amorphous carbon
				}




			for (j = 0; j <= amplitude_grating[0].physical_upper_bound_complex_y; j++)
			{
				y = powf(amplitude_grating[0].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * amplitude_grating[0].fourier_voxel_size_y, 2);
				for (i = 0; i <= amplitude_grating[0].physical_upper_bound_complex_x; i++)
				{
					x = powf(i * amplitude_grating[0].fourier_voxel_size_x, 2);

					// compute squared radius, in units of reciprocal pixels angstroms
					frequency_squared = (x + y) / pixel_size_squared ;

					if (apply_angert)
					{

						amplitude_grating[0].complex_values[pixel_counter] *= angert_b*expf(angert_c*frequency_squared); // FIXME only ~right for 300

					}
					else if (apply_lorentzian)
					{

//
						float  p1 =      0.8235;// (0.7736, 0.8734)
						float   p2 =        47.4;//  (47.13, 47.67)
						float   p3 =           1;//  (fixed at bound)
						float  q1 =        2334 ;// (2322, 2345)
					     float   q2 =       39.22 ;// (39.09, 39.36)
					     float q3 =       1.001 ;// (1.001, 1.002)
	//					amplitude_grating[0].complex_values[pixel_counter] *= ReturnPlasmonConversionFactor(frequency_squared); // FIXME only ~right for 300
	//						amplitude_grating[0].complex_values[pixel_counter] *= expf(energy_spread_bfactor*powf(frequency_squared,2)); // FIXME only ~right for 300

						float frequency = sqrtf(frequency_squared);
						float scale_value = (p1*frequency_squared + p2*frequency + p3) /
											(frequency_squared*frequency + q1*frequency_squared* + q2*frequency + q3);// / whitening_filter.ReturnLinearInterpolationFromX(frequency*pixel_size);




						amplitude_grating[0].complex_values[pixel_counter] *= scale_value;
	//					amplitude_grating[0].complex_values[pixel_counter] += inelatic_scalar_b_complex;
					}
					else
					{
						amplitude_grating[0].complex_values[pixel_counter] *= ((alt_2+expf(alt_1*frequency_squared))/(alt_2+1.0f));
					}
					pixel_counter++;

				}
			}

//			amplitude_grating[0].Compute1DRotationalAverage(whitening_filter,number_of_terms,true);
//			whitening_filter.WriteToFile("PostPS.txt");

			} // if apply filter

//			amplitude_grating[0].MultiplyByConstant(sqrtf(sum_of_squares / amplitude_grating[0].ReturnSumOfSquares()));

			amplitude_grating[0].BackwardFFT();

			} // if condition on amplitude contrast > 0
		} // if contition on pure phase contrast image


		MyDebugAssertFalse(scattering_potential[iSlab].HasNan(),"There is a NAN 1");
		MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2");


		// Up until here the phase gratings are identical. So I've only put values into (0);
		// Now we introduce the aperture:
		phase_grating[1].CopyFrom(&phase_grating[0]);
		amplitude_grating[1].CopyFrom(&amplitude_grating[0]);




		#pragma omp simd
		for ( long iPixel = 0; iPixel < phase_grating[0].real_memory_allocated; iPixel++)
		{
			phase_grating[0].real_values[iPixel] = expf(-amplitude_grating[0].real_values[iPixel]) * std::cos(phase_grating[0].real_values[iPixel]);
		}

		#pragma omp simd
		for ( long iPixel = 0; iPixel < phase_grating[1].real_memory_allocated; iPixel++)
		{
			phase_grating[1].real_values[iPixel] = expf(-amplitude_grating[1].real_values[iPixel]) * std::sin(phase_grating[1].real_values[iPixel]);
		}


		MyDebugAssertFalse(phase_grating[0].HasNan(),"There is a NAN 2a");
		MyDebugAssertFalse(phase_grating[1].HasNan(),"There is a NAN 3a");

		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 4; iPar++)
		{
			t_N[iPar].CopyFrom(&wave_function[copy_from_1[iPar]]);
			t_N[iPar].MultiplyPixelWise(phase_grating[mult_by[iPar]]);

			  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11");
			t_N[iPar].ForwardFFT(true);
			  MyDebugAssertFalse(t_N[iPar].HasNan(),"There is a NAN t11F");

		}

		// Reset the wave function to zero to store the update results
		wave_function[0].SetToConstant(0.0f);
		wave_function[1].SetToConstant(0.0f);

		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 4; iPar++)
		{
			// Get the real part of the new exit wave
			temp_img[iPar].CopyFrom(&t_N[iPar]);
			  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");

				if (do_beam_tilt_full)
				{
					// The propagtor must be normalized by Kz/K, and the current wave-front needs to be shifted BACK opposite the inclined
					// direction of propagtion, so that the next slice is relatively shifted forward along the direction of propagation.

					// Phase shifts here are in pixels & tan(B) ~ B
					float shift_x = 1.0f * propagator_distance[iSlab]*beam_tilt_shift_factor_x/pixel_size;
					float shift_y = 1.0f * propagator_distance[iSlab]*beam_tilt_shift_factor_y/pixel_size;

//					wxPrintf("Shifting by %f %f on slab %d\n", shift_x, shift_y, iSlab);

					temp_img[iPar].MultiplyByConstant(cosf(beam_tilt_magnitude));

					// Do not shift the final slab as there is no following slab for it to be shifted relative to.
//					if (iSlab < nSlabs - 1)
					{
						temp_img[iPar].PhaseShift(shift_x, shift_y ,0.0f);

						if (iPar == 0)
						{
							total_shift_x += shift_x;
							total_shift_y += shift_y;

						}
					}


//					wxPrintf("Shifting b %f slab %d\n",-1.0f* propagator_distance[iSlab]*beam_tilt_y/wanted_pixel_size,iSlab);


				}

			temp_img[iPar].ApplyCTF(fresnel_propagtor[prop_apply_real[iPar]],false);

			  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
			temp_img[iPar].BackwardFFT();
			  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

		}

		for (int iSeq = 0; iSeq < 4; iSeq++)
		{
			if (iSeq == 0)
			{
				wave_function[0].AddImage(&temp_img[iSeq]);
			}
			else
			{
				wave_function[0].SubtractImage(&temp_img[iSeq]);
			}
		}


		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 4; iPar++)
		{


			// Get the real part of the new exit wave
			temp_img[iPar].CopyFrom(&t_N[iPar]);
			  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1");


			temp_img[iPar].ApplyCTF(fresnel_propagtor[prop_apply_imag[iPar]],false);
			MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1CTF");
			temp_img[iPar].BackwardFFT();
			  MyDebugAssertFalse(temp_img[iPar].HasNan(),"There is a NAN temp1BFFT");

		}

		for (int iSeq = 0; iSeq < 4; iSeq++)
		{
			if (iSeq == 1)
			{
				wave_function[1].SubtractImage(&temp_img[iSeq]);
			}
			else
			{
				wave_function[1].AddImage(&temp_img[iSeq]);
			}
		}



	} // end of loop over slabs


	// Now apply the CTF for the center of mass, which also includes an envelopes (but not beam tilt as this depends has an axial dependence);
	#pragma omp parallel for num_threads(local_threads)
	for (int iPar = 0; iPar < 4; iPar++)
	{

		// Re-use t_N[0] through t_N[3]
		t_N[iPar].CopyFrom(&wave_function[copy_from_2[iPar]]);
		t_N[iPar].ForwardFFT(true);


		t_N[iPar].ApplyCTF(ctf[ctf_apply[iPar]],false,do_beam_tilt,do_coherence_envelope);


		if (do_beam_tilt_full)
		{

			t_N[iPar].PhaseShift(-0.5f*total_shift_x,-0.5f*total_shift_y,0.0f);
		}

		t_N[iPar].BackwardFFT();

	}

	wave_function[0].SetToConstant(0.0);
	wave_function[1].SetToConstant(0.0);

	wave_function[0].AddImage(&t_N[0]);
	wave_function[0].SubtractImage(&t_N[1]);
	wave_function[1].AddImage(&t_N[2]);
	wave_function[1].AddImage(&t_N[3]);

	 MyDebugAssertFalse(wave_function[0].HasNan(),"There is a NAN 6");
	  MyDebugAssertFalse(wave_function[1].HasNan(),"There is a NAN 7");

		double probability_total = 0;
		double probability_outside_aperture = 0;

	  if (iContrast > 0)
	  {

		// Re-use the amplitude grating here
		#pragma omp parallel for num_threads(local_threads)
		for (int iPar = 0; iPar < 2; iPar++)
		{
			wave_function[iPar].ForwardFFT();
			// Get the wave function inside and outside the aperture
			wave_function[iPar].MultiplyPixelWise(aperture_grating[0]);

			wave_function[iPar].BackwardFFT();
		}

	  }

	#pragma omp simd
	// Now get the square modulus of the wavefunction
	for (long iPixel = 0; iPixel < wave_function[0].real_memory_allocated; iPixel++)
	{
		 sum_image[tilt_IDX].real_values[iPixel] = (wave_function[0].real_values[iPixel]*wave_function[0].real_values[iPixel] + wave_function[1].real_values[iPixel]*wave_function[1].real_values[iPixel]);
	}




//	// Limit based on objective aperture. Ideally this would be done prior to "imaging" where we take the square modulus. Unfortunately, the division of the complex image into its real and imaginary parts
//	// only makes sense for linear operators (fft) and multiplication by scalars to both parts. Multiplying by the aperture function violates this linearity.
	if (iContrast > 0 && estimate_amplitude_contrast)
	{
		wxPrintf("Limiting the resolution based on an objective aperture of diameter %3.3f micron to %3.3f Angstrom\n", GetObjectiveAperture(), pixel_size/objective_aperture_resolution);


		ReturnImageContrast(sum_image[tilt_IDX], &total_contrast,false, tilt_angle);// - phase_contrast;
		sum_image[tilt_IDX].QuickAndDirtyWriteSlice("total.mrc",1,false,1);

		wxPrintf("Total contrast is %3.3e\n", total_contrast);
	}
	else if(estimate_amplitude_contrast)
	{
		ReturnImageContrast(sum_image[tilt_IDX], &phase_contrast, true, tilt_angle);
		sum_image[tilt_IDX].QuickAndDirtyWriteSlice("phase.mrc",1,false,1);


		wxPrintf("Phase contrast estimate at %3.3e\n",phase_contrast);
	}



	is_set_input_projections = false;

	} // loop on contrast

//	sum_image[tilt_IDX].MultiplyByConstant(expected_dose_per_pixel);
	return total_contrast; //( 1.0f - (total_contrast - phase_contrast)/total_contrast);

}

void WaveFunctionPropagator::ReturnImageContrast(Image &wave_function_sq_modulus, float* contrast, bool is_phase_contrast_image, float tilt_angle)
{
//	*contrast = wave_function_sq_modulus.ReturnAverageOfRealValues(sqrtf(unpadded_x_dimension*unpadded_x_dimension + unpadded_y_dimension*unpadded_y_dimension),false);

//	Image buffer1, buffer2;
//
//	buffer1.CopyFrom(&wave_function_sq_modulus);
////	buffer1.AddConstant(-1.0f*buffer1.ReturnAverageOfRealValues(0.35f*(float)buffer1.logical_x_dimension));
////	*contrast = buffer1.ReturnSumOfSquares(0.35f*(float)buffer1.logical_x_dimension);
//
//
////	wave_function_sq_modulus.ChangePixelSize(&buffer1, pixel_size / 0.375f, 0.05, false);
//	buffer2.CopyFrom(&buffer1);
////	float meanVal = buffer1.ReturnAverageOfRealValues(0.35f*(float)buffer1.logical_x_dimension,false);
////	buffer1.CopyFrom(&wave_function_sq_modulus);
////	buffer2.CopyFrom(&wave_function_sq_modulus);
////	buffer1.MultiplyAddConstant(-1.0f,meanVal);
////	buffer2.AddConstant(meanVal);
//	buffer1.MultiplyAddConstant(-1.0f,expected_dose_per_pixel);
//	buffer2.AddConstant(expected_dose_per_pixel);
//	buffer1.DividePixelWise(buffer2);
//
//	*contrast = buffer1.ReturnAverageOfRealValues(0.35f*(float)buffer1.logical_x_dimension,false);



	Image amplitude_spectrum;
	Image amplitude_spectrum_masked;
	Image buffer;
	RandomNumberGenerator my_rand(PIf);

	bool non_unique_id = true;
	int fileID;
	int hostname;
	char hostbuffer[256];

	std::string script_name;

	hostname = gethostname(hostbuffer, sizeof(hostbuffer));
	std::string file_id = std::to_string(getpid());
	file_id += hostbuffer;
	script_name = "./simulator_ctf_" + file_id + ".sh";


	std::ofstream myfile;

	float original_defocus1 = ctf_for_fitting->GetDefocus1() ; // The defocus is returned in pixels, not Angstrom
	float original_defocus2 = ctf_for_fitting->GetDefocus2();
	float known_astigmatism = (original_defocus1 - original_defocus2) / 2.0f * pixel_size;
	float known_astigmatism_angle = rad_2_deg(ctf_for_fitting->GetAstigmatismAzimuth());

	wxPrintf("Astigmatism params are %3.3e %3.3e\n", known_astigmatism, known_astigmatism_angle);

	wxPrintf("\t\t\n\nSEED is %lde\n\n", (long)my_rand.seed);




//	std::string phase_name = "/dev/shm/phase_" + file_id + ".mrc";
//	std::string text_file_name = "/dev/shm/fit_" + file_id;
//	std::string text_file_name_2 = "/dev/shm/fit_" + file_id + "_2.txt";
	std::string phase_name = "phase_" + file_id + ".mrc";
	std::string text_file_name = "fit_" + file_id;
	std::string text_file_name_2 = "fit_" + file_id + "_2.txt";

	std::string awk_command = "tail -n -1 " + text_file_name + ".txt | awk '{print $2,$3}' > " + text_file_name_2;
	std::string chmod_command = "chmod a=wrx " + script_name;
	std::string run_command = script_name;

	std::string clean_script_command = "rm -f " + script_name;
	std::string clean_image_command = "rm -f " + phase_name;
	std::string clean_diagnostic_command1 = "rm -f " + text_file_name + ".txt";
	std::string clean_diagnostic_command2 = "rm -f " + text_file_name + ".mrc";
	std::string clean_diagnostic_command3 = "rm -f " + text_file_name + "_avrot.txt";
	std::string clean_diagnostic_command4 = "rm -f " + text_file_name_2;


	//

	float found_defocus[2] = {0,0};

	if (is_phase_contrast_image)
	{

		myfile.open(script_name.c_str());
		wave_function_sq_modulus.QuickAndDirtyWriteSlice(phase_name,1,false);

		std::string do_tilt;
		if (fabsf(tilt_angle) > 20.0f)
		{
			do_tilt = "yes\n";
		}
		else
		{
			do_tilt = "no\n";
		}

		// Fixme change floats in params to ints.

		myfile << "#!/bin/bash\n\n";
		myfile << "ctffind << eof\n";
		myfile << phase_name + "\n";
		myfile << text_file_name + ".mrc\n";
		myfile << std::to_string(pixel_size) + "\n";
		myfile << std::to_string(for_ctffind.kv) + "\n";
		myfile << std::to_string(for_ctffind.Cs) + "\n";
		myfile << std::to_string(for_ctffind.AmplitudeContrast) + "\n";
		myfile << std::to_string(for_ctffind.Size) + "\n";
		myfile << std::to_string(myroundint(for_ctffind.min_resolution)) + "\n";
		myfile << std::to_string(myroundint(for_ctffind.max_resolution)) + "\n";
		myfile << std::to_string(myroundint(for_ctffind.min_defocus)) + "\n";
		myfile << std::to_string(myroundint(for_ctffind.max_defocus)) + "\n";
		myfile << std::to_string(myroundint(for_ctffind.defocus_step)) + "\n";
		myfile << "yes\n"; // know astig?
		myfile << "yes\n";
		myfile << std::to_string(0) + "\n"; //astig
		myfile << std::to_string(0) + "\n"; // ang
		myfile << "no\n"; // phase shift
		myfile << do_tilt; // tilt
		myfile << "yes\n"; // expert opt
		myfile << "yes\n"; // resample if too small
		myfile << "no\n"; // know defocus
		myfile << std::to_string(myroundint(std::max(for_ctffind.nThreads,2.0f))) + "\n"; // nThreads
		myfile << "eof\n";
		myfile.close();
//

//		myfile << text_file_name + ".mrc\n";
//		myfile << std::to_string(pixel_size) + "\n";
//		myfile << std::to_string(300) + "\n";
//		myfile << std::to_string(2.7) + "\n";
//		myfile << std::to_string(0) + "\n";
//		myfile << std::to_string(512) + "\n";
//		myfile << std::to_string(20) + "\n";
//		myfile << std::to_string(4) + "\n";
//		myfile << std::to_string(5000) + "\n";
//		myfile << std::to_string(10000) + "\n";
//		myfile << std::to_string(2) + "\n";
//		myfile << "yes\n"; // know astig?
//		myfile << "yes\n";
//		myfile << std::to_string(0) + "\n"; //astig
//		myfile << std::to_string(0) + "\n"; // ang
//		myfile << "no\n"; // phase shift
//		myfile << do_tilt; // tilt
//		myfile << "yes\n"; // expert opt
//		myfile << "yes\n"; // resample if too small
//		myfile << "no\n"; // know defocus
//		myfile << std::to_string(16) + "\n"; // nThreads
//		myfile << "eof\n";
//		myfile.close();
//		myfile << "#" + text_file_name + ".mrc\n";
//		myfile << "#" +std::to_string(pixel_size) + "\n";
//		myfile << "#" +std::to_string(300) + "\n";
//		myfile << "#" +std::to_string(2.7) + "\n";
//		myfile << "#" +std::to_string(0) + "\n";
//		myfile << "#" +std::to_string(512) + "\n";
//		myfile << "#" +std::to_string(20) + "\n";
//		myfile << "#" +std::to_string(4) + "\n";
//		myfile << "#" +std::to_string(5000) + "\n";
//		myfile << "#" +std::to_string(10000) + "\n";
//		myfile << "#" +std::to_string(2) + "\n";
//		myfile << "#yes\n"; // know astig?
//		myfile << "#yes\n";
//		myfile << "#" +std::to_string(0) + "\n"; //astig
//		myfile << "#" +std::to_string(0) + "\n"; // ang
//		myfile << "#no\n"; // phase shift
//		myfile << "#" +do_tilt; // tilt
//		myfile << "#yes\n"; // expert opt
//		myfile << "#yes\n"; // resample if too small
//		myfile << "#no\n"; // know defocus
//		myfile << "#" +std::to_string(16) + "\n"; // nThreads
//		myfile << "#eof\n";
//		myfile.close();

		std::system(chmod_command.c_str());
		std::system(run_command.c_str());
		std::system(awk_command.c_str());

		NumericTextFile myfile_in(text_file_name_2,0,2);
		myfile_in.ReadLine(found_defocus);
		myfile_in.Close();

		wxPrintf("\n\n\t\tFound a defocus of %f %f\n\n", found_defocus[0], found_defocus[1]);

		std::system(clean_script_command.c_str());
		std::system(clean_image_command.c_str());
		std::system(clean_diagnostic_command1.c_str());
		std::system(clean_diagnostic_command2.c_str());
		std::system(clean_diagnostic_command3.c_str());
		std::system(clean_diagnostic_command4.c_str());




	}

		amplitude_spectrum.CopyFrom(&wave_function_sq_modulus);

	amplitude_spectrum.ForwardFFT();

//	float high_pass_from = min_resolution_for_fitting; // Ang
//	amplitude_spectrum.CosineMask(0, 2.0f*pixel_size / high_pass_from, true);
//	amplitude_spectrum.QuickAndDirtyWriteSlice("Amp.mrc",1,false);
	amplitude_spectrum_masked.Allocate(amplitude_spectrum.logical_x_dimension, amplitude_spectrum.logical_y_dimension, 1, true);
	buffer.Allocate(amplitude_spectrum.logical_x_dimension, amplitude_spectrum.logical_y_dimension, 1, true);
	amplitude_spectrum.ComputeAmplitudeSpectrumFull2D(&amplitude_spectrum_masked,false,1.0f);
	wave_function_sq_modulus.ForwardFFT(true);
	wave_function_sq_modulus.ComputeAmplitudeSpectrumFull2D(&amplitude_spectrum);
	wave_function_sq_modulus.BackwardFFT();

//
	float best_score = std::numeric_limits<float>::min();
	double current_score;
	float best_fit_value;
	float precomputed_amplitude_contrast_term;
	int number_to_correlate = 1;
	double norm_image;
	double image_mean;
	int* addresses;
	float* spatial_frequency_squared;
	float* azimuths;

	float min_to_fit;
	float max_to_fit;
	float step_for_fit;

	wxPrintf("def1 %6.6e def2 %6.6e btx/y %3.3e %3.3e CS %3.3e AMP %3.3e ASTIG %3.3e APH %3.3e WAVELENGTH %3.3e\n",
			ctf_for_fitting->GetDefocus1(),ctf_for_fitting->GetDefocus2(), ctf_for_fitting->GetBeamTiltX(),ctf_for_fitting->GetBeamTiltY(),
			ctf_for_fitting->GetSphericalAberration(), ctf_for_fitting->GetAmplitudeContrast(),ctf_for_fitting->GetAstigmatismAzimuth(),
			ctf_for_fitting->GetAdditionalPhaseShift(),ctf_for_fitting->GetWavelength());



	float average;
	float sigma;

	amplitude_spectrum.CopyFrom(&amplitude_spectrum_masked);
	amplitude_spectrum.ComputeFilteredAmplitudeSpectrumFull2D(&amplitude_spectrum_masked, &buffer, average, sigma, 1/min_resolution_for_fitting, 1/max_resolution_for_fitting,pixel_size);




		// fix the defocus offset and fit the amplitude contrast
		min_to_fit = 0.0f;
		max_to_fit = 0.5f;
		step_for_fit = 0.05;


	if (! is_phase_contrast_image)
	{
		for (int iIter = 0; iIter < 4; iIter++)
		{

			for ( float iFit = min_to_fit; iFit <= max_to_fit; iFit+=step_for_fit )
			{
				if (is_phase_contrast_image || iFit >= 0.0f)
				{
					// Set up the CTF object.
					if (is_phase_contrast_image)
					{
						// The error should only be a slight defocus offset, the astigmatism should be correct.
						ctf_for_fitting->SetDefocus(original_defocus1 + iFit, original_defocus2 + iFit, ctf_for_fitting->GetAstigmatismAzimuth());
						ctf_for_fitting->SetAdditionalPhaseShift(0.0f);

					}
					else
					{
						if (fabs(iFit - 1.0) < 1e-3) iFit = PIf / 2.0f;
						else precomputed_amplitude_contrast_term = atanf(iFit/sqrtf(1.0 - powf(iFit, 2)));
						ctf_for_fitting->SetAdditionalPhaseShift(precomputed_amplitude_contrast_term);
					}


					amplitude_spectrum_masked.SetupQuickCorrelationWithCTF(*ctf_for_fitting, number_to_correlate, norm_image, image_mean, NULL, NULL, NULL);
					azimuths = new float[number_to_correlate];
					spatial_frequency_squared = new float[number_to_correlate];
					addresses = new int[number_to_correlate];
					amplitude_spectrum_masked.SetupQuickCorrelationWithCTF(*ctf_for_fitting, number_to_correlate, norm_image, image_mean, addresses, spatial_frequency_squared, azimuths);
					current_score = amplitude_spectrum_masked.QuickCorrelationWithCTF(*ctf_for_fitting, number_to_correlate, norm_image, image_mean, addresses, spatial_frequency_squared, azimuths);

					if (current_score > best_score)
					{
						best_score = current_score;
						best_fit_value = iFit;
					}

//					if (is_phase_contrast_image)
//					{
//						wxPrintf("For iFit %2.4f, score is %3.5e\n", pixel_size*(iFit + original_defocus1), current_score);
//					}
//					{
//						wxPrintf("For iFit %2.4f, score is %3.5e\n", iFit, current_score);
//					}


					delete [] azimuths;
					delete [] spatial_frequency_squared;
					delete [] addresses;
				}


			}



			if (is_phase_contrast_image)
			{
				wxPrintf("\n\n\tBest defocus so far is %2.4f %2.4f, score %3.3e\n", pixel_size*(best_fit_value + original_defocus1),pixel_size*(best_fit_value + original_defocus2), best_score);

			}
			else
			{
				wxPrintf("\n\n\tBest amplitude contrast so far is %2.4f, score %3.3e\n", best_fit_value, best_score);

			}


			// Step down by 1/10
			min_to_fit = best_fit_value - step_for_fit*2;
			max_to_fit = best_fit_value + step_for_fit*2;
			step_for_fit /= 10.0f;
		}

	}




    if (is_phase_contrast_image)
    {
        // Taking the best fit, adjust the ctfs for the amplitude object
//        ctf[0].SetDefocus(ctf[0].GetDefocus1() - best_fit_value,
//        				  ctf[0].GetDefocus2() - best_fit_value,
//    					  ctf[0].GetAstigmatismAzimuth());
//
//        ctf[1].SetDefocus(ctf[1].GetDefocus1() - best_fit_value,
//        				  ctf[1].GetDefocus2() - best_fit_value,
//    					  ctf[1].GetAstigmatismAzimuth());
    	float fit_1 = original_defocus1 - (found_defocus[0] / pixel_size);
    	float fit_2 = original_defocus2 - (found_defocus[1] / pixel_size);
    	wxPrintf("Setting defocus from %f to %f %f\n", ctf[0].GetDefocus1(), fit_1, ctf[0].GetDefocus1() + fit_1);
        ctf[0].SetDefocus(ctf[0].GetDefocus1() + fit_1 ,
        				  ctf[0].GetDefocus2() + fit_2,
    					  ctf[0].GetAstigmatismAzimuth());

        ctf[1].SetDefocus(ctf[1].GetDefocus1() + fit_1 ,
				  	  	  ctf[1].GetDefocus2() + fit_2,
    					  ctf[1].GetAstigmatismAzimuth());


        ctf_for_fitting->SetDefocus(original_defocus1,original_defocus2,ctf_for_fitting->GetAstigmatismAzimuth());

        *contrast = 0.0f;
    }
    else
    {
    	*contrast = best_fit_value;
    }







}



