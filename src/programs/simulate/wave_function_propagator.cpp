/*
 * wave_function_propagator.cpp
 *
 *  Created on: Oct 2, 2019
 *      Author: himesb
 */

#include "../../core/core_headers.h"
#include "wave_function_propagator.h"

WaveFunctionPropagator::WaveFunctionPropagator(float set_real_part_wave_function_in, float wanted_objective_aperture_diameter_micron,
											   float wanted_pixel_size, int wanted_number_threads, float beam_tilt_x, float beam_tilt_y, bool do_beam_tilt_full)
{
	temp_img = new Image[4];
	t_N = new Image[4];
	wave_function = new Image[2];
	phase_grating = new Image[2];
	amplitude_grating = new Image[2];
	aperture_grating = new Image[2];
	ctf = new CTF[2];
	fresnel_propagtor = new CTF[2];

	wave_function_in = new float[2];
	wave_function_in[0] = set_real_part_wave_function_in;
	wave_function_in[1] = 0.0f;

	pixel_size = wanted_pixel_size;
	pixel_size_squared = pixel_size * pixel_size;
	nThreads = wanted_number_threads;

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
}

void WaveFunctionPropagator::SetCTF(float wanted_acceleration_voltage,
									float wanted_spherical_aberration,
									float wanted_defocus_1,
									float wanted_defocus_2,
									float wanted_astigmatism_azimuth,
									float wanted_additional_phase_shift_in_radians)
{
	// defocus values are assumed to include any absolute offsets relative to a slabs position in the full 3d specimen.
	// The amplitude contrast is forced to 1 or 0 to retrieve just the real/imag portion of the conventional CTF
	//

	ctf[0].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				1.0f,
				wanted_defocus_1,
				wanted_defocus_2,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians+PIf);

	ctf[1].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				0.0f,
				wanted_defocus_1,
				wanted_defocus_2,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians+PIf);


	requested_kv = wanted_acceleration_voltage;
	is_set_ctf = true;
}

void WaveFunctionPropagator::SetCTF(float wanted_acceleration_voltage,
									float wanted_spherical_aberration,
									float wanted_defocus_1,
									float wanted_defocus_2,
									float wanted_astigmatism_azimuth,
									float wanted_additional_phase_shift_in_radians,
									float wanted_dose_rate)
{
	// defocus values are assumed to include any absolute offsets relative to a slabs position in the full 3d specimen.
	// The amplitude contrast is forced to 1 or 0 to retrieve just the real/imag portion of the conventional CTF
	//

	ctf[0].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				1.0f,
				wanted_defocus_1,
				wanted_defocus_2,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians+PIf);

	ctf[1].Init(wanted_acceleration_voltage,
				wanted_spherical_aberration,
				0.0,
				wanted_defocus_1,
				wanted_defocus_2,
				wanted_astigmatism_azimuth,
				pixel_size,
				wanted_additional_phase_shift_in_radians+PIf);


	ctf[0].SetEnvelope(wanted_acceleration_voltage,pixel_size,wanted_dose_rate / (pixel_size_squared));
	ctf[1].SetEnvelope(wanted_acceleration_voltage,pixel_size,wanted_dose_rate / (pixel_size_squared));

	requested_kv = wanted_acceleration_voltage;
	do_coherence_envelope = true;
	is_set_ctf = true;
}

void WaveFunctionPropagator::SetFresnelPropagator(float wanted_acceleration_voltage, float propagation_distance)
{

	fresnel_propagtor[0].Init(wanted_acceleration_voltage,
					   0.0,
					   1.0,
					   - propagation_distance,
					   - propagation_distance,
					   0.0,
					   pixel_size,
					   0.0 + PIf);

	fresnel_propagtor[1].Init(wanted_acceleration_voltage,
					   0.0,
					   0.0,
					   - propagation_distance,
					   - propagation_distance,
					   0.0,
					   pixel_size,
					   0.0+PIf);

	is_set_fresnel_propagator = true;

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
										   float* image_mean, float* inelastic_mean, float* propagator_distance)
{
	MyAssertTrue(is_set_ctf && is_set_fresnel_propagator, "Either the ctf or fresnel propagtor are not set")

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



	//
	for (int iContrast = 0 ; iContrast < 2; iContrast++)
	{


	SetInputWaveFunction(size_x, size_y);

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

	float total_shift_x = 0.0f;
	float total_shift_y = 0.0f;


	for (int iSlab = 0; iSlab < nSlabs; iSlab++)
	{


		phase_grating[0].SetToConstant(0.0f);

		scattering_potential[iSlab].ClipInto(&phase_grating[0],image_mean[iSlab]);

		if (do_beam_tilt)
		{
			// For tilted illumination, the scattering plane sees only the z-component of the wave-vector = lower energy = stronger interaction
			// so the interaction constant must be scaled. (Ishizuka 1982 eq 12) cos(B) = K/Kz K = 1/Lambda pointing to the displaced origin of the Ewald Sphere
			phase_grating[0].DivideByConstant(cosf(beam_tilt_magnitude));

			ctf[0].SetBeamTilt(beam_tilt_x, beam_tilt_y, 0.0f, 0.0f);
			ctf[1].SetBeamTilt(beam_tilt_x, beam_tilt_y, 0.0f, 0.0f);
		}




		amplitude_grating[0].SetToConstant(0.0f);

		if (iContrast > 0) // FIXME temp override
		{
			inelastic_potential[iSlab].ClipInto(&amplitude_grating[0],inelastic_mean[iSlab]);

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
			for (j = 0; j <= amplitude_grating[0].physical_upper_bound_complex_y; j++)
			{
				y = powf(amplitude_grating[0].ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * amplitude_grating[0].fourier_voxel_size_y, 2);
				for (i = 0; i <= amplitude_grating[0].physical_upper_bound_complex_x; i++)
				{
					x = powf(i * amplitude_grating[0].fourier_voxel_size_x, 2);

					// compute squared radius, in units of reciprocal pixels angstroms
					frequency_squared = (x + y) / pixel_size_squared ;
//					amplitude_grating[0].complex_values[pixel_counter] *= ReturnPlasmonConversionFactor(frequency_squared); // FIXME only ~right for 300
//						amplitude_grating[0].complex_values[pixel_counter] *= expf(energy_spread_bfactor*powf(frequency_squared,2)); // FIXME only ~right for 300
					inelatic_scalar_b_complex =  inelastic_scalar_a * amplitude_grating[0].complex_values[pixel_counter] / abs(amplitude_grating[0].complex_values[pixel_counter]);
					if ( ! std::isfinite(abs(inelatic_scalar_b_complex)))
					{
						inelatic_scalar_b_complex = 0.0f + I*0.0f;
					}


					amplitude_grating[0].complex_values[pixel_counter] *= inelastic_scalar_b / (sqrtf(frequency_squared) + inelastic_scalar_b);
//					amplitude_grating[0].complex_values[pixel_counter] += inelatic_scalar_b_complex;
					pixel_counter++;

				}
			}
			amplitude_grating[0].BackwardFFT();
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
					float shift_x = shift_sign* propagator_distance[iSlab]*beam_tilt_shift_factor_x/pixel_size;
					float shift_y = shift_sign* propagator_distance[iSlab]*beam_tilt_shift_factor_y/pixel_size;

					wxPrintf("Shifting by %f %f on slab %d\n", shift_x, shift_y, iSlab);

					temp_img[iPar].MultiplyByConstant(cosf(beam_tilt_magnitude));
					temp_img[iPar].PhaseShift(shift_x, shift_y ,0.0f);


					if (iPar == 0)
					{
						total_shift_x += shift_x;
						total_shift_y += shift_y;
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
//			wxPrintf("Shifting back by half the total to keep particle centered ( %3.3f %3.3f) pixels\n",-0.5f*shift_sign*total_shift_x,-0.5f*shift_sign*total_shift_y);
//
//			t_N[iPar].PhaseShift(-0.5f*shift_sign*total_shift_x,-0.5f*shift_sign*total_shift_y,0.0f);
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
	if (iContrast > 0)
	{
		wxPrintf("Limiting the resolution based on an objective aperture of diameter %3.3f micron to %3.3f Angstrom\n", GetObjectiveAperture(), pixel_size/objective_aperture_resolution);


		ReturnImageContrast(sum_image[tilt_IDX], &total_contrast, &mean_value);// - phase_contrast;
		sum_image[tilt_IDX].QuickAndDirtyWriteSlice("total.mrc",1,false,1);

		wxPrintf("Total contrast is %3.3e\n", total_contrast);
	}
	else
	{
		ReturnImageContrast(sum_image[tilt_IDX], &phase_contrast, &mean_value);
		sum_image[tilt_IDX].QuickAndDirtyWriteSlice("phase.mrc",1,false,1);
		wxPrintf("Phase contrast estimate at %3.3e\n",phase_contrast);
	}



	is_set_input_projections = false;

	} // loop on contrast

//	return ((total_contrast/phase_contrast) - 1.0f);
	return ( 1.0f - total_contrast/phase_contrast);


}

void WaveFunctionPropagator::ReturnImageContrast(Image &wave_function_sq_modulus, float* contrast, float* mean_value)
{
	*contrast = wave_function_sq_modulus.ReturnAverageOfRealValues(sqrtf(unpadded_x_dimension*unpadded_x_dimension + unpadded_y_dimension*unpadded_y_dimension),false);
	/*
	Image buffer1;
	buffer1.CopyFrom(&wave_function_sq_modulus);
//	buffer1.CalculateDerivative(0.0f, 0.0f, 0.0f);

	EmpiricalDistribution dist;

	int padVal_y = (buffer1.logical_y_dimension - unpadded_y_dimension)/2;
	int padVal_x = (buffer1.logical_x_dimension - unpadded_x_dimension)/2;


	for (int j = padVal_y ; j <  buffer1.logical_y_dimension - padVal_y - 1 ; j ++)
	{
		for (int i = padVal_x ;i < buffer1.logical_x_dimension - padVal_x - 1; i ++)
		{
			dist.AddSampleValue(buffer1.ReturnRealPixelFromPhysicalCoord(i,j,0));
		}
	}

	wxPrintf("Min max mean %f %f %f\n", dist.GetMinimum(), dist.GetMaximum(), dist.GetSampleMean());
	float firstSubtract = -1.0f*dist.GetMinimum();
	float thenDivide = 1.0f/(dist.GetMaximum() + firstSubtract);
	buffer1.AddMultiplyConstant(firstSubtract, thenDivide);


	dist.Reset();
	for (int j = padVal_y ; j <  buffer1.logical_y_dimension - padVal_y - 1 ; j ++)
	{
		for (int i = padVal_x ;i < buffer1.logical_x_dimension - padVal_x - 1; i ++)
		{
			dist.AddSampleValue(buffer1.ReturnRealPixelFromPhysicalCoord(i,j,0));
		}
	}

	wxPrintf("Min max mean %f %f %f\n", dist.GetMinimum(), dist.GetMaximum(), dist.GetSampleMean());

	*contrast = sqrtf(dist.GetSampleVariance());
*/

//	, buffer2;
//	buffer1.CopyFrom(&wave_function_sq_modulus);
//	buffer2.CopyFrom(&wave_function_sq_modulus);
//
//	*mean_value = buffer1.ReturnAverageOfRealValues(0.0f, false);
//	wxPrintf("\nAverage of wave funct sq is %3.3e\n",*mean_value);
//	buffer1.MultiplyAddConstant(-1.0f,*mean_value);
////	buffer1.DivideByConstant(*mean_value);
//	buffer2.AddConstant(*mean_value);
//	buffer1.DividePixelWise(buffer2);


//	*contrast = (max - min) / (max + min);//buffer1.ReturnAverageOfRealValues(0.0f,false);
}



