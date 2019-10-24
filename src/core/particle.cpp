#include "core_headers.h"

ParameterMap::ParameterMap()
{
	phi = false;
	theta = false;
	psi = false;
	x_shift = false;
	y_shift = false;
}

void ParameterMap::SetAllTrue()
{
	phi = true;
	theta = true;
	psi = true;
	x_shift = true;
	y_shift = true;
}

Particle::Particle()
{
	Init();
}

Particle::Particle(int wanted_logical_x_dimension, int wanted_logical_y_dimension)
{
	Init();

	AllocateImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
}

Particle::~Particle()
{

	if (particle_image != NULL)
	{
		delete particle_image;
	}

	if (ctf_image != NULL)
	{
		delete ctf_image;
	}

	if (beamtilt_image != NULL)
	{
		delete beamtilt_image;
	}

	if (bin_index != NULL)
	{
		delete [] bin_index;
	}
}

void Particle::CopyAllButImages(const Particle *other_particle)
{
   // Check for self assignment
   if(this != other_particle)
   {
	   origin_micrograph = other_particle->origin_micrograph;
	   origin_x_coordinate = other_particle->origin_x_coordinate;
	   origin_y_coordinate = other_particle->origin_y_coordinate;
	   location_in_stack = other_particle->location_in_stack;
	   pixel_size = other_particle->pixel_size;
	   sigma_signal = other_particle->sigma_signal;
	   sigma_noise = other_particle->sigma_noise;
	   snr = other_particle->snr;
	   logp = other_particle->logp;
	   particle_occupancy = other_particle->particle_occupancy;
	   particle_score = other_particle->particle_score;
	   alignment_parameters = other_particle->alignment_parameters;
	   scaled_noise_variance = other_particle->scaled_noise_variance;
	   parameter_constraints = other_particle->parameter_constraints;
	   ctf_parameters = other_particle->ctf_parameters;
	   current_ctf = other_particle->current_ctf;
	   ctf_is_initialized = other_particle->ctf_is_initialized;
	   ctf_image_calculated = false;
	   beamtilt_image_calculated = false;
	   includes_reference_ssnr_weighting = false;
	   is_normalized = false;
	   is_phase_flipped = false;
	   is_masked = false;
	   mask_radius = other_particle->mask_radius;
	   mask_falloff = other_particle->mask_falloff;
	   mask_volume = other_particle->mask_volume;
	   molecular_mass_kDa = other_particle->molecular_mass_kDa;
	   is_filtered = false;
	   filter_radius_low = other_particle->filter_radius_low;
	   filter_radius_high = other_particle->filter_radius_high;
	   filter_falloff = other_particle->filter_falloff;
	   filter_volume = other_particle->filter_volume;
	   signed_CC_limit = other_particle->signed_CC_limit;
	   is_ssnr_filtered = false;
	   is_centered_in_box = true;
	   shift_counter = 0;
	   insert_even = other_particle->insert_even;
	   target_phase_error = other_particle->target_phase_error;
	   current_parameters = other_particle->current_parameters;
	   temp_parameters = other_particle->temp_parameters;
	   parameter_average = other_particle->parameter_average;
	   parameter_variance = other_particle->parameter_variance;
	   parameter_map = other_particle->parameter_map;
	   constraints_used = other_particle->constraints_used;
	   number_of_search_dimensions = other_particle->number_of_search_dimensions;
	   mask_center_2d_x = other_particle->mask_center_2d_x;
	   mask_center_2d_y = other_particle->mask_center_2d_y;
	   mask_center_2d_z = other_particle->mask_center_2d_z;
	   mask_radius_2d = other_particle->mask_radius_2d;
	   apply_2D_masking = other_particle->apply_2D_masking;
	   no_ctf_weighting = false;
	   complex_ctf = other_particle->complex_ctf;

	   if (particle_image != NULL) {delete particle_image; particle_image = NULL;}
	   if (ctf_image != NULL) {delete ctf_image; ctf_image = NULL;}
	   if (beamtilt_image != NULL) {delete beamtilt_image; beamtilt_image = NULL;}
	   if (bin_index != NULL) {delete [] bin_index; bin_index = NULL;}
   }
}

void Particle::Init()
{
	target_phase_error = 45.0;
	origin_micrograph = -1;
	origin_x_coordinate = -1;
	origin_y_coordinate = -1;
	location_in_stack = -1;
	pixel_size = 0.0;
	sigma_signal = 0.0;
	sigma_noise = 0.0;
	snr = 0.0;
	logp = -std::numeric_limits<float>::max();;
	particle_occupancy = 0.0;
	particle_score = 0.0;
	particle_image = NULL;
	scaled_noise_variance = 0.0;
	ctf_is_initialized = false;
	ctf_image = NULL;
	ctf_image_calculated = false;
	beamtilt_image = NULL;
	beamtilt_image_calculated = false;
	includes_reference_ssnr_weighting = false;
	is_normalized = false;
	is_phase_flipped = false;
	is_masked = false;
	mask_radius = 0.0;
	mask_falloff = 0.0;
	mask_volume = 0.0;
	molecular_mass_kDa = 0.0;
	is_filtered = false;
	filter_radius_low = 0.0;
	filter_radius_high = 0.0;
	filter_falloff = 0.0;
	filter_volume = 0.0;
	signed_CC_limit = 0.0;
	is_ssnr_filtered = false;
	is_centered_in_box = true;
	shift_counter = 0;
	insert_even = false;
	number_of_search_dimensions = 0;
	bin_index = NULL;
	mask_center_2d_x = 0.0;
	mask_center_2d_y = 0.0;
	mask_center_2d_z = 0.0;
	mask_radius_2d = 0.0;
	apply_2D_masking = false;
	no_ctf_weighting = false;
	complex_ctf = false;
}

void Particle::AllocateImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension)
{
	if (particle_image == NULL)
	{
		particle_image = new Image;
	}
	particle_image->Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, true);
}

void Particle::AllocateCTFImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension)
{
	if (ctf_image == NULL)
	{
		ctf_image = new Image;
	}
	ctf_image->Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);

	if (beamtilt_image == NULL)
	{
		beamtilt_image = new Image;
	}
	beamtilt_image->Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);
}

void Particle::Allocate(int wanted_logical_x_dimension, int wanted_logical_y_dimension)
{
	AllocateImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
	AllocateCTFImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
}

void Particle::Deallocate()
{
	if (particle_image != NULL) {delete particle_image; particle_image = NULL;}
	if (ctf_image != NULL) {delete ctf_image; ctf_image = NULL;}
	if (beamtilt_image != NULL) {delete beamtilt_image; beamtilt_image = NULL;}
}

void Particle::ResetImageFlags()
{
	includes_reference_ssnr_weighting = false;
	is_normalized = false;
	is_phase_flipped = false;
	if (is_masked) {is_masked = false; mask_radius = 0.0; mask_falloff = 0.0; mask_volume = 0.0;};
	if (is_filtered) {is_filtered = false; filter_radius_low = 0.0; filter_radius_high = 0.0; filter_falloff = 0.0; filter_volume = 0.0;};
	is_ssnr_filtered = false;
	is_centered_in_box = true;
	shift_counter = 0;
	logp = -std::numeric_limits<float>::max();;
	insert_even = false;
	no_ctf_weighting = false;
}

void Particle::PhaseShift()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(abs(shift_counter) < 2, "Image already shifted");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->PhaseShift(alignment_parameters.ReturnShiftX() / pixel_size, alignment_parameters.ReturnShiftY() / pixel_size);
	shift_counter += 1;
}

void Particle::PhaseShiftInverse()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(abs(shift_counter) < 2, "Image already shifted");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->PhaseShift(- alignment_parameters.ReturnShiftX() / pixel_size, - alignment_parameters.ReturnShiftY() / pixel_size);
	shift_counter -= 1;
}

void Particle::Whiten(float resolution_limit)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! particle_image->is_in_real_space, "Image not in Fourier space");

	particle_image->Whiten(resolution_limit);
}

void Particle::ForwardFFT(bool do_scaling)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(particle_image->is_in_real_space, "Image not in real space");

	particle_image->ForwardFFT(do_scaling);
}

void Particle::BackwardFFT()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! particle_image->is_in_real_space, "Image not in Fourier space");

	particle_image->BackwardFFT();
}

void Particle::CosineMask(bool invert, bool force_mask_value, float wanted_mask_value)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_masked, "Image already masked");

	if (! particle_image->is_in_real_space) particle_image->BackwardFFT();
	mask_volume = particle_image->CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, invert, force_mask_value, wanted_mask_value);
	is_masked = true;
}

void Particle::CenterInBox()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! particle_image->object_is_centred_in_box, "Image already centered");
	MyDebugAssertTrue(! is_centered_in_box, "Image already centered");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->SwapRealSpaceQuadrants();
	is_centered_in_box = true;
}

void Particle::CenterInCorner()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(particle_image->object_is_centred_in_box, "Image already centered");
	MyDebugAssertTrue(is_centered_in_box, "Image already in corner");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->SwapRealSpaceQuadrants();
	is_centered_in_box = false;
}

void Particle::InitCTF(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift, float beam_tilt_x, float beam_tilt_y, float particle_shift_x, float particle_shift_y)
{
//	MyDebugAssertTrue(! ctf_is_initialized, "CTF already initialized");

	ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.0, 0.0, pixel_size, phase_shift, beam_tilt_x, beam_tilt_y, particle_shift_x, particle_shift_y);
	ctf_is_initialized = true;
}

void Particle::SetDefocus(float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift)
{
	MyDebugAssertTrue(ctf_is_initialized, "CTF not initialized");

	ctf_parameters.SetDefocus(defocus_1 / pixel_size, defocus_2 / pixel_size, deg_2_rad(astigmatism_angle));
	ctf_parameters.SetAdditionalPhaseShift(phase_shift);
}

void Particle::SetBeamTilt(float beam_tilt_x, float beam_tilt_y, float particle_shift_x, float particle_shift_y)
{
	MyDebugAssertTrue(ctf_is_initialized, "CTF not initialized");

	ctf_parameters.SetBeamTilt(beam_tilt_x, beam_tilt_y, particle_shift_x / pixel_size, particle_shift_y / pixel_size);
}

void Particle::SetLowResolutionContrast(float low_resolution_contrast)
{
	MyDebugAssertTrue(ctf_is_initialized, "CTF not initialized");

	ctf_parameters.SetLowResolutionContrast(low_resolution_contrast);
}

void Particle::InitCTFImage(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift, float beam_tilt_x, float beam_tilt_y, float particle_shift_x, float particle_shift_y, bool calculate_complex_ctf)
{
	MyDebugAssertTrue(ctf_image->is_in_memory, "ctf_image memory not allocated");
	MyDebugAssertTrue(beamtilt_image->is_in_memory, "beamtilt_image memory not allocated");
	MyDebugAssertTrue(! ctf_image->is_in_real_space, "ctf_image not in Fourier space");
	MyDebugAssertTrue(! beamtilt_image->is_in_real_space, "beamtilt_image not in Fourier space");

	InitCTF(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, phase_shift, beam_tilt_x, beam_tilt_y, particle_shift_x, particle_shift_y);
	complex_ctf = calculate_complex_ctf;
	if (ctf_parameters.IsAlmostEqualTo(&current_ctf, 1 / pixel_size) == false || ! ctf_image_calculated)
	// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
	{
		current_ctf = ctf_parameters;
		ctf_image->CalculateCTFImage(current_ctf, complex_ctf);
	}
	if (ctf_parameters.BeamTiltIsAlmostEqualTo(&current_ctf) == false || ! beamtilt_image_calculated)
	// Need to calculate current_beamtilt_image to correct input image for beam tilt
	{
		beamtilt_image->CalculateBeamTiltImage(current_ctf);
	}
	ctf_image_calculated = true;
	beamtilt_image_calculated = true;
}

void Particle::PhaseFlipImage()
{
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not calculated");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->PhaseFlipPixelWise(*ctf_image);
}

void Particle::CTFMultiplyImage()
{
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not calculated");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->MultiplyPixelWiseReal(*ctf_image);
}

void Particle::BeamTiltMultiplyImage()
{
	MyDebugAssertTrue(beamtilt_image_calculated, "Beamtilt image not calculated");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->MultiplyPixelWise(*beamtilt_image);
}

void Particle::SetIndexForWeightedCorrelation(bool limit_resolution)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Image memory not allocated");

	int i;
	int j;
	int k;
	int bin;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;

	float low_limit2;
	float high_limit2 = fminf(powf(pixel_size / filter_radius_high,2),0.25);

	int number_of_bins = particle_image->ReturnLargestLogicalDimension() / 2 + 1;
	int number_of_bins2 = 2 * (number_of_bins - 1);

	long pixel_counter = 0;

	low_limit2 = 0.0;
	if (filter_radius_low != 0.0) low_limit2 = powf(pixel_size / filter_radius_low,2);

	if (bin_index != NULL) delete [] bin_index;
	bin_index = new int [particle_image->real_memory_allocated / 2];

	for (k = 0; k <= particle_image->physical_upper_bound_complex_z; k++)
	{
		z = powf(particle_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * particle_image->fourier_voxel_size_z, 2);

		for (j = 0; j <= particle_image->physical_upper_bound_complex_y; j++)
		{
			y = powf(particle_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * particle_image->fourier_voxel_size_y, 2);

			for (i = 0; i <= particle_image->physical_upper_bound_complex_x; i++)
			{
				x = powf(i * particle_image->fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				if ((frequency_squared >= low_limit2 && frequency_squared <= high_limit2) || ! limit_resolution)
				{
					bin_index[pixel_counter] = int(sqrtf(frequency_squared) * number_of_bins2);
				}
				else
				{
					bin_index[pixel_counter] = -1;
				}
				pixel_counter++;
			}
		}
	}
}

void Particle::WeightBySSNR(Curve &SSNR, int include_reference_weighting, bool no_ctf)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(ctf_image->is_in_memory, "CTF image memory not allocated");
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");
	MyDebugAssertTrue(! is_ssnr_filtered, "Already SSNR filtered");

	int i;
	// mask_volume = number of pixels in 2D mask applied to input images
	// (4.0 * PI / 3.0 * powf(mask_volume / PI, 1.5)) = volume (number of voxels) inside sphere with radius of 2D mask
	// kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3) = volume (number of pixels) inside the particle envelope
	float particle_area_in_pixels = PI * powf(3.0 * (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) / 4.0 / PI, 2.0 / 3.0);
//	float ssnr_scale_factor = particle_area_in_pixels / mask_volume;
	float ssnr_scale_factor = particle_area_in_pixels / particle_image->logical_x_dimension / particle_image->logical_y_dimension;
//	wxPrintf("particle_area_in_pixels = %g, mask_volume = %g\n", particle_area_in_pixels, mask_volume);
//	float ssnr_scale_factor_old = kDa_to_Angstrom3(molecular_mass_kDa) / 4.0 / PI / powf(pixel_size,3) / (4.0 * PI / 3.0 * powf(mask_volume / PI, 1.5));
//	wxPrintf("old = %g, new = %g\n", ssnr_scale_factor_old, ssnr_scale_factor);
//	float ssnr_scale_factor = PI * powf( powf(3.0 * kDa_to_Angstrom3(molecular_mass_kDa) / 4.0 / PI / powf(pixel_size,3) ,1.0 / 3.0) ,2) / mask_volume;
//	float ssnr_scale_factor = particle_image->logical_x_dimension * particle_image->logical_y_dimension / mask_volume;

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();

	Image *snr_image = new Image;
	snr_image->Allocate(ctf_image->logical_x_dimension, ctf_image->logical_y_dimension, false);
	particle_image->Whiten();

//	snr_image->CopyFrom(ctf_image);
	if (no_ctf)
	{
		snr_image->SetToConstant(1.0);
		no_ctf_weighting = true;
	}
	else
	{
		for (i = 0; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conj(ctf_image->complex_values[i]);}
		no_ctf_weighting = false;
	}
	snr_image->MultiplyByWeightsCurve(SSNR, ssnr_scale_factor);
	particle_image->OptimalFilterBySNRImage(*snr_image, include_reference_weighting);
	is_ssnr_filtered = true;
	if (include_reference_weighting != 0) includes_reference_ssnr_weighting = true;
	else includes_reference_ssnr_weighting = false;

	// Apply cosine filter to reduce ringing when resolution limit higher than 7 A
//	if (filter_radius_high > 0.0) particle_image->CosineMask(std::max(pixel_size / filter_radius_high, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);

	delete snr_image;
}

void Particle::WeightBySSNR(Curve &SSNR, Image &projection_image, bool weight_particle_image, bool weight_projection_image)
{
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");
	MyDebugAssertTrue(! is_ssnr_filtered, "Already SSNR filtered");

	particle_image->WeightBySSNR(*ctf_image, molecular_mass_kDa, pixel_size, SSNR, projection_image, weight_particle_image, weight_projection_image);
	if (weight_particle_image) is_ssnr_filtered = true;
	else is_ssnr_filtered = false;
	includes_reference_ssnr_weighting = false;
}

void Particle::CalculateProjection(Image &projection_image, ReconstructedVolume &input_3d)
{
	MyDebugAssertTrue(projection_image.is_in_memory, "Projection image memory not allocated");
	MyDebugAssertTrue(input_3d.density_map->is_in_memory, "3D reconstruction memory not allocated");
	MyDebugAssertTrue(ctf_image->is_in_memory, "CTF image memory not allocated");
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");

	input_3d.CalculateProjection(projection_image, *ctf_image, alignment_parameters, 0.0, 0.0, 1.0, true, true, false, true, false);
	if (current_ctf.GetBeamTiltX() != 0.0f || current_ctf.GetBeamTiltY() != 0.0f) projection_image.ConjugateMultiplyPixelWise(*beamtilt_image);
}

void Particle::GetParameters(cisTEMParameterLine &output_parameters)
{
	output_parameters = current_parameters;
}

void Particle::SetParameters(cisTEMParameterLine &wanted_parameters, bool initialize_scores)
{
	current_parameters = wanted_parameters;

	if (initialize_scores)
	{
		current_parameters.logp = -std::numeric_limits<float>::max();
		current_parameters.sigma = -std::numeric_limits<float>::max();
		current_parameters.score = -std::numeric_limits<float>::max();
	}

	alignment_parameters.Init(current_parameters.phi, current_parameters.theta, current_parameters.psi, current_parameters.x_shift, current_parameters.y_shift);
}

void Particle::SetAlignmentParameters(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x, float wanted_shift_y)
{
	alignment_parameters.Init(wanted_euler_phi, wanted_euler_theta, wanted_euler_psi, wanted_shift_x, wanted_shift_y);
}

void Particle::SetParameterStatistics(cisTEMParameterLine &wanted_averages, cisTEMParameterLine &wanted_variances)
{
	parameter_average = wanted_averages;
	parameter_variance = wanted_variances;

}

void Particle::SetParameterConstraints(float wanted_noise_variance)
{
	MyDebugAssertTrue(! constraints_used.phi || parameter_variance.phi > 0.0, "Phi variance not positive");
	MyDebugAssertTrue(! constraints_used.theta || parameter_variance.theta > 0.0, "Theta variance not positive");
	MyDebugAssertTrue(! constraints_used.psi || parameter_variance.psi > 0.0, "Psi variance not positive");
	MyDebugAssertTrue(! constraints_used.x_shift || parameter_variance.x_shift > 0.0, "Shift_X variance not positive");
	MyDebugAssertTrue(! constraints_used.y_shift || parameter_variance.y_shift > 0.0, "Shift_Y variance not positive");

	scaled_noise_variance = wanted_noise_variance;
	if (constraints_used.phi) parameter_constraints.InitPhi(parameter_average.phi, parameter_variance.phi, scaled_noise_variance);
	if (constraints_used.theta) parameter_constraints.InitTheta(parameter_average.theta, parameter_variance.theta, scaled_noise_variance);
	if (constraints_used.psi) parameter_constraints.InitPsi(parameter_average.psi, parameter_variance.psi, scaled_noise_variance);
	if (constraints_used.x_shift) parameter_constraints.InitShiftX(parameter_average.x_shift, parameter_variance.x_shift, scaled_noise_variance);
	if (constraints_used.y_shift) parameter_constraints.InitShiftY(parameter_average.y_shift, parameter_variance.y_shift, scaled_noise_variance);
}

float Particle::ReturnParameterPenalty(cisTEMParameterLine &parameters)
{
	float penalty = 0.0;

// Assume that sigma_noise is approximately equal to sigma_image, i.e. the SNR in the image is very low
	if (constraints_used.phi) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnPhiAngleLogP(parameters.phi);
	if (constraints_used.theta) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnThetaAngleLogP(parameters.theta);
	if (constraints_used.psi) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnPsiAngleLogP(parameters.psi);
	if (constraints_used.x_shift) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnShiftXLogP(parameters.x_shift);
	if (constraints_used.y_shift) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnShiftYLogP(parameters.y_shift);

	return penalty;
}

float Particle::ReturnParameterLogP(cisTEMParameterLine &parameters)
{
	float logp = 0.0;

	if (constraints_used.phi) logp += parameter_constraints.ReturnPhiAngleLogP(parameters.phi);
	if (constraints_used.theta) logp += parameter_constraints.ReturnThetaAngleLogP(parameters.theta);
	if (constraints_used.psi) logp += parameter_constraints.ReturnPsiAngleLogP(parameters.psi);
	if (constraints_used.x_shift) logp += parameter_constraints.ReturnShiftXLogP(parameters.x_shift);
	if (constraints_used.y_shift) logp += parameter_constraints.ReturnShiftYLogP(parameters.y_shift);

	return logp;
}

int Particle::MapParameterAccuracy(float *accuracies)
{
	cisTEMParameterLine accuracy_line;
	accuracy_line.psi = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	accuracy_line.theta = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	accuracy_line.phi = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	accuracy_line.x_shift = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size) / 5.0;
	accuracy_line.y_shift = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size) / 5.0;
	number_of_search_dimensions = MapParametersFromExternal(accuracy_line, accuracies);
	return number_of_search_dimensions;
}

int Particle::MapParametersFromExternal(cisTEMParameterLine &input_parameters, float *mapped_parameters)
{
	int i;
	int j = 0;

	if (parameter_map.phi == true)
	{
		mapped_parameters[j] = input_parameters.phi;
		j++;
	}

	if (parameter_map.theta == true)
	{
		mapped_parameters[j] = input_parameters.theta;
		j++;
	}

	if (parameter_map.psi == true)
	{
		mapped_parameters[j] = input_parameters.psi;
		j++;
	}

	if (parameter_map.x_shift == true)
	{
		mapped_parameters[j] = input_parameters.x_shift;
		j++;
	}

	if (parameter_map.y_shift == true)
	{
		mapped_parameters[j] = input_parameters.y_shift;
		j++;
	}

	return j;
}


int Particle::MapParameters(float *mapped_parameters)
{
	int i;
	int j = 0;

	if (parameter_map.phi == true)
	{
		mapped_parameters[j] = current_parameters.phi;
		j++;
	}

	if (parameter_map.theta == true)
	{
		mapped_parameters[j] = current_parameters.theta;
		j++;
	}

	if (parameter_map.psi == true)
	{
		mapped_parameters[j] = current_parameters.psi;
		j++;
	}

	if (parameter_map.x_shift == true)
	{
		mapped_parameters[j] = current_parameters.x_shift;
		j++;
	}

	if (parameter_map.y_shift == true)
	{
		mapped_parameters[j] = current_parameters.y_shift;
		j++;
	}

	return j;
}

int Particle::UnmapParametersToExternal(cisTEMParameterLine &output_parameters, float *mapped_parameters)
{
	int i;
	int j = 0;


	if (parameter_map.phi == true)
	{
		output_parameters.phi = mapped_parameters[j];
		j++;
	}

	if (parameter_map.theta == true)
	{
		output_parameters.theta = mapped_parameters[j];
		j++;
	}

	if (parameter_map.psi == true)
	{
		output_parameters.psi = mapped_parameters[j];
		j++;
	}

	if (parameter_map.x_shift == true)
	{
		output_parameters.x_shift = mapped_parameters[j];
		j++;
	}

	if (parameter_map.y_shift == true)
	{
		output_parameters.y_shift = mapped_parameters[j];
		j++;
	}

	return j;
}

int Particle::UnmapParameters(float *mapped_parameters)
{
	int i;
	int j = 0;

	if (parameter_map.phi == true)
	{
		current_parameters.phi = mapped_parameters[j];
		j++;
	}

	if (parameter_map.theta == true)
	{
		current_parameters.theta = mapped_parameters[j];
		j++;
	}

	if (parameter_map.psi == true)
	{
		current_parameters.psi = mapped_parameters[j];
		j++;
	}

	if (parameter_map.x_shift == true)
	{
		current_parameters.x_shift = mapped_parameters[j];
		j++;
	}

	if (parameter_map.y_shift == true)
	{
		current_parameters.y_shift = mapped_parameters[j];
		j++;
	}

	alignment_parameters.Init(current_parameters.phi, current_parameters.theta, current_parameters.psi, current_parameters.x_shift, current_parameters.y_shift);

	return j;
}

float Particle::ReturnLogLikelihood(Image &input_image, Image &padded_unbinned_image, CTF &input_ctf, ReconstructedVolume &input_3d, ResolutionStatistics &statistics, float classification_resolution_limit)
{
//!!!	MyDebugAssertTrue(is_ssnr_filtered, "particle_image not filtered");

	float number_of_independent_pixels;
	float variance_masked;
	float variance_difference;
	float variance_particle;
	float variance_projection;
	float rotated_center_x;
	float rotated_center_y;
	float rotated_center_z;
	float alpha;
	float sigma;
	float original_pixel_size = pixel_size * float(input_3d.density_map->logical_x_dimension) / float(padded_unbinned_image.logical_x_dimension);
//	float effective_bfactor;

	float pixel_center_2d_x = mask_center_2d_x / original_pixel_size - input_image.physical_address_of_box_center_x;
	float pixel_center_2d_y = mask_center_2d_y / original_pixel_size - input_image.physical_address_of_box_center_y;
	// Assumes cubic reference volume
	float pixel_center_2d_z = mask_center_2d_z / original_pixel_size - input_image.physical_address_of_box_center_x;
	float pixel_radius_2d = mask_radius_2d / original_pixel_size;

	Image *temp_image1 = new Image;
	temp_image1->Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
	Image *temp_image2 = new Image;
	temp_image2->Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
	Image *projection_image = new Image;
	projection_image->Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, false);
	Image *temp_projection = new Image;
	temp_projection->Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, false);
	Image *temp_particle = new Image;
	temp_particle->Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, false);
	Image *ctf_input_image = new Image;
	ctf_input_image->Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, false);
	Image *beamtilt_input_image = new Image;
	beamtilt_input_image->Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, false);

//	if (filter_radius_high != 0.0)
//	{
//		effective_bfactor = 2.0 * powf(original_pixel_size / filter_radius_high, 2);
//	}
//	else
//	{
//		effective_bfactor = 0.0;
//	}

//	ResetImageFlags();
//	mask_volume = PI * powf(mask_radius / original_pixel_size, 2);
//	is_ssnr_filtered = false;
//	is_centered_in_box = true;
//	CenterInCorner();
//	input_3d.CalculateProjection(*projection_image, *ctf_image, alignment_parameters, mask_radius, mask_falloff, original_pixel_size / filter_radius_high, false, true);
	input_3d.density_map->ExtractSlice(*temp_image1, alignment_parameters, pixel_size / filter_radius_high);
	temp_image1->SwapRealSpaceQuadrants();
	temp_image1->BackwardFFT();
	temp_image1->AddConstant(- temp_image1->ReturnAverageOfRealValues(mask_radius / pixel_size, true));
	temp_image1->CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
	temp_image1->ForwardFFT();
	temp_image2->CopyFrom(temp_image1);

	ctf_input_image->CalculateCTFImage(input_ctf);
	beamtilt_input_image->CalculateBeamTiltImage(input_ctf);
	if (includes_reference_ssnr_weighting) temp_image1->Whiten(pixel_size / filter_radius_high);
//	temp_image1->PhaseFlipPixelWise(*ctf_image);
//	if (input_3d.density_map->logical_x_dimension != padded_unbinned_image.logical_x_dimension) temp_image1->CosineMask(0.5 - pixel_size / 20.0, pixel_size / 10.0);
	if (input_3d.density_map->logical_x_dimension != padded_unbinned_image.logical_x_dimension) temp_image1->CosineMask(0.45, 0.1);
	temp_image1->ClipInto(&padded_unbinned_image);
	padded_unbinned_image.BackwardFFT();
	padded_unbinned_image.ClipInto(projection_image);
	projection_image->ForwardFFT();
	projection_image->PhaseFlipPixelWise(*ctf_input_image);
	projection_image->MultiplyPixelWise(*beamtilt_input_image);

//	temp_image2->MultiplyPixelWiseReal(*ctf_image);
//	if (input_3d.density_map->logical_x_dimension != padded_unbinned_image.logical_x_dimension) temp_image2->CosineMask(0.5 - pixel_size / 20.0, pixel_size / 10.0);
	if (input_3d.density_map->logical_x_dimension != padded_unbinned_image.logical_x_dimension) temp_image2->CosineMask(0.45, 0.1);
	temp_image2->ClipInto(&padded_unbinned_image);
	padded_unbinned_image.BackwardFFT();
	padded_unbinned_image.ClipInto(temp_projection);
	temp_projection->ForwardFFT();
	temp_projection->MultiplyPixelWiseReal(*ctf_input_image);
	temp_projection->MultiplyPixelWise(*beamtilt_input_image);
//	temp_projection->CopyFrom(projection_image);
//	projection_image->PhaseFlipPixelWise(*ctf_image);
//	temp_projection->MultiplyPixelWiseReal(*ctf_image);
//	temp_projection->CopyFrom(projection_image);
	if (input_image.is_in_real_space) input_image.ForwardFFT();
	input_image.PhaseShift(- current_parameters.x_shift / original_pixel_size, - current_parameters.y_shift / original_pixel_size);
	temp_particle->CopyFrom(&input_image);

//	if (includes_reference_ssnr_weighting) temp_projection->Whiten(pixel_size / filter_radius_high);
//	WeightBySSNR(statistics.part_SSNR, *temp_projection, false, includes_reference_ssnr_weighting);
	input_image.WeightBySSNR(*ctf_input_image, molecular_mass_kDa, original_pixel_size, statistics.part_SSNR, *projection_image, true, includes_reference_ssnr_weighting);

//	if (includes_reference_ssnr_weighting) projection_image->Whiten(original_pixel_size / filter_radius_high);
//	WeightBySSNR(statistics.part_SSNR, *projection_image, true, includes_reference_ssnr_weighting);

//	particle_image->SwapRealSpaceQuadrants();
//	particle_image->PhaseShift(- current_parameters[4] / pixel_size, - current_parameters[5] / pixel_size);
	input_image.BackwardFFT();
//	temp_particle->BackwardFFT();

//	projection_image->SwapRealSpaceQuadrants();
	projection_image->BackwardFFT();
	// Apply some low-pass filtering to improve classification
//	temp_projection->ApplyBFactor(effective_bfactor);
//	temp_projection->BackwardFFT();
//	input_image.QuickAndDirtyWriteSlice("part.mrc", 1);
//	projection_image->QuickAndDirtyWriteSlice("proj.mrc", 1);
//	temp_particle->QuickAndDirtyWriteSlice("part2.mrc", 1);
//	temp_projection->QuickAndDirtyWriteSlice("proj2.mrc", 1);
//	exit(0);

	// Calculate LogP
//	variance_masked = temp_particle->ReturnVarianceOfRealValues(mask_radius / pixel_size, 0.0, 0.0, 0.0, true);
//	wxPrintf("variance_masked = %g\n", variance_masked);
//	temp_particle->MultiplyByConstant(1.0 / sqrtf(variance_masked));
//	alpha = temp_particle->ReturnImageScale(*temp_projection, mask_radius / pixel_size);
//	temp_projection->MultiplyByConstant(alpha);
	// This scaling according to the average sqrtf(SNR) should take care of variable signal strength in the images
	// However, it seems to lead to some oscillatory behavior of the occupancies (from cycle to cycle)
//	if (current_parameters[7] >= 0 && current_parameters[14] > 0.0) temp_projection->MultiplyByConstant(parameter_average[14] / current_parameters[14]);
//	wxPrintf("alpha for logp, scaling factor = %g %g\n", alpha, parameter_average[14] / current_parameters[14]);

	if (apply_2D_masking)
	{
		AnglesAndShifts	reverse_alignment_parameters;
		reverse_alignment_parameters.Init(- current_parameters.psi, - current_parameters.theta, - current_parameters.phi, 0.0, 0.0);
		reverse_alignment_parameters.euler_matrix.RotateCoords(pixel_center_2d_x, pixel_center_2d_y, pixel_center_2d_z, rotated_center_x, rotated_center_y, rotated_center_z);
//		variance_masked = particle_image->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0);
	}
//	else
//	{
//		variance_masked = particle_image->ReturnVarianceOfRealValues(mask_radius / pixel_size);
//	}
	temp_particle->BackwardFFT();
	temp_projection->BackwardFFT();
//	temp_particle->QuickAndDirtyWriteSlice("temp_particle.mrc", 1);
//	temp_projection->QuickAndDirtyWriteSlice("temp_projection.mrc", 1);
//	phase_difference->QuickAndDirtyWriteSlice("phase_difference.mrc", 1);
//	exit(0);
	temp_particle->SubtractImage(temp_projection);
//	particle_image->QuickAndDirtyWriteSlice("diff.mrc", 1);
	// This low-pass filter reduces the number of independent pixels. It should therefore be applied only to
	// the reference (temp_projection), and not to the difference (temp_particle - temp_projection), as is done here...
	if (classification_resolution_limit > 0.0)
	{
		temp_particle->ForwardFFT();
		temp_particle->CosineMask(original_pixel_size / classification_resolution_limit, original_pixel_size / mask_falloff);
		temp_particle->BackwardFFT();
	}
	if (apply_2D_masking)
	{
		variance_difference = temp_particle->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + temp_particle->physical_address_of_box_center_x,
				rotated_center_y + temp_particle->physical_address_of_box_center_y, 0.0);
//		sigma = sqrtf(variance_difference / temp_projection->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + temp_particle->physical_address_of_box_center_x,
//				rotated_center_y + temp_particle->physical_address_of_box_center_y, 0.0));
		number_of_independent_pixels = PI * powf(pixel_radius_2d, 2);
	}
	else
	{
		variance_difference = temp_particle->ReturnSumOfSquares(mask_radius / original_pixel_size);
//		sigma = sqrtf(variance_difference / temp_projection->ReturnVarianceOfRealValues(mask_radius / pixel_size));
		number_of_independent_pixels = PI * powf(mask_radius / original_pixel_size, 2);
//		number_of_independent_pixels = mask_volume;
	}

	logp = - 0.5 * (variance_difference + logf(2.0 * PI)) * number_of_independent_pixels;
	// This penalty term assumes a Gaussian x,y distribution that is probably not correct in most cases. Better to leave it out.
//			+ ReturnParameterLogP(current_parameters);

	// Calculate SNR used for particle weighting during reconstruction
	input_image.CosineMask(mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
//	alpha = input_image.ReturnImageScale(*projection_image);
//	variance_masked = projection_image->ReturnVarianceOfRealValues(mask_radius / original_pixel_size);
//	projection_image->MultiplyByConstant(1.0 / sqrtf(variance_masked));
//	wxPrintf("var = %g\n", variance_masked);
//	alpha = input_image.ReturnImageScale(*projection_image, mask_radius / original_pixel_size);
	alpha = input_image.ReturnImageScale(*projection_image);
	projection_image->MultiplyByConstant(alpha);

//	if (origin_micrograph < 0) origin_micrograph = 0;
//	origin_micrograph++;
//	input_image.QuickAndDirtyWriteSlice("part.mrc", origin_micrograph);
//	projection_image->QuickAndDirtyWriteSlice("proj.mrc", origin_micrograph);

	input_image.SubtractImage(projection_image);
//	input_image.QuickAndDirtyWriteSlice("diff.mrc", origin_micrograph);
//	exit(0);
//	variance_difference = input_image.ReturnVarianceOfRealValues();
//	sigma = sqrtf(variance_difference / projection_image->ReturnVarianceOfRealValues());
//	variance_difference = input_image.ReturnVarianceOfRealValues(mask_radius / original_pixel_size);
//	sigma = sqrtf(variance_difference / projection_image->ReturnVarianceOfRealValues(mask_radius / original_pixel_size));
	variance_difference = input_image.ReturnVarianceOfRealValues();
	sigma = sqrtf(variance_difference / projection_image->ReturnVarianceOfRealValues());
//	sigma = sqrtf(variance_difference / powf(alpha, 2));
//	wxPrintf("variance_difference, alpha for sigma, sigma = %g %g %g\n", variance_difference, alpha, sigma);
	// Prevent rare occurrences of unrealistically high sigmas
	if (sigma > 100.0) sigma = 100.0;
	if (sigma > 0.0) snr = powf(1.0 / sigma, 2);
	else snr = 0.0;

//	wxPrintf("number_of_independent_pixels = %g, variance_difference = %g, variance_masked = %g, logp = %g\n", number_of_independent_pixels,
//			variance_difference, variance_masked, -number_of_independent_pixels * variance_difference / variance_masked / 2.0);
//	exit(0);
//	return 	- number_of_independent_pixels * variance_difference / variance_masked / 2.0
//			+ ReturnParameterLogP(current_parameters);

	delete temp_image1;
	delete temp_image2;
	delete projection_image;
	delete temp_particle;
	delete temp_projection;
	delete ctf_input_image;
	delete beamtilt_input_image;

	return 	logp;
}

void Particle::CalculateMaskedLogLikelihood(Image &projection_image, ReconstructedVolume &input_3d, float classification_resolution_limit)
{
//!!!	MyDebugAssertTrue(is_ssnr_filtered, "particle_image not filtered");

//	float ssq_XA, ssq_A2;
	float alpha;
	float variance_masked;
	float variance_difference;
	float rotated_center_x;
	float rotated_center_y;
	float rotated_center_z;

	float pixel_center_2d_x = mask_center_2d_x / pixel_size - particle_image->physical_address_of_box_center_x;
	float pixel_center_2d_y = mask_center_2d_y / pixel_size - particle_image->physical_address_of_box_center_y;
	// Assumes cubic reference volume
	float pixel_center_2d_z = mask_center_2d_z / pixel_size - particle_image->physical_address_of_box_center_x;
	float pixel_radius_2d = mask_radius_2d / pixel_size;

	AnglesAndShifts	reverse_alignment_parameters;

	reverse_alignment_parameters.Init(- current_parameters.psi, - current_parameters.theta, - current_parameters.phi, 0.0, 0.0);
	reverse_alignment_parameters.euler_matrix.RotateCoords(pixel_center_2d_x, pixel_center_2d_y, pixel_center_2d_z, rotated_center_x, rotated_center_y, rotated_center_z);

	input_3d.CalculateProjection(projection_image, *ctf_image, alignment_parameters, 0.0, 0.0, pixel_size / classification_resolution_limit, false, false, false, true, is_phase_flipped);

	particle_image->PhaseShift(- current_parameters.x_shift / pixel_size, - current_parameters.y_shift / pixel_size);
	particle_image->BackwardFFT();
//	wxPrintf("ssq part = %g var part = %g\n", particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//			rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0), particle_image->ReturnVarianceOfRealValues());

	projection_image.SwapRealSpaceQuadrants();
	projection_image.BackwardFFT();
//	particle_image->QuickAndDirtyWriteSlice("part2.mrc", 1);
//	projection_image.QuickAndDirtyWriteSlice("proj2.mrc", 1);
//	exit(0);

//	variance_masked = particle_image->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0);

//	particle_image->QuickAndDirtyWriteSlice("part.mrc", 1);
//	projection_image.AddConstant(- projection_image.ReturnAverageOfRealValues(0.45 * projection_image.logical_x_dimension, true));
//	projection_image.MultiplyByConstant(0.02);
//	float min = 100.0;
//	for (int i = 0; i < 100; i++)
//	{
//		particle_image->SubtractImage(&projection_image);
//		sigma_signal = sqrtf(projection_image.ReturnVarianceOfRealValues(mask_radius / pixel_size));
//		sigma_noise = sqrtf(particle_image->ReturnVarianceOfRealValues(mask_radius / pixel_size));
//		if (sigma_noise < min) {min = sigma_noise; wxPrintf("i, sigma_noise = %i %g\n", i, sigma_noise);}
//	}
//	ssq_XA = particle_image->ReturnPixelWiseProduct(projection_image);
//	ssq_A2 = projection_image.ReturnPixelWiseProduct(projection_image);
//	alpha = ssq_XA / ssq_A2;
	alpha = particle_image->ReturnImageScale(projection_image, mask_radius / pixel_size);
	projection_image.MultiplyByConstant(alpha);
	particle_image->SubtractImage(&projection_image);
	sigma_signal = sqrtf(projection_image.ReturnVarianceOfRealValues(mask_radius / pixel_size));
	sigma_noise = sqrtf(particle_image->ReturnVarianceOfRealValues(mask_radius / pixel_size));
	if (sigma_noise > 0.0) snr = powf(sigma_signal / sigma_noise, 2);
	else snr = 0.0;
//	wxPrintf("mask_radius, pixel_size, alpha, sigma_noise = %g %g %g %g\n", mask_radius, pixel_size, alpha, sigma_noise);
//	particle_image->QuickAndDirtyWriteSlice("diff.mrc", 1);
//	exit(0);

//	wxPrintf("number_of_independent_pixels = %g, variance_difference = %g, variance_masked = %g, logp = %g\n", number_of_independent_pixels,
//			variance_difference, variance_masked, -number_of_independent_pixels * variance_difference / variance_masked / 2.0);
//	wxPrintf("sum = %g pix = %li penalty = %g indep = %g\n", particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//			rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0), particle_image->number_of_real_space_pixels,
//			ReturnParameterLogP(current_parameters), mask_volume);
//	exit(0);
	if (mask_radius_2d > 0.0)
	{
		logp = - 0.5 * (particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0) + logf(2.0 * PI)) * PI * powf(pixel_radius_2d, 2)
				+ ReturnParameterLogP(current_parameters);
	}
	else
	{
		logp = - 0.5 * (particle_image->ReturnSumOfSquares(mask_radius / pixel_size) + logf(2.0 * PI)) * PI * powf(mask_radius / pixel_size, 2)
				+ ReturnParameterLogP(current_parameters);
	}
}

float Particle::MLBlur(Image *input_classes_cache, float ssq_X, Image &cropped_input_image, Image *rotation_cache, Image &blurred_image,
		int current_class, int number_of_rotations, float psi_step, float psi_start, float smoothing_factor, float &max_logp_particle,
		int best_class, float best_psi, Image &best_correlation_map, bool calculate_correlation_map_only, bool uncrop, bool apply_ctf_to_classes,
		Image *image_to_blur, Image *diff_image_to_blur, float max_shift_in_angstroms)
{
	MyDebugAssertTrue(cropped_input_image.is_in_memory, "cropped_input_image: memory not allocated");
	MyDebugAssertTrue(rotation_cache[0].is_in_memory, "rotation_cache: memory not allocated");
	MyDebugAssertTrue(blurred_image.is_in_memory, "blurred_image: memory not allocated");
	MyDebugAssertTrue(input_classes_cache[0].is_in_memory, "input_classes_cache: memory not allocated");
	MyDebugAssertTrue(! ctf_image->is_in_real_space, "ctf_image in real space");
	MyDebugAssertTrue(! input_classes_cache[0].is_in_real_space, "input_classes_cache not in Fourier space");
	
	int i, j;
	int pixel_counter;
	int current_rotation;
	int non_zero_pixels;
	float binning_factor;
	float snr_psi = -std::numeric_limits<float>::max();
	float snr_class;
	float log_threshold;
	float old_max_logp;
	float log_range = 20.0;
	float var_A;
	float ssq_A;
	float ssq_A_rot0;
	float ssq_XA2;
	float psi;
	float rmdr;
	float number_of_pixels = particle_image->number_of_real_space_pixels * smoothing_factor;
	bool new_max_found;
	bool use_best_psi;
	float dx, dy;
	float mid_x;
	float mid_y;
	float rvar2_x = powf(pixel_size, 2) / parameter_variance.x_shift / 2.0;
	float rvar2_y = powf(pixel_size, 2) / parameter_variance.y_shift / 2.0;
	float penalty_x, penalty_y;
	float number_of_independent_pixels;
	float norm_X, norm_A;
	double sump_psi;
	double sump_class;
	double min_float = std::numeric_limits<float>::min();
	double scale;
	AnglesAndShifts rotation_angle;
	Image *correlation_map = new Image;
	correlation_map->Allocate(particle_image->logical_x_dimension, particle_image->logical_y_dimension, false);
	Image *temp_image = new Image;
	temp_image->Allocate(particle_image->logical_x_dimension, particle_image->logical_y_dimension, false);
	Image *sum_image = new Image;
	sum_image->Allocate(particle_image->logical_x_dimension, particle_image->logical_y_dimension, true);

//	wxPrintf("Max shift in angstoms = %f\n", max_shift_in_angstroms);
	float max_radius_squared = powf(max_shift_in_angstroms / pixel_size, 2);
	float current_squared_radius;

#ifndef MKL
	float *temp_k1 = new float [particle_image->real_memory_allocated];
	float *temp_k2; temp_k2 = temp_k1 + 1;
	float *real_a;
	float *real_b;
	float *real_c;
	float *real_d;
	float *real_r;
	float *real_i;
#endif

	if (is_filtered) number_of_independent_pixels = filter_volume;
	else number_of_independent_pixels = particle_image->number_of_real_space_pixels;
	if (is_masked) number_of_independent_pixels *= mask_volume / particle_image->number_of_real_space_pixels;

	// Determine sum of squares of reference after CTF multiplication
	temp_image->CopyFrom(&input_classes_cache[current_class]);
	if (apply_ctf_to_classes) temp_image->MultiplyPixelWiseReal(*ctf_image);
	ssq_A = temp_image->ReturnSumOfSquares();
	temp_image->BackwardFFT();
	var_A = temp_image->ReturnVarianceOfRealValues();

	norm_A = 0.5 * number_of_pixels * ssq_A;
	ssq_XA2 = sqrtf(ssq_X * ssq_A);

	// Prevent collapse of x,y distribution to 0 due to limited resolution
	if (rvar2_x > 1.0 ) rvar2_x = 1.0;
	if (rvar2_y > 1.0 ) rvar2_y = 1.0;

	rvar2_x *= smoothing_factor;
	rvar2_y *= smoothing_factor;
	snr_class = - std::numeric_limits<float>::max();
	sum_image->SetToConstant(0.0);
	sump_class = 0.0;
	old_max_logp = max_logp_particle;
	if (log_range == 0.0) {log_range = 0.0001;}
	for (current_rotation = 0; current_rotation < number_of_rotations; current_rotation++)
	{
		if (calculate_correlation_map_only) {psi = best_psi; current_rotation = number_of_rotations;}
		else psi = 360.0 - current_rotation * psi_step - psi_start;
		rotation_angle.GenerateRotationMatrix2D(psi);
		rotation_angle.euler_matrix.RotateCoords2D(parameter_average.x_shift, parameter_average.y_shift, mid_x, mid_y);
		mid_x /= pixel_size;
		mid_y /= pixel_size;
		rotation_angle.GenerateRotationMatrix2D(- psi);

//		wxPrintf("current_rotation = %i ssq_X = %g ssq_A = %g\n", current_rotation, rotation_cache[current_rotation].ReturnSumOfSquares(), input_classes_cache[current_class].ReturnSumOfSquares());
//		wxPrintf("number_of_pixels = %g, ssq_X = %g ssq_A = %g\n", number_of_pixels, ssq_X, ssq_A);
		// Calculate X.A
#ifdef MKL
		vmcMulByConj(particle_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (input_classes_cache[current_class].complex_values),reinterpret_cast <MKL_Complex8 *> (rotation_cache[current_rotation].complex_values),reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
		real_a = input_classes_cache[current_class].real_values;
		real_b = input_classes_cache[current_class].real_values + 1;
		real_c = rotation_cache[current_rotation].real_values;
		real_d = rotation_cache[current_rotation].real_values + 1;
		real_r = correlation_map->real_values;
		real_i = correlation_map->real_values + 1;
		for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
		for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
		for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];};
		for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
#endif
		correlation_map->is_in_real_space = false;
		correlation_map->BackwardFFT();
		temp_image->CopyFrom(correlation_map);

		// Calculate LogP (excluding -0.5 * number_of_independent_pixels * (logf(2.0 * PI) + ssq_X) and apply hierarchical prior f(x,y)
		// ssq_X_minus_A = (ssq_X - 2.0 * correlation_map->real_values[0] + ssq_A) / 2
		pixel_counter = 0;
		penalty_x = 0.0;
		penalty_y = 0.0;
		// The following is divided by 2 according to the LogP formula
		for (j = 0; j < particle_image->logical_y_dimension; j++)
		{
			if (constraints_used.y_shift)
			{
				if (j > particle_image->physical_address_of_box_center_y) dy = j - particle_image->logical_y_dimension;
				else dy = j;
				penalty_y = powf(dy - mid_y, 2) * rvar2_y;				
			}
			for (i = 0; i < particle_image->logical_x_dimension; i++)
			{
				if (constraints_used.x_shift)
				{
					if (i > particle_image->physical_address_of_box_center_x) dx = i - particle_image->logical_y_dimension;
					else dx = i;
					penalty_x = powf(dx - mid_x, 2) * rvar2_x;					
				}
				correlation_map->real_values[pixel_counter] = correlation_map->real_values[pixel_counter] * number_of_pixels - norm_A - penalty_x - penalty_y;
				pixel_counter++;
			}
			pixel_counter += particle_image->padding_jump_value;
		}
		// Find correlation maximum to threshold LogP, and find best alignment parameters
		pixel_counter = 0;
		new_max_found = false;
		for (j = 0; j < particle_image->logical_y_dimension; j++)
		{
			if (j > particle_image->physical_address_of_box_center_y) dy = j - particle_image->logical_y_dimension;
			else dy = j;
			for (i = 0; i < particle_image->logical_x_dimension; i++)
			{
				if (i > particle_image->physical_address_of_box_center_x) dx = i - particle_image->logical_y_dimension;
				else dx = i;

				current_squared_radius = powf(dx, 2) + powf(dy, 2);
				if (current_squared_radius > max_radius_squared)
				{
					correlation_map->real_values[pixel_counter] = 0.0f;
				}
				else if(correlation_map->real_values[pixel_counter] > max_logp_particle)
				{
					new_max_found = true;
					max_logp_particle = correlation_map->real_values[pixel_counter];
					// Store correlation coefficient that corresponds to highest likelihood
					snr_psi = temp_image->real_values[pixel_counter];
					rotation_angle.euler_matrix.RotateCoords2D(dx, dy, current_parameters.x_shift, current_parameters.y_shift);
					current_parameters.x_shift *= pixel_size;
					current_parameters.y_shift *= pixel_size;

					current_parameters.psi = psi;
					current_parameters.best_2d_class = current_class + 1;
				}
				pixel_counter++;
			}
			pixel_counter += particle_image->padding_jump_value;
		}
//		// To get normalized correlation coefficient, need to divide by sigmas of particle and reference
		// To get sigma^2, need to calculate ssq_X - 2XA + ssq_A
		if (new_max_found)
		{
//			wxPrintf("ssq_X, snr_psi, ssq_A, number_of_real_space_pixels,  number_of_independent_pixels, var_A = %g %g %g %li %g %g\n",
//					ssq_X, snr_psi, ssq_A, particle_image->number_of_real_space_pixels,  number_of_independent_pixels, var_A);
			snr_psi = (ssq_X - 2.0 * snr_psi + ssq_A) * particle_image->number_of_real_space_pixels / number_of_independent_pixels / var_A;
			// Update SIGMA (SNR)
			if (snr_psi >= 0.0) current_parameters.sigma = sqrtf(snr_psi);
			// Update SCORE
			current_parameters.score = 100.0 * (max_logp_particle + norm_A) / number_of_pixels / ssq_XA2;
		}

		rmdr = remainderf(best_psi - psi, 360.0);
		use_best_psi = false;
		if (! calculate_correlation_map_only && best_class == current_class && ! new_max_found && rmdr < psi_step / 2.0 && rmdr >= - psi_step / 2.0)
		{
			use_best_psi = true;
			snr_psi = snr;
			correlation_map->CopyFrom(&best_correlation_map);
		}

		if (calculate_correlation_map_only)
		{
			snr = snr_psi;
			best_correlation_map.CopyFrom(correlation_map);
			// Update SIGMA (SNR)
			if (snr_psi >= 0.0) current_parameters.sigma = sqrtf(snr_psi);
			// Update SCORE
			current_parameters.score = 100.0 * (max_logp_particle + norm_A) / number_of_pixels / ssq_XA2;
			break;
		}
		else
		{
			// Calculate thresholded LogP
			log_threshold = max_logp_particle - log_range;
			pixel_counter = 0;
			non_zero_pixels = 0;
			sump_psi = 0.0;


			for (j = 0; j < particle_image->logical_y_dimension; j++)
			{
				if (j > particle_image->physical_address_of_box_center_y) dy = particle_image->logical_y_dimension - j;
				else dy = j;
				for (i = 0; i < particle_image->logical_x_dimension; i++)
				{
					if (i > particle_image->physical_address_of_box_center_x) dx = particle_image->logical_y_dimension - i;
					else dx = i;

					if (correlation_map->real_values[pixel_counter] >= log_threshold && correlation_map->real_values[pixel_counter] != 0.0f)
					{
						correlation_map->real_values[pixel_counter] = exp(correlation_map->real_values[pixel_counter] - max_logp_particle);
						sump_psi += correlation_map->real_values[pixel_counter];
						non_zero_pixels++;
					}
					else
					{
						correlation_map->real_values[pixel_counter] = 0.0;
					}
					pixel_counter++;
				}
				pixel_counter += particle_image->padding_jump_value;
			}

			if (non_zero_pixels > 0)
			{
//				correlation_map->QuickAndDirtyWriteSlice("corr.mrc", 1);
//				exit(0);
//				correlation_map->SetToConstant(0.0);
//				correlation_map->real_values[0] = 1.0;
//				sump_psi = 1.0;
//				if (! uncrop) correlation_map->real_values[0] += 0.01;
				correlation_map->ForwardFFT();

				if (use_best_psi) i = number_of_rotations;
				else i = current_rotation;

				if (image_to_blur == NULL)
				{
#ifdef MKL
					vmcMul(particle_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),reinterpret_cast <MKL_Complex8 *> (rotation_cache[i].complex_values),reinterpret_cast <MKL_Complex8 *> (temp_image->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
					real_a = correlation_map->real_values;
					real_b = correlation_map->real_values + 1;
					real_c = rotation_cache[i].real_values;
					real_d = rotation_cache[i].real_values + 1;
					real_r = temp_image->real_values;
					real_i = temp_image->real_values + 1;
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) - real_d[pixel_counter] * temp_k1[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
#endif
				}
				else
				{
					if (diff_image_to_blur != NULL)
					{
#ifdef MKL
						vmcMul(particle_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),reinterpret_cast <MKL_Complex8 *> (diff_image_to_blur->complex_values),reinterpret_cast <MKL_Complex8 *> (temp_image->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
						real_a = correlation_map->real_values;
						real_b = correlation_map->real_values + 1;
						real_c = diff_image_to_blur->real_values;
						real_d = diff_image_to_blur->real_values + 1;
						real_r = temp_image->real_values;
						real_i = temp_image->real_values + 1;
						for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
						for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
						for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) - real_d[pixel_counter] * temp_k1[pixel_counter];};
						for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
#endif
						diff_image_to_blur->CopyFrom(temp_image);
					}
#ifdef MKL
					vmcMul(particle_image->real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (correlation_map->complex_values),reinterpret_cast <MKL_Complex8 *> (image_to_blur->complex_values),reinterpret_cast <MKL_Complex8 *> (temp_image->complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
					real_a = correlation_map->real_values;
					real_b = correlation_map->real_values + 1;
					real_c = image_to_blur->real_values;
					real_d = image_to_blur->real_values + 1;
					real_r = temp_image->real_values;
					real_i = temp_image->real_values + 1;
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) - real_d[pixel_counter] * temp_k1[pixel_counter];};
					for (pixel_counter = 0; pixel_counter < particle_image->real_memory_allocated; pixel_counter += 2) {real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] + real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];};
#endif
				}

				// max_logp_particle is the current best LogP found over all tested references, angles and x,y positions.
				// It is also the offset of the likelihoods in correlation_map and sump_psi (the sum of likelihoods).
				// old_max_log is the previous best logP and correlation_map offset used to calculate sum_image and sump_class.

				// First deal with case where the old offset is so low that the previous sums are not significant compared with the current reference.
				temp_image->is_in_real_space = false;
				temp_image->BackwardFFT();

				if (old_max_logp < max_logp_particle - log_range)
				{
					sum_image->CopyFrom(temp_image);
					old_max_logp = max_logp_particle;
					snr_class = snr_psi;
					sump_class = sump_psi;
				}
				else
				// If the old and new offsets are similar, need to calculate a weighted sum
				if (fabsf(old_max_logp - max_logp_particle) <= log_range)
				{
					// Case of old offset smaller than or equal new offset
					if (old_max_logp <= max_logp_particle)
					{
						scale = expf(old_max_logp - max_logp_particle);
						pixel_counter = 0;
						for (j = 0; j < sum_image->logical_y_dimension; j++)
						{
							for (i = 0; i < sum_image->logical_x_dimension; i++)
							{
								sum_image->real_values[pixel_counter] = sum_image->real_values[pixel_counter] * scale + temp_image->real_values[pixel_counter];
								pixel_counter++;
							}
							pixel_counter += sum_image->padding_jump_value;
						}
						old_max_logp = max_logp_particle;
						snr_class = snr_psi;
						sump_class = sump_class * scale + sump_psi;
					}
					// Case of old offset larger than new offset
					else
					{
						scale = expf(max_logp_particle - old_max_logp);
						pixel_counter = 0;
						for (j = 0; j < sum_image->logical_y_dimension; j++)
						{
							for (i = 0; i < sum_image->logical_x_dimension; i++)
							{
								sum_image->real_values[pixel_counter] = sum_image->real_values[pixel_counter] + temp_image->real_values[pixel_counter] * scale;
								pixel_counter++;
							}
							pixel_counter += sum_image->padding_jump_value;
						}
						sump_class = sump_class + sump_psi * scale;
					}
				}
			}
		}
	}

	if (sump_class > 0.0)
	{
		// Divide rotationally & translationally blurred image by sum of probabilities
		binning_factor = float(cropped_input_image.logical_x_dimension) / float(sum_image->logical_x_dimension);
		sum_image->MultiplyByConstant(sum_image->number_of_real_space_pixels / binning_factor / sump_class);
		if (diff_image_to_blur != NULL) diff_image_to_blur->MultiplyByConstant(sum_image->number_of_real_space_pixels / binning_factor / sump_class);
		sum_image->ForwardFFT();
		if (uncrop)
		{
			sum_image->CosineMask(0.45, 0.1);
		//	sum_image->CosineMask(0.3, 0.4);
			sum_image->ClipInto(&cropped_input_image);
			cropped_input_image.BackwardFFT();
			cropped_input_image.ClipIntoLargerRealSpace2D(&blurred_image, cropped_input_image.ReturnAverageOfRealValuesOnEdges());
		}
		else
		{
			blurred_image.CopyFrom(sum_image);
			temp_image->CopyFrom(image_to_blur);
			temp_image->MultiplyByConstant(sump_class / temp_image->number_of_real_space_pixels / temp_image->number_of_real_space_pixels);
			blurred_image.AddImage(temp_image);
		}

//		wxPrintf("log sump_class = %g old_max_logp = %g number_of_independent_pixels = %g ssq_X = %g\n", logf(sump_class), old_max_logp, number_of_independent_pixels, ssq_X);
		logp = logf(sump_class) + old_max_logp - 0.5 * (number_of_independent_pixels * logf(2.0 * PI) + number_of_pixels * ssq_X);
		current_parameters.logp = logp;
	}
	else
	{
		blurred_image.SetToConstant(0.0);
		if (diff_image_to_blur != NULL) diff_image_to_blur->SetToConstant(0.0);
		logp = -std::numeric_limits<float>::max();
	}

//	wxPrintf("log_sum = %g, old = %g, n = %g, var = %g\n", logf(sump_class), old_max_log,  number_of_independent_pixels, norm_A);
	delete correlation_map;
	delete temp_image;
	delete sum_image;
#ifndef MKL
	delete [] temp_k1;
#endif

	return logp;
}

void Particle::EstimateSigmaNoise()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "particle_image memory not allocated");

	sigma_noise = particle_image->ReturnSigmaNoise();
}
