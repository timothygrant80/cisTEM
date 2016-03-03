#include "core_headers.h"

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
	delete [] temp_float;
	delete [] input_parameters;
	delete [] parameter_average;
	delete [] parameter_variance;
	delete [] refined_parameters;
	delete [] parameter_map;
	delete [] constraints_used;

	if (particle_image != NULL)
	{
		delete particle_image;
	}

	if (ctf_image != NULL)
	{
		delete ctf_image;
	}
}

void Particle::Init()
{
	number_of_parameters = 16;
	target_phase_error = 45.0;
	origin_micrograph = -1;
	origin_x_coordinate = -1;
	origin_y_coordinate = -1;
	location_in_stack = -1;
	pixel_size = 0.0;
	sigma_signal = 0.0;
	sigma_noise = 0.0;
	logp = 0.0;
	particle_occupancy = 0.0;
	particle_score = 0.0;
	particle_image = NULL;
	euler_matrix = &alignment_parameters.euler_matrix;
/*	phi_average = 0.0;
	theta_average = 0.0;
	psi_average = 0.0;
	shift_x_average = 0.0;
	shift_y_average = 0.0;
	phi_variance = 0.0;
	theta_variance = 0.0;
	psi_variance = 0.0;
	shift_x_variance = 0.0;
	shift_y_variance = 0.0;
*/
	scaled_noise_variance = 0.0;
	ctf_is_initialized = false;
	ctf_image = NULL;
	ctf_image_calculated = false;
	is_normalized = false;
	normalized_sigma = 0.0;
	is_masked = false;
	mask_radius = 0.0;
	mask_falloff = 0.0;
	mask_volume = 0.0;
	is_filtered = false;
	filter_radius_low = 0.0;
	filter_radius_high = 0.0;
	filter_falloff = 0.0;
	is_ssnr_filtered = false;
	is_centered_in_box = true;
	shift_counter = 0;
	insert_even = false;
	temp_float = new float [number_of_parameters];
	input_parameters = new float [number_of_parameters];
	parameter_average = new float [number_of_parameters];
	parameter_variance = new float [number_of_parameters];
	refined_parameters = new float [number_of_parameters];
	parameter_map = new bool [number_of_parameters];
	constraints_used = new bool [number_of_parameters];
	ZeroFloatArray(temp_float, number_of_parameters);
	ZeroFloatArray(input_parameters, number_of_parameters);
	ZeroFloatArray(parameter_average, number_of_parameters);
	ZeroFloatArray(parameter_variance, number_of_parameters);
	ZeroFloatArray(refined_parameters, number_of_parameters);
	ZeroBoolArray(parameter_map, number_of_parameters);
	ZeroBoolArray(constraints_used, number_of_parameters);
	number_of_search_dimensions = 0;
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
}

void Particle::Allocate(int &wanted_logical_x_dimension, int &wanted_logical_y_dimension)
{
	AllocateImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
	AllocateCTFImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
}

void Particle::ResetImageFlags()
{
	if (is_normalized) {is_normalized = false; normalized_sigma = 0.0;};
	if (is_masked) {is_masked = false; mask_radius = 0.0; mask_falloff = 0.0; mask_volume = 0.0;};
	if (is_filtered) {is_filtered = false; filter_radius_low = 0.0; filter_radius_high = 0.0; filter_falloff = 0.0;};
	is_ssnr_filtered = false;
	is_centered_in_box = true;
	shift_counter = 0;
	insert_even = false;
}

void Particle::PhaseShift()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(abs(shift_counter) < 2, "Image already shifted");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->PhaseShift(alignment_parameters.ReturnShiftX() / pixel_size, alignment_parameters.ReturnShiftX() / pixel_size);
	shift_counter += 1;
}

void Particle::PhaseShiftInverse()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(abs(shift_counter) < 2, "Image already shifted");

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();
	particle_image->PhaseShift(- alignment_parameters.ReturnShiftX() / pixel_size, - alignment_parameters.ReturnShiftX() / pixel_size);
	shift_counter -= 1;
}

void Particle::ForwardFFT()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(particle_image->is_in_real_space, "Image not in real space");

	particle_image->ForwardFFT();
}

void Particle::BackwardFFT()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! particle_image->is_in_real_space, "Image not in Fourier space");

	particle_image->BackwardFFT();
}

void Particle::CosineMask()
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_masked, "Image already masked");

	if (! particle_image->is_in_real_space) particle_image->BackwardFFT();
	mask_volume = particle_image->CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
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

void Particle::InitCTF(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle)
{
	MyDebugAssertTrue(! ctf_is_initialized, "CTF already initialized");

	ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.0, 0.0, pixel_size, 0.0);
	ctf_is_initialized = true;
}

void Particle::InitCTFImage(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle)
{
	MyDebugAssertTrue(ctf_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! ctf_image->is_in_real_space, "CTF image not in Fourier space");

	if (! ctf_is_initialized) ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.0, 0.0, pixel_size, 0.0);
	ctf_image->CalculateCTFImage(ctf_parameters);
	ctf_image_calculated = true;
}

void Particle::WeightBySSNR(Curve &SSNR)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(ctf_image->is_in_memory, "CTF image memory not allocated");
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");
	MyDebugAssertTrue(! is_ssnr_filtered, "Already SSNR filtered");

	int i;
	Image *snr_image = new Image;
	snr_image->Allocate(ctf_image->logical_x_dimension, ctf_image->logical_y_dimension, 1, false);
	particle_image->Whiten();

//	snr_image->CopyFrom(ctf_image);
	for (i = 1; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conjf(ctf_image->complex_values[i]);}
	snr_image->MultiplyByWeightsCurve(SSNR);
	particle_image->OptimalFilterBySNRImage(*snr_image);
	is_ssnr_filtered = true;

	delete snr_image;
}

void Particle::CalculateProjection(Image &projection_image, ReconstructedVolume &input_3d)
{
	MyDebugAssertTrue(projection_image.is_in_memory, "Projection image memory not allocated");
	MyDebugAssertTrue(input_3d.density_map.is_in_memory, "3D reconstruction memory not allocated");
	MyDebugAssertTrue(ctf_image->is_in_memory, "CTF image memory not allocated");
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");

	input_3d.CalculateProjection(projection_image, *ctf_image, alignment_parameters);
}

void Particle::SetParameters(float *wanted_parameters)
{
	for (int i = 0; i < number_of_parameters; i++) {input_parameters[i] = wanted_parameters[i];};

	alignment_parameters.Init(input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4], input_parameters[5]);
}

void Particle::SetAlignmentParameters(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x, float wanted_shift_y)
{
	alignment_parameters.Init(wanted_euler_phi, wanted_euler_theta, wanted_euler_psi, wanted_shift_x, wanted_shift_y);
}

void Particle::SetParameterStatistics(float *wanted_averages, float *wanted_variances)
{
	int i;
	for (i = 1; i < number_of_parameters; i++) {parameter_average[i] = wanted_averages[i];}
	for (i = 1; i < number_of_parameters; i++) {parameter_variance[i] = wanted_variances[i];}
}

void Particle::SetParameterConstraints(float wanted_noise_variance)
{
	MyDebugAssertTrue(! constraints_used[1] || parameter_variance[1] > 0.0, "Phi variance not positive");
	MyDebugAssertTrue(! constraints_used[2] || parameter_variance[2] > 0.0, "Theta variance not positive");
	MyDebugAssertTrue(! constraints_used[3] || parameter_variance[3] > 0.0, "Psi variance not positive");
	MyDebugAssertTrue(! constraints_used[4] || parameter_variance[4] > 0.0, "Shift_X variance not positive");
	MyDebugAssertTrue(! constraints_used[5] || parameter_variance[5] > 0.0, "Shift_Y variance not positive");

	scaled_noise_variance = wanted_noise_variance;
	if (constraints_used[1]) parameter_constraints.InitPhi(parameter_average[1], parameter_variance[1], scaled_noise_variance);
	if (constraints_used[2]) parameter_constraints.InitTheta(parameter_average[2], parameter_variance[2], scaled_noise_variance);
	if (constraints_used[3]) parameter_constraints.InitPsi(parameter_average[3], parameter_variance[3], scaled_noise_variance);
	if (constraints_used[4]) parameter_constraints.InitShiftX(parameter_average[4], parameter_variance[4], scaled_noise_variance);
	if (constraints_used[5]) parameter_constraints.InitShiftY(parameter_average[5], parameter_variance[5], scaled_noise_variance);
}

float Particle::ReturnParameterPenalty(float *parameters)
{
	float penalty = 0.0;

/*	if (constraints_used[1]) penalty += parameter_constraints.ReturnPhiAnglePenalty(parameters[1]);
	if (constraints_used[2]) penalty += parameter_constraints.ReturnThetaAnglePenalty(parameters[2]);
	if (constraints_used[3]) penalty += parameter_constraints.ReturnPsiAnglePenalty(parameters[3]);
	if (constraints_used[4]) penalty += parameter_constraints.ReturnShiftXPenalty(parameters[4]);
	if (constraints_used[5]) penalty += parameter_constraints.ReturnShiftYPenalty(parameters[5]);
*/
// Assume that sigma_noise is approximately equal to sigma_image, i.e. the SNR in the image is very low
	if (constraints_used[1]) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnPhiAngleLogP(parameters[1]);
	if (constraints_used[2]) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnThetaAngleLogP(parameters[2]);
	if (constraints_used[3]) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnPsiAngleLogP(parameters[3]);
	if (constraints_used[4]) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnShiftXLogP(parameters[4]);
	if (constraints_used[5]) penalty += sigma_noise / mask_volume * parameter_constraints.ReturnShiftYLogP(parameters[5]);

	return penalty;
}

float Particle::ReturnParameterLogP(float *parameters)
{
	float logp = 0.0;

	if (constraints_used[1]) logp += parameter_constraints.ReturnPhiAngleLogP(parameters[1]);
	if (constraints_used[2]) logp += parameter_constraints.ReturnThetaAngleLogP(parameters[2]);
	if (constraints_used[3]) logp += parameter_constraints.ReturnPsiAngleLogP(parameters[3]);
	if (constraints_used[4]) logp += parameter_constraints.ReturnShiftXLogP(parameters[4]);
	if (constraints_used[5]) logp += parameter_constraints.ReturnShiftYLogP(parameters[5]);

	return logp;
}

int Particle::MapParameterAccuracy(float *accuracies)
{
	ZeroFloatArray(temp_float, number_of_parameters);
	temp_float[1] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius);
	temp_float[2] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius);
	temp_float[3] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius);
	temp_float[4] = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size);
	temp_float[5] = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size);
	number_of_search_dimensions = MapParametersFromExternal(temp_float, accuracies);
	return number_of_search_dimensions;
}

int Particle::MapParametersFromExternal(float *input_parameters, float *mapped_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			mapped_parameters[j] = input_parameters[i];
			j++;
		}
	}

	return j;
}


int Particle::MapParameters(float *mapped_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			mapped_parameters[j] = input_parameters[i];
			j++;
		}
	}

	return j;
}

int Particle::UnmapParametersToExternal(float *output_parameters, float *mapped_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			output_parameters[i] = mapped_parameters[j];
			j++;
		}
	}

	return j;
}

int Particle::UnmapParameters(float *mapped_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			input_parameters[i] = mapped_parameters[j];
			j++;
		}
	}

	alignment_parameters.Init(input_parameters[1], input_parameters[2], input_parameters[3], input_parameters[4], input_parameters[5]);

	return j;
}
