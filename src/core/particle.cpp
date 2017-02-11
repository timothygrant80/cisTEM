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
	delete [] current_parameters;
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

	if (bin_index != NULL)
	{
		delete [] bin_index;
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
	snr = 0.0;
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
	temp_float = new float [number_of_parameters];
	current_parameters = new float [number_of_parameters];
	parameter_average = new float [number_of_parameters];
	parameter_variance = new float [number_of_parameters];
	refined_parameters = new float [number_of_parameters];
	parameter_map = new bool [number_of_parameters];
	constraints_used = new bool [number_of_parameters];
	ZeroFloatArray(temp_float, number_of_parameters);
	ZeroFloatArray(current_parameters, number_of_parameters);
	ZeroFloatArray(parameter_average, number_of_parameters);
	ZeroFloatArray(parameter_variance, number_of_parameters);
	ZeroFloatArray(refined_parameters, number_of_parameters);
	ZeroBoolArray(parameter_map, number_of_parameters);
	ZeroBoolArray(constraints_used, number_of_parameters);
	number_of_search_dimensions = 0;
	bin_index = NULL;
	mask_center_2d_x = 0.0;
	mask_center_2d_y = 0.0;
	mask_center_2d_z = 0.0;
	mask_radius_2d = 0.0;
	apply_2D_masking = false;
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

void Particle::Allocate(int wanted_logical_x_dimension, int wanted_logical_y_dimension)
{
	AllocateImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
	AllocateCTFImage(wanted_logical_x_dimension, wanted_logical_y_dimension);
}

void Particle::ResetImageFlags()
{
	if (is_normalized) {is_normalized = false; normalized_sigma = 0.0;};
	if (is_masked) {is_masked = false; mask_radius = 0.0; mask_falloff = 0.0; mask_volume = 0.0;};
	if (is_filtered) {is_filtered = false; filter_radius_low = 0.0; filter_radius_high = 0.0; filter_falloff = 0.0; filter_volume = 0.0;};
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

void Particle::Whiten(float resolution_limit)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! particle_image->is_in_real_space, "Image not in Fourier space");

	particle_image->Whiten(resolution_limit);
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

void Particle::InitCTF(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle)
{
//	MyDebugAssertTrue(! ctf_is_initialized, "CTF already initialized");

	ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle, 0.0, 0.0, 0.0, pixel_size, 0.0);
	ctf_is_initialized = true;
}

void Particle::SetDefocus(float defocus_1, float defocus_2, float astigmatism_angle)
{
	MyDebugAssertTrue(ctf_is_initialized, "CTF not initialized");

	ctf_parameters.SetDefocus(defocus_1 / pixel_size, defocus_2 / pixel_size, deg_2_rad(astigmatism_angle));
}

void Particle::InitCTFImage(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle)
{
	MyDebugAssertTrue(ctf_image->is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! ctf_image->is_in_real_space, "CTF image not in Fourier space");

	InitCTF(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, astigmatism_angle);
	if (ctf_parameters.IsAlmostEqualTo(&current_ctf, 40.0 / pixel_size) == false || ! ctf_image_calculated)
	// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
	{
		current_ctf = ctf_parameters;
		ctf_image->CalculateCTFImage(current_ctf);
	}
	ctf_image_calculated = true;
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

void Particle::SetIndexForWeightedCorrelation()
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

				if (frequency_squared >= low_limit2 && frequency_squared <= high_limit2)
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

void Particle::WeightBySSNR(Curve &SSNR, int include_reference_weighting)
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
	for (i = 0; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conj(ctf_image->complex_values[i]);}
	snr_image->MultiplyByWeightsCurve(SSNR, ssnr_scale_factor);
	particle_image->OptimalFilterBySNRImage(*snr_image, include_reference_weighting);
	is_ssnr_filtered = true;

	delete snr_image;
}

void Particle::WeightBySSNR(Curve &SSNR, Image &projection_image)
{
	MyDebugAssertTrue(particle_image->is_in_memory, "particle_image memory not allocated");
	MyDebugAssertTrue(projection_image.is_in_memory, "projection_image memory not allocated");
	MyDebugAssertTrue(ctf_image->is_in_memory, "CTF image memory not allocated");
	MyDebugAssertTrue(ctf_image_calculated, "CTF image not initialized");
	MyDebugAssertTrue(! is_ssnr_filtered, "Already SSNR filtered");

	int i;
	float particle_area_in_pixels = PI * powf(3.0 * (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) / 4.0 / PI, 2.0 / 3.0);
//	float ssnr_scale_factor = particle_area_in_pixels / mask_volume;
	float ssnr_scale_factor = particle_area_in_pixels / particle_image->logical_x_dimension / particle_image->logical_y_dimension;
//	wxPrintf("particle_area_in_pixels = %g, mask_volume = %g\n", particle_area_in_pixels, mask_volume);
//	float ssnr_scale_factor_old = kDa_to_Angstrom3(molecular_mass_kDa) / 4.0 / PI / powf(pixel_size,3) / (4.0 * PI / 3.0 * powf(mask_volume / PI, 1.5));
//	wxPrintf("old = %g, new = %g\n", ssnr_scale_factor_old, ssnr_scale_factor);
//	float ssnr_scale_factor = PI * powf((3.0 * kDa_to_Angstrom3(molecular_mass_kDa) / 4.0 / PI / powf(pixel_size,3),1.0 / 3.0) ,2) / mask_volume;
//	float ssnr_scale_factor = particle_image->logical_x_dimension * particle_image->logical_y_dimension / mask_volume;

	if (particle_image->is_in_real_space) particle_image->ForwardFFT();

	Image *snr_image = new Image;
	snr_image->Allocate(ctf_image->logical_x_dimension, ctf_image->logical_y_dimension, false);
	particle_image->Whiten();

//	snr_image->CopyFrom(ctf_image);
	for (i = 0; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conj(ctf_image->complex_values[i]);}
	snr_image->MultiplyByWeightsCurve(SSNR, ssnr_scale_factor);
	particle_image->OptimalFilterBySNRImage(*snr_image, 0);
	projection_image.OptimalFilterBySNRImage(*snr_image, -1);
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

void Particle::GetParameters(float *output_parameters)
{
	for (int i = 0; i < number_of_parameters; i++) {output_parameters [i] = current_parameters[i];}
}

void Particle::SetParameters(float *wanted_parameters, bool initialize_scores)
{
	int i;
	for (i = 0; i < number_of_parameters; i++) {current_parameters[i] = wanted_parameters[i];}
	if (initialize_scores) for (i = 12; i < 15; i++) {current_parameters[i] = -std::numeric_limits<float>::max();}

	alignment_parameters.Init(current_parameters[1], current_parameters[2], current_parameters[3], current_parameters[4], current_parameters[5]);
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
	temp_float[1] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	temp_float[2] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	temp_float[3] = target_phase_error / (1.0 / filter_radius_high * 2.0 * PI * mask_radius) / 5.0;
	temp_float[4] = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size) / 5.0;
	temp_float[5] = deg_2_rad(target_phase_error) / (1.0 / filter_radius_high * 2.0 * PI * pixel_size) / 5.0;
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
			mapped_parameters[j] = current_parameters[i];
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
			current_parameters[i] = mapped_parameters[j];
			j++;
		}
	}

	alignment_parameters.Init(current_parameters[1], current_parameters[2], current_parameters[3], current_parameters[4], current_parameters[5]);

	return j;
}

float Particle::ReturnLogLikelihood(ReconstructedVolume &input_3d, ResolutionStatistics &statistics, Image &projection_image, float classification_resolution_limit, float &alpha, float &sigma)
{
//!!!	MyDebugAssertTrue(is_ssnr_filtered, "particle_image not filtered");

	float number_of_independent_pixels;
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

//	ResetImageFlags();
//	mask_volume = PI * powf(mask_radius / pixel_size,2);
	is_ssnr_filtered = false;
	is_centered_in_box = true;
	CenterInCorner();
	input_3d.CalculateProjection(projection_image, *ctf_image, alignment_parameters, mask_radius, mask_falloff, pixel_size / filter_radius_high, false, true);
	projection_image.PhaseFlipPixelWise(*ctf_image);
	WeightBySSNR(statistics.part_SSNR, projection_image);

	particle_image->SwapRealSpaceQuadrants();
	particle_image->PhaseShift(- current_parameters[4] / pixel_size, - current_parameters[5] / pixel_size);
	particle_image->BackwardFFT();

	projection_image.SwapRealSpaceQuadrants();
	projection_image.BackwardFFT();
//	particle_image->QuickAndDirtyWriteSlice("part.mrc", 1);
//	projection_image.QuickAndDirtyWriteSlice("proj.mrc", 1);
	alpha = particle_image->ReturnImageScale(projection_image, mask_radius / pixel_size);
	projection_image.MultiplyByConstant(alpha);

	if (apply_2D_masking)
	{
		AnglesAndShifts	reverse_alignment_parameters;
		reverse_alignment_parameters.Init(- current_parameters[3], - current_parameters[2], - current_parameters[1], 0.0, 0.0);
		reverse_alignment_parameters.euler_matrix.RotateCoords(pixel_center_2d_x, pixel_center_2d_y, pixel_center_2d_z, rotated_center_x, rotated_center_y, rotated_center_z);
		variance_masked = particle_image->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0);
	}
	else
	{
		variance_masked = particle_image->ReturnVarianceOfRealValues(mask_radius / pixel_size);
	}

	particle_image->SubtractImage(&projection_image);
//	particle_image->QuickAndDirtyWriteSlice("diff.mrc", 1);
	if (classification_resolution_limit > 0.0)
	{
		particle_image->ForwardFFT();
		particle_image->CosineMask(pixel_size / classification_resolution_limit, pixel_size / mask_falloff);
		particle_image->BackwardFFT();
	}
	if (apply_2D_masking)
	{
		variance_difference = particle_image->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0);
		sigma = sqrtf(variance_difference / projection_image.ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0));
		number_of_independent_pixels = PI * powf(pixel_radius_2d,2);
	}
	else
	{
		variance_difference = particle_image->ReturnVarianceOfRealValues(mask_radius / pixel_size);
		sigma = sqrtf(variance_difference / projection_image.ReturnVarianceOfRealValues(mask_radius / pixel_size));
		number_of_independent_pixels = PI * powf(mask_radius / pixel_size,2);
//		number_of_independent_pixels = mask_volume;
	}

	// Prevent rare occurrences of unrealistically high sigmas
	if (sigma > 100.0) sigma = 100.0;

//	wxPrintf("number_of_independent_pixels = %g, variance_difference = %g, variance_masked = %g, logp = %g\n", number_of_independent_pixels,
//			variance_difference, variance_masked, -number_of_independent_pixels * variance_difference / variance_masked / 2.0);
//	exit(0);
	return 	- number_of_independent_pixels * variance_difference / variance_masked / 2.0
			+ ReturnParameterLogP(current_parameters);
}

float Particle::ReturnMaskedLogLikelihood(ReconstructedVolume &input_3d, Image &projection_image, float classification_resolution_limit)
{
//!!!	MyDebugAssertTrue(is_ssnr_filtered, "particle_image not filtered");

	float number_of_independent_pixels;
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

	reverse_alignment_parameters.Init(- current_parameters[3], - current_parameters[2], - current_parameters[1], 0.0, 0.0);
	reverse_alignment_parameters.euler_matrix.RotateCoords(pixel_center_2d_x, pixel_center_2d_y, pixel_center_2d_z, rotated_center_x, rotated_center_y, rotated_center_z);

	input_3d.CalculateProjection(projection_image, *ctf_image, alignment_parameters, 0.0, 0.0, pixel_size / classification_resolution_limit, false, false);

	particle_image->PhaseShift(- current_parameters[4] / pixel_size, - current_parameters[5] / pixel_size);
	particle_image->BackwardFFT();
//	wxPrintf("ssq part = %g var part = %g\n", particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//			rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0), particle_image->ReturnVarianceOfRealValues());

	projection_image.SwapRealSpaceQuadrants();
	projection_image.BackwardFFT();
//	particle_image->QuickAndDirtyWriteSlice("part2.mrc", 1);
//	projection_image.QuickAndDirtyWriteSlice("proj2.mrc", 1);

//	variance_masked = particle_image->ReturnVarianceOfRealValues(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//				rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0);

	particle_image->SubtractImage(&projection_image);
//	particle_image->QuickAndDirtyWriteSlice("diff.mrc", 1);

//	wxPrintf("number_of_independent_pixels = %g, variance_difference = %g, variance_masked = %g, logp = %g\n", number_of_independent_pixels,
//			variance_difference, variance_masked, -number_of_independent_pixels * variance_difference / variance_masked / 2.0);
//	wxPrintf("sum = %g pix = %li penalty = %g indep = %g\n", particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
//			rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0), particle_image->number_of_real_space_pixels,
//			ReturnParameterLogP(current_parameters), mask_volume);
//	exit(0);
	return 	- 0.5 * (particle_image->ReturnSumOfSquares(pixel_radius_2d, rotated_center_x + particle_image->physical_address_of_box_center_x,
			rotated_center_y + particle_image->physical_address_of_box_center_y, 0.0) + logf(2.0 * PI)) * PI * powf(pixel_radius_2d, 2)
			+ ReturnParameterLogP(current_parameters);
}

float Particle::MLBlur(Image *input_classes_cache, float ssq_X, Image &cropped_input_image, Image *rotation_cache, Image &blurred_image,
		int current_class, int number_of_rotations, float psi_step, float psi_start, float smoothing_factor, float &max_logp_particle,
		int best_class, float best_psi, Image &best_correlation_map, bool calculate_correlation_map_only, bool uncrop, bool apply_ctf_to_classes)
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
	float rvar2_x = powf(pixel_size, 2) / parameter_variance[4] / 2.0;
	float rvar2_y = powf(pixel_size, 2) / parameter_variance[5] / 2.0;
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
		rotation_angle.euler_matrix.RotateCoords2D(parameter_average[4], parameter_average[4], mid_x, mid_y);
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
			if (constraints_used[5])
			{
				if (j > particle_image->physical_address_of_box_center_y) dy = particle_image->logical_y_dimension - j;
				else dy = j;
				penalty_y = powf(dy - mid_y, 2) * rvar2_y;
			}
			for (i = 0; i < particle_image->logical_x_dimension; i++)
			{
				if (constraints_used[4])
				{
					if (i > particle_image->physical_address_of_box_center_x) dx = particle_image->logical_y_dimension - i;
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
			if (j > particle_image->physical_address_of_box_center_y) dy = particle_image->logical_y_dimension - j;
			else dy = j;
			for (i = 0; i < particle_image->logical_x_dimension; i++)
			{
				if (i > particle_image->physical_address_of_box_center_x) dx = particle_image->logical_y_dimension - i;
				else dx = i;
				if(correlation_map->real_values[pixel_counter] > max_logp_particle)
				{
					new_max_found = true;
					max_logp_particle = correlation_map->real_values[pixel_counter];
					// Store correlation coefficient that corresponds to highest likelihood
					snr_psi = temp_image->real_values[pixel_counter];
					rotation_angle.euler_matrix.RotateCoords2D(dx, dy, current_parameters[4], current_parameters[5]);
					current_parameters[3] = psi;
					current_parameters[7] = current_class + 1;
				}
				pixel_counter++;
			}
			pixel_counter += particle_image->padding_jump_value;
		}
//		// To get normalized correlation coefficient, need to divide by sigmas of particle and reference
		// To get sigma^2, need to calculate ssq_X - 2XA + ssq_A
		if (new_max_found)
		{
			snr_psi = (ssq_X - 2.0 * snr_psi + ssq_A) * particle_image->number_of_real_space_pixels / number_of_independent_pixels / var_A;
			// Update SIGMA (SNR)
			if (snr_psi >= 0.0) current_parameters[13] = sqrtf(snr_psi);
			// Update SCORE
			current_parameters[14] = 100.0 * (max_logp_particle + norm_A) / number_of_pixels / ssq_XA2;
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
			if (snr_psi >= 0.0) current_parameters[13] = sqrtf(snr_psi);
			// Update SCORE
			current_parameters[14] = 100.0 * (max_logp_particle + norm_A) / number_of_pixels / ssq_XA2;
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
					if (correlation_map->real_values[pixel_counter] >= log_threshold)
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
				correlation_map->ForwardFFT();

				if (use_best_psi) i = number_of_rotations;
				else i = current_rotation;
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
		sum_image->ForwardFFT();
		if (uncrop)
		{
			sum_image->ClipInto(&cropped_input_image);
			cropped_input_image.BackwardFFT();
			cropped_input_image.ClipIntoLargerRealSpace2D(&blurred_image, cropped_input_image.ReturnAverageOfRealValuesOnEdges());
		}
		else
		{
			blurred_image.CopyFrom(sum_image);
		}

//		wxPrintf("log sump_class = %g old_max_logp = %g number_of_independent_pixels = %g ssq_X = %g\n", logf(sump_class), old_max_logp, number_of_independent_pixels, ssq_X);
		logp = logf(sump_class) + old_max_logp - 0.5 * (number_of_independent_pixels * logf(2.0 * PI) + number_of_pixels * ssq_X);
		current_parameters[12] = logp;
	}
	else
	{
		blurred_image.SetToConstant(0.0);
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
