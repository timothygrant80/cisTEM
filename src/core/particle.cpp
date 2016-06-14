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
	if (ctf_parameters.IsAlmostEqualTo(&current_ctf, 40.0 / pixel_size) == false)
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
	for (i = 0; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conjf(ctf_image->complex_values[i]);}
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
	for (i = 0; i < ctf_image->real_memory_allocated / 2; i++) {snr_image->complex_values[i] = ctf_image->complex_values[i] * conjf(ctf_image->complex_values[i]);}
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

void Particle::SetParameters(float *wanted_parameters)
{
	for (int i = 0; i < number_of_parameters; i++) {current_parameters[i] = wanted_parameters[i];};

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

float Particle::ReturnLogLikelihood(ReconstructedVolume &input_3d, Image &projection_image, float classification_resolution_limit, float &alpha, float &sigma)
{
	MyDebugAssertTrue(is_ssnr_filtered, "particle_image not filtered");

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
	WeightBySSNR(input_3d.statistics.part_SSNR, projection_image);

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
		number_of_independent_pixels = mask_volume;
	}

	// Prevent rare occurrences of unrealistically high sigmas
	if (sigma > 100.0) sigma = 100.0;

//	wxPrintf("number_of_independent_pixels = %g, variance_difference = %g, variance_masked = %g, logp = %g\n", number_of_independent_pixels,
//			variance_difference, variance_masked, -number_of_independent_pixels * variance_difference / variance_masked / 2.0);
//	exit(0);
	return 	- number_of_independent_pixels * variance_difference / variance_masked / 2.0
			+ ReturnParameterLogP(current_parameters);
}
