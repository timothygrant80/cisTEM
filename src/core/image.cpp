#include "core_headers.h"

Image::Image()
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;

	is_in_real_space = true;
	object_is_centred_in_box = true;

	physical_upper_bound_complex_x = 0;
	physical_upper_bound_complex_y = 0;
	physical_upper_bound_complex_z = 0;

	physical_address_of_box_center_x = 0;
	physical_address_of_box_center_y = 0;
	physical_address_of_box_center_z = 0;

	//physical_index_of_first_negative_frequency_x = 0;
	physical_index_of_first_negative_frequency_y = 0;
	physical_index_of_first_negative_frequency_z = 0;

	fourier_voxel_size_x = 0.0;
	fourier_voxel_size_y = 0.0;
	fourier_voxel_size_z = 0.0;

	logical_upper_bound_complex_x = 0;
	logical_upper_bound_complex_y = 0;
	logical_upper_bound_complex_z = 0;

	logical_lower_bound_complex_x = 0;
	logical_lower_bound_complex_y = 0;
	logical_lower_bound_complex_z = 0;

	logical_upper_bound_real_x = 0;
	logical_upper_bound_real_y = 0;
	logical_upper_bound_real_z = 0;

	logical_lower_bound_real_x = 0;
	logical_lower_bound_real_y = 0;
	logical_lower_bound_real_z = 0;

	insert_into_which_reconstruction = 0;

	real_values = NULL;
	complex_values = NULL;

	is_in_memory = false;
	real_memory_allocated = 0;

	plan_fwd = NULL;
	plan_bwd = NULL;

	planned = false;

	padding_jump_value = 0;
}

Image::Image( const Image &other_image) // copy constructor
{
	MyDebugPrint("Warning: copying an image object");
	 *this = other_image;
}

Image::~Image()
{
	Deallocate();
}


int Image::ReturnSmallestLogicalDimension()
{
	if (logical_z_dimension == 1)
	{
		return std::min(logical_x_dimension, logical_y_dimension);
	}
	else
	{
		int temp_int;
		temp_int = std::min(logical_x_dimension, logical_y_dimension);
		return std::min(temp_int, logical_z_dimension);
	}
}


int Image::ReturnLargestLogicalDimension()
{
	if (logical_z_dimension == 1)
	{
		return std::max(logical_x_dimension, logical_y_dimension);
	}
	else
	{
		int temp_int;
		temp_int = std::max(logical_x_dimension, logical_y_dimension);
		return std::max(temp_int, logical_z_dimension);
	}
}

float Image::ReturnSumOfSquares()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	int i;
	int j;
	int k;
	long address = 0;

	double sum = 0.0;

	if (is_in_real_space)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			for (j = 0; j < logical_y_dimension; j++)
			{
				for (i = 0; i < logical_x_dimension; i++)
				{
					sum += powf(real_values[address],2);
					address++;
				}
				address += padding_jump_value;
			}
		}

	}

	return sum / logical_x_dimension / logical_y_dimension / logical_z_dimension;
}

float Image::ReturnSigmaNoise(Image &matching_projection, float mask_radius)
{
	float correlation_coefficient;
	float alpha;
	float sigma;
	float pixel_variance_image;
	float pixel_variance_projection;

	correlation_coefficient = ReturnCorrelationCoefficientUnnormalized(matching_projection, mask_radius);
	pixel_variance_image = ReturnVarianceOfRealValues(mask_radius);
	pixel_variance_projection = matching_projection.ReturnVarianceOfRealValues(mask_radius);
	alpha = fabs(correlation_coefficient / pixel_variance_projection);
//	sigma = sqrtf(pixel_variance_image - powf(alpha,2) * pixel_variance_projection);
//	wxPrintf("var_input = %f, var_output = %g, alpha = %f, sigma_signal = %f, sigma_noise = %f\n", pixel_variance_image, pixel_variance_projection, alpha, sqrtf(pixel_variance_image - powf(sigma,2)), sigma);

	return sqrtf(pixel_variance_image - powf(alpha,2) * pixel_variance_projection);
}

float Image::ReturnImageScale(Image &matching_projection, float mask_radius)
{
	float correlation_coefficient;
	float alpha;
	float sigma;
	float pixel_variance_image;
	float pixel_variance_projection;

	correlation_coefficient = ReturnCorrelationCoefficientUnnormalized(matching_projection, mask_radius);
	pixel_variance_image = ReturnVarianceOfRealValues(mask_radius);
	pixel_variance_projection = matching_projection.ReturnVarianceOfRealValues(mask_radius);

	return fabs(correlation_coefficient / pixel_variance_projection);
}

float Image::ReturnCorrelationCoefficientUnnormalized(Image &other_image, float wanted_mask_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");
	MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");
	MyDebugAssertTrue(other_image.is_in_real_space == is_in_real_space, "Images not in the same space");

	int i;
	int j;
	int k;
	int number_of_pixels = 0;

	float x;
	float y;
	float z;

	long pixel_counter = 0;

	float distance_from_center_squared;
	float mask_radius_squared;
	float edge;
	double cross_terms = 0.0;

	if (wanted_mask_radius > 0.0)
	{
		mask_radius_squared = powf(wanted_mask_radius, 2);
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared <= mask_radius_squared)
					{
						cross_terms += real_values[pixel_counter] * other_image.real_values[pixel_counter];
						number_of_pixels++;
					}
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			for (j = 0; j < logical_y_dimension; j++)
			{
				for (i = 0; i < logical_x_dimension; i++)
				{
					cross_terms += real_values[pixel_counter] * other_image.real_values[pixel_counter];
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		number_of_pixels = logical_x_dimension * logical_y_dimension * logical_z_dimension;
	}

	return float(cross_terms / number_of_pixels);
}

// Frealign weighted correlation coefficient
float Image::GetWeightedCorrelationWithImage(Image &projection_image, float low_limit, float high_limit)
{
	MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
	MyDebugAssertTrue(projection_image.is_in_memory, "projection_image memory not allocated");
	MyDebugAssertTrue(! is_in_real_space, "Image not in Fourier space");
	MyDebugAssertTrue(! projection_image.is_in_real_space, "projection_image not in Fourier space");
	MyDebugAssertTrue(object_is_centred_in_box == projection_image.object_is_centred_in_box, "Image quadrants not in the same locations");
//	MyDebugAssertTrue(! projection_image.object_is_centred_in_box, "projection_image quadrants have not been swapped");
	MyDebugAssertTrue(HasSameDimensionsAs(&projection_image), "Images do not have the same dimensions");

	int i;
	int j;
	int k;
	int bin;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;
	float score;
	float sum1;
	float sum2;
	float sum3;
	float r;

	float low_limit2 = powf(low_limit,2);
	float high_limit2 = fminf(powf(high_limit,2),0.25);

	int number_of_bins = ReturnLargestLogicalDimension() / 2 + 1;
	int number_of_bins2 = 2 * (number_of_bins - 1);

	double *sum_a = new double[number_of_bins];
	double *sum_b = new double[number_of_bins];
	double *cross_terms = new double[number_of_bins];

	long pixel_counter = 0;

	fftwf_complex temp_c;

	ZeroDoubleArray(sum_a, number_of_bins);
	ZeroDoubleArray(sum_b, number_of_bins);
	ZeroDoubleArray(cross_terms, number_of_bins);

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				if (frequency_squared >= low_limit2 && frequency_squared <= high_limit2)
				{
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					temp_c = creal(complex_values[pixel_counter] * conjf(projection_image.complex_values[pixel_counter]));
					if (temp_c != 0.0)
					{
						sum_a[bin] += creal(complex_values[pixel_counter] * conjf(complex_values[pixel_counter]));
						sum_b[bin] += creal(projection_image.complex_values[pixel_counter] * conjf(projection_image.complex_values[pixel_counter]));
						cross_terms[bin] += creal(temp_c);
					}
				}
				pixel_counter++;
			}
		}
	}

	sum1 = 0.0;
	sum2 = 0.0;
	sum3 = 0.0;
	for (i = 1; i < number_of_bins; i++)
	{
		if (sum_b[i] != 0.0)
		{
			r = cross_terms[i] / sqrtf(sum_b[i]);
			sum1 += 1.0;
			sum2 += sum_a[i];
			sum3 += r;
		}
	}
	sum1 *= sum2;
	if (sum1 != 0.0) sum3 /= sqrtf(sum1);

	delete [] sum_a;
	delete [] sum_b;
	delete [] cross_terms;

	return sum3;
}

void Image::PhaseFlipPixelWise(Image &phase_image)
{
	MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
	MyDebugAssertTrue(phase_image.is_in_memory, "Other image memory not allocated");

	int i;
	long pixel_counter;

	if (is_in_real_space)
	{
		for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
		{
			if (phase_image.real_values[pixel_counter] < 0.0) real_values[pixel_counter] = - real_values[pixel_counter];
		}
	}
	else
	{
		for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++)
		{
			if (crealf(phase_image.complex_values[pixel_counter]) < 0.0) complex_values[pixel_counter] = - complex_values[pixel_counter];
		}
	}
}

void Image::MultiplyPixelWise(Image &other_image)
{
	MyDebugAssertTrue(is_in_memory, "Image memory not allocated");
	MyDebugAssertTrue(other_image.is_in_memory, "Other image memory not allocated");

	int i;
	long pixel_counter;

	if (is_in_real_space)
	{
		for (pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
		{
			real_values[pixel_counter] *= other_image.real_values[pixel_counter];
		}
	}
	else
	{
		for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++)
		{
			complex_values[pixel_counter] *= other_image.complex_values[pixel_counter];
		}
	}
}

void Image::AddGaussianNoise(float wanted_sigma_value)
{
	MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] += wanted_sigma_value * global_random_number_generator.GetNormalRandom();
	}
}

void Image::Normalize(float wanted_sigma_value, float wanted_mask_radius)
{
	MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");

	float variance = ReturnVarianceOfRealValues(wanted_mask_radius);
	float average = ReturnAverageOfRealValues(wanted_mask_radius);

	AddConstant(-average);
	DivideByConstant(sqrtf(variance));
}

float Image::ReturnVarianceOfRealValues(float wanted_mask_radius, float wanted_center_x, float wanted_center_y, float wanted_center_z)
{
	MyDebugAssertTrue(is_in_real_space == true, "Image must be in real space");
	int i;
	int j;
	int k;
	int number_of_pixels = 0;

	float x;
	float y;
	float z;

	long pixel_counter = 0;

	float distance_from_center_squared;
	float mask_radius_squared;
	float edge;
	float center_x;
	float center_y;
	float center_z;
	double pixel_sum = 0.0;
	double pixel_sum_squared = 0.0;

	if (wanted_center_x == 0.0 && wanted_center_y == 0.0 && wanted_center_z == 0.0)
	{
		center_x = physical_address_of_box_center_x;
		center_y = physical_address_of_box_center_y;
		center_z = physical_address_of_box_center_z;
	}
	else
	{
		center_x = wanted_center_x;
		center_y = wanted_center_y;
		center_z = wanted_center_z;
	}

	if (wanted_mask_radius > 0.0)
	{
		mask_radius_squared = powf(wanted_mask_radius, 2);
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared <= mask_radius_squared)
					{
						pixel_sum += real_values[pixel_counter];
						pixel_sum_squared += powf(real_values[pixel_counter],2);
						number_of_pixels++;
					}
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			for (j = 0; j < logical_y_dimension; j++)
			{
				for (i = 0; i < logical_x_dimension; i++)
				{
					pixel_sum += real_values[pixel_counter];
					pixel_sum_squared += powf(real_values[pixel_counter],2);
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		number_of_pixels = logical_x_dimension * logical_y_dimension * logical_z_dimension;
	}

	return fabs(float(pixel_sum_squared / number_of_pixels - powf(pixel_sum / number_of_pixels, 2)));
}

void Image::WhitenTwo(Image &other_image)
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to whiten is not in Fourier space");

	int i;
	int j;
	int k;
	int bin;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;
	float temp_float;

	int number_of_bins = ReturnLargestLogicalDimension() / 2 + 1;
// Extend table to include corners in 3D Fourier space
	int number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;
	double *sum = new double[number_of_bins_extended];
	int number_of_bins2 = 2 * (number_of_bins - 1);
	int *non_zero_count = new int[number_of_bins_extended];
	ZeroDoubleArray(sum, number_of_bins_extended);
	ZeroIntArray(non_zero_count, number_of_bins_extended);

	long pixel_counter = 0;

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				temp_float = powf(cabsf(complex_values[pixel_counter]),2);
				if (temp_float != 0.0)
				{
					x = powf(i * fourier_voxel_size_x, 2);
					frequency_squared = x + y + z;

					// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					sum[bin] += temp_float;
					non_zero_count[bin] += 1;
				}
				pixel_counter++;
			}
		}
	}

	for (i = 0; i < number_of_bins_extended; i++)
	{
		if (non_zero_count[i] != 0) sum[i] = sqrtf(sum[i] / non_zero_count[i]);
	}

	pixel_counter = 0;
	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				// compute radius, in units of physical Fourier pixels
				bin = int(sqrtf(frequency_squared) * number_of_bins2);

				if (sum[bin] != 0.0)
				{
					complex_values[pixel_counter] /= sum[bin];
					other_image.complex_values[pixel_counter] /= sum[bin];
				}
				pixel_counter++;
			}
		}
	}

	delete [] sum;
	delete [] non_zero_count;
}

void Image::Whiten()
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to whiten is not in Fourier space");

	int i;
	int j;
	int k;
	int bin;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;
	float temp_float;

	int number_of_bins = ReturnLargestLogicalDimension() / 2 + 1;
// Extend table to include corners in 3D Fourier space
	int number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;
	double *sum = new double[number_of_bins_extended];
	int number_of_bins2 = 2 * (number_of_bins - 1);
	int *non_zero_count = new int[number_of_bins_extended];
	ZeroDoubleArray(sum, number_of_bins_extended);
	ZeroIntArray(non_zero_count, number_of_bins_extended);

	long pixel_counter = 0;

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				temp_float = powf(cabsf(complex_values[pixel_counter]),2);
				if (temp_float != 0.0)
				{
					x = powf(i * fourier_voxel_size_x, 2);
					frequency_squared = x + y + z;

					// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					sum[bin] += temp_float;
					non_zero_count[bin] += 1;
				}
				pixel_counter++;
			}
		}
	}

	for (i = 0; i < number_of_bins_extended; i++)
	{
		if (non_zero_count[i] != 0) sum[i] = sqrtf(sum[i] / non_zero_count[i]);
	}

	pixel_counter = 0;
	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				// compute radius, in units of physical Fourier pixels
				bin = int(sqrtf(frequency_squared) * number_of_bins2);

				if (sum[bin] != 0.0) complex_values[pixel_counter] /= sum[bin];
				pixel_counter++;
			}
		}
	}

	delete [] sum;
	delete [] non_zero_count;
}

void Image::MultiplyByWeightsCurve(Curve &weights)
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
	MyDebugAssertTrue(weights.number_of_points > 0, "weights curve not calculated");

	int i;
	int j;
	int k;
	int bin;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;

	int number_of_bins2 = ReturnLargestLogicalDimension();

	long pixel_counter = 0;

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				if (frequency_squared <= 0.25)
				{
					// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					complex_values[pixel_counter] *= fabs(weights.data_y[bin]);
				}
				else
				{
					complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}
}

void Image::OptimalFilterBySNRImage(Image &SNR_image)
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
	MyDebugAssertTrue(SNR_image.is_in_real_space == false, "SNR image not in Fourier space");

	int i;
	float snr;

	long pixel_counter = 0;

	for (i = 0; i < real_memory_allocated / 2; i++)
	{
		snr = cabsf(SNR_image.complex_values[pixel_counter]);
		complex_values[pixel_counter] *= snr / (1.0 + snr);
		pixel_counter++;
	}
}

void Image::OptimalFilterSSNR(Curve &SSNR)
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
	MyDebugAssertTrue(SSNR.number_of_points > 0, "SSNR curve not calculated");

	int i;
	int j;
	int k;
	int bin;
	int number_of_bins2;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;
	float snr;

	long pixel_counter = 0;

	number_of_bins2 = 2 * (SSNR.number_of_points - 1);

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				if (frequency_squared <= 0.25)
				{
					// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					snr = fabs(SSNR.data_y[bin]);
					complex_values[pixel_counter] *= snr / (1.0 + snr);
				}
				else
				{
					complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}
}

void Image::OptimalFilterFSC(Curve &FSC)
{
	MyDebugAssertTrue(is_in_real_space == false, "Image to filter not in Fourier space");
	MyDebugAssertTrue(FSC.number_of_points > 0, "FSC curve not calculated");

	int i;
	int j;
	int k;
	int bin;
	int number_of_bins2;

	float x;
	float y;
	float z;
	float frequency;
	float frequency_squared;

	long pixel_counter = 0;

	number_of_bins2 = 2 * (FSC.number_of_points - 1);

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x = powf(i * fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

				if (frequency_squared <= 0.25)
				{
					// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					if (FSC.data_y[bin] != 0.0) complex_values[pixel_counter] /= (1.0 + 0.5 * (1.0 - fabs(FSC.data_y[bin])) / fabs(FSC.data_y[bin]));
				}
				else
				{
					complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}
}

float Image::Correct3D(float mask_radius)
{
	MyDebugAssertTrue(is_in_real_space == true, "reconstruction to correct not in real space");

	int i;
	int j;
	int k;
	int int_x_coordinate;
	int int_y_coordinate;
	int int_z_coordinate;

	float x;
	float y;
	float z;
	float distance_from_center_squared;
	float mask_radius_squared = powf(mask_radius,2);
	double pixel_sum;

	long pixel_counter = 0;

	float weight;
	float weight_y;
	float weight_z;
	float scale_x = PI / logical_x_dimension;
	float scale_y = PI / logical_y_dimension;
	float scale_z = PI / logical_z_dimension;

	for (k = 0; k < logical_z_dimension; k++)
	{
		int_z_coordinate = k - physical_address_of_box_center_z;
		z = powf(int_z_coordinate, 2);
		weight_z = sinc(float(int_z_coordinate) * scale_z);

		for (j = 0; j < logical_y_dimension; j++)
		{
			int_y_coordinate = j - physical_address_of_box_center_y;
			y = powf(int_y_coordinate, 2);
			weight_y = sinc(float(int_y_coordinate) * scale_y);

			for (i = 0; i < logical_x_dimension; i++)
			{
				int_x_coordinate = i - physical_address_of_box_center_x;
				x = powf(int_x_coordinate, 2);

				weight = powf(sinc(float(int_x_coordinate) * scale_x) * weight_y * weight_z,2);

				real_values[pixel_counter] /= weight;

				distance_from_center_squared = x + y + z;

				if (distance_from_center_squared <= mask_radius_squared)
				{
					pixel_sum += weight;
				}

				pixel_counter++;
			}
			pixel_counter += padding_jump_value;
		}
	}
	if (pixel_sum == 0.0) return 0.0;
	return pixel_sum / (4.0 / 3.0 * PI * powf(mask_radius,3));
}

void Image::MirrorXFourier2D(Image &mirrored_image)
{
	MyDebugAssertTrue(mirrored_image.logical_z_dimension == 1, "Error: attempting to mirrored_image into 3D image");
	MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to mirrored_image from 3D image");
	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
	MyDebugAssertTrue(mirrored_image.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(mirrored_image.logical_x_dimension == logical_x_dimension && mirrored_image.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");

	int pixel_counter;
	int mirrored_counter;
	int x_counter;
	int y_counter;
	int fft_dim_x = physical_upper_bound_complex_x + 1;

	x_counter = 0;
	y_counter = 0;
	for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++)
	{
		mirrored_counter = y_counter * fft_dim_x + x_counter;
	//					correlation_map->complex_values[pixel_counter] = rotated_image->complex_values[mirrored_counter] * conjf(particle.particle_image->complex_values[pixel_counter]);
	//					correlation_map->complex_values[pixel_counter] = conjf(rotated_image->complex_values[pixel_counter]);
		mirrored_image.complex_values[pixel_counter] = complex_values[mirrored_counter];
	//					wxPrintf("x_counter = %i, y_counter = %i; pixel_counter = %i, mirror_counter = %i\n", x_counter, y_counter, pixel_counter, mirrored_counter);
		x_counter++;
		if (x_counter >= fft_dim_x)
		{
			x_counter -= fft_dim_x;
			y_counter--;
			if (y_counter < 0) y_counter += logical_y_dimension;
		}
	}
	mirrored_image.is_in_real_space = false;
}

void Image::MirrorYFourier2D(Image &mirrored_image)
{
	MyDebugAssertTrue(mirrored_image.logical_z_dimension == 1, "Error: attempting to mirrored_image into 3D image");
	MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to mirrored_image from 3D image");
	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
	MyDebugAssertTrue(mirrored_image.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(mirrored_image.logical_x_dimension == logical_x_dimension && mirrored_image.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");

	int pixel_counter;
	int mirrored_counter;
	int x_counter;
	int y_counter;
	int fft_dim_x = physical_upper_bound_complex_x + 1;

	x_counter = 0;
	y_counter = 0;
	for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++)
	{
		mirrored_counter = y_counter * fft_dim_x + x_counter;
		mirrored_image.complex_values[pixel_counter] = conjf(complex_values[mirrored_counter]);
		x_counter++;
		if (x_counter >= fft_dim_x)
		{
			x_counter -= fft_dim_x;
			y_counter--;
			if (y_counter < 0) y_counter += logical_y_dimension;
		}
	}
	mirrored_image.is_in_real_space = false;
}

void Image::RotateQuadrants(Image &rotated_image, int quad_i)
{
	MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
	MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
	MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
	MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
	MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension && rotated_image.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(quad_i == 0 || quad_i == 90 || quad_i == 180 || quad_i == 270, "Selected rotation invalid");

	int i;
	int j;

	long pixel_counter;
	long pixel_counter2;

	rotated_image.object_is_centred_in_box = object_is_centred_in_box;
	rotated_image.is_in_real_space = false;

	if (quad_i == 0)
	{
		for (i = 0; i < real_memory_allocated / 2; i++) {rotated_image.complex_values[i] = complex_values[i];};
		return;
	}

	if (quad_i == 180)
	{
		for (i = 0; i < real_memory_allocated / 2; i++) {rotated_image.complex_values[i] = conjf(complex_values[i]);};
		return;
	}

	if (quad_i == 90)
	{
		for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++)
		{
			for (i = 0; i <= logical_upper_bound_complex_x; i++)
			{
				pixel_counter = ReturnFourier1DAddressFromLogicalCoord(i,j,0);

				if (i <= logical_upper_bound_complex_y)
				{
					pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j,i,0);
				}
				else
				{
					pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j,-i,0);
				}

				if (j <= 0)
				{
					rotated_image.complex_values[pixel_counter2] = conjf(complex_values[pixel_counter]);
				}
				else
				{
					rotated_image.complex_values[pixel_counter2] = complex_values[pixel_counter];
				}
			}
		}
		return;
	}

	if (quad_i == 270)
	{
		for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++)
		{
			for (i = 0; i <= logical_upper_bound_complex_x; i++)
			{
				pixel_counter = ReturnFourier1DAddressFromLogicalCoord(i,j,0);

				if (i <= logical_upper_bound_complex_y)
				{
					pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j,i,0);
				}
				else
				{
					pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(-j,-i,0);
				}

				if (j <= 0)
				{
					rotated_image.complex_values[pixel_counter2] = complex_values[pixel_counter];
				}
				else
				{
					rotated_image.complex_values[pixel_counter2] = conjf(complex_values[pixel_counter]);
				}
			}
		}
		return;
	}

	rotated_image.is_in_real_space = false;
}

void Image::GenerateReferenceProjections(Image *projections, EulerSearch &parameters)
{
	int i;
	AnglesAndShifts angles;

	for (i = 0; i < parameters.number_of_search_positions; i++)
	{
		angles.Init(parameters.list_of_search_parameters[0][i], parameters.list_of_search_parameters[1][i], 0.0, 0.0, 0.0);
		ExtractSlice(projections[i], angles, parameters.resolution_limit);
	}
}

void Image::RotateFourier2DGenerateIndex(Kernel2D **&kernel_index, float psi_max, float psi_step)
{
	int psi_i;
	float psi;
	AnglesAndShifts angles;

	psi_i = myroundint(psi_max / psi_step);
	wxPrintf("psi_max = %f, psi_step = %f, psi_i = %i\n", psi_max, psi_step, psi_i);
	kernel_index = new Kernel2D* [psi_i];									// dynamic array of pointers to float
	psi_i = 0;
	for (psi = 0.0; psi < psi_max; psi += psi_step)
	{
		kernel_index[psi_i] = new Kernel2D [real_memory_allocated / 2];		// each i-th pointer is now pointing to dynamic array (size number_of_positions) of actual float values
		psi_i++;
	}

	psi_i = 0;
	for (psi = 0.0; psi < psi_max; psi += psi_step)
	{
		angles.GenerateRotationMatrix2D(psi);
		RotateFourier2DIndex(kernel_index[psi_i], angles);
		psi_i++;
	}
}

void Image::RotateFourier2DDeleteIndex(Kernel2D **&kernel_index, float psi_max, float psi_step)
{
	int psi_i = 0;
	float psi;

	for (psi = 0.0; psi < psi_max; psi += psi_step)
	{
		delete [] kernel_index[psi_i];				// delete inner arrays of floats
		psi_i++;
	}
	delete [] kernel_index;							// delete array of pointers to float arrays

	kernel_index = NULL;
}

void Image::RotateFourier2DFromIndex(Image &rotated_image, Kernel2D *kernel_index)
{
	MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
	MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
	MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
	MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
	MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension && rotated_image.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(! object_is_centred_in_box, "Image volume quadrants not swapped");

	int i;
	int index;
	float weight;
	int pixel_counter;

	for (pixel_counter = 0; pixel_counter < rotated_image.real_memory_allocated / 2; pixel_counter++)
	{
		rotated_image.complex_values[pixel_counter] = 0.0;
		if (kernel_index[pixel_counter].pixel_index[0] != kernel_index[pixel_counter].pixel_index[3])
		{
			for (i = 0; i < 4; i++)
			{
				index = kernel_index[pixel_counter].pixel_index[i];
				weight = kernel_index[pixel_counter].pixel_weight[i];
				if (weight < 0.0)
				{
					rotated_image.complex_values[pixel_counter] -= conjf(complex_values[index]) * weight;
				}
				else
				{
					rotated_image.complex_values[pixel_counter] += complex_values[index] * weight;
				}
			}
		}
	}
	rotated_image.is_in_real_space = false;
}

void Image::RotateFourier2DIndex(Kernel2D *kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit, float padding_factor)
{
	MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");

	int i;
	int j;

	int pixel_counter;
	int pixel_counter2;

	float x_coordinate_2d;
	float y_coordinate_2d;

	float x_coordinate_3d;
	float y_coordinate_3d;

	float resolution_limit_sq = powf(resolution_limit * logical_x_dimension,2);
	float y_coord_sq;

	Kernel2D null_kernel;

	null_kernel.pixel_index[0] = 0;
	null_kernel.pixel_index[1] = 0;
	null_kernel.pixel_index[2] = 0;
	null_kernel.pixel_index[3] = 0;
	null_kernel.pixel_weight[0] = 0.0;
	null_kernel.pixel_weight[1] = 0.0;
	null_kernel.pixel_weight[2] = 0.0;
	null_kernel.pixel_weight[3] = 0.0;

	for (j = logical_lower_bound_complex_y; j <= logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j * padding_factor;
		y_coord_sq = powf(y_coordinate_2d,2);
		for (i = 1; i <= logical_upper_bound_complex_x; i++)
		{
			x_coordinate_2d = i * padding_factor;
			pixel_counter = ReturnFourier1DAddressFromLogicalCoord(i,j,0);
			if (powf(x_coordinate_2d,2) + y_coord_sq <= resolution_limit_sq)
			{
				rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
//				rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
				kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
			}
			else
			{
//				rotated_image.complex_values[pixel_counter] = 0.0;
				kernel_index[pixel_counter] = null_kernel;
			}
		}
	}
// Now deal with special case of i = 0
	for (j = 1; j <= logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j * padding_factor;
		x_coordinate_2d = 0;
		if (powf(y_coordinate_2d,2) <= resolution_limit_sq)
		{
			rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
			pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0,j,0);
//			rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
			kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
			pixel_counter2 = ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
//			rotated_image.complex_values[pixel_counter2] = conjf(rotated_image.complex_values[pixel_counter]);
			kernel_index[pixel_counter2] = kernel_index[pixel_counter];
			kernel_index[pixel_counter2].pixel_weight[0] = - kernel_index[pixel_counter2].pixel_weight[0];
			kernel_index[pixel_counter2].pixel_weight[1] = - kernel_index[pixel_counter2].pixel_weight[1];
			kernel_index[pixel_counter2].pixel_weight[2] = - kernel_index[pixel_counter2].pixel_weight[2];
			kernel_index[pixel_counter2].pixel_weight[3] = - kernel_index[pixel_counter2].pixel_weight[3];
		}
		else
		{
			pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0,j,0);
//			rotated_image.complex_values[pixel_counter] = 0.0;
			kernel_index[pixel_counter] = null_kernel;
			pixel_counter2 = ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
//			rotated_image.complex_values[pixel_counter2] = 0.0;
			kernel_index[pixel_counter2] = null_kernel;
		}
	}
// Deal with pixel at edge if image dimensions are even
	if (-logical_lower_bound_complex_y != logical_upper_bound_complex_y)
	{
		y_coordinate_2d = logical_lower_bound_complex_y * padding_factor;
		x_coordinate_2d = 0;
		rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
		pixel_counter = ReturnFourier1DAddressFromLogicalCoord(0,logical_lower_bound_complex_y,0);
//		rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
		kernel_index[pixel_counter] = ReturnLinearInterpolatedFourierKernel2D(x_coordinate_3d, y_coordinate_3d);
	}

// Set origin to zero to generate a projection with average set to zero
//	rotated_image.complex_values[0] = 0.0;
	kernel_index[0] = null_kernel;
}

Kernel2D Image::ReturnLinearInterpolatedFourierKernel2D(float &x, float &y)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_in_real_space, "Must be in Fourier space");

//	fftwf_complex sum = 0.0;
	int i;
	int j;
	int i_start;
	int i_end;
	int j_start;
	int j_end;
	int physical_y_address;
	int jj;
	int i_coeff = 0;
	Kernel2D kernel;

	float weight;
	float y_dist;

	kernel.pixel_index[0] = 0;
	kernel.pixel_index[1] = 0;
	kernel.pixel_index[2] = 0;
	kernel.pixel_index[3] = 0;
	kernel.pixel_weight[0] = 0.0;
	kernel.pixel_weight[1] = 0.0;
	kernel.pixel_weight[2] = 0.0;
	kernel.pixel_weight[3] = 0.0;

	if (x >= 0.0)
	{
		i_start = int(floorf(x));
		i_end = i_start + 1;
		if (i_end > logical_upper_bound_complex_x) return kernel;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return kernel;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return kernel;

		for (j = j_start; j <= j_end; j++)
		{
			if (j >= 0)
			{
				physical_y_address = j;
			}
			else
			{
				physical_y_address = logical_y_dimension + j;
			}
			jj = (physical_upper_bound_complex_x + 1) * physical_y_address;
			y_dist = fabs(y - float(j));
			for (i = i_start; i <= i_end; i++)
			{
				weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist);
				kernel.pixel_index[i_coeff] = jj + i;
				kernel.pixel_weight[i_coeff] = weight;
				i_coeff++;
//				sum = sum + complex_values[jj + i] * weight;
			}
		}
	}
	else
	{
		i_start = int(floorf(x));
		if (i_start < logical_lower_bound_complex_x) return kernel;
		i_end = i_start + 1;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return kernel;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return kernel;

		for (j = j_start; j <= j_end; j++)
		{
			if (j > 0)
			{
				physical_y_address = logical_y_dimension - j;
			}
			else
			{
				physical_y_address = -j;
			}
			jj = (physical_upper_bound_complex_x + 1) * physical_y_address;
			y_dist = fabs(y - float(j));
			for (i = i_start; i <= i_end; i++)
			{
				weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist);
				weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist);
				kernel.pixel_index[i_coeff] = jj - i;
				kernel.pixel_weight[i_coeff] = - weight;
				i_coeff++;
//				sum = sum + conjf(complex_values[jj - i]) * weight;
			}
		}
	}
	return kernel;
}

void Image::RotateFourier2D(Image &rotated_image, AnglesAndShifts &rotation_angle, float resolution_limit, bool use_nearest_neighbor)
{
	MyDebugAssertTrue(rotated_image.logical_z_dimension == 1, "Error: attempting to rotate into 3D image");
	MyDebugAssertTrue(logical_z_dimension == 1, "Error: attempting to rotate from 3D image");
	MyDebugAssertTrue(rotated_image.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(! is_in_real_space, "Image is in real space");
	MyDebugAssertTrue(IsSquare(), "Image to rotate is not square");
	MyDebugAssertTrue(rotated_image.logical_x_dimension == logical_x_dimension && rotated_image.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(! object_is_centred_in_box, "Image volume quadrants not swapped");

	int i;
	int j;

	long pixel_counter;
	long pixel_counter2;

	float x_coordinate_2d;
	float y_coordinate_2d;

	float x_coordinate_3d;
	float y_coordinate_3d;

	float resolution_limit_sq = powf(resolution_limit * logical_x_dimension,2);
	float y_coord_sq;
	float padding_factor = logical_x_dimension / rotated_image.logical_x_dimension;

	rotated_image.object_is_centred_in_box = false;
	rotated_image.is_in_real_space = false;
//	image_to_extract.SetToConstant(0.0);

	if (use_nearest_neighbor)
	{
		for (j = rotated_image.logical_lower_bound_complex_y; j <= rotated_image.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j * padding_factor;
			y_coord_sq = powf(y_coordinate_2d,2);
			for (i = 1; i <= rotated_image.logical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i * padding_factor;
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				if (powf(x_coordinate_2d,2) + y_coord_sq <= resolution_limit_sq)
				{
					rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
					rotated_image.complex_values[pixel_counter] = ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
				}
				else
				{
					rotated_image.complex_values[pixel_counter] = 0.0;
				}
			}
		}
	// Now deal with special case of i = 0
		for (j = 1; j <= rotated_image.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j * padding_factor;
			x_coordinate_2d = 0;
			if (powf(y_coordinate_2d,2) <= resolution_limit_sq)
			{
				rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
				rotated_image.complex_values[pixel_counter] = ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
				pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
				rotated_image.complex_values[pixel_counter2] = conjf(rotated_image.complex_values[pixel_counter]);
			}
			else
			{
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
				rotated_image.complex_values[pixel_counter] = 0.0;
				pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
				rotated_image.complex_values[pixel_counter2] = 0.0;
			}
		}
	// Deal with pixel at edge if image dimensions are even
		if (-rotated_image.logical_lower_bound_complex_y != rotated_image.logical_upper_bound_complex_y)
		{
			y_coordinate_2d = rotated_image.logical_lower_bound_complex_y * padding_factor;
			x_coordinate_2d = 0;
			rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
			pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,rotated_image.logical_lower_bound_complex_y,0);
			rotated_image.complex_values[pixel_counter] = ReturnNearestFourier2D(x_coordinate_3d, y_coordinate_3d);
		}
	}
	else
	{
		for (j = rotated_image.logical_lower_bound_complex_y; j <= rotated_image.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j * padding_factor;
			y_coord_sq = powf(y_coordinate_2d,2);
			for (i = 1; i <= rotated_image.logical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i * padding_factor;
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				if (powf(x_coordinate_2d,2) + y_coord_sq <= resolution_limit_sq)
				{
					rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
					rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
				}
				else
				{
					rotated_image.complex_values[pixel_counter] = 0.0;
				}
			}
		}
	// Now deal with special case of i = 0
		for (j = 1; j <= rotated_image.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j * padding_factor;
			x_coordinate_2d = 0;
			if (powf(y_coordinate_2d,2) <= resolution_limit_sq)
			{
				rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
				rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
				pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
				rotated_image.complex_values[pixel_counter2] = conjf(rotated_image.complex_values[pixel_counter]);
			}
			else
			{
				pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
				rotated_image.complex_values[pixel_counter] = 0.0;
				pixel_counter2 = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
				rotated_image.complex_values[pixel_counter2] = 0.0;
			}
		}
	// Deal with pixel at edge if image dimensions are even
		if (-rotated_image.logical_lower_bound_complex_y != rotated_image.logical_upper_bound_complex_y)
		{
			y_coordinate_2d = rotated_image.logical_lower_bound_complex_y * padding_factor;
			x_coordinate_2d = 0;
			rotation_angle.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
			pixel_counter = rotated_image.ReturnFourier1DAddressFromLogicalCoord(0,rotated_image.logical_lower_bound_complex_y,0);
			rotated_image.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier2D(x_coordinate_3d, y_coordinate_3d);
		}
	}

// Set origin to zero to generate a projection with average set to zero
	rotated_image.complex_values[0] = 0.0;

	rotated_image.is_in_real_space = false;
}

void Image::ExtractSlice(Image &image_to_extract, AnglesAndShifts &angles_and_shifts_of_image, float resolution_limit)
{
	MyDebugAssertTrue(image_to_extract.logical_x_dimension == logical_x_dimension && image_to_extract.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(image_to_extract.logical_z_dimension == 1, "Error: attempting to extract 3D image from 3D reconstruction");
	MyDebugAssertTrue(image_to_extract.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(IsCubic(), "Image volume to project is not cubic");
	MyDebugAssertTrue(! object_is_centred_in_box, "Image volume quadrants not swapped");

	int i;
	int j;

	long pixel_counter;
	long pixel_counter2;

	float x_coordinate_2d;
	float y_coordinate_2d;
	float z_coordinate_2d = 0.0;

	float x_coordinate_3d;
	float y_coordinate_3d;
	float z_coordinate_3d;

	float resolution_limit_sq = powf(resolution_limit * logical_x_dimension,2);
	float y_coord_sq;

	image_to_extract.object_is_centred_in_box = false;
	image_to_extract.is_in_real_space = false;
//	image_to_extract.SetToConstant(0.0);

	for (j = image_to_extract.logical_lower_bound_complex_y; j <= image_to_extract.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		y_coord_sq = powf(y_coordinate_2d,2);
		for (i = 1; i <= image_to_extract.logical_upper_bound_complex_x; i++)
		{
			x_coordinate_2d = i;
			pixel_counter = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
			if (powf(x_coordinate_2d,2) + y_coord_sq <= resolution_limit_sq)
			{
				angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
				image_to_extract.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			}
			else
			{
				image_to_extract.complex_values[pixel_counter] = 0.0;
			}
		}
	}
// Now deal with special case of i = 0
	for (j = 1; j <= image_to_extract.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		x_coordinate_2d = 0;
		if (powf(y_coordinate_2d,2) <= resolution_limit_sq)
		{
			angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			image_to_extract.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter2 = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
			image_to_extract.complex_values[pixel_counter2] = conjf(image_to_extract.complex_values[pixel_counter]);
		}
		else
		{
			pixel_counter = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			image_to_extract.complex_values[pixel_counter] = 0.0;
			pixel_counter2 = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(0,-j,0);
			image_to_extract.complex_values[pixel_counter2] = 0.0;
		}
	}
// Deal with pixel at edge if image dimensions are even
	if (-image_to_extract.logical_lower_bound_complex_y != image_to_extract.logical_upper_bound_complex_y)
	{
		y_coordinate_2d = image_to_extract.logical_lower_bound_complex_y;
		x_coordinate_2d = 0;
		angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
		pixel_counter = image_to_extract.ReturnFourier1DAddressFromLogicalCoord(0,image_to_extract.logical_lower_bound_complex_y,0);
		image_to_extract.complex_values[pixel_counter] = ReturnLinearInterpolatedFourier(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
	}

// Set origin to zero to generate a projection with average set to zero
	image_to_extract.complex_values[0] = 0.0;

	image_to_extract.is_in_real_space = false;
}

fftwf_complex Image::ReturnNearestFourier2D(float &x, float &y)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_in_real_space, "Must be in Fourier space");

	int i_nearest;
	int j_nearest;
	int physical_y_address;

	if (x >= 0.0)
	{
		i_nearest = myroundint(x);
		if (i_nearest > logical_upper_bound_complex_x) return 0.0;

		j_nearest = myroundint(y);
		if (j_nearest < logical_lower_bound_complex_y) return 0.0;
		if (j_nearest > logical_upper_bound_complex_y) return 0.0;

		if (j_nearest >= 0)
		{
			physical_y_address = j_nearest;
		}
		else
		{
			physical_y_address = logical_y_dimension + j_nearest;
		}
		return complex_values[(physical_upper_bound_complex_x + 1) * physical_y_address + i_nearest];
	}
	else
	{
		i_nearest = myroundint(x);
		if (i_nearest < logical_lower_bound_complex_x) return 0.0;

		j_nearest = myroundint(y);
		if (j_nearest < logical_lower_bound_complex_y) return 0.0;
		if (j_nearest > logical_upper_bound_complex_y) return 0.0;

		if (j_nearest > 0)
		{
			physical_y_address = logical_y_dimension - j_nearest;
		}
		else
		{
			physical_y_address = -j_nearest;
		}
		return conjf(complex_values[(physical_upper_bound_complex_x + 1) * physical_y_address - i_nearest]);
	}
}

// Implementation of Frealign's ainterpo3ds
fftwf_complex Image::ReturnLinearInterpolatedFourier2D(float &x, float &y)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_in_real_space, "Must be in Fourier space");

	fftwf_complex sum = 0.0;
	int i;
	int j;
	int i_start;
	int i_end;
	int j_start;
	int j_end;
	int physical_y_address;
	int jj;

	float weight;
	float y_dist;

	if (x >= 0.0)
	{
		i_start = int(floorf(x));
		i_end = i_start + 1;
		if (i_end > logical_upper_bound_complex_x) return 0.0;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return 0.0;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return 0.0;

		for (j = j_start; j <= j_end; j++)
		{
			if (j >= 0)
			{
				physical_y_address = j;
			}
			else
			{
				physical_y_address = logical_y_dimension + j;
			}
			jj = (physical_upper_bound_complex_x + 1) * physical_y_address;
			y_dist = fabs(y - float(j));
			for (i = i_start; i <= i_end; i++)
			{
				weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist);
				sum = sum + complex_values[jj + i] * weight;
			}
		}
	}
	else
	{
		i_start = int(floorf(x));
		if (i_start < logical_lower_bound_complex_x) return 0.0;
		i_end = i_start + 1;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return 0.0;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return 0.0;

		for (j = j_start; j <= j_end; j++)
		{
			if (j > 0)
			{
				physical_y_address = logical_y_dimension - j;
			}
			else
			{
				physical_y_address = -j;
			}
			jj = (physical_upper_bound_complex_x + 1) * physical_y_address;
			y_dist = fabs(y - float(j));
			for (i = i_start; i <= i_end; i++)
			{
				weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist);
				sum = sum + conjf(complex_values[jj - i]) * weight;
			}
		}
	}
	return sum;
}

fftwf_complex Image::ReturnLinearInterpolatedFourier(float &x, float &y, float &z)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(! is_in_real_space, "Must be in Fourier space");

	fftwf_complex sum = 0.0;
	int i;
	int j;
	int k;
	int i_start;
	int i_end;
	int j_start;
	int j_end;
	int k_start;
	int k_end;
	int physical_y_address;
	int physical_z_address;
	int jj;
	int kk;

	float weight;
	float y_dist;
	float z_dist;

	if (x >= 0.0)
	{
		i_start = int(floorf(x));
		i_end = i_start + 1;
		if (i_end > logical_upper_bound_complex_x) return 0.0;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return 0.0;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return 0.0;

		k_start = int(floorf(z));
		if (k_start < logical_lower_bound_complex_z) return 0.0;
		k_end = k_start + 1;
		if (k_end > logical_upper_bound_complex_z) return 0.0;

		for (k = k_start; k <= k_end; k++)
		{
			if (k >= 0)
			{
				physical_z_address = k;
			}
			else
			{
				physical_z_address = logical_z_dimension + k;
			}
			kk = (physical_upper_bound_complex_y + 1) * physical_z_address;
			z_dist = fabs(z - float(k));
			for (j = j_start; j <= j_end; j++)
			{
				if (j >= 0)
				{
					physical_y_address = j;
				}
				else
				{
					physical_y_address = logical_y_dimension + j;
				}
				jj = (physical_upper_bound_complex_x + 1) * (kk + physical_y_address);
				y_dist = fabs(y - float(j));
				for (i = i_start; i <= i_end; i++)
				{
					weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist) * (1.0 - z_dist);
					sum = sum + complex_values[jj + i] * weight;
				}
			}
		}
	}
	else
	{
		i_start = int(floorf(x));
		if (i_start < logical_lower_bound_complex_x) return 0.0;
		i_end = i_start + 1;

		j_start = int(floorf(y));
		if (j_start < logical_lower_bound_complex_y) return 0.0;
		j_end = j_start + 1;
		if (j_end > logical_upper_bound_complex_y) return 0.0;

		k_start = int(floorf(z));
		if (k_start < logical_lower_bound_complex_z) return 0.0;
		k_end = k_start + 1;
		if (k_end > logical_upper_bound_complex_z) return 0.0;

		for (k = k_start; k <= k_end; k++)
		{
			if (k > 0)
			{
				physical_z_address = logical_z_dimension - k;
			}
			else
			{
				physical_z_address = -k;
			}
			kk = (physical_upper_bound_complex_y + 1) * physical_z_address;
			z_dist = fabs(z - float(k));
			for (j = j_start; j <= j_end; j++)
			{
				if (j > 0)
				{
					physical_y_address = logical_y_dimension - j;
				}
				else
				{
					physical_y_address = -j;
				}
				jj = (physical_upper_bound_complex_x + 1) * (kk + physical_y_address);
				y_dist = fabs(y - float(j));
				for (i = i_start; i <= i_end; i++)
				{
					weight = (1.0 - fabs(x - float(i))) * (1.0 - y_dist) * (1.0 - z_dist);
					sum = sum + conjf(complex_values[jj - i]) * weight;
				}
			}
		}
	}
	return sum;
}

void Image::AddByLinearInterpolationReal(float &wanted_physical_x_coordinate, float &wanted_physical_y_coordinate, float &wanted_physical_z_coordinate, float &wanted_value)
{
	int i;
	int j;
	int k;
	int int_x_coordinate;
	int int_y_coordinate;
	int int_z_coordinate;

	long physical_coord;

	float weight_x;
	float weight_y;
	float weight_z;

	MyDebugAssertTrue(is_in_real_space == true, "Error: attempting REAL insertion into COMPLEX image");
	MyDebugAssertTrue(wanted_physical_x_coordinate >= 0 && wanted_physical_x_coordinate <= logical_x_dimension, "Error: attempting insertion outside image X boundaries");
	MyDebugAssertTrue(wanted_physical_y_coordinate >= 0 && wanted_physical_y_coordinate <= logical_y_dimension, "Error: attempting insertion outside image Y boundaries");
	MyDebugAssertTrue(wanted_physical_z_coordinate >= 0 && wanted_physical_z_coordinate <= logical_z_dimension, "Error: attempting insertion outside image Z boundaries");

	int_x_coordinate = int(wanted_physical_x_coordinate);
	int_y_coordinate = int(wanted_physical_y_coordinate);
	int_z_coordinate = int(wanted_physical_z_coordinate);

	for (k = int_z_coordinate; k <= int_z_coordinate + 1; k++)
	{
		weight_z = (1.0 - fabs(wanted_physical_z_coordinate - k));
		for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++)
		{
			weight_y = (1.0 - fabs(wanted_physical_y_coordinate - j));
			for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++)
			{
				weight_x = (1.0 - fabs(wanted_physical_x_coordinate - i));
				physical_coord = ReturnReal1DAddressFromPhysicalCoord(i, j, k);
				real_values[physical_coord] = real_values[physical_coord] + wanted_value * weight_x * weight_y * weight_z;
			}
		}
	}
}

void Image::AddByLinearInterpolationFourier2D(float &wanted_logical_x_coordinate, float &wanted_logical_y_coordinate, fftwf_complex &wanted_value)
{
	int i;
	int j;
	int int_x_coordinate;
	int int_y_coordinate;

	long physical_coord;

	float weight_y;
	float weight;

	fftwf_complex conjugate;

	MyDebugAssertTrue(is_in_real_space != true, "Error: attempting COMPLEX insertion into REAL image");

	int_x_coordinate = int(floor(wanted_logical_x_coordinate));
	int_y_coordinate = int(floor(wanted_logical_y_coordinate));

	for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++)
	{
		weight_y = (1.0 - fabs(wanted_logical_y_coordinate - j));
		for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++)
		{
			if (i >= logical_lower_bound_complex_x && i <= logical_upper_bound_complex_x
			 && j >= logical_lower_bound_complex_y && j <= logical_upper_bound_complex_y)
			{
				weight = weight_y * (1.0 - fabs(wanted_logical_x_coordinate - i));
				physical_coord = ReturnFourier1DAddressFromLogicalCoord(i, j, 0);
				if (i < 0)
				{
					conjugate = conjf(wanted_value);
					complex_values[physical_coord] = complex_values[physical_coord] + conjugate * weight;
				}
				else
				if (i == 0 && j != logical_lower_bound_complex_y && j != 0)
				{
					complex_values[physical_coord] = complex_values[physical_coord] + wanted_value * weight;
					physical_coord = ReturnFourier1DAddressFromLogicalCoord(i, -j, 0);
					conjugate = conjf(wanted_value);
					complex_values[physical_coord] = complex_values[physical_coord] + conjugate * weight;
				}
				else
				{
					complex_values[physical_coord] = complex_values[physical_coord] + wanted_value * weight;
				}
			}
		}
	}
}

void Image::CalculateCTFImage(CTF &ctf_of_image)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated for CTF image");
	MyDebugAssertTrue(is_in_real_space == false, "CTF image not in Fourier space");

	int i;
	int j;

	long pixel_counter = 0;

	float x_coordinate_2d;
	float y_coordinate_2d;

	float y_coord_sq;

	float frequency_squared;
	float azimuth;

	for (j = 0; j <= physical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
		y_coord_sq = powf(y_coordinate_2d, 2);

		for (i = 0; i <= physical_upper_bound_complex_x; i++)
		{
			x_coordinate_2d = i * fourier_voxel_size_x;
			// Compute the azimuth
			if ( i == 0 && j == 0 )
			{
				azimuth = 0.0;
			}
			else
			{
				azimuth = atan2f(y_coordinate_2d,x_coordinate_2d);
			}
			// Compute the square of the frequency
			frequency_squared = powf(x_coordinate_2d, 2) + y_coord_sq;

			complex_values[pixel_counter] = ctf_of_image.Evaluate(frequency_squared,azimuth);
			pixel_counter++;
		}
	}
}

// Apply a cosine-edge mask. By default, pixels on the outside of the mask radius are flattened. If invert=true, the pixels near the center are flattened. This does not currently work when quadrants are swapped.
float Image::CosineRingMask(float wanted_inner_radius, float wanted_outer_radius, float wanted_mask_edge)
{
//	MyDebugAssertTrue(! is_in_real_space || object_is_centred_in_box, "Image in real space but not centered");
	MyDebugAssertTrue(wanted_mask_edge > 1, "Edge width too small");
	MyDebugAssertTrue(wanted_inner_radius <= wanted_outer_radius, "Inner radius larger than outer radius");

	int i;
	int j;
	int k;
	int ii;
	int jj;
	int kk;
	int outer_number_of_pixels;
	int inner_number_of_pixels;

	float x;
	float y;
	float z;

	long pixel_counter = 0;

	float distance_from_center;
	float distance_from_center_squared;
	float outer_mask_radius;
	float outer_mask_radius_squared;
	float outer_mask_radius_plus_edge;
	float outer_mask_radius_plus_edge_squared;
	float inner_mask_radius;
	float inner_mask_radius_squared;
	float inner_mask_radius_minus_edge;
	float inner_mask_radius_minus_edge_squared;
	float edge;
	double outer_pixel_sum;
	double inner_pixel_sum;

	float frequency;
	float frequency_squared;

	double mask_volume = 0.0;

	outer_mask_radius = wanted_outer_radius - wanted_mask_edge / 2;
	if (outer_mask_radius < 0.0) outer_mask_radius = 0.0;
	outer_mask_radius_plus_edge = outer_mask_radius + wanted_mask_edge;

	outer_mask_radius_squared = powf(outer_mask_radius, 2);
	outer_mask_radius_plus_edge_squared = powf(outer_mask_radius_plus_edge, 2);

	inner_mask_radius = wanted_inner_radius + wanted_mask_edge / 2;
	inner_mask_radius_minus_edge = inner_mask_radius - wanted_mask_edge;

	inner_mask_radius_squared = powf(inner_mask_radius, 2);
	inner_mask_radius_minus_edge_squared = powf(inner_mask_radius_minus_edge, 2);

	outer_pixel_sum = 0.0;
	outer_number_of_pixels = 0;
	inner_pixel_sum = 0.0;
	inner_number_of_pixels = 0;
	if (is_in_real_space && object_is_centred_in_box)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= outer_mask_radius_squared && distance_from_center_squared <= outer_mask_radius_plus_edge_squared)
					{
						outer_pixel_sum += real_values[pixel_counter];
						outer_number_of_pixels++;
					}
					if (distance_from_center_squared <= inner_mask_radius_squared && distance_from_center_squared >= inner_mask_radius_minus_edge_squared)
					{
						inner_pixel_sum += real_values[pixel_counter];
						inner_number_of_pixels++;
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		outer_pixel_sum /= outer_number_of_pixels;
		inner_pixel_sum /= inner_number_of_pixels;

		pixel_counter = 0.0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= outer_mask_radius_squared && distance_from_center_squared <= outer_mask_radius_plus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (distance_from_center - outer_mask_radius) / wanted_mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * outer_pixel_sum;
						mask_volume += powf(edge,2);
					}
					else
					if (distance_from_center_squared >= outer_mask_radius_plus_edge_squared)
					{
						real_values[pixel_counter] = outer_pixel_sum;
					}
					else
					if (distance_from_center_squared <= inner_mask_radius_squared && distance_from_center_squared >= inner_mask_radius_minus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (inner_mask_radius - distance_from_center) / wanted_mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * inner_pixel_sum;
						mask_volume += powf(edge,2);
					}
					else
					if (distance_from_center_squared <= inner_mask_radius_minus_edge_squared)
					{
						real_values[pixel_counter] = inner_pixel_sum;
					}
					else
					{
						mask_volume += 1.0;
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	if (is_in_real_space)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			kk = k;
			if (kk >= physical_address_of_box_center_z) kk -= logical_z_dimension;
			z = powf(kk, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				jj = j;
				if (jj >= physical_address_of_box_center_y) jj -= logical_y_dimension;
				y = powf(jj, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					ii = i;
					if (ii >= physical_address_of_box_center_x) ii -= logical_x_dimension;
					x = powf(ii, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= outer_mask_radius_squared && distance_from_center_squared <= outer_mask_radius_plus_edge_squared)
					{
						outer_pixel_sum += real_values[pixel_counter];
						outer_number_of_pixels++;
					}
					if (distance_from_center_squared <= inner_mask_radius_squared && distance_from_center_squared >= inner_mask_radius_minus_edge_squared)
					{
						inner_pixel_sum += real_values[pixel_counter];
						inner_number_of_pixels++;
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		outer_pixel_sum /= outer_number_of_pixels;
		inner_pixel_sum /= inner_number_of_pixels;

		pixel_counter = 0.0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			kk = k;
			if (kk >= physical_address_of_box_center_z) kk -= logical_z_dimension;
			z = powf(kk, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				jj = j;
				if (jj >= physical_address_of_box_center_y) jj -= logical_y_dimension;
				y = powf(jj, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					ii = i;
					if (ii >= physical_address_of_box_center_x) ii -= logical_x_dimension;
					x = powf(ii, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= outer_mask_radius_squared && distance_from_center_squared <= outer_mask_radius_plus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (distance_from_center - outer_mask_radius) / wanted_mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * outer_pixel_sum;
						mask_volume += powf(edge,2);
					}
					else
					if (distance_from_center_squared >= outer_mask_radius_plus_edge_squared)
					{
						real_values[pixel_counter] = outer_pixel_sum;
					}
					else
					if (distance_from_center_squared <= inner_mask_radius_squared && distance_from_center_squared >= inner_mask_radius_minus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (inner_mask_radius - distance_from_center) / wanted_mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * inner_pixel_sum;
						mask_volume += powf(edge,2);
					}
					else
					if (distance_from_center_squared <= inner_mask_radius_minus_edge_squared)
					{
						real_values[pixel_counter] = inner_pixel_sum;
					}
					else
					{
						mask_volume += 1.0;
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	{
		for (k = 0; k <= physical_upper_bound_complex_z; k++)
		{
			z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

			for (j = 0; j <= physical_upper_bound_complex_y; j++)
			{
				y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

				for (i = 0; i <= physical_upper_bound_complex_x; i++)
				{
					x = powf(i * fourier_voxel_size_x, 2);

					// compute squared radius, in units of reciprocal pixels

					frequency_squared = x + y + z;

					if (frequency_squared >= outer_mask_radius_squared && frequency_squared <= outer_mask_radius_plus_edge_squared)
					{
						frequency = sqrtf(frequency_squared);
						edge = (1.0 + cosf(PI * (frequency - outer_mask_radius) / wanted_mask_edge)) / 2.0;
						complex_values[pixel_counter] *= edge;
					}
					if (frequency_squared <= inner_mask_radius_squared && frequency_squared >= inner_mask_radius_minus_edge_squared)
					{
						frequency = sqrtf(frequency_squared);
						edge = (1.0 + cosf(PI * (outer_mask_radius - frequency) / wanted_mask_edge)) / 2.0;
						complex_values[pixel_counter] *= edge;
					}
					if (frequency_squared >= outer_mask_radius_plus_edge_squared) complex_values[pixel_counter] = 0.0;
					if (frequency_squared <= inner_mask_radius_minus_edge_squared) complex_values[pixel_counter] = 0.0;

					pixel_counter++;
				}
			}

		}

	}

	return float(mask_volume);
}

void Image::CircleMask(float wanted_mask_radius, bool invert)
{
	MyDebugAssertTrue(is_in_real_space,"Image not in real space");
	MyDebugAssertTrue(object_is_centred_in_box,"Object not centered in box");

	long pixel_counter;
	int i,j,k;
	float x,y,z;
	float distance_from_center_squared;
	const float wanted_mask_radius_squared = powf(wanted_mask_radius,2);
	double average_value = 0.0;
	long number_of_pixels = 0;

	pixel_counter = 0;
	for (k = 0; k < logical_z_dimension; k++)
	{
		z = powf(k - physical_address_of_box_center_z, 2);

		for (j = 0; j < logical_y_dimension; j++)
		{
			y = powf(j - physical_address_of_box_center_y, 2);

			for (i = 0; i < logical_x_dimension; i++)
			{
				x = powf(i - physical_address_of_box_center_x, 2);

				distance_from_center_squared = x + y + z;

				if (abs(distance_from_center_squared-wanted_mask_radius_squared) <= 2.0)
				{
					number_of_pixels++;
					average_value += real_values[pixel_counter];
				}

				pixel_counter++;

			}
			pixel_counter += padding_jump_value;
		}
	}

	// Now we know what value to mask with
	average_value /= float(number_of_pixels);

	// Let's mask
	pixel_counter = 0;
	for (k = 0; k < logical_z_dimension; k++)
	{
		z = powf(k - physical_address_of_box_center_z, 2);

		for (j = 0; j < logical_y_dimension; j++)
		{
			y = powf(j - physical_address_of_box_center_y, 2);

			for (i = 0; i < logical_x_dimension; i++)
			{
				x = powf(i - physical_address_of_box_center_x, 2);

				distance_from_center_squared = x + y + z;

				if (invert)
				{
					if ( distance_from_center_squared <= wanted_mask_radius_squared)
					{
						real_values[pixel_counter] = average_value;
					}
				}
				else
				{
					if ( distance_from_center_squared > wanted_mask_radius_squared)
					{
						real_values[pixel_counter] = average_value;
					}
				}

				pixel_counter++;

			}
			pixel_counter += padding_jump_value;
		}
	}


}


float Image::CosineMask(float wanted_mask_radius, float wanted_mask_edge, bool invert)
{
//	MyDebugAssertTrue(! is_in_real_space || object_is_centred_in_box, "Image in real space but not centered");
	MyDebugAssertTrue(wanted_mask_edge > 1, "Edge width too small");

	int i;
	int j;
	int k;
	int ii;
	int jj;
	int kk;
	int number_of_pixels;

	float x;
	float y;
	float z;

	long pixel_counter = 0;

	float distance_from_center;
	float mask_radius_plus_edge;
	float distance_from_center_squared;
	float mask_radius;
	float mask_radius_squared;
	float mask_radius_plus_edge_squared;
	float edge;
	double pixel_sum;

	float frequency;
	float frequency_squared;

	double mask_volume = 0.0;

	mask_radius = wanted_mask_radius - wanted_mask_edge / 2;
	if (mask_radius < 0.0) mask_radius = 0.0;
	mask_radius_plus_edge = mask_radius + wanted_mask_edge;

	mask_radius_squared = powf(mask_radius, 2);
	mask_radius_plus_edge_squared = powf(mask_radius_plus_edge, 2);

	pixel_sum = 0.0;
	number_of_pixels = 0;
	if (is_in_real_space && object_is_centred_in_box)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						pixel_sum += real_values[pixel_counter];
						number_of_pixels++;
					}
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		pixel_sum /= number_of_pixels;

		pixel_counter = 0.0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (distance_from_center - mask_radius) / wanted_mask_edge)) / 2.0;
						if (invert)
						{
							real_values[pixel_counter] = real_values[pixel_counter] * (1.0 - edge) + edge * pixel_sum;
							mask_volume += powf(1.0 - edge,2);
						}
						else
						{
							real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * pixel_sum;
							mask_volume += powf(edge,2);
						}
					}
					else
					if (invert)
					{
						if (distance_from_center_squared <= mask_radius)
						{
							real_values[pixel_counter] = pixel_sum;
						}
						else
						{
							mask_volume += 1.0;
						}
					}
					else
					{
						if (distance_from_center_squared >= mask_radius_plus_edge_squared)
						{
							real_values[pixel_counter] = pixel_sum;
						}
						else
						{
							mask_volume += 1.0;
						}
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	if (is_in_real_space)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			kk = k;
			if (kk >= physical_address_of_box_center_z) kk -= logical_z_dimension;
			z = powf(kk, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				jj = j;
				if (jj >= physical_address_of_box_center_y) jj -= logical_y_dimension;
				y = powf(jj, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					ii = i;
					if (ii >= physical_address_of_box_center_x) ii -= logical_x_dimension;
					x = powf(ii, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						pixel_sum += real_values[pixel_counter];
						number_of_pixels++;
					}
					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
		pixel_sum /= number_of_pixels;

		pixel_counter = 0.0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			kk = k;
			if (kk >= physical_address_of_box_center_z) kk -= logical_z_dimension;
			z = powf(kk, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				jj = j;
				if (jj >= physical_address_of_box_center_y) jj -= logical_y_dimension;
				y = powf(jj, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					ii = i;
					if (ii >= physical_address_of_box_center_x) ii -= logical_x_dimension;
					x = powf(ii, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						distance_from_center = sqrtf(distance_from_center_squared);
						edge = (1.0 + cosf(PI * (distance_from_center - mask_radius) / wanted_mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * pixel_sum;
						mask_volume += powf(edge,2);
					}
					else
						if (distance_from_center_squared >= mask_radius_plus_edge_squared) real_values[pixel_counter] = pixel_sum;
					else
					{
						mask_volume += 1.0;
					}

					pixel_counter++;
				}
				pixel_counter += padding_jump_value;
			}
		}
	}
	else
	{
		for (k = 0; k <= physical_upper_bound_complex_z; k++)
		{
			z = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

			for (j = 0; j <= physical_upper_bound_complex_y; j++)
			{
				y = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

				for (i = 0; i <= physical_upper_bound_complex_x; i++)
				{
					x = powf(i * fourier_voxel_size_x, 2);

					// compute squared radius, in units of reciprocal pixels

					frequency_squared = x + y + z;

					if (frequency_squared >= mask_radius_squared && frequency_squared <= mask_radius_plus_edge_squared)
					{
						frequency = sqrtf(frequency_squared);
						edge = (1.0 + cosf(PI * (frequency - mask_radius) / wanted_mask_edge)) / 2.0;
						if (invert)
						{
							complex_values[pixel_counter] *= (1.0 - edge);
						}
						else
						{
							complex_values[pixel_counter] *= edge;
						}
					}
					if (invert)
					{
						if (frequency_squared <= mask_radius) complex_values[pixel_counter] = 0.0;
					}
					else
					{
						if (frequency_squared >= mask_radius_plus_edge_squared) complex_values[pixel_counter] = 0.0;
					}

					pixel_counter++;
				}
			}

		}

	}
	
	return float(mask_volume);
}

Image & Image::operator = (const Image &other_image)
{
	*this = &other_image;
	return *this;
}


Image & Image::operator = (const Image *other_image)
{
   // Check for self assignment
   if(this != other_image)
   {
		MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

		if (is_in_memory == true)
		{

			if (logical_x_dimension != other_image->logical_x_dimension || logical_y_dimension != other_image->logical_y_dimension || logical_z_dimension != other_image->logical_z_dimension)
			{
				Deallocate();
				Allocate(other_image->logical_x_dimension, other_image->logical_y_dimension, other_image->logical_z_dimension, other_image->is_in_real_space);
			}
		}
		else
		{
			Allocate(other_image->logical_x_dimension, other_image->logical_y_dimension, other_image->logical_z_dimension, other_image->is_in_real_space);
		}

		// by here the memory allocation should be ok..

		is_in_real_space = other_image->is_in_real_space;
		object_is_centred_in_box = other_image->object_is_centred_in_box;

		for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
		{
			real_values[pixel_counter] = other_image->real_values[pixel_counter];
		}
   }

   return *this;
}



//!>  \brief  Deallocate all the memory.  The idea is to keep this safe in the case that something isn't
//    allocated, so it can always be safely called.  I.e. it should check whatever it is deallocating is
//    in fact allocated.
//


void Image::Deallocate()
{
	if (is_in_memory == true)
	{
		fftwf_free(real_values);
		is_in_memory = false;
	}

	if (planned == true)
	{
		fftwf_destroy_plan(plan_fwd);
		fftwf_destroy_plan(plan_bwd);
		planned = false;
	}

}

//!>  \brief  Allocate memory for the Image object.
//
//  If the object is already allocated with correct dimensions, nothing happens. Otherwise, object is deallocated first.

void Image::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space)
{

	// check to see if we need to do anything?

	if (is_in_memory == true)
	{
		is_in_real_space = should_be_in_real_space;

		if (wanted_x_size == logical_x_dimension && wanted_y_size == logical_y_dimension && wanted_z_size == logical_z_dimension)
		{
			// everything is already done..

			is_in_real_space = should_be_in_real_space;
//			wxPrintf("returning\n");

			return;
		}
		else
		Deallocate();
	}

	// if we got here we need to do allocation..

	SetLogicalDimensions(wanted_x_size, wanted_y_size, wanted_z_size);
	is_in_real_space = should_be_in_real_space;

	// first_x_dimension
	if (IsEven(wanted_x_size) == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	complex_values = (fftwf_complex*) real_values;  // Set the complex_values to point at the newly allocated real values;

	is_in_memory = true;

	// Update addresses etc..

    UpdateLoopingAndAddressing();

    // Prepare the plans for FFTW

    if (planned == false)
    {
    	if (logical_z_dimension > 1)
    	{
    		plan_fwd = fftwf_plan_dft_r2c_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension, real_values, complex_values, FFTW_ESTIMATE);
    		plan_bwd = fftwf_plan_dft_c2r_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension, complex_values, real_values, FFTW_ESTIMATE);
    	}
    	else
    	{
    		plan_fwd = fftwf_plan_dft_r2c_2d(logical_y_dimension, logical_x_dimension, real_values, complex_values, FFTW_ESTIMATE);
    	    plan_bwd = fftwf_plan_dft_c2r_2d(logical_y_dimension, logical_x_dimension, complex_values, real_values, FFTW_ESTIMATE);

    	}

    	planned = true;
    }

    // set the loop junk value..

	if (IsEven(logical_x_dimension) == true) padding_jump_value = 2;
	else padding_jump_value = 1;
}

//!>  \brief  Allocate memory for the Image object.
//
//  Overloaded version of allocate to cover the supplying just 2 dimensions along with the should_be_in_real_space bool.

void Image::Allocate(int wanted_x_size, int wanted_y_size, bool should_be_in_real_space)
{
	Allocate(wanted_x_size, wanted_y_size, 1, should_be_in_real_space);
}

//!>  \brief  Change the logical dimensions of an image and update all related values

void Image::SetLogicalDimensions(int wanted_x_size, int wanted_y_size, int wanted_z_size)
{
	logical_x_dimension = wanted_x_size;
	logical_y_dimension = wanted_y_size;
	logical_z_dimension = wanted_z_size;
}

//!>  \brief  Update all properties related to looping & addressing in real & Fourier space, given the current logical dimensions.

void Image::UpdateLoopingAndAddressing()
{

	physical_upper_bound_complex_x = logical_x_dimension / 2;
	physical_upper_bound_complex_y = logical_y_dimension - 1;
	physical_upper_bound_complex_z = logical_z_dimension - 1;

	UpdatePhysicalAddressOfBoxCenter();

	//physical_index_of_first_negative_frequency_x = logical_x_dimension / 2 + 1;
	if (IsEven(logical_y_dimension) == true)
	{
		physical_index_of_first_negative_frequency_y = logical_y_dimension / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency_y = logical_y_dimension / 2 + 1;
	}

	if (IsEven(logical_z_dimension) == true)
	{
		physical_index_of_first_negative_frequency_z = logical_z_dimension / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency_z = logical_z_dimension / 2 + 1;
	}


    // Update the Fourier voxel size

	fourier_voxel_size_x = 1.0 / double(logical_x_dimension);
	fourier_voxel_size_y = 1.0 / double(logical_y_dimension);
	fourier_voxel_size_z = 1.0 / double(logical_z_dimension);

	// Logical bounds
	if (IsEven(logical_x_dimension) == true)
	{
		logical_lower_bound_complex_x = -logical_x_dimension / 2;
		logical_upper_bound_complex_x =  logical_x_dimension / 2;
	    logical_lower_bound_real_x    = -logical_x_dimension / 2;
	    logical_upper_bound_real_x    =  logical_x_dimension / 2 - 1;
	}
	else
	{
		logical_lower_bound_complex_x = -(logical_x_dimension-1) / 2;
		logical_upper_bound_complex_x =  (logical_x_dimension-1) / 2;
		logical_lower_bound_real_x    = -(logical_x_dimension-1) / 2;
		logical_upper_bound_real_x    =  (logical_x_dimension-1) / 2;
	}


	if (IsEven(logical_y_dimension) == true)
	{
	    logical_lower_bound_complex_y = -logical_y_dimension / 2;
	    logical_upper_bound_complex_y =  logical_y_dimension / 2 - 1;
	    logical_lower_bound_real_y    = -logical_y_dimension / 2;
	    logical_upper_bound_real_y    =  logical_y_dimension / 2 - 1;
	}
	else
	{
	    logical_lower_bound_complex_y = -(logical_y_dimension-1) / 2;
	    logical_upper_bound_complex_y =  (logical_y_dimension-1) / 2;
	    logical_lower_bound_real_y    = -(logical_y_dimension-1) / 2;
	    logical_upper_bound_real_y    =  (logical_y_dimension-1) / 2;
	}

	if (IsEven(logical_z_dimension) == true)
	{
		logical_lower_bound_complex_z = -logical_z_dimension / 2;
		logical_upper_bound_complex_z =  logical_z_dimension / 2 - 1;
		logical_lower_bound_real_z    = -logical_z_dimension / 2;
		logical_upper_bound_real_z    =  logical_z_dimension / 2 - 1;

	}
	else
	{
		logical_lower_bound_complex_z = -(logical_z_dimension - 1) / 2;
		logical_upper_bound_complex_z =  (logical_z_dimension - 1) / 2;
		logical_lower_bound_real_z    = -(logical_z_dimension - 1) / 2;
		logical_upper_bound_real_z    =  (logical_z_dimension - 1) / 2;
	}
}

//!>  \brief  Returns the physical address of the image origin

void Image::UpdatePhysicalAddressOfBoxCenter()
{
	/*
    if (IsEven(logical_x_dimension)) physical_address_of_box_center_x = logical_x_dimension / 2;
    else physical_address_of_box_center_x = (logical_x_dimension - 1) / 2;

    if (IsEven(logical_y_dimension)) physical_address_of_box_center_y = logical_y_dimension / 2;
    else physical_address_of_box_center_y = (logical_y_dimension - 1) / 2;

    if (IsEven(logical_z_dimension)) physical_address_of_box_center_z = logical_z_dimension / 2;
    else physical_address_of_box_center_z = (logical_z_dimension - 1) / 2;
*/
	physical_address_of_box_center_x = logical_x_dimension / 2;
	physical_address_of_box_center_y = logical_y_dimension / 2;
	physical_address_of_box_center_z = logical_z_dimension / 2;
}


// Work out whether a given Fourier component has a Hermitian mate which is also described explicitely by the FFTW
bool Image::FourierComponentHasExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z)
{
	bool explicit_mate;

	explicit_mate = physical_index_x == 0 && ! ( physical_index_y == 0 && physical_index_x == 0);

	// We assume that the Y dimension is the non-flat one
	if (IsEven(logical_y_dimension))
	{
		explicit_mate = explicit_mate && physical_index_y != physical_index_of_first_negative_frequency_y-1;
	}

	if (logical_z_dimension > 1)
	{
		if (IsEven(logical_z_dimension))
		{
			explicit_mate = explicit_mate && physical_index_z != physical_index_of_first_negative_frequency_z - 1;
		}
	}

	return explicit_mate;
}


//!> \brief   Apply a forward FT to the Image object. The FT is scaled.
//   The DC component is at (self%DIM(1)/2+1,self%DIM(2)/2+1,self%DIM(3)/2+1) (assuming even dimensions) or at (1,1,1) by default.
//
//
//   For details on FFTW, see http://www.fftw.org/
//   A helpful page for understanding the output format: http://www.dsprelated.com/showmessage/102675/1.php
//   A helpful page to learn about vectorization and FFTW benchmarking: http://www.dsprelated.com/showmessage/76779/1.php
//   \todo   Check this: http://objectmix.com/fortran/371439-ifort-openmp-fftw-problem.html

void Image::ForwardFFT(bool should_scale)
{

	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Image already in Fourier space");
	MyDebugAssertTrue(planned, "FFT's not planned");

	fftwf_execute_dft_r2c(plan_fwd, real_values, complex_values);

	if (should_scale == true)
	{
		DivideByConstant(float(logical_x_dimension * logical_y_dimension * logical_z_dimension));
	}

	// Set the image type

	is_in_real_space = false;
}

void Image::BackwardFFT()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertFalse(is_in_real_space, "Image already in real space");

	fftwf_execute_dft_c2r(plan_bwd, complex_values, real_values);

    // Set the image type

    is_in_real_space = true;
}

//!> \brief Divide all voxels by a constant value (this is actually done as a multiplication by the inverse)

void Image::DivideByConstant(float constant_to_divide_by)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	float inverse = 1. / constant_to_divide_by;
	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] *= inverse;
	}
}

//!> \brief Multiply all voxels by a constant value

//inline
void Image::MultiplyByConstant(float constant_to_multiply_by)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] *= constant_to_multiply_by;
	}
}

bool Image::IsConstant()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		if (real_values[pixel_counter] != real_values[0]) return false;
	}
	return true;
}

//!> \brief Read a set of slices from disk (FFTW padding is done automatically)

void Image::ReadSlices(MRCFile *input_file, long start_slice, long end_slice)
{

	MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
	MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
	MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(), "End slice is greater than number of slices in the file!");
	MyDebugAssertTrue(input_file->my_file.is_open(), "MRCFile not open!");


	// check the allocations..

	int number_of_slices = (end_slice - start_slice) + 1;

	if (logical_x_dimension != input_file->ReturnXSize() || logical_y_dimension != input_file->ReturnYSize() || logical_z_dimension != number_of_slices || is_in_memory == false)
	{
		Deallocate();
		Allocate(input_file->ReturnXSize(), input_file->ReturnYSize(), number_of_slices);

	}

	// We should be set up - so read in the correct values.
	// AT THE MOMENT, WE CAN ONLY READ REAL SPACE IMAGES, SO OVERRIDE THIS!!!!!

	is_in_real_space = true;
	object_is_centred_in_box = true;

	input_file->ReadSlicesFromDisk(start_slice, end_slice, real_values);

	// we need to respace this to take into account the FFTW padding..

	AddFFTWPadding();

}

//!> \brief Read a set of slices from disk (FFTW padding is done automatically)

void Image::ReadSlices(DMFile *input_file, long start_slice, long end_slice)
{

	MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
	MyDebugAssertTrue(start_slice > 0, "Start slice is less than 0, the first slice is 1!");
	MyDebugAssertTrue(end_slice <= input_file->ReturnNumberOfSlices(), "End slice is greater than number of slices in the file!");
	MyDebugAssertTrue(start_slice == end_slice, "Can only read one slice at a time from DM files. Sorry.")


	// check the allocations..

	int number_of_slices = (end_slice - start_slice) + 1;

	if (logical_x_dimension != input_file->sizeX() || logical_y_dimension != input_file->sizeY() || logical_z_dimension != number_of_slices || is_in_memory == false)
	{
		Deallocate();
		Allocate(input_file->sizeX(), input_file->sizeY(), number_of_slices);

	}

	// We should be set up - so read in the correct values.
	// AT THE MOMENT, WE CAN ONLY READ REAL SPACE IMAGES, SO OVERRIDE THIS!!!!!

	is_in_real_space = true;
	object_is_centred_in_box = true;

	input_file->ReadSliceFromDisk(start_slice - 1, real_values); // DM indexes slices starting at 0

	// we need to respace this to take into account the FFTW padding..

	AddFFTWPadding();

}

//!> \brief Write a set of slices from disk (FFTW padding is done automatically)

void Image::WriteSlices(MRCFile *input_file, long start_slice, long end_slice)
{
	MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
	MyDebugAssertTrue(input_file->my_file.is_open(), "MRCFile not open!");

	// THIS PROBABLY NEEDS ATTENTION..

	if (start_slice == 1) // if the start slice is one, we set the header to match the image
	{
		input_file->SetXSize(logical_x_dimension);
		input_file->SetYSize(logical_y_dimension);

		if (end_slice > input_file->ReturnNumberOfSlices())
		{
			input_file->SetNumberOfSlices(end_slice);
		}

		//input_file->WriteHeader();
		input_file->rewrite_header_on_close = true;
	}
	else // if the last slice is bigger than the current max number of slices, increase the max number of slices
	{
		if (end_slice > input_file->ReturnNumberOfSlices())
		{
			input_file->SetNumberOfSlices(end_slice);
		}

		input_file->rewrite_header_on_close = true;
	}


	MyDebugAssertTrue(logical_x_dimension == input_file->ReturnXSize() || logical_y_dimension == input_file->ReturnYSize(), "Image dimensions and file dimensions differ!");

	// if the image is complex.. make a temp image and transform it..

	int number_of_slices = (end_slice - start_slice) + 1;

	if (is_in_real_space == false)
	{
		Image temp_image;
		temp_image.CopyFrom(this);
		temp_image.BackwardFFT();
		temp_image.RemoveFFTWPadding();
		input_file->WriteSlicesToDisk(start_slice, end_slice, temp_image.real_values);

	}
	else // real space
	{
		RemoveFFTWPadding();
		input_file->WriteSlicesToDisk(start_slice, end_slice, real_values);
		AddFFTWPadding(); // to go back
	}
}

void Image::QuickAndDirtyWriteSlices(std::string filename, long first_slice_to_write, long last_slice_to_write)
{
	MyDebugAssertTrue(first_slice_to_write >0, "Slice is less than 1, first slice is 1");
	MRCFile output_file(filename, false);
	WriteSlices(&output_file,first_slice_to_write,last_slice_to_write);
}


void Image::QuickAndDirtyWriteSlice(std::string filename, long slice_to_write)
{
	MyDebugAssertTrue(slice_to_write >0, "Slice is less than 1, first slice is 1");
	MRCFile output_file(filename, false);
	WriteSlice(&output_file, slice_to_write);
}

void Image::QuickAndDirtyReadSlice(std::string filename, long slice_to_read)
{
	wxFileName wx_filename = wxString(filename);
	wxString extension = wx_filename.GetExt();
	//wxPrintf("(QuickAndDirtyReadSlice): found the following extension: %s",extension);
	if (extension.Find("dm") == 0)
	{
		DMFile input_file(filename);
		ReadSlices(&input_file,slice_to_read,slice_to_read);
	}
	else
	{
		MRCFile input_file(filename, false);

		MyDebugAssertTrue(slice_to_read <= input_file.ReturnNumberOfSlices(), "End slices is greater than number of slices in the file!");
		MyDebugAssertTrue(slice_to_read >0, "Slice is less than 1, first slice is 1");

		ReadSlice(&input_file, slice_to_read);
	}
}

//!> \brief Take a contiguous set of values, and add the FFTW padding.

void Image::AddFFTWPadding()
{
	MyDebugAssertTrue(is_in_memory, "Image not allocated!");

	int x,y,z;

	long current_write_position = real_memory_allocated - (1 + padding_jump_value);
	long current_read_position = (logical_x_dimension * logical_y_dimension * logical_z_dimension) - 1;

	for (z = 0; z < logical_z_dimension; z++)
	{
		for (y = 0; y < logical_y_dimension; y++)
		{
			for (x = 0; x < logical_x_dimension; x++)
			{
				real_values[current_write_position] = real_values[current_read_position];
				current_write_position--;
				current_read_position--;
			}

			current_write_position -= padding_jump_value;
		}
	}
}

//!> \brief Take a set of FFTW padded values, and remove the padding.

void Image::RemoveFFTWPadding()
{
	MyDebugAssertTrue(is_in_memory, "Image not allocated!");

	int x,y,z;

	long current_write_position = 0;
	long current_read_position = 0;

	for (z = 0; z < logical_z_dimension; z++)
	{
		for (y = 0; y < logical_y_dimension; y++)
		{
			for (x = 0; x < logical_x_dimension; x++)
			{
				real_values[current_write_position] = real_values[current_read_position];
				current_write_position++;
				current_read_position++;
			}

			current_read_position +=padding_jump_value;
		}
	}
}

void Image::SetToConstant(float wanted_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] = wanted_value;
	}
}

void Image::AddConstant(float wanted_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] += wanted_value;
	}
}

void Image::SetMaximumValue(float new_maximum_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	for (long address = 0; address < real_memory_allocated; address++)
	{
		real_values[address] = std::min(new_maximum_value,real_values[address]);
	}
}

void Image::SetMinimumAndMaximumValues(float new_minimum_value, float new_maximum_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	for (long address = 0; address < real_memory_allocated; address++)
	{
		real_values[address] = std::max(std::min(new_maximum_value,real_values[address]),new_minimum_value);
	}
}

int Image::ReturnMaximumDiagonalRadius()
{
	if (is_in_real_space)
	{
		return sqrt(pow(physical_address_of_box_center_x,2)+pow(physical_address_of_box_center_y,2)+pow(physical_address_of_box_center_z,2));
	}
	else
	{
		return sqrt(pow(logical_upper_bound_complex_x * fourier_voxel_size_x , 2) + pow(logical_upper_bound_complex_y * fourier_voxel_size_y , 2) + pow(logical_upper_bound_complex_z * fourier_voxel_size_z , 2) );
	}
}

void Image::GetMinMax(float &min_value, float &max_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Only real space supported");

	min_value = FLT_MAX;
	max_value = -FLT_MAX;

	int i, j, k;
	long address = 0;

	for (k = 0; k < logical_z_dimension; k++)
		{
			for (j = 0; j < logical_y_dimension; j++)
			{
				for (i = 0; i < logical_x_dimension; i++)
				{
					if (real_values[address] < min_value) min_value = real_values[address];
					if (real_values[address] > max_value) max_value = real_values[address];

					address++;
				}
				address += padding_jump_value;
			}
		}
}

/*
!>  \brief  return the maximum radius possible along a diagonal
pure function GetMaximumDiagonalRadius(self)  result(maximum_radius)

    ! Arguments

    class(image),   intent(in)  ::  self

    ! Variables

    real                        ::  maximum_radius  !<  In pixels (real-space) or reciprocal pixels (Fourier space)

    ! Start work
    if (self%IsInRealSpace()) then
        maximum_radius = sqrt(real(sum((self%physical_address_of_box_center-1)**2)))
    else
        maximum_radius = sqrt(  (self%logical_upper_bound_complex(1)*self%fourier_voxel_size(1))**2 &
                            +   (self%logical_upper_bound_complex(2)*self%fourier_voxel_size(2))**2 &
                            +   (self%logical_upper_bound_complex(3)*self%fourier_voxel_size(3))**2)
    endif


end function GetMaximumDiagonalRadius
*/

float Image::ReturnAverageOfRealValuesOnEdges()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	double sum;
	long number_of_pixels;
	int pixel_counter;
	int line_counter;
	int plane_counter;
	long address;

	sum = 0.0;
	number_of_pixels = 0;
	address = 0;

	if (logical_z_dimension == 1)
	{
		// Two-dimensional image

		// First line
		for (pixel_counter=0; pixel_counter < logical_x_dimension; pixel_counter++)
		{
			sum += real_values[address];
			address++;
		}
		number_of_pixels += logical_x_dimension;
		address += padding_jump_value;
		// Other lines
		for (line_counter=1; line_counter < logical_y_dimension-1; line_counter++)
		{
			sum += real_values[address];
			address += logical_x_dimension-1;
			sum += real_values[address];
			address += padding_jump_value + 1;
			number_of_pixels += 2;
		}
		// Last line
		for (pixel_counter=0; pixel_counter < logical_x_dimension; pixel_counter++)
		{
			sum += real_values[address];
			address++;
		}
		number_of_pixels += logical_x_dimension;
	}
	else
	{
		// Three-dimensional volume

		// First plane
		for (line_counter=0; line_counter < logical_y_dimension; line_counter++)
		{
			for (pixel_counter=0; pixel_counter < logical_x_dimension; pixel_counter++)
			{
				sum += real_values[address];
				address++;
			}
			address += padding_jump_value;
		}
		number_of_pixels += logical_x_dimension * logical_y_dimension;
		// Other planes
		for (plane_counter = 1; plane_counter < logical_z_dimension - 1; plane_counter++)
		{
			for (line_counter=0; line_counter< logical_y_dimension; line_counter++)
			{
				if (line_counter == 0 || line_counter == logical_y_dimension-1)
				{
					// First and last line of that section
					for (pixel_counter=0; pixel_counter < logical_x_dimension; pixel_counter++)
					{
						sum += real_values[address];
						address++;
					}
					address += padding_jump_value;
					number_of_pixels += logical_x_dimension;
				}
				else
				{
					// All other lines (only count first and last pixel)
					sum += real_values[address];
					address += logical_x_dimension-1;
					sum += real_values[address];
					address += padding_jump_value + 1;
					number_of_pixels += 2;
				}
			}
		}
		// Last plane
		for (line_counter=0; line_counter < logical_y_dimension; line_counter++)
		{
			for (pixel_counter=0; pixel_counter < logical_x_dimension; pixel_counter++)
			{
				sum += real_values[address];
				address++;
			}
			address += padding_jump_value;
		}
		number_of_pixels += logical_x_dimension * logical_y_dimension;
	}
	return sum/float(number_of_pixels);
}

// Find the largest voxel value, only considering voxels which are at least a certain distance from the center and from the edge in each dimension
float Image::ReturnMaximumValue(float minimum_distance_from_center, float minimum_distance_from_edge)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	int i,j,k;
	int i_dist_from_center, j_dist_from_center, k_dist_from_center;
	float maximum_value = - std::numeric_limits<float>::max();
	const int last_acceptable_address_x = logical_x_dimension - minimum_distance_from_edge - 1;
	const int last_acceptable_address_y = logical_y_dimension - minimum_distance_from_edge - 1;
	const int last_acceptable_address_z = logical_z_dimension - minimum_distance_from_edge - 1;
	long address = 0;


	for (k=0;k<logical_z_dimension;k++)
	{
		if (logical_z_dimension > 1)
		{
			k_dist_from_center = abs(k - physical_address_of_box_center_z);
			if (k_dist_from_center < minimum_distance_from_center || k < minimum_distance_from_edge || k > last_acceptable_address_z)
			{
				address += logical_y_dimension * (logical_x_dimension + padding_jump_value);
				continue;
			}
		}
		for (j=0;j<logical_y_dimension;j++)
		{
			j_dist_from_center = abs(j - physical_address_of_box_center_y);
			if (j_dist_from_center < minimum_distance_from_center || j < minimum_distance_from_edge || j > last_acceptable_address_y)
			{
				address += logical_x_dimension + padding_jump_value;
				continue;
			}
			for (i=0;i<logical_x_dimension;i++)
			{
				i_dist_from_center = abs(i - physical_address_of_box_center_x);
				if (i_dist_from_center < minimum_distance_from_center || i < minimum_distance_from_edge || i > last_acceptable_address_x)
				{
					address++;
					continue;
				}

				maximum_value = std::max(maximum_value,real_values[address]);
				address++;
			}
			address += padding_jump_value;
		}
	}

	return maximum_value;
}

float Image::ReturnAverageOfRealValues(float wanted_mask_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	double sum = 0.0;
	long address = 0;
	long number_of_pixels = 0;
	int		i;
	int		j;
	int 	k;
	float	x;
	float	y;
	float	z;
	float   mask_radius_squared;
	float	distance_from_center_squared;

	if (wanted_mask_radius > 0.0)
	{
		mask_radius_squared = powf(wanted_mask_radius, 2);
		number_of_pixels = 0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = powf(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = powf(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = powf(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared <= mask_radius_squared)
					{
						sum += real_values[address];
						number_of_pixels++;
					}
					address++;
				}
				address += padding_jump_value;
			}
		}
		return float(sum / number_of_pixels);
	}
	else
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			for (j = 0; j < logical_y_dimension; j++)
			{
				for (i = 0; i < logical_x_dimension; i++)
				{
					sum += real_values[address];
					address++;
				}
				address += padding_jump_value;
			}
		}

	}
	return float(sum / (logical_x_dimension * logical_y_dimension * logical_z_dimension));
}


void Image::ComputeAverageAndSigmaOfValuesInSpectrum(float minimum_radius, float maximum_radius, float &average, float &sigma, int cross_half_width)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");
	MyDebugAssertTrue(maximum_radius > minimum_radius,"Maximum radius must be greater than minimum radius");
	MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

	// Private variables
	int i, j;
	float x_sq, y_sq, rad_sq;
	EmpiricalDistribution my_distribution(false);
	const float min_rad_sq = powf(minimum_radius,2);
	const float max_rad_sq = powf(maximum_radius,2);
	const float cross_half_width_sq = powf(cross_half_width,2);
	long address = -1;

	for (j=0;j<logical_y_dimension;j++)
	{
		y_sq = powf(j-physical_address_of_box_center_y,2);
		if (y_sq <= cross_half_width_sq)
		{
			address += logical_x_dimension + padding_jump_value;
			continue;
		}
		for (i=0;i<logical_x_dimension;i++)
		{
			address++;
			x_sq = powf(i-physical_address_of_box_center_x,2);
			if (x_sq <= cross_half_width_sq) continue;
			rad_sq = x_sq + y_sq;
			if (rad_sq > min_rad_sq && rad_sq < max_rad_sq)
			{
				my_distribution.AddSampleValue(real_values[address]);
			}

		}
		address += padding_jump_value;
	}
	average = my_distribution.GetSampleMean();
	sigma = sqrtf(my_distribution.GetSampleVariance());

}


void Image::SetMaximumValueOnCentralCross(float maximum_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

	int i,j;
	long address = 0;

	for (j=0;j<logical_y_dimension;j++)
	{
		for (i=0;i<logical_x_dimension;i++)
		{
			if (j==physical_address_of_box_center_y || i==physical_address_of_box_center_x)
			{
				real_values[address] = std::min(maximum_value,real_values[address]);
			}
			address++;
		}
		address += padding_jump_value;
	}

}

// The image is assumed to be an amplitude spectrum, which we want to correlate with a set of CTF parameters
float Image::GetCorrelationWithCTF(CTF ctf)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");
	MyDebugAssertTrue(ctf.GetLowestFrequencyForFitting() > 0, "Will not work with lowest frequency for fitting of 0.");

	// Local variables
	double 			cross_product = 0.0;
	double 			norm_image = 0.0;
	double 			norm_ctf = 0.0;
	long			number_of_values = 0;
	int 			i,j;
	float 			i_logi, j_logi;
	float			i_logi_sq, j_logi_sq;
	const float		inverse_logical_x_dimension = 1.0 / float(logical_x_dimension);
	const float 	inverse_logical_y_dimension = 1.0 / float(logical_y_dimension);
	float			current_spatial_frequency_squared;
	const float		lowest_freq = powf(ctf.GetLowestFrequencyForFitting(),2);
	const float		highest_freq = powf(ctf.GetHighestFrequencyForFitting(),2);
	long			address = 0;
	float			current_azimuth;
	float			current_ctf_value;
	const int		central_cross_half_width = 10;
	float			astigmatism_penalty;

	// Loop over half of the image (ignore Friedel mates)
	for (j=0;j<logical_y_dimension;j++)
	{
		j_logi = float(j-physical_address_of_box_center_y)*inverse_logical_y_dimension;
		j_logi_sq = powf(j_logi,2);
		for (i=0;i<physical_address_of_box_center_x;i++)
		{
			i_logi = float(i-physical_address_of_box_center_x)*inverse_logical_x_dimension;
			i_logi_sq = powf(i_logi,2);

			// Where are we?
			current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

			if (current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared < highest_freq)
			{
				current_azimuth = atan2f(j_logi,i_logi);
				current_ctf_value = fabsf(ctf.Evaluate(current_spatial_frequency_squared,current_azimuth));
				// accumulate results
				if ( i < physical_address_of_box_center_x - central_cross_half_width && (j < physical_address_of_box_center_y - central_cross_half_width || j > physical_address_of_box_center_y + central_cross_half_width))
				{
					number_of_values++;
					cross_product += real_values[address] * current_ctf_value;
					norm_image    += pow(real_values[address],2);
					norm_ctf      += pow(current_ctf_value,2);
				}

			} // end of test whether within min,max frequency range

			// We're going to the next pixel
			address++;
		}
		// We're going to the next line
		address += padding_jump_value + physical_address_of_box_center_x;
	}

	// Compute the penalty due to astigmatism
	if (ctf.GetAstigmatismTolerance() > 0.0)
	{
		astigmatism_penalty = powf(ctf.GetAstigmatism(),2) * 0.5 / powf(ctf.GetAstigmatismTolerance(),2) / float(number_of_values);
	}
	else
	{
		astigmatism_penalty = 0.0;
	}

	// The final score
	return cross_product / sqrt(norm_image * norm_ctf) - astigmatism_penalty;
}

void Image::ApplyMirrorAlongY()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Meant for images, not volumes");

	int i,j;
	long address = logical_x_dimension + padding_jump_value;
	int j_dist;
	float temp_value;

	for (j = 1; j < physical_address_of_box_center_y; j++)
	{
		j_dist = 2 * ( physical_address_of_box_center_y - j ) * (logical_x_dimension + padding_jump_value);

		for (i = 0; i < logical_x_dimension; i++)
		{
			temp_value = real_values[address];
			real_values[address] = real_values[address+j_dist];
			real_values[address+j_dist] = temp_value;
			address++;
		}
		address += padding_jump_value;
	}

	// The column j=0 is undefined, we set it to the average of the values that were there before the mirror operation was applied
	temp_value = 0;
	for (i=0 ; i < logical_x_dimension; i++) {
		temp_value += real_values[i];
	}
	temp_value /= float(logical_x_dimension);
	for (i=0 ; i < logical_x_dimension; i++) {
		real_values[i] = temp_value;
	}
}


void Image::AddImage(Image *other_image)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] += other_image->real_values[pixel_counter];
	}

}

void Image::SubtractImage(Image *other_image)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] -= other_image->real_values[pixel_counter];
	}

}


int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_x, "index out of bounds");

    //if (physical_index >= physical_index_of_first_negative_frequency_x)
    if (physical_index > physical_address_of_box_center_x)
    {
    	 return physical_index - logical_x_dimension;
    }
    else return physical_index;
}


int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_y, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_y)
    {
    	 return physical_index - logical_y_dimension;
    }
    else return physical_index;
}

int Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(physical_index <= physical_upper_bound_complex_z, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_z)
    {
    	 return physical_index - logical_z_dimension;
    }
    else return physical_index;
}

//  \brief  Compute the 1D rotational average
//          Each bin is 1 pixel wide and there are as many bins as fit in the diagonal of the image
//          If the image is in Fourier space, compute the average of amplitudes.
//			It is assumed that the average array has already been allocated with number_of_bins elements.
//			The first element will be the value at the center/origin of the image.
void Image::Compute1DRotationalAverage(double average[], int number_of_bins)
{

	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	int i;
	int j;
	int k;
	int i_logi;
	int j_logi;
	int k_logi;
	float rad;
	double number_of_values[number_of_bins];
	long address;

	// Initialise
	for (i=0;i<number_of_bins;i++)
	{
		average[i] = 0.0;
		number_of_values[i] = 0.0;
	}
	address = 0;


	//
	if (is_in_real_space)
	{
		for (k=0;k<logical_z_dimension;k++)
		{
			k_logi = pow((k-physical_address_of_box_center_z),2);
			for (j=0;j<logical_y_dimension;j++)
			{
				j_logi = pow((j-physical_address_of_box_center_y),2) + k_logi;
				for (i=0;i<logical_x_dimension;i++)
				{
					i_logi = pow((i-physical_address_of_box_center_x),2) + j_logi;
					//
					rad = sqrt(float(i_logi));
					//
					MyDebugAssertTrue(int(rad)+1 < number_of_bins,"Bad radius: %f",rad);
					average[int(rad)  ] += (int(rad)-rad+1.0) * real_values[address];
					average[int(rad)+1] += (rad-int(rad)    ) * real_values[address];
					//
					number_of_values[int(rad)  ] += (rad-int(rad)    );
					number_of_values[int(rad)+1] += (int(rad)-rad+1.0);

					// Increment the address
					address ++;
				}
				// End of the line in real space
				address += padding_jump_value;
			}
		}
	}
	else
	{
		for (k=0;k<logical_z_dimension;k++)
		{
			k_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k),2);
			for (j=0;j<logical_y_dimension;j++)
			{
				j_logi = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j),2) + k_logi;
				for (i=0;i<physical_upper_bound_complex_x;i++)
				{
					i_logi = pow(i,2) + j_logi;
					//
					if (FourierComponentHasExplicitHermitianMate(i,j,k)) continue;
					rad = sqrt(float(i_logi));
					//
					average[int(rad)  ] += (rad-int(rad)    ) * cabs(complex_values[address]);
					average[int(rad)+1] += (int(rad)-rad+1.0) * cabs(complex_values[address]);
					//
					number_of_values[int(rad)  ] += (rad-int(rad)    );
					number_of_values[int(rad)+1] += (int(rad)-rad+1.0);

					// Increment the address
					address ++;
				}
			}
		}
	}

	// Do the actual averaging
	for (i=0;i<number_of_bins;i++)
	{
		if (number_of_values[i] > 0.0)
		{
			average[i] /= number_of_values[i];
		}
		else
		{
			average[i] = 0.0;
		}
	}
}

// The output image will be allocated to the correct dimensions (half-volume, a la FFTW)
void Image::ComputeAmplitudeSpectrum(Image *amplitude_spectrum)
{
	MyDebugAssertTrue(is_in_memory,"Memory not allocated");
	MyDebugAssertFalse(is_in_real_space,"Image not in Fourier space");

	//
	int i,j,k;
	long address_in_amplitude_spectrum = 0;
	long address_in_self = 0;
	const int spectrum_logical_dim_x = physical_upper_bound_complex_x + 1;
	const int spectrum_logical_dim_y = physical_upper_bound_complex_y + 1;
	const int spectrum_logical_dim_z = physical_upper_bound_complex_z + 1;


	//
	if (logical_z_dimension > 1)
	{
		amplitude_spectrum->Allocate(spectrum_logical_dim_x,spectrum_logical_dim_y,spectrum_logical_dim_z,true);
	}
	else
	{
		amplitude_spectrum->Allocate(spectrum_logical_dim_x,spectrum_logical_dim_y,true);
	}

	// Loop over the amplitude spectrum
	for (k = 0; k < amplitude_spectrum->logical_z_dimension; k++)
	{
		for (j = 0; j < amplitude_spectrum->logical_y_dimension; j++)
		{
			for (i = 0; i < amplitude_spectrum->logical_x_dimension; i++)
			{
				address_in_self = ReturnFourier1DAddressFromPhysicalCoord(i,j,k);
				amplitude_spectrum->real_values[address_in_amplitude_spectrum] = cabsf(complex_values[address_in_self]);
				address_in_amplitude_spectrum++;
			}
			address_in_amplitude_spectrum += amplitude_spectrum->padding_jump_value;
		}
	}
}


void Image::ComputeAmplitudeSpectrumFull2D(Image *amplitude_spectrum)
{
	MyDebugAssertTrue(is_in_memory,"Memory not allocated");
	MyDebugAssertTrue(amplitude_spectrum->is_in_memory, "Other image Memory not allocated");
	MyDebugAssertTrue(HasSameDimensionsAs(amplitude_spectrum), "Images do not have same dimensions");
	MyDebugAssertFalse(is_in_real_space,"Image not in Fourier space");

	int ampl_addr_i;
	int ampl_addr_j;
	int image_addr_i;
	int image_addr_j;
	int i_mate;
	int j_mate;

	long address_in_amplitude_spectrum = 0;
	long address_in_self;

	// Loop over the amplitude spectrum
	for (ampl_addr_j = 0; ampl_addr_j < amplitude_spectrum->logical_y_dimension; ampl_addr_j++)
	{
		for (ampl_addr_i = 0; ampl_addr_i < amplitude_spectrum->logical_x_dimension; ampl_addr_i++)
		{
			address_in_self = ReturnFourier1DAddressFromLogicalCoord(ampl_addr_i-amplitude_spectrum->physical_address_of_box_center_x,ampl_addr_j-amplitude_spectrum->physical_address_of_box_center_y,0);
			amplitude_spectrum->real_values[address_in_amplitude_spectrum] = cabsf(complex_values[address_in_self]);
			address_in_amplitude_spectrum++;
		}
		address_in_amplitude_spectrum += amplitude_spectrum->padding_jump_value;
	}

	// Done
	amplitude_spectrum->is_in_real_space = true;
	amplitude_spectrum->object_is_centred_in_box = true;
}

/*
 * Real-space box convolution meant for 2D amplitude spectra
 *
 * This is adapted from the MSMOOTH subroutine from CTFFIND3, with a different wrap-around behaviour
 */
void Image::SpectrumBoxConvolution(Image *output_image, int box_size, float minimum_radius)
{
	MyDebugAssertTrue(IsEven(box_size) == false,"Box size must be odd");
	MyDebugAssertTrue(logical_z_dimension == 1,"Volumes not supported");
	MyDebugAssertTrue(output_image->is_in_memory == true,"Output image not allocated");
	MyDebugAssertTrue(HasSameDimensionsAs(output_image),"Output image does not have same dimensions as image");

	// Variables
	int half_box_size = (box_size-1)/2;
	int cross_half_width_to_ignore = 1;
	int i;
	int i_friedel;
	int i_sq;
	int ii;
	int ii_friedel;
	int ii_sq;
	int iii;
	int j;
	int j_friedel;
	int j_sq;
	int jj;
	int jj_friedel;
	int jj_sq;
	int jjj;
	float radius;
	int num_voxels;
	int m;
	int l;

	// Addresses
	long address_within_output = 0;
	long address_within_input;

	// Loop over the output image. To save time, we only loop over one half of the image
	for (j = 0; j < logical_y_dimension; j++)
	{
		j_friedel = 2 * physical_address_of_box_center_y - j;
		j_sq = powf((j - physical_address_of_box_center_y),2);

		for (i = 0; i < logical_x_dimension; i++)
		{
			i_friedel = 2 * physical_address_of_box_center_x - i;
			i_sq = powf((i - physical_address_of_box_center_x),2);

			//address_within_output = ReturnReal1DAddressFromPhysicalCoord(i,j,0);

			radius = sqrt(float(i_sq+j_sq));

			if ( radius <= minimum_radius )
			{
				output_image->real_values[address_within_output] = real_values[address_within_output];
			}
			else
			{
				output_image->real_values[address_within_output] = 0.0e0;
				num_voxels = 0;

				for ( m = - half_box_size; m <= half_box_size; m++)
				{
					jj = j + m;
					if (jj < 0) { jj += logical_y_dimension; }
					if (jj >= logical_y_dimension) { jj -= logical_y_dimension; }
					jj_friedel = 2 * physical_address_of_box_center_y - jj;
					jj_sq = powf((jj - physical_address_of_box_center_y),2);

					for ( l = - half_box_size; l <= half_box_size; l++)
					{
						ii = i + l;
						if (ii < 0) { ii += logical_x_dimension; }
						if (ii >= logical_x_dimension) { ii -= logical_x_dimension; }
						ii_friedel = 2 * physical_address_of_box_center_x - ii;
						ii_sq = powf((ii - physical_address_of_box_center_x),2);

						// Friedel or not?
						if ( ii > physical_address_of_box_center_x)
						{
							iii = ii_friedel;
							jjj = jj_friedel;
							if (jjj > logical_y_dimension - 1 || iii > logical_x_dimension - 1) { continue; }
						}
						else
						{
							iii = ii;
							jjj = jj;
						}

						// In central cross?
						if ( abs(iii - physical_address_of_box_center_x) <= cross_half_width_to_ignore || abs(jjj - physical_address_of_box_center_y) <= cross_half_width_to_ignore ) { continue; }

						address_within_input = ReturnReal1DAddressFromPhysicalCoord(iii,jjj,0);

						if ( iii < logical_x_dimension && jjj < logical_y_dimension ) // it sometimes happens that we end up on Nyquist Friedel mates that we don't have (perhaps this can be fixed)
						{
							output_image->real_values[address_within_output] += real_values[address_within_input];
						}
						num_voxels++; // not sure why this is not within the if branch, like the addition itself - is this a bug?

					}
				} // end of loop over the box

				if (num_voxels == 0)
				{
					output_image->real_values[address_within_output] = real_values[address_within_input];
				}
				else
				{
					output_image->real_values[address_within_output] /= float(num_voxels);
				}
			}

			if (j_friedel < logical_y_dimension && i_friedel < logical_x_dimension)
			{
				output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,j_friedel,0)] = output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,j,0)];
			}
			address_within_output++;
		}
		address_within_output += output_image->padding_jump_value;
	}

	// There are a few pixels that are not set by the logical above
	for (i = physical_address_of_box_center_x + 1; i < logical_x_dimension; i++)
	{
		i_friedel = 2 * physical_address_of_box_center_x - i;
		output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,0,0)]                     = output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,0,0)];
		output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,logical_y_dimension-1,0)] = output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,logical_y_dimension - 1,0)];
	}
}


//#pragma GCC push_options
//#pragma GCC optimize ("O0")
// Taper edges of image so that there are no sharp discontinuities in real space
// This is a re-implementation of the MRC program taperedgek.for (Richard Henderson, 1987)
void Image::TaperEdges()
{
	MyDebugAssertTrue(is_in_memory,"Image not in memory");
	MyDebugAssertTrue(is_in_real_space,"Image not in real space");

	// Private variables
	const float				fraction_averaging = 30.0;
	const float				fraction_tapering  = 30.0;
	const int				averaging_strip_width_x	=	int(logical_x_dimension/fraction_averaging); //100
	const int				averaging_strip_width_y	=	int(logical_y_dimension/fraction_averaging);
	const int				averaging_strip_width_z =   int(logical_z_dimension/fraction_averaging);
	const int				tapering_strip_width_x	=	int(logical_x_dimension/fraction_tapering); //500
	const int				tapering_strip_width_y	=	int(logical_y_dimension/fraction_tapering);
	const int				tapering_strip_width_z	=	int(logical_z_dimension/fraction_tapering);
	const int				smoothing_half_width_x	=	1; // 1
	const int				smoothing_half_width_y	=	1;
	const int				smoothing_half_width_z	=	1;
	int						current_dimension;
	int						number_of_dimensions;
	int						second_dimension;
	int						third_dimension;
	int						logical_current_dimension;
	int						logical_second_dimension;
	int						logical_third_dimension;
	int						current_tapering_strip_width;
	int						i,j,k;
	int						j_shift,k_shift;
	int						jj,kk;
	int						number_of_values_in_running_average;
	long					address;
	int						smoothing_half_width_third_dimension;
	int						smoothing_half_width_second_dimension;
	// 2D arrays
	float					*average_for_current_edge_start = NULL;
	float					*average_for_current_edge_finish = NULL;
	float					*average_for_current_edge_average = NULL;
	float					*smooth_average_for_current_edge_start = NULL;
	float					*smooth_average_for_current_edge_finish = NULL;

	// Start work


	// Check dimensions of image are OK
	if (logical_x_dimension < 2 * tapering_strip_width_x || logical_y_dimension < 2 * tapering_strip_width_y)
	{
		MyPrintWithDetails("X,Y dimensions of input image are too small: %i %i\n", logical_x_dimension,logical_y_dimension);
		abort();
	}
	if (logical_z_dimension > 1 && logical_z_dimension < 2 * tapering_strip_width_z)
	{
		MyPrintWithDetails("Z dimension is too small: %i\n",logical_z_dimension);
		abort();
	}

	if ( logical_z_dimension > 1 )
	{
		number_of_dimensions = 3;
	}
	else
	{
		number_of_dimensions = 2;
	}


	for (current_dimension=1; current_dimension <= number_of_dimensions; current_dimension++)
	{
		switch(current_dimension)
		{
		case(1):
			second_dimension = 2;
			third_dimension = 3;
			logical_current_dimension = logical_x_dimension;
			logical_second_dimension = logical_y_dimension;
			logical_third_dimension = logical_z_dimension;
			current_tapering_strip_width = tapering_strip_width_x;
			smoothing_half_width_second_dimension = smoothing_half_width_y;
			smoothing_half_width_third_dimension = smoothing_half_width_z;
			break;
		case(2):
			second_dimension = 1;
			third_dimension = 3;
			logical_current_dimension = logical_y_dimension;
			logical_second_dimension = logical_x_dimension;
			logical_third_dimension = logical_z_dimension;
			current_tapering_strip_width = tapering_strip_width_y;
			smoothing_half_width_second_dimension = smoothing_half_width_x;
			smoothing_half_width_third_dimension = smoothing_half_width_z;
			break;
		case(3):
			second_dimension = 1;
			third_dimension = 2;
			logical_current_dimension = logical_z_dimension;
			logical_second_dimension = logical_x_dimension;
			logical_third_dimension = logical_y_dimension;
			current_tapering_strip_width = tapering_strip_width_z;
			smoothing_half_width_second_dimension = smoothing_half_width_x;
			smoothing_half_width_third_dimension = smoothing_half_width_y;
			break;
		}

		// Allocate memory
		if (average_for_current_edge_start != NULL) {
			delete [] average_for_current_edge_start;
			delete [] average_for_current_edge_finish;
			delete [] average_for_current_edge_average;
			delete [] smooth_average_for_current_edge_start;
			delete [] smooth_average_for_current_edge_finish;
		}
		average_for_current_edge_start 			= new float[logical_second_dimension*logical_third_dimension];
		average_for_current_edge_finish 		= new float[logical_second_dimension*logical_third_dimension];
		average_for_current_edge_average 		= new float[logical_second_dimension*logical_third_dimension];
		smooth_average_for_current_edge_start	= new float[logical_second_dimension*logical_third_dimension];
		smooth_average_for_current_edge_finish	= new float[logical_second_dimension*logical_third_dimension];

		// Initialise memory
		for(i=0;i<logical_second_dimension*logical_third_dimension;i++)
		{
			average_for_current_edge_start[i] = 0.0;
			average_for_current_edge_finish[i] = 0.0;
			average_for_current_edge_average[i] = 0.0;
			smooth_average_for_current_edge_start[i] = 0.0;
			smooth_average_for_current_edge_finish[i] = 0.0;
		}

		/*
		 * Deal with X=0 and X=logical_x_dimension edges
		 */
		i = 1;
		for (k=1;k<=logical_third_dimension;k++)
		{
			for (j=1;j<=logical_second_dimension;j++)
			{
				switch(current_dimension)
				{
				case(1):
						for (i=1;i<=averaging_strip_width_x;i++)
						{
							average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(i-1,j-1,k-1)];
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension-i,j-1,k-1)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_x);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_x);
						break;
				case(2):
						for (i=1;i<=averaging_strip_width_y;i++)
						{
							average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,i-1,k-1)];
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,logical_y_dimension-i,k-1)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_y);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_y);
						break;
				case(3):
						for (i=1;i<=averaging_strip_width_z;i++)
						{
							average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,i-1)];
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,logical_z_dimension-i)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_z);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_z);
						break;
				}
			}
		}

		for (address=0;address<logical_second_dimension*logical_third_dimension;address++)
		{
			average_for_current_edge_average[address] = 0.5 * ( average_for_current_edge_finish[address] + average_for_current_edge_start[address]);
			average_for_current_edge_start[address] -= average_for_current_edge_average[address];
			average_for_current_edge_finish[address] -= average_for_current_edge_average[address];
		}

		// Apply smoothing parallel to edge in the form of a running average
		for (k=1;k<=logical_third_dimension;k++)
		{
			for (j=1;j<=logical_second_dimension;j++)
			{
				number_of_values_in_running_average = 0;
				// Loop over neighbourhood of non-smooth arrays
				for (k_shift=-smoothing_half_width_third_dimension;k_shift<=smoothing_half_width_third_dimension;k_shift++)
				{
					kk = k+k_shift;
					if (kk < 1 || kk > logical_third_dimension) continue;
					for (j_shift=-smoothing_half_width_second_dimension;j_shift<=smoothing_half_width_second_dimension;j_shift++)
					{
						jj = j+j_shift;
						if (jj<1 || jj > logical_second_dimension) continue;
						number_of_values_in_running_average++;

						smooth_average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += average_for_current_edge_start [(jj-1)+(kk-1)*logical_second_dimension];
						smooth_average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += average_for_current_edge_finish[(jj-1)+(kk-1)*logical_second_dimension];
					}
				}
				// Now we can compute the average
				smooth_average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] /= float(number_of_values_in_running_average);
				smooth_average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] /= float(number_of_values_in_running_average);
			}
		}

		// Taper the image
		for (i=1;i<=logical_current_dimension;i++)
		{
			if (i<=current_tapering_strip_width)
			{
				switch(current_dimension)
				{
				case(1):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(i-1,j-1,k-1)] -= smooth_average_for_current_edge_start[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width-i+1) / float(current_tapering_strip_width);
							}
						}
						break;
				case(2):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,i-1,k-1)] -= smooth_average_for_current_edge_start[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width-i+1) / float(current_tapering_strip_width);
							}
						}
						break;
				case(3):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,i-1)] -= smooth_average_for_current_edge_start[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width-i+1) / float(current_tapering_strip_width);
							}
						}
						break;
				}
			}
			else if(i >= logical_current_dimension - current_tapering_strip_width+1)
			{
				switch(current_dimension)
				{
				case(1):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(i-1,j-1,k-1)] -= smooth_average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width+i-logical_current_dimension) / float(current_tapering_strip_width);
							}
						}
						break;
				case(2):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,i-1,k-1)] -= smooth_average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width+i-logical_current_dimension) / float(current_tapering_strip_width);
							}
						}
						break;
				case(3):
						for (k=1;k<=logical_third_dimension;k++)
						{
							for (j=1;j<=logical_second_dimension;j++)
							{
								real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,i-1)] -= smooth_average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] * float(current_tapering_strip_width+i-logical_current_dimension) / float(current_tapering_strip_width);
							}
						}
						break;
				}
			}
		}

	} // end of loop over dimensions

	// Cleanup
	delete [] average_for_current_edge_start;
	delete []average_for_current_edge_finish;
	delete [] average_for_current_edge_average;
	delete [] smooth_average_for_current_edge_start;
	delete [] smooth_average_for_current_edge_finish;


}
//#pragma GCC pop_options


void Image::Sine1D(int number_of_periods)
{
	int i;
	int j;
	float sine_value_i;
	float sine_value_j;

	for (j=0;j<logical_y_dimension;j++)
	{
		sine_value_j = sin(2*PI*(j-physical_address_of_box_center_y)*number_of_periods/logical_y_dimension);
		for (i=0;i<logical_x_dimension;i++)
		{
			sine_value_i = sin(2*PI*(i-physical_address_of_box_center_x)*number_of_periods/logical_x_dimension);
			real_values[ReturnReal1DAddressFromPhysicalCoord(i,j,0)] = sine_value_i + sine_value_j;
		}
	}
}


void Image::ClipInto(Image *other_image, float wanted_padding_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

	long pixel_counter = 0;
	int array_address = 0;

	int temp_logical_x;
	int temp_logical_y;
	int temp_logical_z;

	int kk;
	int k;
	int kk_logi;

	int jj;
	int jj_logi;
	int j;

	int ii;
	int ii_logi;
	int i;

	double junk;

	// take other following attributes

	other_image->is_in_real_space = is_in_real_space;
	other_image->object_is_centred_in_box = object_is_centred_in_box;

	if (is_in_real_space == true)
	{
		MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

		for (kk = 0; kk < other_image->logical_z_dimension; kk++)
		{
			kk_logi = kk - other_image->physical_address_of_box_center_z;
			k = physical_address_of_box_center_z + kk_logi;

			for (jj = 0; jj < other_image->logical_y_dimension; jj++)
			{
				jj_logi = jj - other_image->physical_address_of_box_center_y;
				j = physical_address_of_box_center_y + jj_logi;

				for (ii = 0; ii < other_image->logical_x_dimension; ii++)
				{
					ii_logi = ii - other_image->physical_address_of_box_center_x;
					i = physical_address_of_box_center_x + ii_logi;

					if (k < 0 || k >= logical_z_dimension || j < 0 || j >= logical_y_dimension || i < 0 || i >= logical_x_dimension)
					{
						other_image->real_values[pixel_counter] = wanted_padding_value;
					}
					else
					{
						other_image->real_values[pixel_counter] = ReturnRealPixelFromPhysicalCoord(i, j, k);
					}

					pixel_counter++;
				}

				pixel_counter+=other_image->padding_jump_value;
			}
		}
	}
	else
	{
		for (kk = 0; kk <= other_image->physical_upper_bound_complex_z; kk++)
		{
			temp_logical_z = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(kk);

			//if (temp_logical_z > logical_upper_bound_complex_z || temp_logical_z < logical_lower_bound_complex_z) continue;

			for (jj = 0; jj <= other_image->physical_upper_bound_complex_y; jj++)
			{
				temp_logical_y = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(jj);

				//if (temp_logical_y > logical_upper_bound_complex_y || temp_logical_y < logical_lower_bound_complex_y) continue;

				for (ii = 0; ii <= other_image->physical_upper_bound_complex_x; ii++)
				{
					temp_logical_x = ii;

					//if (temp_logical_x > logical_upper_bound_complex_x || temp_logical_x < logical_lower_bound_complex_x) continue;

					other_image->complex_values[pixel_counter] = ReturnComplexPixelFromLogicalCoord(temp_logical_x, temp_logical_y, temp_logical_z, wanted_padding_value);
					pixel_counter++;

				}

			}
		}


		// When we are clipping into a larger volume in Fourier space, there is a half-plane (vol) or half-line (2D image) at Nyquist for which FFTW
		// does not explicitly tell us the values. We need to fill them in.
		if (logical_y_dimension < other_image->logical_y_dimension || logical_z_dimension < other_image->logical_z_dimension)
		{
			// For a 2D image
			if (logical_z_dimension == 1)
			{
				jj = physical_index_of_first_negative_frequency_y;
				for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
				{
					other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,0)] = complex_values[ReturnFourier1DAddressFromPhysicalCoord(ii,jj,0)];
				}
			}
			// For a 3D volume
			else
			{

				// Deal with the positive Nyquist of the 2nd dimension
				for (kk_logi = logical_lower_bound_complex_z; kk_logi <= logical_upper_bound_complex_z; kk_logi ++)
				{
					jj = physical_index_of_first_negative_frequency_y;
					jj_logi = logical_lower_bound_complex_y;
					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
					{
						other_image->complex_values[other_image->ReturnFourier1DAddressFromLogicalCoord(ii,jj,kk_logi)] = complex_values[ReturnFourier1DAddressFromLogicalCoord(ii,jj_logi,kk_logi)];
					}
				}


				// Deal with the positive Nyquist in the 3rd dimension
				kk = physical_index_of_first_negative_frequency_z;
				int kk_mirror = other_image->logical_z_dimension - physical_index_of_first_negative_frequency_z;
				//wxPrintf("\nkk = %i; kk_mirror = %i\n",kk,kk_mirror);
				int jj_mirror;
				//wxPrintf("Will loop jj from %i to %i\n",1,physical_index_of_first_negative_frequency_y);
				for (jj = 1; jj <= physical_index_of_first_negative_frequency_y; jj ++ )
				{
					//jj_mirror = other_image->logical_y_dimension - jj;
					jj_mirror = jj;
					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++ )
					{
						//wxPrintf("(1) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror = %i\n",ii,jj,kk,jj_mirror,kk_mirror);
						other_image->complex_values[other_image-> ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj_mirror,kk_mirror)];
					}
				}
				//wxPrintf("Will loop jj from %i to %i\n", other_image->logical_y_dimension - physical_index_of_first_negative_frequency_y, other_image->logical_y_dimension - 1);
				for (jj = other_image->logical_y_dimension - physical_index_of_first_negative_frequency_y; jj <= other_image->logical_y_dimension - 1; jj ++)
				{
					//jj_mirror = other_image->logical_y_dimension - jj;
					jj_mirror = jj;
					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++ )
					{
						//wxPrintf("(2) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror = %i\n",ii,jj,kk,jj_mirror,kk_mirror);
						other_image->complex_values[other_image-> ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj_mirror,kk_mirror)];
					}
				}
				jj = 0;
				for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
				{
					other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk_mirror)];
				}

			}
		}


	}

}

// Bilinear interpolation in real space, at point (x,y) where x and y are physical coordinates (i.e. first pixel has x,y = 0,0)
void Image::GetRealValueByLinearInterpolationNoBoundsCheckImage(float &x, float &y, float &interpolated_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(logical_z_dimension == 1, "Not for volumes");
	MyDebugAssertTrue(is_in_real_space, "Need to be in real space");

	const int i_start = int(x);
	const int j_start = int(y);
	const float x_dist = x - float(i_start);
	const float y_dist = y - float(j_start);
	const float x_dist_m = 1.0 - x_dist;
	const float y_dist_m = 1.0 - y_dist;

	const int address_1 = j_start * (logical_x_dimension + padding_jump_value) + i_start;
	const int address_2 = address_1 + logical_x_dimension + padding_jump_value;

	MyDebugAssertTrue(address_1+1 <= real_memory_allocated && address_1 >= 0,"Out of bounds, address 1\n");
	MyDebugAssertTrue(address_2+1 <= real_memory_allocated && address_2 >= 0,"Out of bounds, address 2\n");

	interpolated_value =    x_dist_m * y_dist_m * real_values[address_1]
						+	x_dist   * y_dist_m * real_values[address_1 + 1]
						+   x_dist_m * y_dist   * real_values[address_2]
						+   x_dist   * y_dist   * real_values[address_2 + 1];

}


void Image::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	Image temp_image;

	temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);
	ClipInto(&temp_image, wanted_padding_value);

	CopyFrom(&temp_image);
	//Consume(&temp_image);
}

void Image::CopyFrom(Image *other_image)
{
	*this = other_image;
}

void Image::CopyLoopingAndAddressingFrom(Image *other_image)
{
	object_is_centred_in_box = other_image->object_is_centred_in_box;
	logical_x_dimension = other_image->logical_x_dimension;
	logical_y_dimension = other_image->logical_y_dimension;
	logical_z_dimension = other_image->logical_z_dimension;

	physical_upper_bound_complex_x = other_image->physical_upper_bound_complex_x;
	physical_upper_bound_complex_y = other_image->physical_upper_bound_complex_y;
	physical_upper_bound_complex_z = other_image->physical_upper_bound_complex_z;

	physical_address_of_box_center_x = other_image->physical_address_of_box_center_x;
	physical_address_of_box_center_y = other_image->physical_address_of_box_center_y;
	physical_address_of_box_center_z = other_image->physical_address_of_box_center_z;

	//physical_index_of_first_negative_frequency_x = other_image->physical_index_of_first_negative_frequency_x;
	physical_index_of_first_negative_frequency_y = other_image->physical_index_of_first_negative_frequency_y;
	physical_index_of_first_negative_frequency_z = other_image->physical_index_of_first_negative_frequency_z;

	fourier_voxel_size_x = other_image->fourier_voxel_size_x;
	fourier_voxel_size_y = other_image->fourier_voxel_size_y;
	fourier_voxel_size_z = other_image->fourier_voxel_size_z;

	logical_upper_bound_complex_x = other_image->logical_upper_bound_complex_x;
	logical_upper_bound_complex_y = other_image->logical_upper_bound_complex_y;
	logical_upper_bound_complex_z = other_image->logical_upper_bound_complex_z;

	logical_lower_bound_complex_x = other_image->logical_lower_bound_complex_x;
	logical_lower_bound_complex_y = other_image->logical_lower_bound_complex_y;
	logical_lower_bound_complex_z = other_image->logical_lower_bound_complex_z;

	logical_upper_bound_real_x = other_image->logical_upper_bound_complex_x;
	logical_upper_bound_real_y = other_image->logical_upper_bound_complex_y;
	logical_upper_bound_real_z = other_image->logical_upper_bound_complex_z;

	logical_lower_bound_real_x = other_image->logical_lower_bound_complex_x;
	logical_lower_bound_real_y = other_image->logical_lower_bound_complex_y;
	logical_lower_bound_real_z = other_image->logical_lower_bound_complex_z;
}

void Image::Consume(Image *other_image) // copy the parameters then directly steal the memory of another image, leaving it an empty shell
{
	MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

	if (is_in_memory == true)
	{
		Deallocate();
	}

	is_in_real_space = other_image->is_in_real_space;
	real_memory_allocated = other_image->real_memory_allocated;
	CopyLoopingAndAddressingFrom(other_image);

	real_values = other_image->real_values;
	complex_values = other_image->complex_values;
	is_in_memory = other_image->is_in_memory;

	plan_fwd = other_image->plan_fwd;
	plan_bwd = other_image->plan_bwd;
	planned = other_image->planned;

	other_image->real_values = NULL;
	other_image->complex_values = NULL;
	other_image->is_in_memory = false;

	other_image->plan_fwd = NULL;
	other_image->plan_bwd = NULL;
	other_image->planned = false;

}


void Image::PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");


	bool need_to_fft = false;

	long pixel_counter = 0;

	int k;
	int k_logical;

	int j;
	int j_logical;

	int i;

	float phase_z;
	float phase_y;
	float phase_x;

	fftw_complex total_phase_shift;

	if (is_in_real_space == true)
	{
		ForwardFFT();
		need_to_fft = true;
	}

	for (k=0; k <= physical_upper_bound_complex_z; k++)
	{
		k_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
		phase_z = ReturnPhaseFromShift(wanted_z_shift, k_logical, logical_z_dimension);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			j_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
			phase_y = ReturnPhaseFromShift(wanted_y_shift, j_logical, logical_y_dimension);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{

				phase_x = ReturnPhaseFromShift(wanted_x_shift, i, logical_x_dimension);

				total_phase_shift = Return3DPhaseFromIndividualDimensions(phase_x, phase_y, phase_z);
				complex_values[pixel_counter] *= total_phase_shift;

				pixel_counter++;
			}
		}
	}

	if (need_to_fft == true) BackwardFFT();

}

void Image::ApplyCTFPhaseFlip(CTF ctf_to_apply)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

	int j;
	int i;

	long pixel_counter = 0;

	float y_coord_sq;
	float x_coord_sq;

	float y_coord;
	float x_coord;

	float frequency_squared;
	float azimuth;
	float ctf_value;

	for (j = 0; j <= physical_upper_bound_complex_y; j++)
	{
		y_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
		y_coord_sq = powf(y_coord, 2.0);

		for (i = 0; i <= physical_upper_bound_complex_x; i++)
		{
			x_coord = i * fourier_voxel_size_x;
			x_coord_sq = powf(x_coord, 2);

			// Compute the azimuth
			if ( i == 0 && j == 0 ) {
				azimuth = 0.0;
			} else {
				azimuth = atan2f(y_coord,x_coord);
			}

			// Compute the square of the frequency
			frequency_squared = x_coord_sq + y_coord_sq;

			ctf_value = ctf_to_apply.Evaluate(frequency_squared,azimuth);

			if (ctf_value < 0.0) complex_values[pixel_counter] = - complex_values[pixel_counter];
			pixel_counter++;
		}
	}

}

void Image::ApplyCTF(CTF ctf_to_apply)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Volumes not supported");

	int j;
	int i;

	long pixel_counter = 0;

	float y_coord_sq;
	float x_coord_sq;

	float y_coord;
	float x_coord;

	float frequency_squared;
	float azimuth;
	float ctf_value;

	for (j = 0; j <= physical_upper_bound_complex_y; j++)
	{
		y_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y;
		y_coord_sq = powf(y_coord, 2.0);

		for (i = 0; i <= physical_upper_bound_complex_x; i++)
		{
			x_coord = i * fourier_voxel_size_x;
			x_coord_sq = powf(x_coord, 2);

			// Compute the azimuth
			if ( i == 0 && j == 0 ) {
				azimuth = 0.0;
			} else {
				azimuth = atan2f(y_coord,x_coord);
			}

			// Compute the square of the frequency
			frequency_squared = x_coord_sq + y_coord_sq;

			ctf_value = ctf_to_apply.Evaluate(frequency_squared,azimuth);

			complex_values[pixel_counter] *= ctf_value;
			pixel_counter++;
		}
	}

}

void Image::ApplyBFactor(float bfactor) // add real space and windows later, probably to an overloaded function
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == false, "image not in Fourier space");

	int k;
	int j;
	int i;

	long pixel_counter = 0;

	float z_coord;
	float y_coord;
	float x_coord;

	float frequency_squared;
	float filter_value;
	//float frequency;

	for (k = 0; k <= physical_upper_bound_complex_z; k++)
	{
		z_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y_coord = powf(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x_coord = powf(i * fourier_voxel_size_x, 2);

				// compute squared radius, in units of reciprocal pixels

				frequency_squared = x_coord + y_coord + z_coord;
				//frequency = sqrt(frequency_squared);
				filter_value = exp(-bfactor * frequency_squared * 0.25);
				complex_values[pixel_counter] *= filter_value;
				pixel_counter++;
			}
		}

	}
}

void Image::MaskCentralCross(int vertical_half_width, int horizontal_half_width)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D currently supported");

	int pixel_counter;
	int width_counter;
	bool must_fft = false;

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT();
	}

	for (pixel_counter = logical_lower_bound_complex_y; pixel_counter <= logical_upper_bound_complex_y; pixel_counter++)
	{
		for (width_counter = -(horizontal_half_width - 1); width_counter <= (horizontal_half_width - 1); width_counter++)
		{
			complex_values[ReturnFourier1DAddressFromLogicalCoord(width_counter, pixel_counter, 0)] = 0.0 + 0.0 * I;
		}
	}


	for (pixel_counter = 0; pixel_counter <= logical_upper_bound_complex_x; pixel_counter++)
	{
		for (width_counter = -(vertical_half_width - 1); width_counter <=  (vertical_half_width - 1); width_counter++)
		{
			complex_values[ReturnFourier1DAddressFromLogicalCoord(pixel_counter, width_counter, 0)] = 0.0 + 0.0 * I;

		}
	}

	if (must_fft == true) BackwardFFT();

}

bool Image::HasSameDimensionsAs(Image *other_image)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

	if (logical_x_dimension == other_image->logical_x_dimension && logical_y_dimension == other_image->logical_y_dimension && logical_z_dimension == other_image->logical_z_dimension) return true;
	else return false;
}

void Image::SwapRealSpaceQuadrants()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	bool must_fft = false;

	float x_shift_to_apply;
	float y_shift_to_apply;
	float z_shift_to_apply;

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT();
	}

	if (object_is_centred_in_box == true)
	{
		x_shift_to_apply = float(physical_address_of_box_center_x);
		y_shift_to_apply = float(physical_address_of_box_center_y);
		z_shift_to_apply = float(physical_address_of_box_center_z);
	}
	else
	{
		if (IsEven(logical_x_dimension) == true)
		{
			x_shift_to_apply = float(physical_address_of_box_center_x);
		}
		else
		{
			x_shift_to_apply = float(physical_address_of_box_center_x) - 1.0;
		}

		if (IsEven(logical_y_dimension) == true)
		{
			y_shift_to_apply = float(physical_address_of_box_center_y);
		}
		else
		{
			y_shift_to_apply = float(physical_address_of_box_center_y) - 1.0;
		}

		if (IsEven(logical_z_dimension) == true)
		{
			z_shift_to_apply = float(physical_address_of_box_center_z);
		}
		else
		{
			z_shift_to_apply = float(physical_address_of_box_center_z) - 1.0;
		}
	}


	if (logical_z_dimension == 1)
	{
		z_shift_to_apply = 0.0;
	}

	PhaseShift(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

	if (must_fft == true) BackwardFFT();


	// keep track of center;
	if (object_is_centred_in_box == true) object_is_centred_in_box = false;
	else object_is_centred_in_box = true;


}


void Image::CalculateCrossCorrelationImageWith(Image *other_image)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == other_image->is_in_real_space, "Images are in different spaces");
	MyDebugAssertTrue(HasSameDimensionsAs(other_image) == true, "Images are different sizes");

	long pixel_counter;
	bool must_fft = false;

	// do we have to fft..

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT();
		other_image->ForwardFFT();
	}

	// multiply by the complex conjugate

	for (pixel_counter = 0; pixel_counter < real_memory_allocated / 2; pixel_counter++)
	{
		complex_values[pixel_counter] *= conjf(other_image->complex_values[pixel_counter]);
	}

	if (object_is_centred_in_box == true)
	{
		object_is_centred_in_box = false;
		SwapRealSpaceQuadrants();
	}

	BackwardFFT();

	if (must_fft == true) other_image->BackwardFFT();

}

Peak Image::FindPeakAtOriginFast2D(int wanted_max_1d_distance)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
	MyDebugAssertTrue(! object_is_centred_in_box, "Peak centered in image");

	int j;
	int i;
	int jj;
	int pixel_counter;
	int y_dim = logical_y_dimension + padding_jump_value;
	int max_1d_distance = wanted_max_1d_distance;

	if (max_1d_distance > physical_address_of_box_center_x) max_1d_distance = physical_address_of_box_center_x;

	Peak found_peak;
	found_peak.value = -FLT_MAX;
	found_peak.x = 0.0;
	found_peak.y = 0.0;
	found_peak.z = 0.0;

	for (j = 0; j <= max_1d_distance; j++)
	{
		jj = j * y_dim;
		for (i = 0; i <= max_1d_distance; i++)
		{
			pixel_counter = jj + i;
			if (real_values[pixel_counter] > found_peak.value)
			{
				found_peak.value = real_values[pixel_counter];
				found_peak.x = i;
				found_peak.y = j;
			}
		}
	}

	for (j = logical_y_dimension - max_1d_distance - 1; j <= logical_y_dimension - 1; j++)
	{
		jj = j * y_dim;
		for (i = 0; i <= max_1d_distance; i++)
		{
			pixel_counter = jj + i;
			if (real_values[pixel_counter] > found_peak.value)
			{
				found_peak.value = real_values[pixel_counter];
				found_peak.x = i;
				found_peak.y = j;
			}
		}
	}

	for (j = 0; j <= max_1d_distance; j++)
	{
		jj = j * y_dim;
		for (i = logical_x_dimension - max_1d_distance - 1; i <= logical_x_dimension - 1; i++)
		{
			pixel_counter = jj + i;
			if (real_values[pixel_counter] > found_peak.value)
			{
				found_peak.value = real_values[pixel_counter];
				found_peak.x = i;
				found_peak.y = j;
			}
		}
	}

	for (j = logical_y_dimension - max_1d_distance - 1; j <= logical_y_dimension - 1; j++)
	{
		jj = j * y_dim;
		for (i = logical_x_dimension - max_1d_distance - 1; i <= logical_x_dimension - 1; i++)
		{
			pixel_counter = jj + i;
			if (real_values[pixel_counter] > found_peak.value)
			{
				found_peak.value = real_values[pixel_counter];
				found_peak.x = i;
				found_peak.y = j;
			}
		}
	}

	if (found_peak.x > physical_address_of_box_center_x) found_peak.x -= logical_x_dimension;
	if (found_peak.y > physical_address_of_box_center_y) found_peak.y -= logical_y_dimension;
	return found_peak;
}

Peak Image::FindPeakWithIntegerCoordinates(float wanted_min_radius, float wanted_max_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

	int k;
	int j;
	int i;
	int kk;
	int jj;
	int ii;

	float z;
	float y;
	float x;

	long pixel_counter = 0;

	float inv_max_radius_sq_x;
	float inv_max_radius_sq_y;
	float inv_max_radius_sq_z;

	float distance_from_origin;

	bool radii_are_fractional;

	Peak found_peak;
	found_peak.value = -FLT_MAX;
	found_peak.x = 0.0;
	found_peak.y = 0.0;
	found_peak.z = 0.0;

	// if they are > 0.0 and < 1.0 then the radius will be interpreted as fraction, otherwise
	// it is interpreted as absolute.

	if (wanted_min_radius >= 0.0 && wanted_min_radius < 1.0 && wanted_max_radius >= 0.0 && wanted_max_radius < 1.0) radii_are_fractional = true;
	else radii_are_fractional = false;

	wanted_min_radius = powf(wanted_min_radius, 2);
	wanted_max_radius = powf(wanted_max_radius, 2);

	if (object_is_centred_in_box)
	{
		if (radii_are_fractional == true)
		{
			inv_max_radius_sq_x = 1.0 / powf(physical_address_of_box_center_x, 2);
			inv_max_radius_sq_y = 1.0 / powf(physical_address_of_box_center_y, 2);

			if (logical_z_dimension == 1) inv_max_radius_sq_z = 0;
			else inv_max_radius_sq_z = 1.0 / powf(physical_address_of_box_center_z, 2);

			for (k = 0; k < logical_z_dimension; k++)
				{
					z = powf(k - physical_address_of_box_center_z, 2) * inv_max_radius_sq_z;

					for (j = 0; j < logical_y_dimension; j++)
					{
						y = powf(j - physical_address_of_box_center_y, 2) * inv_max_radius_sq_y;

						for (i = 0; i < logical_x_dimension; i++)
						{
							x = powf(i - physical_address_of_box_center_x, 2) * inv_max_radius_sq_x;

							distance_from_origin = x + y + z;

							if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius)
							{
								if (real_values[pixel_counter] > found_peak.value)
								{
									found_peak.value = real_values[pixel_counter];
									found_peak.x = i - physical_address_of_box_center_x;
									found_peak.y = j - physical_address_of_box_center_y;
									found_peak.z = k - physical_address_of_box_center_z;
								}

							}

							//wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z, found_peak.value);
							//wxPrintf("value %f, %f, %f (%f)\n", x, y, z, real_values[pixel_counter]);

							pixel_counter++;
						}

						pixel_counter+=padding_jump_value;
					}


				}
		}
		else
		{
			for (k = 0; k < logical_z_dimension; k++)
			{
				z = powf(k - physical_address_of_box_center_z, 2);

				for (j = 0; j < logical_y_dimension; j++)
				{
					y = powf(j - physical_address_of_box_center_y, 2);

					for (i = 0; i < logical_x_dimension; i++)
					{
						x = powf(i - physical_address_of_box_center_x, 2);

						distance_from_origin = x + y + z;

						if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius)
						{
							if (real_values[pixel_counter] > found_peak.value)
							{
								found_peak.value = real_values[pixel_counter];
								found_peak.x = i - physical_address_of_box_center_x;
								found_peak.y = j - physical_address_of_box_center_y;
								found_peak.z = k - physical_address_of_box_center_z;
								//wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z, found_peak.value);
								//wxPrintf("new peak %i, %i, %i (%f)\n", i, j, k, found_peak.value);
							}
						}

						pixel_counter++;
					}

					pixel_counter+=padding_jump_value;
				}
			}
		}
	}
	else
	{
		if (radii_are_fractional == true)
		{
			inv_max_radius_sq_x = 1.0 / powf(physical_address_of_box_center_x, 2);
			inv_max_radius_sq_y = 1.0 / powf(physical_address_of_box_center_y, 2);

			if (logical_z_dimension == 1) inv_max_radius_sq_z = 0;
			else inv_max_radius_sq_z = 1.0 / powf(physical_address_of_box_center_z, 2);

			for (k = 0; k < logical_z_dimension; k++)
			{
				kk = k;
				if (kk > physical_address_of_box_center_z) kk -= logical_z_dimension;
				z = powf(float(kk), 2) * inv_max_radius_sq_z;

				for (j = 0; j < logical_y_dimension; j++)
				{
					jj = j;
					if (jj > physical_address_of_box_center_y) jj -= logical_y_dimension;
					y = powf(float(jj), 2) * inv_max_radius_sq_y;

					for (i = 0; i < logical_x_dimension; i++)
					{
						ii = i;
						if (ii > physical_address_of_box_center_x) ii -= logical_x_dimension;
						x = powf(float(ii), 2) * inv_max_radius_sq_x;

						distance_from_origin = x + y + z;

						if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius)
						{
							if (real_values[pixel_counter] > found_peak.value)
							{
								found_peak.value = real_values[pixel_counter];
								found_peak.x = ii;
								found_peak.y = jj;
								found_peak.z = kk;
							}
						}

						//wxPrintf("new peak %f, %f, %f (%f)\n", found_peak.x, found_peak.y, found_peak.z, found_peak.value);
						//wxPrintf("value %f, %f, %f (%f)\n", x, y, z, real_values[pixel_counter]);

						pixel_counter++;
					}

					pixel_counter+=padding_jump_value;
				}
			}
		}
		else
		{
			for (k = 0; k < logical_z_dimension; k++)
			{
				kk = k;
				if (kk > physical_address_of_box_center_z) kk -= logical_z_dimension;
				z = powf(float(kk), 2);

				for (j = 0; j < logical_y_dimension; j++)
				{
					jj = j;
					if (jj > physical_address_of_box_center_y) jj -= logical_y_dimension;
					y = powf(float(jj), 2);

					for (i = 0; i < logical_x_dimension; i++)
					{
						ii = i;
						if (ii > physical_address_of_box_center_x) ii -= logical_x_dimension;
						x = powf(float(ii), 2);

						distance_from_origin = x + y + z;

						if (distance_from_origin >= wanted_min_radius && distance_from_origin <= wanted_max_radius)
						{
							if (real_values[pixel_counter] > found_peak.value)
							{
								found_peak.value = real_values[pixel_counter];
								found_peak.x = ii;
								found_peak.y = jj;
								found_peak.z = kk;
							}
						}

						pixel_counter++;
					}

					pixel_counter+=padding_jump_value;
				}
			}
		}
	}

	return found_peak;
}

Peak Image::FindPeakWithParabolaFit(float wanted_min_radius, float wanted_max_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D images supported for now");

	Peak integer_peak;
	Peak found_peak;

	int best_x;
	int best_y;
	int x_counter;
	int y_counter;
	int current_x;
	int current_y;

	float average_of_square = 0.0;
	float scale_factor;
	float scaled_square[3][3];

	float coefficient_one;
	float coefficient_two;
	float coefficient_three;
	float coefficient_four;
	float coefficient_five;
	float coefficient_six;
	float denominator;

	float x_max;
	float y_max;



	integer_peak = FindPeakWithIntegerCoordinates(wanted_min_radius, wanted_max_radius);

	//wxPrintf("Integer Peak = %f, %f\n", integer_peak.x, integer_peak.y);

	best_x = integer_peak.x + physical_address_of_box_center_x;
	best_y = integer_peak.y + physical_address_of_box_center_y;

	for (y_counter = -1; y_counter <= 1; y_counter++)
	{
		for (x_counter = -1; x_counter <= 1; x_counter++)
		{
			current_x = best_x + x_counter;
			current_y = best_y + y_counter;

			if (current_x < 0 || current_x >= logical_x_dimension || current_y < 0 || current_y >= logical_y_dimension) scaled_square[x_counter + 1][y_counter + 1] = 0.0;
			else scaled_square[x_counter + 1][y_counter + 1] = ReturnRealPixelFromPhysicalCoord(current_x, current_y , 0);

			average_of_square += scaled_square[x_counter + 1][y_counter + 1];
		}
	}

	average_of_square /= 9.0;

	if (average_of_square != 0.0) scale_factor = 1.0 / average_of_square;
	else scale_factor = 1.0;

	for (y_counter = 0; y_counter <  3; y_counter++)
	{
		for (x_counter = 0; x_counter < 3; x_counter++)
		{
			scaled_square[x_counter][y_counter]  *= scale_factor;
		}
	}

    coefficient_one = (26.0e0 * scaled_square[0][0] - scaled_square[0][1] + 2.0e0 * scaled_square[0][2] - scaled_square[1][0] - 19.0e0 * scaled_square[1][1] - 7.0e0 * scaled_square[1][2] + 2.0e0 * scaled_square[2][0] - 7.0e0 * scaled_square[2][1] + 14.0e0 * scaled_square[2][2]) / 9.0e0;
    coefficient_two = ( 8.0e0 * scaled_square[0][0] - 8.0e0 * scaled_square[0][1] + 5.0e0 * scaled_square[1][0] - 8.0e0 * scaled_square[1][1] + 3.0e0 * scaled_square[1][2] + 2.0e0 * scaled_square[2][0] - 8.0e0 * scaled_square[2][1] + 6.0e0 * scaled_square[2][2]) / (-1.0e0 * 6.0e0);
    coefficient_three = (scaled_square[0][0] - 2.0e0 * scaled_square[0][1] + scaled_square[0][2] + scaled_square[1][0] - 2.0e0 * scaled_square[1][1] + scaled_square[1][2] + scaled_square[2][0] - 2.0e0 * scaled_square[2][1] + scaled_square[2][2]) / 6.0e0;
    coefficient_four = (8.0e0 * scaled_square[0][0] + 5.0e0 * scaled_square[0][1] + 2.0e0 * scaled_square[0][2] - 8.0e0 * scaled_square[1][0] - 8.0e0 * scaled_square[1][1] - 8.0e0 * scaled_square[1][2] + 3.0e0 * scaled_square[2][1] + 6.0e0 * scaled_square[2][2]) / (-1.0e0 * 6.0e0);
    coefficient_five = (scaled_square[0][0] - scaled_square[0][2] - scaled_square[2][0] + scaled_square[2][2]) / 4.0e0;
    coefficient_six = (scaled_square[0][0] + scaled_square[0][1] + scaled_square[0][2] - 2.0e0 * scaled_square[1][0] - 2.0e0 * scaled_square[1][1] - 2.0e0 * scaled_square[1][2] + scaled_square[2][0] + scaled_square[2][1] + scaled_square[2][2]) / 6.0e0;
    denominator = 4.0e0 * coefficient_three * coefficient_six - powf(coefficient_five, 2);

    if (denominator == 0.0) found_peak = integer_peak;
    else
    {
    	y_max = (coefficient_four * coefficient_five - 2.0e0 * coefficient_two * coefficient_six) / denominator;
    	x_max = (coefficient_two * coefficient_five - 2.0e0 * coefficient_four * coefficient_three) / denominator;

        y_max = y_max - 2.0e0;
        x_max = x_max - 2.0e0;

        if (y_max > 1.05e0 || y_max < -1.05e0) y_max = 0.0e0;

        if (x_max > 1.05e0 || x_max < -1.05e0) x_max = 0.0e0;

        found_peak = integer_peak;

        found_peak.x += x_max;
        found_peak.y += y_max;

        found_peak.value = 4.0e0 * coefficient_one * coefficient_three * coefficient_six - coefficient_one * powf(coefficient_five,2) - powf(coefficient_two,2) * coefficient_six + coefficient_two * coefficient_four*coefficient_five - powf(coefficient_four,2) * coefficient_three;
        found_peak.value = found_peak.value * average_of_square / denominator;

        if (fabs((found_peak.value - integer_peak.value) / (found_peak.value + integer_peak.value)) > 0.15) found_peak.value = integer_peak.value;
    }

	//wxPrintf("%f %f %f %f\n", integer_peak.x, integer_peak.y, found_peak.x, found_peak.y);
    return found_peak;
}


/*
Peak Image::FindPeakWithParabolaFit(float wanted_min_radius, float wanted_max_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
	MyDebugAssertTrue(logical_z_dimension == 1, "Only 2D images supported for now");

	int x;
	int y;

	int current_x;
	int current_y;

	float ImportantSquare[3][3];

	float average = 0.0;
	float scale;

	float c1,c2,c3,c4,c5,c6;
	float xmax, ymax;
	float denomin;

	Peak integer_peak;
	Peak found_peak;

	// first off find the integer peak..

	integer_peak = FindPeakWithIntegerCoordinates(wanted_min_radius, wanted_max_radius);

	// Fit to Parabola

	for (y = 0; y <= 2; y++)
	{
		for (x = 0; x <= 2; x++)
		{
			current_x = integer_peak.x + physical_address_of_box_center_x - 1 + x;
			current_y = integer_peak.y + physical_address_of_box_center_y - 1 + y;

			if (current_x < 0 || current_x >= logical_x_dimension || current_y < 0 || current_y >= logical_y_dimension) ImportantSquare[x][y] = 0.0;
			else ImportantSquare[x][y] = ReturnRealPixelFromPhysicalCoord(current_x, current_y , 0);

			average += ImportantSquare[x][y];
		}
	}

	average /= 9.0;
	scale = 1./ average;

	// scale

	for (y = 0; y <= 2; y++)
	{
		for (x = 0; x <= 2; x++)
		{
			ImportantSquare[x][y] *= scale;

		}
	}

	c1 = (26. * ImportantSquare[0][0] - ImportantSquare[0][1] + 2. * ImportantSquare[0][2] - ImportantSquare[1][0] - 19. * ImportantSquare[1][1] - 7. * ImportantSquare[1][2] + 2. * ImportantSquare[2][0] - 7. * ImportantSquare[2][1] + 14 * ImportantSquare[2][2]) / 9.;
	c2 = (8.*ImportantSquare[0][0] - 8.*ImportantSquare[0][1] + 5.*ImportantSquare[1][0] - 8.*ImportantSquare[1][1] + 3.*ImportantSquare[1][2] + 2.*ImportantSquare[2][0] - 8.*ImportantSquare[2][1] + 6.*ImportantSquare[2][2]) / (-6.);
	c3 = (ImportantSquare[0][0] - 2.*ImportantSquare[0][1] + ImportantSquare[0][2] + ImportantSquare[1][0] - 2.*ImportantSquare[1][1] + ImportantSquare[1][2] + ImportantSquare[2][0] - 2.*ImportantSquare[2][1] + ImportantSquare[2][2]) / 6.;
	c4 = (8.*ImportantSquare[0][0] + 5.*ImportantSquare[0][1] + 2.*ImportantSquare[0][2] - 8.*ImportantSquare[1][0] - 8.*ImportantSquare[1][1] - 8.*ImportantSquare[1][2] + 3.*ImportantSquare[2][1]+ 6.*ImportantSquare[2][2]) / (-6.);
	c5 = (ImportantSquare[0][0] - ImportantSquare[0][2] - ImportantSquare[2][0] + ImportantSquare[2][2]) / 4.;
	c6 = (ImportantSquare[0][0] + ImportantSquare[0][1] + ImportantSquare[0][2] - 2.*ImportantSquare[1][0] - 2.*ImportantSquare[1][1] - 2.*ImportantSquare[1][2] + ImportantSquare[2][0] + ImportantSquare[2][1] + ImportantSquare[2][2]) / 6.;

    denomin   = 4. * c3 * c6 - c5 * c5;

    if (denomin == 0.)
    {
    	found_peak = integer_peak;
    }
    else
    {
    	ymax      = (c4 * c5 - 2.* c2 * c6) / denomin;
	    xmax      = (c2 * c5 - 2.* c4 * c3) / denomin;
	    ymax-= 2.;
	    xmax-= 2.;

	    if (ymax > 1.05 || ymax < -1.05) ymax = 0.0;
	    if (xmax > 1.05 || xmax < -1.05) xmax = 0.0;

	    found_peak.x = integer_peak.x + xmax;
	    found_peak.y = integer_peak.y + ymax;


	    found_peak.value = 4.*c1*c3*c6 - c1*c5*c5 - c2*c2*c6 + c2*c4*c5 - c4*c4*c3;
	    found_peak.value *= (average / denomin);

	    if (fabs((found_peak.value - integer_peak.value) / (found_peak.value + integer_peak.value)) > 0.15) found_peak.value = integer_peak.value;

    }

    return found_peak;
}*/




