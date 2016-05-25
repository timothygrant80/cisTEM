#include "core_headers.h"

ResolutionStatistics::ResolutionStatistics()
{
	pixel_size = 0.0;
	number_of_bins = -1;
	number_of_bins_extended = 0;
}

ResolutionStatistics::ResolutionStatistics(float wanted_pixel_size, int wanted_number_of_bins)
{
	Init(wanted_pixel_size, wanted_number_of_bins);
}

ResolutionStatistics & ResolutionStatistics::operator = (const ResolutionStatistics &other_statistics)
{
	*this = &other_statistics;
	return *this;
}


ResolutionStatistics & ResolutionStatistics::operator = (const ResolutionStatistics *other_statistics)
{
   // Check for self assignment
   if(this != other_statistics)
   {
		MyDebugAssertTrue(other_statistics->number_of_bins >= 0, "Other statistics not initialized");

		FSC = other_statistics->FSC;
		part_FSC = other_statistics->part_FSC;
		part_SSNR = other_statistics->part_SSNR;
		rec_SSNR = other_statistics->rec_SSNR;

		pixel_size = other_statistics->pixel_size;
		number_of_bins = other_statistics->number_of_bins;
		number_of_bins_extended = other_statistics->number_of_bins_extended;
   }

   return *this;
}

void ResolutionStatistics::Init(float wanted_pixel_size, int wanted_number_of_bins)
{
	pixel_size = wanted_pixel_size;
	number_of_bins = wanted_number_of_bins;
}

void ResolutionStatistics::CalculateFSC(Image &reconstructed_volume_1, Image &reconstructed_volume_2)
{
	MyDebugAssertTrue(reconstructed_volume_1.is_in_real_space == false, "reconstructed_volume_1 not in Fourier space");
	MyDebugAssertTrue(reconstructed_volume_2.is_in_real_space == false, "reconstructed_volume_2 not in Fourier space");
	MyDebugAssertTrue(reconstructed_volume_1.HasSameDimensionsAs(&reconstructed_volume_2), "reconstructions do not have equal size");

	int i, j, k;
	int yi, zi;
	int bin;

	float x, y, z;
	float frequency;
	float frequency_squared;

	if (number_of_bins == 0) number_of_bins = reconstructed_volume_1.ReturnSmallestLogicalDimension() / 2 + 1;
	int number_of_bins2 = 2 * (number_of_bins - 1);

// Extend table to include corners in 3D Fourier space
	number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;
	double *sum1 = new double[number_of_bins_extended];
	double *sum2 = new double[number_of_bins_extended];
	double *cross_terms = new double[number_of_bins_extended];
	int *non_zero_count = new int[number_of_bins_extended];
	double temp_double;

	long pixel_counter = 0;

	fftwf_complex temp_c;

	ZeroDoubleArray(sum1, number_of_bins_extended);
	ZeroDoubleArray(sum2, number_of_bins_extended);
	ZeroDoubleArray(cross_terms, number_of_bins_extended);
	ZeroIntArray(non_zero_count, number_of_bins_extended);

	for (k = 0; k <= reconstructed_volume_1.physical_upper_bound_complex_z; k++)
	{
		zi = reconstructed_volume_1.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
		z = powf(zi * reconstructed_volume_1.fourier_voxel_size_z, 2);

		for (j = 0; j <= reconstructed_volume_1.physical_upper_bound_complex_y; j++)
		{
			yi = reconstructed_volume_1.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
			y = powf(yi * reconstructed_volume_1.fourier_voxel_size_y, 2);

			for (i = 0; i <= reconstructed_volume_1.physical_upper_bound_complex_x; i++)
			{
				x = powf(i * reconstructed_volume_1.fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

//				if (frequency_squared <= 0.25)
//				{
					temp_c = crealf(reconstructed_volume_1.complex_values[pixel_counter] * conjf(reconstructed_volume_2.complex_values[pixel_counter]));
					if (temp_c != 0.0)
					{
						if ((i != 0) || (i == 0 && zi > 0) || (i == 0 && yi > 0 && zi == 0))
						{
							// compute radius, in units of physical Fourier pixels
							bin = int(sqrtf(frequency_squared) * number_of_bins2);
							if ((i == 0 && yi != 0) || pixel_counter == 0)
							{
								sum1[bin] += crealf(reconstructed_volume_1.complex_values[pixel_counter] * conjf(reconstructed_volume_1.complex_values[pixel_counter])) * 0.25;
								sum2[bin] += crealf(reconstructed_volume_2.complex_values[pixel_counter] * conjf(reconstructed_volume_2.complex_values[pixel_counter])) * 0.25;
								cross_terms[bin] += crealf(temp_c) * 0.25;
								non_zero_count[bin] += 1;
							}
							else
							{
								sum1[bin] += crealf(reconstructed_volume_1.complex_values[pixel_counter] * conjf(reconstructed_volume_1.complex_values[pixel_counter]));
								sum2[bin] += crealf(reconstructed_volume_2.complex_values[pixel_counter] * conjf(reconstructed_volume_2.complex_values[pixel_counter]));
								cross_terms[bin] += crealf(temp_c);
								non_zero_count[bin] += 1;
							}
						}
					}
//				}
				pixel_counter++;
			}
		}
	}

	for (i = 0; i < number_of_bins_extended; i++)
	{
		temp_double = sum1[i] * sum2[i];
		if (temp_double != 0.0)
		{
//			if (i < 8 || non_zero_count[i] > 440)
//			{
				FSC.AddPoint(pixel_size / float(i) * float(number_of_bins2), float(cross_terms[i] / sqrtf(temp_double)));
//			}
//			else
//			{
//				FSC.AddPoint(pixel_size / float(i) * float(number_of_bins2), 0.0);
//			}
		}
		else
		{
			FSC.AddPoint(pixel_size / float(i) * float(number_of_bins2), 0.0);
		}
	}
	delete [] sum1;
	delete [] sum2;
	delete [] cross_terms;
	delete [] non_zero_count;
}

void ResolutionStatistics::CalculateParticleFSCandSSNR(float mask_volume_in_voxels, float molecular_mass_kDa)
{
	MyDebugAssertTrue(FSC.number_of_points > 0, "FSC curve must be calculated first");

	int i;

	float volume_fraction = kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3) / mask_volume_in_voxels;

	for (i = 0; i < number_of_bins_extended; i++)
	{
		part_FSC.AddPoint(FSC.data_x[i],  FSC.data_y[i] / volume_fraction / (1.0 + (1.0 / volume_fraction - 1.0) * fabsf(FSC.data_y[i])));
	}
	for (i = 0; i < number_of_bins_extended; i++)
	{
		if (part_FSC.data_y[i] > 0.0)
		{
			rec_SSNR.AddPoint(FSC.data_x[i], fabsf(2.0 * part_FSC.data_y[i] / (1.00001 - part_FSC.data_y[i])));
		}
		else
		{
			rec_SSNR.AddPoint(FSC.data_x[i],0.0);
		}
	}
}

void ResolutionStatistics::CalculateParticleSSNR(Image &image_reconstruction, float *ctf_reconstruction, float mask_volume_fraction)
{
	MyDebugAssertTrue(FSC.number_of_points > 0, "FSC curve must be calculated first");
	MyDebugAssertTrue(mask_volume_fraction > 0.0, "mask_volume_fraction invalid");

	int i, j, k;
	int yi, zi;
	int bin;
	int number_of_bins2;

	float x, y, z;
	float frequency;
	float frequency_squared;
//	float pssnr_scaling_factor = mask_volume_fraction / average_occupancy * 100.0;
//	float pssnr_scaling_factor = 1.0 / mask_volume_fraction;

	double	*sum_double	= new double[number_of_bins_extended];
	long 	*sum_int 	= new long[number_of_bins_extended];

	long pixel_counter = 0;

	ZeroDoubleArray(sum_double, number_of_bins_extended);
	ZeroLongArray(sum_int, number_of_bins_extended);
	number_of_bins2 = 2 * (number_of_bins - 1);

	for (k = 0; k <= image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		zi = image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
		z = powf(zi * image_reconstruction.fourier_voxel_size_z, 2);

		for (j = 0; j <= image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			yi = image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
			y = powf(yi * image_reconstruction.fourier_voxel_size_y, 2);

			for (i = 0; i <= image_reconstruction.physical_upper_bound_complex_x; i++)
			{
				x = powf(i * image_reconstruction.fourier_voxel_size_x, 2);
				frequency_squared = x + y + z;

//				if (frequency_squared <= 0.25)
//				{
					if (ctf_reconstruction[pixel_counter] != 0.0)
					{
						if ((i != 0) || (i == 0 && zi > 0) || (i == 0 && yi > 0 && zi == 0))
						{
							// compute radius, in units of physical Fourier pixels
							bin = int(sqrtf(frequency_squared) * number_of_bins2);
							if ((i == 0 && yi != 0) || pixel_counter == 0)
							{
								sum_double[bin]	+= ctf_reconstruction[pixel_counter] * 0.5;
							}
							else
							{
								sum_double[bin]	+= ctf_reconstruction[pixel_counter];
							}
							sum_int[bin] += 1;
						}
					}
//				}
				pixel_counter++;
			}
		}
	}

	for (i = 0; i < number_of_bins_extended; i++)
	{
		if (sum_double[i] > 0.0)
		{
//			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), float(sum_double[i]) / float(sum_int[i]) / 0.5);
//			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), float(sum_double[i]) / float(sum_int[i]) / 0.25 / 10000 * i);
			// Average sum of CTF^2 from N 2D images contributing to a 3D voxel = float(sum_double[i]) / float(sum_int[i]) = N * 0.5 / 2i
			// The factor of 0.5 is due to the average value for CTF^2.
			// Divide volume SSNR by  float(sum_double[i]) / float(sum_int[i])
			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), mask_volume_fraction* fabsf(2.0 * fabsf(FSC.data_y[i]) / (1.00001 - fabsf(FSC.data_y[i])) * float(sum_int[i]) / float(sum_double[i])));
		}
		else
		{
			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), 0.0);
		}
	}
	delete [] sum_double;
	delete [] sum_int;
}
