#include "core_headers.h"

ResolutionStatistics::ResolutionStatistics()
{
	pixel_size = 0.0;
	number_of_bins = -1;
	number_of_bins_extended = 0;
}

ResolutionStatistics::ResolutionStatistics(float wanted_pixel_size, int box_size)
{
	Init(wanted_pixel_size, box_size);
}

ResolutionStatistics::ResolutionStatistics( const ResolutionStatistics &other_statistics) // copy constructor
{
	MyDebugPrint("Warning: copying a resolution statistics object");
	*this = other_statistics;
	//abort();
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
		//MyDebugAssertTrue(other_statistics->number_of_bins >= 0, "Other statistics not initialized");

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

void ResolutionStatistics::ResampleFrom(ResolutionStatistics &other_statistics, int wanted_number_of_bins)
{
	MyDebugAssertTrue(other_statistics.number_of_bins > 0 || wanted_number_of_bins > 0, "Other statistics not initialized");

	int extended = int(wanted_number_of_bins * sqrtf(3.0)) + 1;
	if (wanted_number_of_bins == 0) extended = number_of_bins_extended;

	if (other_statistics.FSC.number_of_points > 0) FSC.ResampleCurve(&other_statistics.FSC, extended);
	if (other_statistics.part_FSC.number_of_points > 0) part_FSC.ResampleCurve(&other_statistics.part_FSC, extended);
	if (other_statistics.part_SSNR.number_of_points > 0) part_SSNR.ResampleCurve(&other_statistics.part_SSNR, extended);
	if (other_statistics.rec_SSNR.number_of_points > 0) rec_SSNR.ResampleCurve(&other_statistics.rec_SSNR, extended);
}

void ResolutionStatistics::CopyFrom(ResolutionStatistics &other_statistics, int wanted_number_of_bins)
{
	MyDebugAssertTrue(other_statistics.number_of_bins > 0 || wanted_number_of_bins > 0, "Other statistics not initialized");
//	MyDebugAssertTrue(other_statistics.number_of_bins > wanted_number_of_bins, "Other statistics has more bins than requested");

	int i;
	int number_of_bins2 = 2 * (number_of_bins - 1);
	float resolution;
	int extended = int(wanted_number_of_bins * sqrtf(3.0)) + 1;
	if (wanted_number_of_bins == 0) extended = number_of_bins_extended;

	FSC.ClearData();
	part_FSC.ClearData();
	part_SSNR.ClearData();
	rec_SSNR.ClearData();

	for (i = 0; i < std::min(extended,number_of_bins_extended); i++)
	{
		if (i > 0)
		{
			resolution = pixel_size / float(i) * float(number_of_bins2);
		}
		else
		{
			resolution = 0.0;
		}
		if (other_statistics.FSC.number_of_points > 0) FSC.AddPoint(resolution, other_statistics.FSC.data_y[i]);
		if (other_statistics.part_FSC.number_of_points > 0) part_FSC.AddPoint(resolution, other_statistics.part_FSC.data_y[i]);
		if (other_statistics.part_SSNR.number_of_points > 0) part_SSNR.AddPoint(resolution, other_statistics.part_SSNR.data_y[i]);
		if (other_statistics.rec_SSNR.number_of_points > 0) rec_SSNR.AddPoint(resolution, other_statistics.rec_SSNR.data_y[i]);
	}

	for (i = std::min(extended,number_of_bins_extended); i < extended; i++)
	{
		if (i > 0)
		{
			resolution = pixel_size / float(i) * float(number_of_bins2);
		}
		else
		{
			resolution = 0.0;
		}
		if (other_statistics.FSC.number_of_points > 0) FSC.AddPoint(resolution, 0.0);
		if (other_statistics.part_FSC.number_of_points > 0) part_FSC.AddPoint(resolution, 0.0);
		if (other_statistics.part_SSNR.number_of_points > 0) part_SSNR.AddPoint(resolution, 0.0);
		if (other_statistics.rec_SSNR.number_of_points > 0) rec_SSNR.AddPoint(resolution, 0.0);
	}
}

void ResolutionStatistics::CopyParticleSSNR(ResolutionStatistics &other_statistics, int wanted_number_of_bins)
{
	MyDebugAssertTrue(other_statistics.number_of_bins > 0 || wanted_number_of_bins > 0, "Other statistics not initialized");
//	MyDebugAssertTrue(other_statistics.number_of_bins > wanted_number_of_bins, "Other statistics has more bins than requested");

	int i;
	int number_of_bins2 = 2 * (number_of_bins - 1);
	float resolution;
	int extended = int(wanted_number_of_bins * sqrtf(3.0)) + 1;
	if (wanted_number_of_bins == 0) extended = number_of_bins_extended;

	part_SSNR.ClearData();

	for (i = 0; i < std::min(extended,number_of_bins_extended); i++)
	{
		if (i > 0)
		{
			resolution = pixel_size / float(i) * float(number_of_bins2);
		}
		else
		{
			resolution = 0.0;
		}
		if (other_statistics.part_SSNR.number_of_points > 0) part_SSNR.AddPoint(resolution, other_statistics.part_SSNR.data_y[i]);
	}

	for (i = std::min(extended,number_of_bins_extended); i < extended; i++)
	{
		if (i > 0)
		{
			resolution = pixel_size / float(i) * float(number_of_bins2);
		}
		else
		{
			resolution = 0.0;
		}
		if (other_statistics.part_SSNR.number_of_points > 0) part_SSNR.AddPoint(resolution, 0.0);
	}
}

void ResolutionStatistics::ResampleParticleSSNR(ResolutionStatistics &other_statistics, int wanted_number_of_bins)
{
	MyDebugAssertTrue(other_statistics.number_of_bins > 0 || wanted_number_of_bins > 0, "Other statistics not initialized");

	int extended = int(wanted_number_of_bins * sqrtf(3.0)) + 1;
	if (wanted_number_of_bins == 0) extended = number_of_bins_extended;

	if (other_statistics.part_SSNR.number_of_points > 0)
	{
		part_SSNR.ResampleCurve(&other_statistics.part_SSNR, extended);
		part_SSNR.MultiplyByConstant(float(extended) / float(other_statistics.number_of_bins_extended));
	}
}

float ResolutionStatistics::ReturnEstimatedResolution(bool use_part_fsc)
{
	float estimated_resolution = 0.0f;

	if (use_part_fsc == true)
	{
		for (int counter = 1; counter < part_FSC.number_of_points; counter++)
		{
			if (part_FSC.data_y[counter] < 0.143)
			{
				estimated_resolution = (part_FSC.data_x[counter - 1] + part_FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}
	else
	{
		for (int counter = 1; counter < FSC.number_of_points; counter++)
		{
			if (FSC.data_y[counter] < 0.143)
			{
				estimated_resolution = (FSC.data_x[counter - 1] +  FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}

	if (estimated_resolution < 2.0f * pixel_size) estimated_resolution = 2.0f * pixel_size;

	return estimated_resolution;
}

float ResolutionStatistics::Return0p8Resolution(bool use_part_fsc)
{
	float estimated_resolution = 0.0f;

	if (use_part_fsc == true)
	{
		for (int counter = 1; counter < part_FSC.number_of_points; counter++)
		{
			if (part_FSC.data_y[counter] < 0.8)
			{
				estimated_resolution = (part_FSC.data_x[counter - 1] + part_FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}
	else
	{
		for (int counter = 1; counter < FSC.number_of_points; counter++)
		{
			if (FSC.data_y[counter] < 0.8)
			{
				estimated_resolution = (FSC.data_x[counter - 1] +  FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}

	if (estimated_resolution < 2.0f * pixel_size) estimated_resolution = 2.0f * pixel_size;

	return estimated_resolution;
}


float ResolutionStatistics::Return0p5Resolution(bool use_part_fsc)
{
	float estimated_resolution = 0.0f;

	if (use_part_fsc == true)
	{
		for (int counter = 1; counter < part_FSC.number_of_points; counter++)
		{
			if (part_FSC.data_y[counter] < 0.5)
			{
				estimated_resolution = (part_FSC.data_x[counter - 1] + part_FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}
	else
	{
		for (int counter = 1; counter < FSC.number_of_points; counter++)
		{
			if (FSC.data_y[counter] < 0.5)
			{
				estimated_resolution = (FSC.data_x[counter - 1] +  FSC.data_x[counter]) / 2.0;
				break;
			}
		}
	}

	if (estimated_resolution < 2.0f * pixel_size) estimated_resolution = 2.0f * pixel_size;

	return estimated_resolution;
}

float ResolutionStatistics::ReturnResolutionNShellsAfter(float wanted_resolution, int number_of_shells)
{
	int resolution_shell = -1;

	for (int counter = 1; counter < FSC.number_of_points; counter++)
	{
		if (FSC.data_x[counter] < wanted_resolution)
		{
			resolution_shell = counter;
			break;
		}
	}

	if (resolution_shell == -1) return 0;
	else
	{
		resolution_shell += number_of_shells;
		if (resolution_shell >= FSC.number_of_points) return pixel_size * 2.0;
		else return FSC.data_x[resolution_shell];
	}
}

int ResolutionStatistics::ReturnResolutionShellNumber(float wanted_resolution)
{
	int resolution_shell = -1;

	for (int counter = 1; counter < FSC.number_of_points; counter++)
	{
		if (FSC.data_x[counter] < wanted_resolution)
		{
			resolution_shell = counter;
			break;
		}
	}

	return resolution_shell;
}

float ResolutionStatistics::ReturnResolutionNShellsBefore(float wanted_resolution, int number_of_shells)
{
	int resolution_shell = -1;

	for (int counter = FSC.number_of_points - 1; counter >= 1; counter--)
	{
		if (FSC.data_x[counter] > wanted_resolution)
		{
			resolution_shell = counter;
			break;
		}
	}

	if (resolution_shell == -1) return 0;
	else
	{
		resolution_shell -= number_of_shells;
		if (resolution_shell < 1) return 0;
		else return FSC.data_x[resolution_shell];
	}
}




void ResolutionStatistics::Init(float wanted_pixel_size, int box_size)
{
	pixel_size = wanted_pixel_size;
	number_of_bins = box_size / 2 + 1;
	number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;
	FSC.ClearData();
	part_FSC.ClearData();
	part_SSNR.ClearData();
	rec_SSNR.ClearData();
}

void ResolutionStatistics::NormalizeVolumeWithParticleSSNR(Image &reconstructed_volume)
{
//	MyDebugAssertTrue(reconstructed_volume.is_in_real_space == false, "reconstructed_volume not in Fourier space");
	MyDebugAssertTrue(number_of_bins_extended == int((reconstructed_volume.logical_x_dimension / 2 + 1) * sqrtf(3.0)) + 1, "reconstructed_volume not compatible with resolution statistics");
	MyDebugAssertTrue(part_SSNR.number_of_points > 0, "part_SSNR curve not calculated");

	float ssq_in;
	float ssq_out;
	bool need_fft;
	Curve temp_curve;

	part_SSNR.SquareRoot();
	need_fft = reconstructed_volume.is_in_real_space;
	if (need_fft) reconstructed_volume.ForwardFFT();
	ssq_in = reconstructed_volume.ReturnSumOfSquares();
	reconstructed_volume.Whiten(0.5);
	reconstructed_volume.MultiplyByWeightsCurve(part_SSNR);
	ssq_out = reconstructed_volume.ReturnSumOfSquares();
	reconstructed_volume.MultiplyByConstant(sqrtf(ssq_in / ssq_out));
	if (need_fft) reconstructed_volume.BackwardFFT();
	// If part_SSNR is used after this, it needs to be squared again
//	part_SSNR.Square();
}

void ResolutionStatistics::CalculateFSC(Image &reconstructed_volume_1, Image &reconstructed_volume_2, bool smooth_curve)
{
	MyDebugAssertTrue(reconstructed_volume_1.is_in_real_space == false, "reconstructed_volume_1 not in Fourier space");
	MyDebugAssertTrue(reconstructed_volume_2.is_in_real_space == false, "reconstructed_volume_2 not in Fourier space");
	MyDebugAssertTrue(reconstructed_volume_1.HasSameDimensionsAs(&reconstructed_volume_2), "reconstructions do not have equal size");

	int i, j, k;
	int yi, zi;
	float bin;
	int ibin;
	int window = myroundint(20.0 / pixel_size);

	double difference;
	double current_sum1;
	double current_sum2;

	float x, y, z;
	float frequency;
	float frequency_squared;

	if (number_of_bins <= 1 ) number_of_bins = reconstructed_volume_1.ReturnSmallestLogicalDimension() / 2 + 1;
	int number_of_bins2 = 2 * (number_of_bins - 1);

// Extend table to include corners in 3D Fourier space
	if (reconstructed_volume_1.logical_z_dimension == 1) number_of_bins_extended = int(number_of_bins * sqrtf(2.0)) + 1;
	else number_of_bins_extended = int(number_of_bins * sqrtf(3.0)) + 1;

	double *sum1 = new double[number_of_bins_extended];
	double *sum2 = new double[number_of_bins_extended];
	double *cross_terms = new double[number_of_bins_extended];
	double *non_zero_count = new double[number_of_bins_extended];
	double temp_double;
	float  temp_float;

	long pixel_counter = 0;

	std::complex<double> temp_c;

	FSC.ClearData();

	ZeroDoubleArray(sum1, number_of_bins_extended);
	ZeroDoubleArray(sum2, number_of_bins_extended);
	ZeroDoubleArray(cross_terms, number_of_bins_extended);
	ZeroDoubleArray(non_zero_count, number_of_bins_extended);

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
					temp_c = real(reconstructed_volume_1.complex_values[pixel_counter] * conj(reconstructed_volume_2.complex_values[pixel_counter])) + I * 0.0f;
					if (temp_c != 0.0)
					{
						if ((i != 0) || (i == 0 && zi > 0) || (i == 0 && yi > 0 && zi == 0))
						{
							// compute radius, in units of physical Fourier pixels
							bin = sqrtf(frequency_squared) * number_of_bins2;
							ibin = int(bin);
							difference = bin - float(ibin);

							//bin = int(sqrtf(frequency_squared) * number_of_bins2);

							if ((i == 0 && yi != 0) || pixel_counter == 0)
							{
								current_sum1 = real(reconstructed_volume_1.complex_values[pixel_counter] * conj(reconstructed_volume_1.complex_values[pixel_counter])) * 0.25;
								current_sum2 = real(reconstructed_volume_2.complex_values[pixel_counter] * conj(reconstructed_volume_2.complex_values[pixel_counter])) * 0.25;

								sum1[ibin] += current_sum1 * (1 - difference);
								sum1[ibin + 1] += current_sum1 * difference;

								sum2[ibin] += current_sum2 * (1 - difference);
								sum2[ibin + 1] += current_sum2 * difference;

								cross_terms[ibin] += real(temp_c) * (1 - difference) * 0.25;
								cross_terms[ibin + 1] += real(temp_c) * difference * 0.25;

								non_zero_count[ibin] += 1 - difference * 0.25;
								non_zero_count[ibin + 1] += difference * 0.25;
							}
							else
							{
								//sum1[bin] += real(reconstructed_volume_1.complex_values[pixel_counter] * conj(reconstructed_volume_1.complex_values[pixel_counter]));
								//sum2[bin] += real(reconstructed_volume_2.complex_values[pixel_counter] * conj(reconstructed_volume_2.complex_values[pixel_counter]));
								//cross_terms[bin] += crealf(temp_c);
								//non_zero_count[bin] += 1;

								current_sum1 = real(reconstructed_volume_1.complex_values[pixel_counter] * conj(reconstructed_volume_1.complex_values[pixel_counter]));
								current_sum2 = real(reconstructed_volume_2.complex_values[pixel_counter] * conj(reconstructed_volume_2.complex_values[pixel_counter]));

								sum1[ibin] += current_sum1 * (1 - difference);
								sum1[ibin + 1] += current_sum1 * difference;

								sum2[ibin] += current_sum2 * (1 - difference);
								sum2[ibin + 1] += current_sum2 * difference;

								cross_terms[ibin] += real(temp_c) * (1 - difference);
								cross_terms[ibin + 1] += real(temp_c) * difference;

								non_zero_count[ibin] += 1 - difference;
								non_zero_count[ibin + 1] += difference;

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
		if (i > 0)
		{
			temp_float = pixel_size / float(i) * float(number_of_bins2);
		}
		else
		{
			temp_float = 0.0;
		}
		if (temp_double != 0.0)
		{
//			if (i < 8 || non_zero_count[i] > 440)
//			{
				FSC.AddPoint(temp_float, float(cross_terms[i] / sqrtf(temp_double)));
//			}
//			else
//			{
//				FSC.AddPoint(pixel_size / float(i) * float(number_of_bins2), 0.0);
//			}
		}
		else
		{
			FSC.AddPoint(temp_float, 0.0);
		}
	}

	if (smooth_curve)
	{
		if (window < 5) window = 5;
		if (window > number_of_bins_extended / 10) window = number_of_bins_extended / 10;

		if (IsOdd(window) == false) window++;

		FSC.data_y[0] = FSC.data_y[1];
		FSC.FitSavitzkyGolayToData(window, 3);
		for (i = 0; i < number_of_bins_extended; i++)
		{
//			wxPrintf("FSC,fit = %i %g %g\n", i, FSC.data_y[i], FSC.savitzky_golay_fit[i]);
//			if (FSC.data_y[i] < 0.8) FSC.data_y[i] = FSC.savitzky_golay_fit[i];
			// Make a smooth transition between original FSC curve and smoothed curve
//			else FSC.data_y[i] = FSC.data_y[i] * (1.0 - (1.0 - FSC.data_y[i]) / 0.2) + FSC.savitzky_golay_fit[i] * (1.0 - FSC.data_y[i]) / 0.2;
			// Make a smooth transition between original FSC curve and smoothed curve
//			if (FSC.data_y[i] < 0.5) FSC.data_y[i] = FSC.data_y[i] * (1.0 - fabsf(1.0 - FSC.data_y[i])) + FSC.savitzky_golay_fit[i] * fabsf(1.0 - FSC.data_y[i]);
			FSC.data_y[i] = FSC.data_y[i] * (1.0 - (1.0 - fabsf(FSC.data_y[i]))) + FSC.savitzky_golay_fit[i] * (1.0 - fabsf(FSC.data_y[i]));
			if (FSC.data_y[i] > 1.0) FSC.data_y[i] = 1.0;
			if (FSC.data_y[i] < -1.0) FSC.data_y[i] = -1.0;
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

	part_FSC.ClearData();
	rec_SSNR.ClearData();

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

	part_SSNR.ClearData();

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
		if (sum_double[i] > 0.0 && i > 0)
		{
//			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), float(sum_double[i]) / float(sum_int[i]) / 0.5);
//			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), float(sum_double[i]) / float(sum_int[i]) / 0.25 / 10000 * i);
			// Average sum of CTF^2 from N 2D images contributing to a 3D voxel = float(sum_double[i]) / float(sum_int[i]) = N * 0.5 / 2i
			// The factor of 0.5 is due to the average value for CTF^2.
			// The factor of 1 / 2 is due to the average weight in the trilinear interpolation
			// Divide volume SSNR by float(sum_double[i]) / float(sum_int[i])
			// Factor of 8.0 is due to 8-point trilinear interpolation
//			wxPrintf("i = %i, fsc, sum_int, sum_double, ratio = %g %g %g %g\n", i, FSC.data_y[i], float(sum_int[i]), float(sum_double[i]), float(sum_int[i]) / float(sum_double[i]));
//			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), 8.0 * 4.0 * mask_volume_fraction * fabsf(2.0 * fabsf(FSC.data_y[i]) / (1.00001 - fabsf(FSC.data_y[i])) * float(sum_int[i]) / float(sum_double[i])));
			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), mask_volume_fraction * fabsf(2.0 * fabsf(FSC.data_y[i]) / (1.00001 - fabsf(FSC.data_y[i])) * float(sum_int[i]) / float(sum_double[i])));
//			wxPrintf("x = %g, y = %g\n", FSC.data_x[i], FSC.data_y[i]);
		}
		else
		if (i > 0)
		{
			part_SSNR.AddPoint(pixel_size / float(i) * float(number_of_bins2), 0.0);
		}
		else
		{
//			part_SSNR.AddPoint(pixel_size * powf(float(number_of_bins2), 2), 0.0);
			part_SSNR.AddPoint(0.0, 0.0);
		}
	}

	// Set value at i = 0 to 8 * value at i = 1 to allow reconstructions with non-zero offset
	part_SSNR.data_y[0] = 8.0 * part_SSNR.data_y[1];

	delete [] sum_double;
	delete [] sum_int;
//	wxPrintf("number_of_bins = %i, number_of_bins_extended = %i, ssnr = %i\n", number_of_bins, number_of_bins_extended, part_SSNR.number_of_points);
}

void ResolutionStatistics::ZeroToResolution(float resolution_limit)
{
	MyDebugAssertTrue(number_of_bins > 0, "Statistics not initialized");

	int   number_of_bins2 = 2 * (number_of_bins - 1);
	int   index;

	if (resolution_limit > 0.0)
	{
		index = int(pixel_size / resolution_limit * float(number_of_bins2));
		FSC.ZeroAfterIndex(index);
		part_FSC.ZeroAfterIndex(index);
		part_SSNR.ZeroAfterIndex(index);
		rec_SSNR.ZeroAfterIndex(index);
	}
}

void ResolutionStatistics::PrintStatistics()
{
	MyDebugAssertTrue(FSC.number_of_points > 0, "Resolution statistics have not been fully calculated");

	wxPrintf("\nC                                             Sqrt       Sqrt  \n");
	wxPrintf("C NO.   RESOL  RING_RAD       FSC  Part_FSC Part_SSNR  Rec_SSNR\n");
	for (int i = 1; i < number_of_bins; i++)
	{
		wxPrintf("%5i%8.2f%10.4f%10.4f%10.4f%10.4f%10.4f\n",i + 1, FSC.data_x[i], pixel_size / FSC.data_x[i],
															FSC.data_y[i], part_FSC.data_y[i],
															sqrtf(part_SSNR.data_y[i]), sqrtf(rec_SSNR.data_y[i]));
	}
}

void ResolutionStatistics::WriteStatisticsToFloatArray(float *float_array, int wanted_class) // this is a hack for pre-embo course rush, but i predict it will last for a long time after...
{
	float_array[0] = FSC.number_of_points;
	float_array[1] = wanted_class;
	int position_counter = 2;

	for (int i = 0; i < FSC.number_of_points; i++)
	{
		float_array[position_counter] = FSC.data_x[i];
		position_counter++;
		float_array[position_counter] = FSC.data_y[i];
		position_counter++;
		float_array[position_counter] = part_FSC.data_y[i];
		position_counter++;
		float_array[position_counter] = part_SSNR.data_y[i];
		position_counter++;
		float_array[position_counter] = rec_SSNR.data_y[i];
		position_counter++;
	}
}

void ResolutionStatistics::WriteStatisticsToFile(NumericTextFile &output_statistics_file, float pssnr_division_factor)
{
	MyDebugAssertTrue(FSC.number_of_points > 0, "Resolution statistics have not been fully calculated");

	float temp_float[7];

//	NumericTextFile output_statistics_file(output_file, OPEN_TO_WRITE, 7);
	output_statistics_file.WriteCommentLine("C        SHELL     RESOLUTION    RING_RADIUS            FSC       Part_FSC  Part_SSNR^0.5   Rec_SSNR^0.5");
	for (int i = 1; i < number_of_bins; i++)
	{
		temp_float[0] = float(i+1);
		temp_float[1] = FSC.data_x[i];
		temp_float[2] = pixel_size / FSC.data_x[i];
		temp_float[3] = FSC.data_y[i];
		temp_float[4] = part_FSC.data_y[i];
		temp_float[5] = sqrtf(part_SSNR.data_y[i] / pssnr_division_factor);
		temp_float[6] = sqrtf(rec_SSNR.data_y[i]);

		output_statistics_file.WriteLine(temp_float);
	}
}

void ResolutionStatistics::ReadStatisticsFromFile(wxString input_file)
{
	int i;
	float temp_float[10];
	float resolution;
	int number_of_bins2 = 2 * (number_of_bins - 1);

	if (! DoesFileExist(input_file))
	{
		MyPrintWithDetails("Error: Statistics file not found\n");
		abort();
	}
	NumericTextFile my_statistics(input_file, OPEN_TO_READ);

	FSC.ClearData();
	part_FSC.ClearData();
	part_SSNR.ClearData();
	rec_SSNR.ClearData();

	FSC.AddPoint(0.0, 1.0);
	part_FSC.AddPoint(0.0, 1.0);
	part_SSNR.AddPoint(0.0, 1000.0);
	rec_SSNR.AddPoint(0.0, 1000.0);

	for (i = 1; i <= my_statistics.number_of_lines; i++)
	{
		my_statistics.ReadLine(temp_float);
		resolution = pixel_size / float(i) * float(number_of_bins2);
		if (fabsf(resolution - temp_float[1]) > 0.1)
		{
			MyPrintWithDetails("Statistics file not compatible with input reconstruction\n");
			abort();
		}
		FSC.AddPoint(temp_float[1], temp_float[3]);
		part_FSC.AddPoint(temp_float[1], temp_float[4]);
		part_SSNR.AddPoint(temp_float[1], powf(temp_float[5],2));
		rec_SSNR.AddPoint(temp_float[1], powf(temp_float[6],2));
	}

	for (i = my_statistics.number_of_lines + 1; i <= number_of_bins_extended; i++)
	{
		resolution = pixel_size / float(i) * float(number_of_bins2);
		FSC.AddPoint(resolution, 0.0);
		part_FSC.AddPoint(resolution, 0.0);
		part_SSNR.AddPoint(resolution, 0.0);
		rec_SSNR.AddPoint(resolution, 0.0);
	}
}

void ResolutionStatistics::GenerateDefaultStatistics(float molecular_mass_in_kDa)
{
	int i;
	float resolution;
	float ssnr;
	float fsc;
//	float particle_diameter = 2.0 * powf(3.0 * kDa_to_Angstrom3(molecular_mass_in_kDa) / 4.0 / PI / powf(pixel_size,3) ,1.0 / 3.0);
	float particle_diameter = 2.0 * powf(3.0 * kDa_to_Angstrom3(molecular_mass_in_kDa) / 4.0 / PI,1.0 / 3.0);
	int number_of_bins2 = 2 * (number_of_bins - 1);
	int number_of_bins_extended = int((number_of_bins2 / 2 + 1) * sqrtf(3.0)) + 1;

	FSC.ClearData();
	part_FSC.ClearData();
	part_SSNR.ClearData();
	rec_SSNR.ClearData();

	FSC.AddPoint(0.0, 1.0);
	part_FSC.AddPoint(0.0, 1.0);
	part_SSNR.AddPoint(0.0, 1000.0);
	rec_SSNR.AddPoint(0.0, 1000.0);

	for (i = 1; i <= number_of_bins_extended; i++)
	{
		resolution = pixel_size / float(i) * float(number_of_bins2);
		// Approximate formula derived from part_SSNR curve for VSV-L
		ssnr = powf(molecular_mass_in_kDa,1.5) / 2200.0 * (800.0 * expf(-3.5 * particle_diameter / resolution) + expf(-25.0 / resolution));
		fsc = ssnr / (2.0 + ssnr);
		FSC.AddPoint(resolution, fsc);
		part_FSC.AddPoint(resolution, fsc);
		part_SSNR.AddPoint(resolution, ssnr);
//		wxPrintf("i = %i, res = %g, sqrt(pssnr) = %g\n", i, resolution, sqrtf(ssnr));
		rec_SSNR.AddPoint(resolution, ssnr);
	}
}
