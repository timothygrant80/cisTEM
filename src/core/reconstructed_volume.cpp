#include "core_headers.h"

ReconstructedVolume::ReconstructedVolume()
{
	has_been_initialized = false;
	mask_volume_in_voxels = 0.0;
	molecular_mass_in_kDa = 0.0;
	has_masked_applied = false;
	was_corrected = false;
	has_statistics = false;
	has_been_filtered = false;
	pixel_size = 0.0;

//	MyPrintWithDetails("Error: Constructor must be called with volume dimensions and pixel size");
//	abort();
}

void ReconstructedVolume::Init(Reconstruct3d &image_reconstruction, float wanted_pixel_size)
{
	density_map.Allocate(image_reconstruction.logical_x_dimension, image_reconstruction.logical_y_dimension, image_reconstruction.logical_z_dimension, false);
	density_map.object_is_centred_in_box = false;
	pixel_size = wanted_pixel_size;
//	symmetry = symmetry;
	statistics.Init(wanted_pixel_size);
	has_been_initialized = true;
}

void ReconstructedVolume::Calculate3DSimple(Reconstruct3d &reconstruction)
{
	MyDebugAssertTrue(has_been_initialized, "Error: reconstruction volume has not been initialized");

	int i;
	int j;
	int k;

	long pixel_counter = 0;

	reconstruction.CompleteEdges();

// Now do the division by the CTF volume
	for (k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		for (j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			for (i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++)
			{
				if (reconstruction.ctf_reconstruction[pixel_counter] != 0.0)
				{
//					if (reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_X(i)==40 && reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j)==20 && reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k)==4)
//					{
//						wxPrintf("counter = %li, image = %g, ctf = %g, weights = %g, n = %i\n", pixel_counter, cabsf(reconstruction.image_reconstruction.complex_values[pixel_counter]),
//						reconstruction.ctf_reconstruction[pixel_counter], reconstruction.weights_reconstruction[pixel_counter], reconstruction.number_of_measurements[pixel_counter]);
//					}

					density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + 1.0)
																		/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
				}
				else
				{
					density_map.complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}
}

void ReconstructedVolume::Calculate3DOptimal(Reconstruct3d &reconstruction, float pssnr_correction_factor)
{
	MyDebugAssertTrue(has_been_initialized, "Error: reconstruction volume has not been initialized");
	MyDebugAssertTrue(has_statistics, "Error: 3D statistics have not been calculated");
	MyDebugAssertTrue(int((reconstruction.image_reconstruction.ReturnSmallestLogicalDimension() / 2 + 1) * sqrtf(3.0)) + 1 == statistics.part_SSNR.number_of_points, "Error: part_SSNR table incompatible with volume");

	int i;
	int j;
	int k;
	int bin;

	long pixel_counter = 0;

	float x;
	float y;
	float z;
	float frequency_squared;

	int number_of_bins2 = reconstruction.image_reconstruction.ReturnSmallestLogicalDimension();

	float *wiener_constant = new float[statistics.part_SSNR.number_of_points];

	reconstruction.CompleteEdges();

// Now do the division by the CTF volume
	pixel_counter = 0;

	for (i = 0; i < statistics.part_SSNR.number_of_points; i++)
	{
		wiener_constant[i] = pssnr_correction_factor / statistics.part_SSNR.data_y[i];
	}

	for (k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		z = pow(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * reconstruction.image_reconstruction.fourier_voxel_size_z, 2);

		for (j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			y = pow(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * reconstruction.image_reconstruction.fourier_voxel_size_y, 2);

			for (i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++)
			{
				if (reconstruction.ctf_reconstruction[pixel_counter] != 0.0)
				{
					x = pow(i * reconstruction.image_reconstruction.fourier_voxel_size_x, 2);
					frequency_squared = x + y + z;

// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					if (statistics.part_SSNR.data_y[bin] != 0.0)
					{
//						if (bin == 50) wxPrintf("n = %i, ctf_sum = %g, pssnr = %g\n", reconstruction.number_of_measurements[pixel_counter], reconstruction.ctf_reconstruction[pixel_counter], statistics.part_SSNR.data_y[bin]);
						density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter]
									/(reconstruction.ctf_reconstruction[pixel_counter] + wiener_constant[bin])
									/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
					}
					else
					{
						density_map.complex_values[pixel_counter] = 0.0;
					}
				}
				else
				{
					density_map.complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}

	delete [] wiener_constant;
}

void ReconstructedVolume::CosineMask(float wanted_mask_radius, float wanted_mask_edge)
{
	mask_volume_in_voxels = density_map.CosineMask(wanted_mask_radius, wanted_mask_edge);
	has_masked_applied = true;
}

float ReconstructedVolume::Correct3D(float mask_radius)
{
	was_corrected = true;
	return density_map.Correct3D(mask_radius);
}

void ReconstructedVolume::OptimalFilter()
{
	density_map.OptimalFilter(statistics.part_FSC);
	was_corrected = true;
}

void ReconstructedVolume::PrintStatistics()
{
	MyDebugAssertTrue(has_statistics, "Resolution statistics have not been fully calculated");

	int number_of_bins = int(statistics.FSC.number_of_points / sqrtf(3.0));
	wxPrintf("                                             Sqrt       Sqrt  \n");
	wxPrintf(" NO.   RESOL  RING_RAD       FSC  Part_FSC Part_SSNR  Rec_SSNR\n");
	for (int i = 1; i < number_of_bins; i++)
	{
		wxPrintf("%4i%8.2f%10.4f%10.4f%10.4f%10.4f%10.4f\n",i + 1, statistics.FSC.data_x[i], pixel_size / statistics.FSC.data_x[i],
															statistics.FSC.data_y[i], statistics.part_FSC.data_y[i],
															sqrt(statistics.part_SSNR.data_y[i]), sqrt(statistics.rec_SSNR.data_y[i]));
	}
}
