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
	symmetry_symbol = "C1";
	current_resolution_limit = -1.0;
	current_ctf = 0.0;
	current_phi = 0.0;
	current_theta = 0.0;
	current_psi = 0.0;
	current_shift_x = 0.0;
	current_shift_y = 0.0;
	current_mask_radius = 0.0;
	current_mask_falloff = 0.0;
	current_whitening = false;
	current_swap_quadrants = false;
	whitened_projection = false;

//	MyPrintWithDetails("Error: Constructor must be called with volume dimensions and pixel size");
//	abort();
}

ReconstructedVolume::~ReconstructedVolume()
{
	Deallocate();
}


ReconstructedVolume & ReconstructedVolume::operator = (const ReconstructedVolume &other_volume)
{
	*this = &other_volume;
	return *this;
}

ReconstructedVolume & ReconstructedVolume::operator = (const ReconstructedVolume *other_volume)
{
   // Check for self assignment
   if(this != other_volume)
   {
		MyDebugAssertTrue(other_volume->has_been_initialized, "Other volume has not been initialized");

		if (has_been_initialized == true)
		{

			if (density_map.logical_x_dimension != other_volume->density_map.logical_x_dimension || density_map.logical_y_dimension != other_volume->density_map.logical_y_dimension || density_map.logical_z_dimension != other_volume->density_map.logical_z_dimension)
			{
				Deallocate();
				InitWithDimensions(other_volume->density_map.logical_x_dimension, other_volume->density_map.logical_y_dimension, other_volume->density_map.logical_z_dimension, other_volume->pixel_size, other_volume->symmetry_symbol);
			}
		}
		else
		{
			InitWithDimensions(other_volume->density_map.logical_x_dimension, other_volume->density_map.logical_y_dimension, other_volume->density_map.logical_z_dimension, other_volume->pixel_size, other_volume->symmetry_symbol);
		}

		// by here the memory allocation should be OK...

		density_map = other_volume->density_map;

		mask_volume_in_voxels = other_volume->mask_volume_in_voxels;
		molecular_mass_in_kDa = other_volume->molecular_mass_in_kDa;
		statistics = other_volume->statistics;

		has_masked_applied = other_volume->has_masked_applied;
		was_corrected = other_volume->was_corrected;
		has_statistics = other_volume->has_statistics;
		has_been_filtered = other_volume->has_been_filtered;

   }

   return *this;
}

void ReconstructedVolume::Deallocate()
{
	if (has_been_initialized)
	{
		density_map.Deallocate();
		current_projection.Deallocate();
		has_been_initialized = false;
	}
}

void ReconstructedVolume::InitWithReconstruct3D(Reconstruct3D &image_reconstruction, float wanted_pixel_size)
{
	density_map.Allocate(image_reconstruction.logical_x_dimension, image_reconstruction.logical_y_dimension, image_reconstruction.logical_z_dimension, false);
	density_map.object_is_centred_in_box = false;
	pixel_size = wanted_pixel_size;
	symmetry_symbol = image_reconstruction.symmetry_matrices.symmetry_symbol;
	symmetry_matrices.Init(image_reconstruction.symmetry_matrices.symmetry_symbol);
	statistics.Init(wanted_pixel_size);
	has_been_initialized = true;
	current_projection.Allocate(image_reconstruction.logical_x_dimension, image_reconstruction.logical_y_dimension, 1, false);
	current_projection.object_is_centred_in_box = false;
}

void ReconstructedVolume::InitWithDimensions(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString wanted_symmetry_symbol)
{
	density_map.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, false);
	density_map.object_is_centred_in_box = false;
	pixel_size = wanted_pixel_size;
	symmetry_symbol = wanted_symmetry_symbol;
	symmetry_matrices.Init(wanted_symmetry_symbol);
	statistics.Init(wanted_pixel_size);
	has_been_initialized = true;
	current_projection.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);
	current_projection.object_is_centred_in_box = false;
}

void ReconstructedVolume::PrepareForProjections(float resolution_limit, bool approximate_binning)
{
	float binning_factor;
	int fourier_size_x;
	int fourier_size_y;
	int fourier_size_z;

	density_map.Correct3D();
	density_map.ForwardFFT();
	binning_factor = resolution_limit / pixel_size / 2.0;
	if (approximate_binning)
	{
		fourier_size_x = ReturnClosestFactorizedLower(density_map.logical_x_dimension / binning_factor, 3, true);
		fourier_size_y = ReturnClosestFactorizedLower(density_map.logical_y_dimension / binning_factor, 3, true);
		fourier_size_z = ReturnClosestFactorizedLower(density_map.logical_z_dimension / binning_factor, 3, true);
	}
	else
	{
		fourier_size_x = int(density_map.logical_x_dimension / binning_factor + 0.5);
		if (! IsEven(fourier_size_x)) fourier_size_x++;
//		fourier_size_x += 2;
		fourier_size_y = int(density_map.logical_y_dimension / binning_factor + 0.5);
		if (! IsEven(fourier_size_y)) fourier_size_y++;
//		fourier_size_y += 2;
		fourier_size_z = int(density_map.logical_z_dimension / binning_factor + 0.5);
		if (! IsEven(fourier_size_z)) fourier_size_z++;
//		fourier_size_z += 2;
	}
	// The following line assumes that we have a cubic volume
	binning_factor = float(density_map.logical_x_dimension) / float(fourier_size_x);
	if (binning_factor != 1.0 )
	{
		density_map.Resize(fourier_size_x, fourier_size_y, fourier_size_z);
		pixel_size *= binning_factor;
	}
	density_map.SwapRealSpaceQuadrants();
}

void ReconstructedVolume::CalculateProjection(Image &projection, Image &CTF, AnglesAndShifts &angles_and_shifts_of_projection,
		float mask_radius, float mask_falloff, float resolution_limit, bool swap_quadrants, bool whiten)
{
	MyDebugAssertTrue(projection.logical_x_dimension == density_map.logical_x_dimension && projection.logical_y_dimension == density_map.logical_y_dimension, "Error: Images have different sizes");
	MyDebugAssertTrue(CTF.logical_x_dimension == density_map.logical_x_dimension && CTF.logical_y_dimension == density_map.logical_y_dimension, "Error: CTF image has different size");
	MyDebugAssertTrue(projection.logical_z_dimension == 1, "Error: attempting to extract 3D image from 3D reconstruction");
	MyDebugAssertTrue(projection.is_in_memory, "Memory not allocated for receiving image");
	MyDebugAssertTrue(density_map.IsCubic(), "Image volume to project is not cubic");
	MyDebugAssertTrue(! density_map.object_is_centred_in_box, "Image volume quadrants not swapped");

	if (current_phi != angles_and_shifts_of_projection.ReturnPhiAngle() || current_theta != angles_and_shifts_of_projection.ReturnThetaAngle()
		|| current_psi != angles_and_shifts_of_projection.ReturnPsiAngle() || current_resolution_limit != resolution_limit)
	{
		density_map.ExtractSlice(projection, angles_and_shifts_of_projection, resolution_limit);
		current_projection.CopyFrom(&projection);
		current_phi = angles_and_shifts_of_projection.ReturnPhiAngle();
		current_theta = angles_and_shifts_of_projection.ReturnThetaAngle();
		current_psi = angles_and_shifts_of_projection.ReturnPsiAngle();
		current_shift_x = angles_and_shifts_of_projection.ReturnShiftX();
		current_shift_y = angles_and_shifts_of_projection.ReturnShiftY();
		current_resolution_limit = resolution_limit;
		current_ctf = CTF.real_values[10];
		current_mask_radius = mask_radius;
		current_mask_falloff = mask_falloff;
		current_swap_quadrants = swap_quadrants;
		current_whitening = whiten;

		if (whiten)
		{
			projection.Whiten(resolution_limit);
//			projection.PhaseFlipPixelWise(CTF);
//			projection.BackwardFFT();
//			projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
//			projection.ForwardFFT();
		}
		else
		{
			projection.MultiplyPixelWiseReal(CTF);

			if (mask_radius > 0.0)
			{
				projection.BackwardFFT();
				projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
				projection.ForwardFFT();
			}
		}

		projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX() / pixel_size, angles_and_shifts_of_projection.ReturnShiftY() / pixel_size);
		if (swap_quadrants) projection.SwapRealSpaceQuadrants();
	}
	else
	{
		if (current_ctf != CTF.real_values[10] || current_shift_x != angles_and_shifts_of_projection.ReturnShiftX() || current_shift_y != angles_and_shifts_of_projection.ReturnShiftY()
			|| current_mask_radius != mask_radius || current_mask_falloff != mask_falloff || current_swap_quadrants != swap_quadrants || current_whitening != whiten)
		{
			current_shift_x = angles_and_shifts_of_projection.ReturnShiftX();
			current_shift_y = angles_and_shifts_of_projection.ReturnShiftY();
			current_ctf = CTF.real_values[10];
			current_mask_radius = mask_radius;
			current_mask_falloff = mask_falloff;
			current_swap_quadrants = swap_quadrants;
			current_whitening = whiten;

			projection.CopyFrom(&current_projection);

			if (whiten)
			{
				projection.Whiten(resolution_limit);
//				projection.PhaseFlipPixelWise(CTF);
//				projection.BackwardFFT();
//				projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
//				projection.ForwardFFT();
			}
			else
			{
				projection.MultiplyPixelWiseReal(CTF);

				if (mask_radius > 0.0)
				{
					projection.BackwardFFT();
					projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
					projection.ForwardFFT();
				}
			}

			projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX() / pixel_size, angles_and_shifts_of_projection.ReturnShiftY() / pixel_size);
			if (swap_quadrants) projection.SwapRealSpaceQuadrants();
		}
	}

	whitened_projection = whiten;
}

void ReconstructedVolume::Calculate3DSimple(Reconstruct3D &reconstruction)
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

					density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + 1.0);
//																		/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
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

void ReconstructedVolume::Calculate3DOptimal(Reconstruct3D &reconstruction, float pssnr_correction_factor)
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
		wiener_constant[i] = 1.0 / pssnr_correction_factor / statistics.part_SSNR.data_y[i];
	}

	for (k = 0; k <= reconstruction.image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		z = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * reconstruction.image_reconstruction.fourier_voxel_size_z, 2);

		for (j = 0; j <= reconstruction.image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			y = powf(reconstruction.image_reconstruction.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * reconstruction.image_reconstruction.fourier_voxel_size_y, 2);

			for (i = 0; i <= reconstruction.image_reconstruction.physical_upper_bound_complex_x; i++)
			{
				if (reconstruction.ctf_reconstruction[pixel_counter] != 0.0)
				{
					x = powf(i * reconstruction.image_reconstruction.fourier_voxel_size_x, 2);
					frequency_squared = x + y + z;

// compute radius, in units of physical Fourier pixels
					bin = int(sqrtf(frequency_squared) * number_of_bins2);

					if (statistics.part_SSNR.data_y[bin] != 0.0)
					{
						density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter]
									/(reconstruction.ctf_reconstruction[pixel_counter] + wiener_constant[bin]);
//									/ reconstruction.weights_reconstruction[pixel_counter] * reconstruction.number_of_measurements[pixel_counter];
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

void ReconstructedVolume::CosineRingMask(float wanted_inner_mask_radius, float wanted_outer_mask_radius, float wanted_mask_edge)
{
	mask_volume_in_voxels = density_map.CosineRingMask(wanted_inner_mask_radius, wanted_outer_mask_radius, wanted_mask_edge);
	has_masked_applied = true;
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
	density_map.OptimalFilterFSC(statistics.part_FSC);
	was_corrected = true;
}

void ReconstructedVolume::PrintStatistics()
{
	MyDebugAssertTrue(has_statistics, "Resolution statistics have not been fully calculated");

	int number_of_bins = density_map.ReturnSmallestLogicalDimension() / 2 + 1;
	wxPrintf("C                                             Sqrt       Sqrt  \n");
	wxPrintf("C NO.   RESOL  RING_RAD       FSC  Part_FSC Part_SSNR  Rec_SSNR\n");
	for (int i = 1; i < number_of_bins; i++)
	{
		wxPrintf("%5i%8.2f%10.4f%10.4f%10.4f%10.4f%10.4f\n",i + 1, statistics.FSC.data_x[i], pixel_size / statistics.FSC.data_x[i],
															statistics.FSC.data_y[i], statistics.part_FSC.data_y[i],
															sqrtf(statistics.part_SSNR.data_y[i]), sqrtf(statistics.rec_SSNR.data_y[i]));
	}
}

void ReconstructedVolume::WriteStatisticsToFile(NumericTextFile &output_statistics_file)
{
	MyDebugAssertTrue(has_statistics, "Resolution statistics have not been fully calculated");

	int number_of_bins = density_map.ReturnSmallestLogicalDimension() / 2 + 1;
	float temp_float[7];

//	NumericTextFile output_statistics_file(output_file, OPEN_TO_WRITE, 7);
	output_statistics_file.WriteCommentLine("C        SHELL     RESOLUTION    RING_RADIUS            FSC       Part_FSC  Part_SSNR^0.5   Rec_SSNR^0.5");
	for (int i = 1; i < number_of_bins; i++)
	{
		temp_float[0] = float(i+1);
		temp_float[1] = statistics.FSC.data_x[i];
		temp_float[2] = pixel_size / statistics.FSC.data_x[i];
		temp_float[3] = statistics.FSC.data_y[i];
		temp_float[4] = statistics.part_FSC.data_y[i];
		temp_float[5] = sqrtf(statistics.part_SSNR.data_y[i]);
		temp_float[6] = sqrtf(statistics.rec_SSNR.data_y[i]);

		output_statistics_file.WriteLine(temp_float);
	}
}

void ReconstructedVolume::ReadStatisticsFromFile(wxString input_file)
{
	int i;
	float temp_float[10];
	float resolution;
	int number_of_bins2 = density_map.ReturnSmallestLogicalDimension();
	int number_of_bins_extended = int((number_of_bins2 / 2 + 1) * sqrtf(3.0)) + 1;

	if (! DoesFileExist(input_file))
	{
		MyPrintWithDetails("Error: Statistics file not found\n");
		abort();
	}
	NumericTextFile my_statistics(input_file, OPEN_TO_READ);

	statistics.FSC.AddPoint(0.0, 1.0);
	statistics.part_FSC.AddPoint(0.0, 1.0);
	statistics.part_SSNR.AddPoint(0.0, 1000.0);
	statistics.rec_SSNR.AddPoint(0.0, 1000.0);

	for (i = 1; i <= my_statistics.number_of_lines; i++)
	{
		my_statistics.ReadLine(temp_float);
		resolution = pixel_size / float(i) * float(number_of_bins2);
		if (fabsf(resolution - temp_float[1]) > 0.1)
		{
			MyPrintWithDetails("Statistics file not compatible with input reconstruction\n");
			abort();
		}
		statistics.FSC.AddPoint(temp_float[1], temp_float[3]);
		statistics.part_FSC.AddPoint(temp_float[1], temp_float[4]);
		statistics.part_SSNR.AddPoint(temp_float[1], powf(temp_float[5],2));
		statistics.rec_SSNR.AddPoint(temp_float[1], powf(temp_float[6],2));
	}

	for (i = my_statistics.number_of_lines + 1; i <= number_of_bins_extended; i++)
	{
		resolution = pixel_size / float(i) * float(number_of_bins2);
		statistics.FSC.AddPoint(resolution, 0.0);
		statistics.part_FSC.AddPoint(resolution, 0.0);
		statistics.part_SSNR.AddPoint(resolution, 0.0);
		statistics.rec_SSNR.AddPoint(resolution, 0.0);
	}

	has_statistics = true;
}

void ReconstructedVolume::GenerateDefaultStatistics()
{
	MyDebugAssertTrue(molecular_mass_in_kDa != 0.0, "Molecular mass not set");

	int i;
	float resolution;
	float ssnr;
	float fsc;
	float particle_diameter = 2.0 * powf(3.0 * kDa_to_Angstrom3(molecular_mass_in_kDa) / 4.0 / PI / powf(pixel_size,3) ,1.0 / 3.0);
	int number_of_bins2 = density_map.ReturnSmallestLogicalDimension();
	int number_of_bins_extended = int((number_of_bins2 / 2 + 1) * sqrtf(3.0)) + 1;

	statistics.FSC.AddPoint(0.0, 1.0);
	statistics.part_FSC.AddPoint(0.0, 1.0);
	statistics.part_SSNR.AddPoint(0.0, 1000.0);
	statistics.rec_SSNR.AddPoint(0.0, 1000.0);

	for (i = 1; i <= number_of_bins_extended; i++)
	{
		resolution = pixel_size / float(i) * float(number_of_bins2);
		// Approximate formula derived from part_SSNR curve for VSV-L
		ssnr = powf(molecular_mass_in_kDa,1.5) / 35.0 * (800.0 * expf(-3.5 * particle_diameter / resolution) + expf(-25.0 / resolution));
		fsc = ssnr / (2.0 + ssnr);
		statistics.FSC.AddPoint(resolution, fsc);
		statistics.part_FSC.AddPoint(resolution, fsc);
		statistics.part_SSNR.AddPoint(resolution, ssnr);
//		wxPrintf("i = %i, res = %g, sqrt(pssnr) = %g\n", i, resolution, sqrtf(ssnr));
		statistics.rec_SSNR.AddPoint(resolution, ssnr);
	}

	has_statistics = true;
}
