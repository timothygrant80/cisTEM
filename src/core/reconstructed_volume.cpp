#include "core_headers.h"

ReconstructedVolume::ReconstructedVolume(float wanted_molecular_mass_in_kDa)
{
	has_been_initialized = false;
	mask_volume_in_voxels = 0.0;
	molecular_mass_in_kDa = wanted_molecular_mass_in_kDa;
	has_masked_applied = false;
	was_corrected = false;
//	has_statistics = false;
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
		has_masked_applied = other_volume->has_masked_applied;
		was_corrected = other_volume->was_corrected;
//		has_statistics = other_volume->has_statistics;
		has_been_filtered = other_volume->has_been_filtered;
		pixel_size = other_volume->pixel_size;
		symmetry_symbol = other_volume->symmetry_symbol;
		current_resolution_limit = other_volume->current_resolution_limit;
		current_ctf = other_volume->current_ctf;
		current_phi = other_volume->current_phi;
		current_theta = other_volume->current_theta;
		current_psi = other_volume->current_psi;
		current_shift_x = other_volume->current_shift_x;
		current_shift_y = other_volume->current_shift_y;
		current_whitening = other_volume->current_whitening;
		current_swap_quadrants = other_volume->current_swap_quadrants;
		whitened_projection = other_volume->whitened_projection;
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
//	statistics.Init(wanted_pixel_size);
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
//	statistics.Init(wanted_pixel_size);
	has_been_initialized = true;
	current_projection.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);
	current_projection.object_is_centred_in_box = false;
}

//void ReconstructedVolume::PrepareForProjections(float resolution_limit, bool approximate_binning, bool apply_binning)
void ReconstructedVolume::PrepareForProjections(float low_resolution_limit, float high_resolution_limit, bool approximate_binning, bool apply_binning)
{
	int fourier_size_x;
	int fourier_size_y;
	int fourier_size_z;
	float binning_factor;

	density_map.CorrectSinc();
//	if (mask_radius > 0.0) density_map.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
//	else density_map.CosineMask(0.45 * density_map.logical_x_dimension, 10.0 / pixel_size, false, true, 0.0);
//	density_map.CorrectSinc();
//	if (mask_radius > 0.0) density_map.AddConstant(- density_map.ReturnAverageOfRealValues(mask_radius, true));
//	else density_map.AddConstant(- density_map.ReturnAverageOfRealValues(0.45 * density_map.logical_x_dimension, true));
	density_map.ForwardFFT();

	if (apply_binning && high_resolution_limit > 0.0)
	{
		binning_factor = high_resolution_limit / pixel_size / 2.0;
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
//			fourier_size_x += 2;
			fourier_size_y = int(density_map.logical_y_dimension / binning_factor + 0.5);
			if (! IsEven(fourier_size_y)) fourier_size_y++;
//			fourier_size_y += 2;
			fourier_size_z = int(density_map.logical_z_dimension / binning_factor + 0.5);
			if (! IsEven(fourier_size_z)) fourier_size_z++;
//			fourier_size_z += 2;
		}
		// The following line assumes that we have a cubic volume
		binning_factor = float(density_map.logical_x_dimension) / float(fourier_size_x);
		if (binning_factor != 1.0 )
		{
			density_map.Resize(fourier_size_x, fourier_size_y, fourier_size_z);
			pixel_size *= binning_factor;
		}
	}

	density_map.SwapRealSpaceQuadrants();

	if (high_resolution_limit > 0.0) density_map.CosineMask(pixel_size / high_resolution_limit, pixel_size / 100.0);
	if (low_resolution_limit > 0.0) density_map.CosineMask(pixel_size / low_resolution_limit, pixel_size / 100.0, true);
}

void ReconstructedVolume::CalculateProjection(Image &projection, Image &CTF, AnglesAndShifts &angles_and_shifts_of_projection,
		float mask_radius, float mask_falloff, float resolution_limit, bool swap_quadrants, bool apply_shifts, bool whiten, bool apply_ctf, bool abolute_ctf)
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
//			var_A = projection.ReturnSumOfSquares();
//			projection.MultiplyByConstant(sqrtf(projection.number_of_real_space_pixels / var_A));
			projection.Whiten(resolution_limit);
//			projection.PhaseFlipPixelWise(CTF);
//			projection.BackwardFFT();
//			projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
//			projection.ForwardFFT();
		}
		if (apply_ctf)
		{
//			projection.BackwardFFT();
//			projection.SetToConstant(1.0);
//			projection.real_values[0] = 1.0;
//			projection.CosineMask(20.0, 1.0, false, true, 0.0);
//			projection.ForwardFFT();
//			projection.MultiplyPixelWiseReal(CTF, false);
//			projection.SwapRealSpaceQuadrants();
//			projection.QuickAndDirtyWriteSlice("proj_20_flipped.mrc", 1);
//			exit(0);
			projection.MultiplyPixelWiseReal(CTF, abolute_ctf);

			if (mask_radius > 0.0)
			{
				projection.BackwardFFT();
				projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
				projection.ForwardFFT();
			}
		}

		if (apply_shifts) projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX() / pixel_size, angles_and_shifts_of_projection.ReturnShiftY() / pixel_size);
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
//				var_A = projection.ReturnSumOfSquares();
//				projection.MultiplyByConstant(sqrtf(projection.number_of_real_space_pixels / var_A));
				projection.Whiten(resolution_limit);
//				projection.PhaseFlipPixelWise(CTF);
//				projection.BackwardFFT();
//				projection.ZeroFloatOutside(0.5 * projection.logical_x_dimension - 1.0);
//				projection.ForwardFFT();
			}
			if (apply_ctf)
			{
				projection.MultiplyPixelWiseReal(CTF, abolute_ctf);

				if (mask_radius > 0.0)
				{
					projection.BackwardFFT();
					projection.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
					projection.ForwardFFT();
				}
			}

			if (apply_shifts) projection.PhaseShift(angles_and_shifts_of_projection.ReturnShiftX() / pixel_size, angles_and_shifts_of_projection.ReturnShiftY() / pixel_size);
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

					// Use 100.0 as Wiener constant for 3DSimple since the SSNR near the resolution limit of high-resolution reconstructions is often close to 0.01.
					// The final result is not strongly dependent on this constant and therefore, a value of 100.0 is a sufficiently good estimate.
					density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter] / (reconstruction.ctf_reconstruction[pixel_counter] + 100.0f);
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

void ReconstructedVolume::Calculate3DOptimal(Reconstruct3D &reconstruction, ResolutionStatistics &statistics)
{
	MyDebugAssertTrue(has_been_initialized, "Error: reconstruction volume has not been initialized");
//	MyDebugAssertTrue(has_statistics, "Error: 3D statistics have not been calculated");
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
	float particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
	float pssnr_correction_factor = float(density_map.ReturnVolumeInRealSpace()) / (kDa_to_Angstrom3(molecular_mass_in_kDa) / powf(pixel_size,3))
			* particle_area_in_pixels / float(density_map.logical_x_dimension * density_map.logical_y_dimension);


	int number_of_bins2 = reconstruction.image_reconstruction.ReturnSmallestLogicalDimension();

	float *wiener_constant = new float[statistics.part_SSNR.number_of_points];

	reconstruction.CompleteEdges();

// Now do the division by the CTF volume
	pixel_counter = 0;

	for (i = 0; i < statistics.part_SSNR.number_of_points; i++)
	{
		if (statistics.part_SSNR.data_y[i] > 0.0)
		{
//			wiener_constant[i] = 1.0 / statistics.part_SSNR.data_y[i];
			wiener_constant[i] = 1.0 / pssnr_correction_factor / statistics.part_SSNR.data_y[i];
//			wiener_constant[i] = 32.0 / pssnr_correction_factor / statistics.part_SSNR.data_y[i];
		}
		else wiener_constant[i] = 0.0;
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
	return density_map.CorrectSinc(mask_radius);
}

void ReconstructedVolume::OptimalFilter(ResolutionStatistics &statistics)
{
	density_map.OptimalFilterFSC(statistics.part_FSC);
	was_corrected = true;
}

void ReconstructedVolume::FinalizeSimple(Reconstruct3D &reconstruction, int &original_box_size, float &original_pixel_size, float &pixel_size,
		float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff, wxString &output_volume)
{
	int intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
	int box_size = reconstruction.logical_x_dimension;
	MRCFile output_file;

	InitWithReconstruct3D(reconstruction, original_pixel_size);
	Calculate3DSimple(reconstruction);
	density_map.SwapRealSpaceQuadrants();
	if (intermediate_box_size != box_size)
	{
		density_map.BackwardFFT();
		density_map.Resize(intermediate_box_size, intermediate_box_size,
				intermediate_box_size, density_map.ReturnAverageOfRealValuesOnEdges());
		density_map.ForwardFFT();
	}
	if (pixel_size != original_pixel_size) density_map.Resize(original_box_size, original_box_size, original_box_size);
	density_map.BackwardFFT();
	CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
//	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	output_file.OpenFile(output_volume.ToStdString(), true);
	density_map.WriteSlices(&output_file,1,density_map.logical_z_dimension);
	output_file.SetPixelSize(original_pixel_size);
	output_file.CloseFile();
	density_map.ForwardFFT();
}

void ReconstructedVolume::FinalizeOptimal(Reconstruct3D &reconstruction, Image &density_map_1, Image &density_map_2,
		float &original_pixel_size, float &pixel_size, float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff,
		bool center_mass, wxString &output_volume, NumericTextFile &output_statistics, ResolutionStatistics *copy_of_statistics)
{
	int original_box_size = density_map_1.logical_x_dimension;
	int intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
	int box_size = reconstruction.logical_x_dimension;
	float binning_factor = pixel_size / original_box_size;
	float particle_area_in_pixels;
	float mask_volume_fraction;
	float resolution_limit = 0.0;
	float temp_float;
	MRCFile output_file;
	ResolutionStatistics statistics(original_pixel_size, original_box_size);
	ResolutionStatistics cropped_statistics(pixel_size, box_size);
	ResolutionStatistics temp_statistics(pixel_size, intermediate_box_size);
	Peak center_of_mass;
	wxChar symmetry_type;
	long symmetry_number;

	if (pixel_size != original_pixel_size) resolution_limit = 2.0 * pixel_size;

	statistics.CalculateFSC(density_map_1, density_map_2, true);
	density_map_1.Deallocate();
	density_map_2.Deallocate();
	InitWithReconstruct3D(reconstruction, original_pixel_size);
	statistics.CalculateParticleFSCandSSNR(mask_volume_in_voxels, molecular_mass_in_kDa);
	particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
	mask_volume_fraction = mask_volume_in_voxels / particle_area_in_pixels / original_box_size;
	if (intermediate_box_size != box_size && binning_factor != 1.0)
	{
		temp_statistics.CopyFrom(statistics);
		cropped_statistics.ResampleFrom(temp_statistics);
	}
	else
	{
		cropped_statistics.CopyFrom(statistics);
	}
	cropped_statistics.CalculateParticleSSNR(reconstruction.image_reconstruction, reconstruction.ctf_reconstruction, mask_volume_fraction);
	if (intermediate_box_size != box_size && binning_factor != 1.0)
	{
		temp_statistics.ResampleParticleSSNR(cropped_statistics);
		statistics.CopyParticleSSNR(temp_statistics);
	}
	else
	{
		statistics.CopyParticleSSNR(cropped_statistics);
	}
	statistics.ZeroToResolution(resolution_limit);
	statistics.PrintStatistics();

	statistics.WriteStatisticsToFile(output_statistics);
	if (copy_of_statistics != NULL)
	{
		copy_of_statistics->Init(original_pixel_size, original_box_size);
		copy_of_statistics->CopyFrom(statistics);
	}

	Calculate3DOptimal(reconstruction, cropped_statistics);
	density_map.SwapRealSpaceQuadrants();
	if (intermediate_box_size != box_size)
	{
		density_map.BackwardFFT();
		Correct3D();
		// Scaling factor needed to compensate for FFT normalization for different box sizes
		density_map.MultiplyByConstant(float(intermediate_box_size) / float(box_size));
		density_map.Resize(intermediate_box_size, intermediate_box_size,
				intermediate_box_size, density_map.ReturnAverageOfRealValuesOnEdges());
		density_map.ForwardFFT();
	}
	if (pixel_size != original_pixel_size) density_map.Resize(original_box_size, original_box_size, original_box_size);
	density_map.CosineMask(0.5, 1.0 / 20.0);
	density_map.BackwardFFT();
	if (intermediate_box_size == box_size) Correct3D();
	CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
	if (center_mass)
	{
		temp_float = density_map.ReturnAverageOfRealValuesOnEdges();
		center_of_mass = density_map.CenterOfMass(temp_float, true);
		symmetry_type = symmetry_symbol.Capitalize()[0];
		if (symmetry_type == 'C' && center_of_mass.value > 0.0)
		{
			symmetry_symbol.Mid(1).ToLong(&symmetry_number);
			if (symmetry_number < 2) density_map.RealSpaceIntegerShift(density_map.physical_address_of_box_center_x - int(center_of_mass.x),
					density_map.physical_address_of_box_center_y - int(center_of_mass.y), density_map.physical_address_of_box_center_z - int(center_of_mass.z));
			else density_map.RealSpaceIntegerShift(0, 0, density_map.physical_address_of_box_center_z - int(center_of_mass.z));
		}
	}
	output_file.OpenFile(output_volume.ToStdString(), true);
	density_map.WriteSlices(&output_file,1,density_map.logical_z_dimension);
	output_file.SetPixelSize(original_pixel_size);
	EmpiricalDistribution density_distribution;
	density_map.UpdateDistributionOfRealValues(&density_distribution);
	output_file.SetDensityStatistics(density_distribution.GetMinimum(), density_distribution.GetMaximum(), density_distribution.GetSampleMean(), sqrtf(density_distribution.GetSampleVariance()));
	output_file.CloseFile();
}

void ReconstructedVolume::FinalizeML(Reconstruct3D &reconstruction, Image &density_map_1, Image &density_map_2,
		float &original_pixel_size, float &pixel_size, float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff,
		wxString &output_volume, NumericTextFile &output_statistics, ResolutionStatistics *copy_of_statistics)
{
	int original_box_size = density_map_1.logical_x_dimension;
	int intermediate_box_size = myroundint(original_box_size / pixel_size * original_pixel_size);
	int box_size = reconstruction.logical_x_dimension;
	float binning_factor = pixel_size / original_box_size;
	float particle_area_in_pixels;
	float mask_volume_fraction;
	float resolution_limit = 0.0;
	MRCFile output_file;
	ResolutionStatistics statistics(original_pixel_size, original_box_size);
	ResolutionStatistics cropped_statistics(pixel_size, box_size);
	ResolutionStatistics temp_statistics(pixel_size, intermediate_box_size);

	if (pixel_size != original_pixel_size) resolution_limit = 2.0 * pixel_size;

	InitWithReconstruct3D(reconstruction, original_pixel_size);
	statistics.CalculateFSC(density_map_1, density_map_2, true);
	statistics.CalculateParticleFSCandSSNR(mask_volume_in_voxels, molecular_mass_in_kDa);
	particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
	mask_volume_fraction = mask_volume_in_voxels / particle_area_in_pixels / original_box_size;
	if (intermediate_box_size != box_size && binning_factor != 1.0)
	{
		temp_statistics.CopyFrom(statistics);
		cropped_statistics.ResampleFrom(temp_statistics);
	}
	else
	{
		cropped_statistics.CopyFrom(statistics);
	}
	cropped_statistics.CalculateParticleSSNR(reconstruction.image_reconstruction, reconstruction.ctf_reconstruction, mask_volume_fraction);
	if (intermediate_box_size != box_size && binning_factor != 1.0)
	{
		temp_statistics.ResampleParticleSSNR(cropped_statistics);
		statistics.CopyParticleSSNR(temp_statistics);
	}
	else
	{
		statistics.CopyParticleSSNR(cropped_statistics);
	}
	statistics.ZeroToResolution(resolution_limit);
	statistics.PrintStatistics();

	statistics.WriteStatisticsToFile(output_statistics);
	if (copy_of_statistics != NULL)
	{
		copy_of_statistics->Init(original_pixel_size, original_box_size);
		copy_of_statistics->CopyFrom(statistics);
	}

	// This would have to be added back if FinalizeML was used
//	if (reconstruction.images_processed > 0) reconstruction.noise_power_spectrum->MultiplyByConstant(1.0 / reconstruction.images_processed);
	Calculate3DML(reconstruction);
	density_map.SwapRealSpaceQuadrants();
	if (intermediate_box_size != box_size)
	{
		density_map.BackwardFFT();
		Correct3D();
		density_map.Resize(intermediate_box_size, intermediate_box_size,
				intermediate_box_size, density_map.ReturnAverageOfRealValuesOnEdges());
		density_map.ForwardFFT();
	}
	if (pixel_size != original_pixel_size) density_map.Resize(original_box_size, original_box_size, original_box_size);
	density_map.CosineMask(0.5, 1.0 / 20.0);
	density_map.BackwardFFT();
	if (intermediate_box_size == box_size) Correct3D();
	CosineRingMask(inner_mask_radius / original_pixel_size, outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size);
	output_file.OpenFile(output_volume.ToStdString(), true);
	density_map.WriteSlices(&output_file,1,density_map.logical_z_dimension);
	output_file.SetPixelSize(original_pixel_size);
	output_file.CloseFile();
}

// This calculation is missing the scaling of the CTF2 sums by sigma2 and therefore will not work properly
void ReconstructedVolume::Calculate3DML(Reconstruct3D &reconstruction)
{
	MyDebugAssertTrue(has_been_initialized, "Error: reconstruction volume has not been initialized");
//	MyDebugAssertTrue(has_statistics, "Error: 3D statistics have not been calculated");
	// This would have to be added back if FinalizeML was used
//	MyDebugAssertTrue(int((reconstruction.image_reconstruction.ReturnSmallestLogicalDimension() / 2 + 1) * sqrtf(3.0)) + 1 == reconstruction.signal_power_spectrum->number_of_points, "Error: signal_power_spectrum table incompatible with volume");

	int i;
	int j;
	int k;
	int bin;

	long pixel_counter = 0;

	float x;
	float y;
	float z;
	float frequency_squared;
//	float particle_area_in_pixels = statistics.kDa_to_area_in_pixel(molecular_mass_in_kDa);
//	float pssnr_correction_factor = density_map.ReturnVolumeInRealSpace() / (kDa_to_Angstrom3(molecular_mass_in_kDa) / powf(pixel_size,3))
//			* particle_area_in_pixels / density_map.logical_x_dimension / density_map.logical_y_dimension;


	int number_of_bins2 = reconstruction.image_reconstruction.ReturnSmallestLogicalDimension();

	// This would have to be added back if FinalizeML was used
//	float *wiener_constant = new float[reconstruction.signal_power_spectrum->number_of_points];
	float *wiener_constant = new float[314];

	reconstruction.CompleteEdges();

// Now do the division by the CTF volume
	pixel_counter = 0;

	// This would have to be added back if FinalizeML was used
//	for (i = 0; i < reconstruction.signal_power_spectrum->number_of_points; i++)
//	{
//		if (reconstruction.signal_power_spectrum->data_y[i] != 0.0) wiener_constant[i] = reconstruction.noise_power_spectrum->data_y[i] / reconstruction.signal_power_spectrum->data_y[i];
//		else wiener_constant[i] = - 1.0;
//		wxPrintf("noise, signal, filter = %i %g %g %g\n", i, reconstruction.noise_power_spectrum->data_y[i], reconstruction.signal_power_spectrum->data_y[i], wiener_constant[i]);
//	}

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

					if (wiener_constant[bin] >= 0.0)
					{
						density_map.complex_values[pixel_counter] = reconstruction.image_reconstruction.complex_values[pixel_counter]
									/(reconstruction.ctf_reconstruction[pixel_counter] + wiener_constant[bin]);
//						wxPrintf("i j k = %i %i %i bin = %i pow = %g rec = %g ctf2 = %g filt = %g final = %g\n",i,j,k,bin,signal_power_spectrum.data_y[bin],
//								cabsf(reconstruction.image_reconstruction.complex_values[pixel_counter]),
//								reconstruction.ctf_reconstruction[pixel_counter],wiener_constant[bin], cabsf(density_map.complex_values[pixel_counter]));
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
