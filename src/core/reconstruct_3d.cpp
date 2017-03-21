#include "core_headers.h"

Reconstruct3D::Reconstruct3D(float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_score, float wanted_score_weights_conversion)
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;
	original_x_dimension = 0;
	original_y_dimension = 0;
	original_z_dimension = 0;

	images_processed = 0;

	pixel_size = wanted_pixel_size;
	original_pixel_size = 0.0;
	average_occupancy = wanted_average_occupancy;
	average_score = wanted_average_score;
	score_weights_conversion = wanted_score_weights_conversion;

	ctf_reconstruction = NULL;

	symmetry_matrices.Init("C1");
	edge_terms_were_added = false;
}

Reconstruct3D::Reconstruct3D(float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_score, float wanted_score_weights_conversion, wxString wanted_symmetry)
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;
	original_x_dimension = 0;
	original_y_dimension = 0;
	original_z_dimension = 0;

	images_processed = 0;

	pixel_size = wanted_pixel_size;
	original_pixel_size = wanted_pixel_size;
	average_occupancy = wanted_average_occupancy;
	average_score = wanted_average_score;
	score_weights_conversion = wanted_score_weights_conversion;

	ctf_reconstruction = NULL;

	symmetry_matrices.Init(wanted_symmetry);
	edge_terms_were_added = false;
}

Reconstruct3D::Reconstruct3D(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_score, float wanted_score_weights_conversion, wxString wanted_symmetry)
{
	ctf_reconstruction = NULL;
	Init(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, wanted_pixel_size, wanted_average_occupancy, wanted_average_score, wanted_score_weights_conversion);

	symmetry_matrices.Init(wanted_symmetry);
}

Reconstruct3D::~Reconstruct3D()
{
	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
}

void Reconstruct3D::Init(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_score, float wanted_score_weights_conversion)
{
	logical_x_dimension = wanted_logical_x_dimension;
	logical_y_dimension = wanted_logical_y_dimension;
	logical_z_dimension = wanted_logical_z_dimension;
	original_x_dimension = wanted_logical_x_dimension;
	original_y_dimension = wanted_logical_y_dimension;
	original_z_dimension = wanted_logical_z_dimension;

	images_processed = 0;

	pixel_size = wanted_pixel_size;
	original_pixel_size = wanted_pixel_size;
	average_occupancy = wanted_average_occupancy;
	average_score = wanted_average_score;
	score_weights_conversion = wanted_score_weights_conversion;

	image_reconstruction.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, false);

	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
	ctf_reconstruction = new float[image_reconstruction.real_memory_allocated / 2];
	ZeroFloatArray(ctf_reconstruction, image_reconstruction.real_memory_allocated / 2);

	image_reconstruction.object_is_centred_in_box = false;

	image_reconstruction.SetToConstant(0.0);

//	current_ctf_image.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);

	edge_terms_were_added = false;
}

void Reconstruct3D::InsertSliceWithCTF(Particle &particle_to_insert)
{
	MyDebugAssertTrue(particle_to_insert.particle_image->logical_x_dimension == logical_x_dimension && particle_to_insert.particle_image->logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(particle_to_insert.particle_image->logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(particle_to_insert.ctf_image->is_in_memory, "Memory not allocated for current_ctf_image");
	MyDebugAssertTrue(particle_to_insert.particle_image->IsSquare(), "Image must be square");

	float particle_weight;

//	if (particle_to_insert.particle_image->is_in_real_space == true)
//	{
//		particle_to_insert.particle_image->ForwardFFT();
//		particle_to_insert.particle_image->SwapRealSpaceQuadrants();
//	}

	particle_to_insert.particle_image->PhaseShift(-particle_to_insert.alignment_parameters.ReturnShiftX() / particle_to_insert.pixel_size, -particle_to_insert.alignment_parameters.ReturnShiftY() / particle_to_insert.pixel_size);
	particle_weight = particle_to_insert.particle_occupancy / particle_to_insert.parameter_average[12] / powf(particle_to_insert.sigma_noise / particle_to_insert.parameter_average[14],2);
//	particle_weight = particle_to_insert.particle_occupancy / 100.0 / powf(particle_to_insert.sigma_noise,2);

	images_processed++;

	if (particle_weight > 0.0)
	{
		int i;
		int j;
		int k;

		long pixel_counter = 0;

		float x_coordinate_2d;
		float y_coordinate_2d;
		float z_coordinate_2d = 0.0;

		float x_coordinate_3d;
		float y_coordinate_3d;
		float z_coordinate_3d;

		float y_coord_sq;

		RotationMatrix temp_matrix;

		float frequency_squared;
		float azimuth;
		float weight;
		float score_weights_conversion4 = score_weights_conversion / powf(pixel_size,2) * 0.25;
		float weight_conversion = (particle_to_insert.particle_score - average_score) * score_weights_conversion4;

		// Make sure that the exponentiated conversion factor will not lead to an overflow
		if (weight_conversion > 60.0) {weight_conversion = 60.0;};
		if (weight_conversion < -60.0) {weight_conversion = -60.0;};

//		if (particle_to_insert.ctf_parameters.IsAlmostEqualTo(&current_ctf, 50.0 / pixel_size) == false)
//		// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
//		{
//			current_ctf = particle_to_insert.ctf_parameters;
//			current_ctf_image->CalculateCTFImage(current_ctf);
//		}

	// Now insert into 3D arrays
		for (j = particle_to_insert.particle_image->logical_lower_bound_complex_y; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			y_coord_sq = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
			for (i = 1; i <= particle_to_insert.particle_image->logical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i;
				frequency_squared = powf(x_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_x, 2) + y_coord_sq;
				weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
				if (weight < 0.0) weight = 0.0;
//				weight = particle_weight * expf(weight_conversion * frequency_squared);
				particle_to_insert.alignment_parameters.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
				pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], particle_to_insert.ctf_image->complex_values[pixel_counter], weight);
			}
		}
	// Now deal with special case of i = 0
		for (j = 0; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			x_coordinate_2d = 0;
			frequency_squared = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
			weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
			if (weight < 0.0) weight = 0.0;
//			weight = particle_weight * expf(weight_conversion * frequency_squared);
			particle_to_insert.alignment_parameters.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], particle_to_insert.ctf_image->complex_values[pixel_counter], weight);
		}

		if (symmetry_matrices.number_of_matrices > 1)
		{
			for (k = 1; k < symmetry_matrices.number_of_matrices; k++)
			{
				for (j = particle_to_insert.particle_image->logical_lower_bound_complex_y; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					y_coord_sq = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
					for (i = 1; i <= particle_to_insert.particle_image->logical_upper_bound_complex_x; i++)
					{
						x_coordinate_2d = i;
						frequency_squared = powf(x_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_x, 2) + y_coord_sq;
						weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
						if (weight < 0.0) weight = 0.0;
//						weight = particle_weight * expf(weight_conversion * frequency_squared);
						temp_matrix = symmetry_matrices.rot_mat[k] * particle_to_insert.alignment_parameters.euler_matrix;
						temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(i,j,0);
						AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], particle_to_insert.ctf_image->complex_values[pixel_counter], weight);
					}
				}
			// Now deal with special case of i = 0
				for (j = 0; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					x_coordinate_2d = 0;
					frequency_squared = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
					weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
					if (weight < 0.0) weight = 0.0;
//					weight = particle_weight * expf(weight_conversion * frequency_squared);
					temp_matrix = symmetry_matrices.rot_mat[k] * particle_to_insert.alignment_parameters.euler_matrix;
					temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(0,j,0);
					AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], particle_to_insert.ctf_image->complex_values[pixel_counter], weight);
				}
			}
		}
	}
}

void Reconstruct3D::InsertSliceNoCTF(Particle &particle_to_insert)
{
	MyDebugAssertTrue(particle_to_insert.particle_image->logical_x_dimension == logical_x_dimension && particle_to_insert.particle_image->logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(particle_to_insert.particle_image->logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(particle_to_insert.particle_image->IsSquare(), "Image must be square");

	float particle_weight;

	if (particle_to_insert.particle_image->is_in_real_space == true)
	{
		particle_to_insert.particle_image->ForwardFFT();
		particle_to_insert.particle_image->SwapRealSpaceQuadrants();
	}

	particle_to_insert.particle_image->PhaseShift(-particle_to_insert.alignment_parameters.ReturnShiftX() / particle_to_insert.pixel_size, -particle_to_insert.alignment_parameters.ReturnShiftY() / particle_to_insert.pixel_size);
//	particle_weight = particle_to_insert.particle_occupancy / particle_to_insert.parameter_average[12] / powf(particle_to_insert.sigma_noise / particle_to_insert.parameter_average[14],2);
	particle_weight = particle_to_insert.particle_occupancy / 100.0 / powf(particle_to_insert.sigma_noise,2);

	images_processed++;

	if (particle_weight > 0.0)
	{
		int i;
		int j;
		int k;

		long pixel_counter = 0;

		float x_coordinate_2d;
		float y_coordinate_2d;
		float z_coordinate_2d = 0.0;

		float x_coordinate_3d;
		float y_coordinate_3d;
		float z_coordinate_3d;

		float y_coord_sq;

		RotationMatrix temp_matrix;

		float frequency_squared;
		float weight;
		float score_weights_conversion4 = score_weights_conversion / powf(pixel_size,2) * 0.25;
		float weight_conversion = (particle_to_insert.particle_score - average_score) * score_weights_conversion4;

		// Make sure that the exponentiated conversion factor will not lead to an overflow
		if (weight_conversion > 60.0) {weight_conversion = 60.0;};
		if (weight_conversion < -60.0) {weight_conversion = -60.0;};

		std::complex<float> ctf_value = 1.0;

		for (j = particle_to_insert.particle_image->logical_lower_bound_complex_y; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			y_coord_sq = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
			for (i = 1; i <= particle_to_insert.particle_image->logical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i;
				frequency_squared = powf(x_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_x, 2) + y_coord_sq;
				weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
				if (weight < 0.0) weight = 0.0;
//				weight = particle_weight * expf(weight_conversion * frequency_squared);
				particle_to_insert.alignment_parameters.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
				pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], ctf_value, weight);
			}
		}
// Now deal with special case of i = 0
		for (j = 0; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			x_coordinate_2d = 0;
			frequency_squared = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
			weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
			if (weight < 0.0) weight = 0.0;
//			weight = particle_weight * expf(weight_conversion * frequency_squared);
			particle_to_insert.alignment_parameters.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], ctf_value, weight);
		}

		if (symmetry_matrices.number_of_matrices > 1)
		{
			for (k = 1; k < symmetry_matrices.number_of_matrices; k++)
			{
				for (j = particle_to_insert.particle_image->logical_lower_bound_complex_y; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					y_coord_sq = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
					for (i = 1; i <= particle_to_insert.particle_image->logical_upper_bound_complex_x; i++)
					{
						x_coordinate_2d = i;
						frequency_squared = powf(x_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_x, 2) + y_coord_sq;
						weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
						if (weight < 0.0) weight = 0.0;
//						weight = particle_weight * expf(weight_conversion * frequency_squared);
						temp_matrix = symmetry_matrices.rot_mat[k] * particle_to_insert.alignment_parameters.euler_matrix;
						temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(i,j,0);
						AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], ctf_value, weight);
					}
				}
// Now deal with special case of i = 0
				for (j = 0; j <= particle_to_insert.particle_image->logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					x_coordinate_2d = 0;
					frequency_squared = powf(y_coordinate_2d * particle_to_insert.ctf_image->fourier_voxel_size_y, 2);
					weight = particle_weight * (1.0 + weight_conversion * frequency_squared);
					if (weight < 0.0) weight = 0.0;
//					weight = particle_weight * expf(weight_conversion * frequency_squared);
					temp_matrix = symmetry_matrices.rot_mat[k] * particle_to_insert.alignment_parameters.euler_matrix;
					temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					pixel_counter = particle_to_insert.particle_image->ReturnFourier1DAddressFromLogicalCoord(0,j,0);
					AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, particle_to_insert.particle_image->complex_values[pixel_counter], ctf_value, weight);
				}
			}
		}
	}
}

void Reconstruct3D::AddByLinearInterpolation(float &wanted_logical_x_coordinate, float &wanted_logical_y_coordinate, float &wanted_logical_z_coordinate, std::complex<float> &input_value, std::complex<float> &ctf_value, float wanted_weight)
{
	int i;
	int j;
	int k;
	int int_x_coordinate;
	int int_y_coordinate;
	int int_z_coordinate;
	long physical_x_address;
	long physical_y_address;
	long physical_z_address;
	long upper_x = image_reconstruction.physical_upper_bound_complex_x + 1;
	long upper_xy = upper_x * long(image_reconstruction.physical_upper_bound_complex_y + 1);

	long physical_coord;
	long physical_coord_xy;

	float weight;
	float weight_x;
	float weight_xy;

	std::complex<float> conjugate;

	float ctf_real = real(ctf_value);
	float ctf_squared = powf(ctf_real,2) * wanted_weight;
	std::complex<float> value_to_insert = input_value * fabsf(ctf_real) * wanted_weight;
//	std::complex<float> value_to_insert = input_value * ctf_value * wanted_weight;

	int_x_coordinate = int(floorf(wanted_logical_x_coordinate));
	int_y_coordinate = int(floorf(wanted_logical_y_coordinate));
	int_z_coordinate = int(floorf(wanted_logical_z_coordinate));

	for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++)
	{
		weight_x = (1.0 - fabsf(wanted_logical_x_coordinate - i));
		physical_x_address = i;
		if (i < 0) physical_x_address = - i;
		for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++)
		{
			weight_xy = (1.0 - fabsf(wanted_logical_y_coordinate - j)) * weight_x;
			if (i >= 0)
			{
				physical_x_address = i;
				if (j >= 0)
				{
					physical_y_address = j;
				}
				else
				{
					physical_y_address = image_reconstruction.logical_y_dimension + j;
				}
				physical_coord_xy = upper_x * physical_y_address + physical_x_address;
				for (k = int_z_coordinate; k <= int_z_coordinate + 1; k++)
				{
					if (i <= image_reconstruction.logical_upper_bound_complex_x
					 && j >= image_reconstruction.logical_lower_bound_complex_y && j <= image_reconstruction.logical_upper_bound_complex_y
					 && k >= image_reconstruction.logical_lower_bound_complex_z && k <= image_reconstruction.logical_upper_bound_complex_z)
					{
						if (k >= 0)
						{
							physical_z_address = k;
						}
						else
						{
							physical_z_address = image_reconstruction.logical_z_dimension + k;
						}
						weight = (1.0 - fabsf(wanted_logical_z_coordinate - k)) * weight_xy;
						physical_coord = (upper_xy * physical_z_address) + physical_coord_xy;
						image_reconstruction.complex_values[physical_coord] = image_reconstruction.complex_values[physical_coord] + value_to_insert * weight;
						ctf_reconstruction[physical_coord] = ctf_reconstruction[physical_coord] + ctf_squared * weight;
					}
				}
			}
			else
			{
				physical_x_address = - i;
				if (j > 0)
				{
					physical_y_address = image_reconstruction.logical_y_dimension - j;
				}
				else
				{
					physical_y_address = - j;
				}
				physical_coord_xy = upper_x * physical_y_address + physical_x_address;
				for (k = int_z_coordinate; k <= int_z_coordinate + 1; k++)
				{
					if (i >= image_reconstruction.logical_lower_bound_complex_x
					 && j >= image_reconstruction.logical_lower_bound_complex_y && j <= image_reconstruction.logical_upper_bound_complex_y
					 && k >= image_reconstruction.logical_lower_bound_complex_z && k <= image_reconstruction.logical_upper_bound_complex_z)
					{
						if (k > 0)
						{
							physical_z_address = image_reconstruction.logical_z_dimension - k;
						}
						else
						{
							physical_z_address = - k;
						}
						weight = (1.0 - fabsf(wanted_logical_z_coordinate - k)) * weight_xy;
						physical_coord = (upper_xy * physical_z_address) + physical_coord_xy;
						conjugate = conj(value_to_insert);
						image_reconstruction.complex_values[physical_coord] = image_reconstruction.complex_values[physical_coord] + conjugate * weight;
						ctf_reconstruction[physical_coord] = ctf_reconstruction[physical_coord] + ctf_squared * weight;
					}
				}
			}
		}
	}
}

void Reconstruct3D::CompleteEdges()
{
	int i;
	int j;
	int k;

	long pixel_counter = 0;
	long physical_coord_1;
	long physical_coord_2;

	int temp_int;
	std::complex<float> temp_complex;
	float temp_real;

	if (! edge_terms_were_added)
	{
// Correct missing contributions to slice at j,k = 0
		for (k = 1; k <= image_reconstruction.logical_upper_bound_complex_z; k++)
		{
			for (j = -image_reconstruction.logical_upper_bound_complex_y; j <= image_reconstruction.logical_upper_bound_complex_y; j++)
			{
				if (j != 0)
				{
					physical_coord_1 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, j, k);
					physical_coord_2 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, -j, -k);
					temp_complex = image_reconstruction.complex_values[physical_coord_1];
					image_reconstruction.complex_values[physical_coord_1] = image_reconstruction.complex_values[physical_coord_1] + conj(image_reconstruction.complex_values[physical_coord_2]);
					image_reconstruction.complex_values[physical_coord_2] = image_reconstruction.complex_values[physical_coord_2] + conj(temp_complex);
					temp_real = ctf_reconstruction[physical_coord_1];
					ctf_reconstruction[physical_coord_1] = ctf_reconstruction[physical_coord_1] + ctf_reconstruction[physical_coord_2];
					ctf_reconstruction[physical_coord_2] = ctf_reconstruction[physical_coord_2] + temp_real;
				}
			}
		}
		for (j = 1; j <= image_reconstruction.logical_upper_bound_complex_y; j++)
		{
			physical_coord_1 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
			physical_coord_2 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
			temp_complex = image_reconstruction.complex_values[physical_coord_1];
			image_reconstruction.complex_values[physical_coord_1] = image_reconstruction.complex_values[physical_coord_1] + conj(image_reconstruction.complex_values[physical_coord_2]);
			image_reconstruction.complex_values[physical_coord_2] = image_reconstruction.complex_values[physical_coord_2] + conj(temp_complex);
			temp_real = ctf_reconstruction[physical_coord_1];
			ctf_reconstruction[physical_coord_1] = ctf_reconstruction[physical_coord_1] + ctf_reconstruction[physical_coord_2];
			ctf_reconstruction[physical_coord_2] = ctf_reconstruction[physical_coord_2] + temp_real;
		}
// Deal with term at origin
		physical_coord_1 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, 0, 0);
		image_reconstruction.complex_values[physical_coord_1] = 2.0 * real(image_reconstruction.complex_values[physical_coord_1]);
		ctf_reconstruction[physical_coord_1] = 2.0 * ctf_reconstruction[physical_coord_1];

		edge_terms_were_added = true;
	}
}

float Reconstruct3D::Correct3DCTF(Image &buffer3d)
{
	int i;
	float correction_factor;

	buffer3d.is_in_real_space = false;

	for (i = 0; i <= buffer3d.real_memory_allocated / 2; i++) buffer3d.complex_values[i] = ctf_reconstruction[i];

	buffer3d.SwapRealSpaceQuadrants();
	buffer3d.BackwardFFT();
	correction_factor = buffer3d.CorrectSinc();
	buffer3d.ForwardFFT();
	buffer3d.SwapRealSpaceQuadrants();

	for (i = 0; i <= buffer3d.real_memory_allocated / 2; i++) ctf_reconstruction[i] = real(buffer3d.complex_values[i]);

	return correction_factor;
}

void Reconstruct3D::DumpArrays(wxString filename, bool insert_even)
{
	int i;
	int count = 0;
	int oddeven;
	char temp_char[8 * sizeof(int) + 5 * sizeof(float) + 4];
	char *char_pointer;

	std::ofstream b_stream(filename.c_str(), std::fstream::out | std::fstream::binary);

	char_pointer = (char *) &logical_x_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &logical_y_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &logical_z_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &original_x_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &original_y_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &original_z_dimension;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &images_processed;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &pixel_size;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &original_pixel_size;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &average_occupancy;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &average_score;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &score_weights_conversion;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	for (i = 0; i < 4; i++) temp_char[count + i] = ' ';
	for (i = 0; i < symmetry_matrices.symmetry_symbol.length(); i++) temp_char[count + i] = symmetry_matrices.symmetry_symbol.GetChar(i);
	count += 4;
	oddeven = 1; if (insert_even) {oddeven = 2;};
	char_pointer = (char *) &oddeven;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	b_stream.write(temp_char, count);

	char_pointer = (char *) image_reconstruction.real_values;
	b_stream.write(char_pointer, sizeof(float) * image_reconstruction.real_memory_allocated);

	char_pointer = (char *) ctf_reconstruction;
	b_stream.write(char_pointer, sizeof(float) * image_reconstruction.real_memory_allocated / 2);

	b_stream.close();
}

void Reconstruct3D::ReadArrayHeader(wxString filename, int &logical_x_dimension, int &logical_y_dimension, int &logical_z_dimension,
		int &original_x_dimension, int &original_y_dimension, int &original_z_dimension, int &images_processed, float &pixel_size, float &original_pixel_size,
		float &average_occupancy, float &average_score, float &score_weights_conversion, wxString &symmetry_symbol, bool &insert_even)
{
	int i;
	int count = 8 * sizeof(int) + 5 * sizeof(float) + 4;
	int oddeven;
	char temp_char[count];
	char *char_pointer;

	std::ifstream b_stream(filename.c_str(), std::fstream::in | std::fstream::binary);

	b_stream.read(temp_char, count);
	count = 0;
	char_pointer = (char *) &logical_x_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &logical_y_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &logical_z_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &original_x_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &original_y_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &original_z_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &images_processed;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &original_pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &average_occupancy;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &average_score;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &score_weights_conversion;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	symmetry_symbol = "";
	for (i = 0; i < 4; i++) symmetry_symbol += temp_char[count + i];
	count += 4;
	char_pointer = (char *) &oddeven;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	insert_even = false; if (oddeven == 2) {insert_even = true;};

	symmetry_matrices.Init(symmetry_symbol);

	b_stream.close();
}

void Reconstruct3D::ReadArrays(wxString filename)
{
	int i;
	int count = 8 * sizeof(int) + 5 * sizeof(float) + 4;
	int oddeven;
	char temp_char[count];
	char *char_pointer;
	float input_pixel_size;
	float input_original_pixel_size;
	float input_average_occupancy;
	float input_average_score;
	float input_score_weights_conversion;
	int input_logical_x_dimension;
	int input_logical_y_dimension;
	int input_logical_z_dimension;
	int input_original_x_dimension;
	int input_original_y_dimension;
	int input_original_z_dimension;
//	int input_images_processed;
	wxString input_symmetry_symbol;

	std::ifstream b_stream(filename.c_str(), std::fstream::in | std::fstream::binary);

	b_stream.read(temp_char, count);
	count = 0;
	char_pointer = (char *) &input_logical_x_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_logical_y_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_logical_z_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_original_x_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_original_y_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_original_z_dimension;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &images_processed;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_original_pixel_size;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_average_occupancy;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_average_score;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	char_pointer = (char *) &input_score_weights_conversion;
	for (i = 0; i < sizeof(float); i++) {char_pointer[i] = temp_char[count]; count++;};
	input_symmetry_symbol = "";
	for (i = 0; i < 4; i++) input_symmetry_symbol += temp_char[count + i];
	count += 4;
	char_pointer = (char *) &oddeven;
	for (i = 0; i < sizeof(int); i++) {char_pointer[i] = temp_char[count]; count++;};

	if (input_logical_x_dimension != logical_x_dimension || input_logical_y_dimension != logical_y_dimension || input_logical_z_dimension != logical_z_dimension || input_pixel_size != pixel_size)
	{
		MyPrintWithDetails("Error: Dump file incompatible with 3D reconstruction\n");
		abort();
	}
	char_pointer = (char *) image_reconstruction.real_values;
	b_stream.read(char_pointer, sizeof(float) * image_reconstruction.real_memory_allocated);

	char_pointer = (char *) ctf_reconstruction;
	b_stream.read(char_pointer, sizeof(float) * image_reconstruction.real_memory_allocated / 2);

	b_stream.close();
}

Reconstruct3D &Reconstruct3D::operator = (const Reconstruct3D &other)
{
	*this = &other;
	return *this;
}

Reconstruct3D &Reconstruct3D::operator = (const Reconstruct3D *other)
{
	// Check for self assignment
	if (this != other)
	{
		int i;
		int j;
		int k;

		long pixel_counter = 0;

		for (k = 0; k <= image_reconstruction.physical_upper_bound_complex_z; k++)
		{
			for (j = 0; j <= image_reconstruction.physical_upper_bound_complex_y; j++)
			{
				for (i = 0; i <= image_reconstruction.physical_upper_bound_complex_x; i++)
				{
					this->image_reconstruction.complex_values[pixel_counter] = other->image_reconstruction.complex_values[pixel_counter];
					this->ctf_reconstruction[pixel_counter] = other->ctf_reconstruction[pixel_counter];
					pixel_counter++;
				}
			}
		}
	}

	images_processed = other->images_processed;

	return *this;
}

Reconstruct3D Reconstruct3D::operator + (const Reconstruct3D &other)
{
	Reconstruct3D temp_3d(other.logical_x_dimension, other.logical_y_dimension, other.logical_z_dimension, other.pixel_size, other.symmetry_matrices.symmetry_symbol);
	temp_3d += other;

    return temp_3d;
}

Reconstruct3D &Reconstruct3D::operator += (const Reconstruct3D &other)
{
	*this += &other;
	return *this;
}

Reconstruct3D &Reconstruct3D::operator += (const Reconstruct3D *other)
{
	MyDebugAssertTrue(other->pixel_size == pixel_size, "Pixel sizes differ");
	MyDebugAssertTrue(other->symmetry_matrices.symmetry_symbol == symmetry_matrices.symmetry_symbol, "Symmetries differ");
	MyDebugAssertTrue(other->edge_terms_were_added == edge_terms_were_added, "Edge terms in one of the reconstructions not corrected");
	MyDebugAssertTrue(other->logical_x_dimension == image_reconstruction.logical_x_dimension && other->logical_y_dimension == image_reconstruction.logical_y_dimension && other->logical_z_dimension == image_reconstruction.logical_z_dimension, "Reconstruct3D objects have different dimensions");

	int i;
	int j;
	int k;

	long pixel_counter = 0;

	for (k = 0; k <= image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		for (j = 0; j <= image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			for (i = 0; i <= image_reconstruction.physical_upper_bound_complex_x; i++)
			{
				this->image_reconstruction.complex_values[pixel_counter] += other->image_reconstruction.complex_values[pixel_counter];
				this->ctf_reconstruction[pixel_counter] += other->ctf_reconstruction[pixel_counter];
				pixel_counter++;
			}
		}
	}

	images_processed += other->images_processed;

	return *this;
}

