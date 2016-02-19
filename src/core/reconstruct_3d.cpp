#include "core_headers.h"

Reconstruct3D::Reconstruct3D(float wanted_pixel_size)
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;

	pixel_size = wanted_pixel_size;

	ctf_reconstruction = NULL;
//	weights_reconstruction = NULL;
//	number_of_measurements = NULL;

	symmetry_matrices.Init("C1");
	edge_terms_were_added = false;
}

Reconstruct3D::Reconstruct3D(float wanted_pixel_size, wxString wanted_symmetry)
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;

	pixel_size = wanted_pixel_size;

	ctf_reconstruction = NULL;
//	weights_reconstruction = NULL;
//	number_of_measurements = NULL;

	symmetry_matrices.Init(wanted_symmetry);
	edge_terms_were_added = false;
}

Reconstruct3D::Reconstruct3D(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString wanted_symmetry)
{
	ctf_reconstruction = NULL;
//	weights_reconstruction = NULL;
//	number_of_measurements = NULL;
	Init(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, wanted_pixel_size);

	symmetry_matrices.Init(wanted_symmetry);
}

Reconstruct3D::~Reconstruct3D()
{
	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
/*	if (weights_reconstruction != NULL)
	{
		delete [] weights_reconstruction;
	}

	if (number_of_measurements != NULL)
	{
		delete [] number_of_measurements;
	}
*/
}

void Reconstruct3D::Init(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size)
{
	logical_x_dimension = wanted_logical_x_dimension;
	logical_y_dimension = wanted_logical_y_dimension;
	logical_z_dimension = wanted_logical_z_dimension;

	pixel_size = wanted_pixel_size;

	image_reconstruction.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, false);

	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
	ctf_reconstruction = new float[image_reconstruction.real_memory_allocated / 2];
	ZeroFloatArray(ctf_reconstruction, image_reconstruction.real_memory_allocated / 2);

/*	if (weights_reconstruction != NULL)
	{
		delete [] weights_reconstruction;
	}
	weights_reconstruction = new float[image_reconstruction.real_memory_allocated / 2];
	ZeroFloatArray(weights_reconstruction, image_reconstruction.real_memory_allocated / 2);

	if (number_of_measurements != NULL)
	{
		delete [] number_of_measurements;
	}
	number_of_measurements = new int[image_reconstruction.real_memory_allocated / 2];
	ZeroIntArray(number_of_measurements, image_reconstruction.real_memory_allocated / 2);
*/
	image_reconstruction.object_is_centred_in_box = false;

	image_reconstruction.SetToConstant(0.0);

	current_ctf_image.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);

	edge_terms_were_added = false;
}

void Reconstruct3D::InsertSlice(Image &image_to_insert, CTF &ctf_of_image, AnglesAndShifts &angles_and_shifts_of_image, float &particle_weight, float &particle_score, float &average_score, float &score_bfactor_conversion)
{
	MyDebugAssertTrue(image_to_insert.logical_x_dimension == logical_x_dimension && image_to_insert.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(image_to_insert.logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(current_ctf_image.is_in_memory, "Memory not allocated for current_ctf_image");
	MyDebugAssertTrue(image_to_insert.is_in_real_space == false, "image not in Fourier space");

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
	//	float x_coord_sq;

		RotationMatrix temp_matrix;

		float frequency_squared;
		float azimuth;
		float weight;
		float score_bfactor_conversion4 = score_bfactor_conversion / powf(pixel_size,2) * 0.25;

		if (ctf_of_image.IsAlmostEqualTo(&current_ctf) == false)
		// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
		{
			current_ctf = ctf_of_image;
			current_ctf_image.CalculateCTFImage(current_ctf);
		}

	// Now insert into 3D arrays
		for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			y_coord_sq = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
			for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
			{
//				if (image_to_insert.ReturnFourierLogicalCoordGivenPhysicalCoord_X(i)==20 && image_to_insert.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j)==20)
//				{
//					wxPrintf("counter = %li, image = %g\n", pixel_counter, cabsf(image_to_insert.complex_values[pixel_counter]));
//				}

				x_coordinate_2d = i;
				frequency_squared = powf(x_coordinate_2d * current_ctf_image.fourier_voxel_size_x, 2) + y_coord_sq;
				weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
				angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
				pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter], weight);
			}
		}
	// Now deal with special case of i = 0
		for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			x_coordinate_2d = 0;
			frequency_squared = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
			weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
			angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter], weight);
		}

		if (symmetry_matrices.number_of_matrices > 1)
		{
			for (k = 1; k < symmetry_matrices.number_of_matrices; k++)
			{
				for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					y_coord_sq = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
					for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
					{
						x_coordinate_2d = i;
						frequency_squared = powf(x_coordinate_2d * current_ctf_image.fourier_voxel_size_x, 2) + y_coord_sq;
						weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
	//					angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						temp_matrix = symmetry_matrices.rot_mat[k] * angles_and_shifts_of_image.euler_matrix;
						temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
						AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter], weight);
					}
				}
			// Now deal with special case of i = 0
				for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					x_coordinate_2d = 0;
					frequency_squared = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
					weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
	//				angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					temp_matrix = symmetry_matrices.rot_mat[k] * angles_and_shifts_of_image.euler_matrix;
					temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
					AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter], weight);
				}
			}
		}
	}
}

void Reconstruct3D::InsertSlice(Image &image_to_insert, AnglesAndShifts &angles_and_shifts_of_image, float &particle_weight, float &particle_score, float &average_score, float &score_bfactor_conversion)
{
	MyDebugAssertTrue(image_to_insert.logical_x_dimension == logical_x_dimension && image_to_insert.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(image_to_insert.logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(image_to_insert.is_in_real_space == false, "image not in Fourier space");
	MyDebugAssertTrue(image_to_insert.IsSquare(), "Image must be square");

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
//		float x_coord_sq;

		RotationMatrix temp_matrix;

		float frequency_squared;
		float weight;
		float score_bfactor_conversion4 = score_bfactor_conversion / powf(pixel_size,2) * 0.25;

		fftwf_complex ctf_value = 1.0;

		for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			y_coord_sq = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
			for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i;
				frequency_squared = powf(x_coordinate_2d * current_ctf_image.fourier_voxel_size_x, 2) + y_coord_sq;
				weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
				angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
				pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
				AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value, weight);
			}
		}
// Now deal with special case of i = 0
		for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = j;
			x_coordinate_2d = 0;
			frequency_squared = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
			weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
			angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value, weight);
		}

		if (symmetry_matrices.number_of_matrices > 1)
		{
			for (k = 1; k < symmetry_matrices.number_of_matrices; k++)
			{
				for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					y_coord_sq = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
					for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
					{
						x_coordinate_2d = i;
						frequency_squared = powf(x_coordinate_2d * current_ctf_image.fourier_voxel_size_x, 2) + y_coord_sq;
						weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
//						angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						temp_matrix = symmetry_matrices.rot_mat[k] * angles_and_shifts_of_image.euler_matrix;
						temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
						pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
						AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value, weight);
					}
				}
// Now deal with special case of i = 0
				for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
				{
					y_coordinate_2d = j;
					x_coordinate_2d = 0;
					frequency_squared = powf(y_coordinate_2d * current_ctf_image.fourier_voxel_size_y, 2);
					weight = particle_weight * expf((particle_score - average_score) * score_bfactor_conversion4 * frequency_squared);
//					angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					temp_matrix = symmetry_matrices.rot_mat[k] * angles_and_shifts_of_image.euler_matrix;
					temp_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
					pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
					AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value, weight);
				}
			}
		}
	}
}

void Reconstruct3D::AddByLinearInterpolation(float &wanted_logical_x_coordinate, float &wanted_logical_y_coordinate, float &wanted_logical_z_coordinate, fftwf_complex &input_value, fftwf_complex &ctf_value, float wanted_weight)
{
	int i;
	int j;
	int k;
	int int_x_coordinate;
	int int_y_coordinate;
	int int_z_coordinate;

	long physical_coord;

	float weight;
	float weight_y;
	float weight_z;

	fftwf_complex conjugate;
	fftwf_complex value_to_insert = input_value * ctf_value * wanted_weight;

//	float ctf_squared = powf(creal(ctf_value),2);
	float ctf_squared = powf(creal(ctf_value),2) * wanted_weight;

	int_x_coordinate = int(floorf(wanted_logical_x_coordinate));
	int_y_coordinate = int(floorf(wanted_logical_y_coordinate));
	int_z_coordinate = int(floorf(wanted_logical_z_coordinate));

	for (k = int_z_coordinate; k <= int_z_coordinate + 1; k++)
	{
		weight_z = (1.0 - fabs(wanted_logical_z_coordinate - k));
		for (j = int_y_coordinate; j <= int_y_coordinate + 1; j++)
		{
			weight_y = (1.0 - fabs(wanted_logical_y_coordinate - j));
			for (i = int_x_coordinate; i <= int_x_coordinate + 1; i++)
			{
				if (i >= image_reconstruction.logical_lower_bound_complex_x && i <= image_reconstruction.logical_upper_bound_complex_x
				 && j >= image_reconstruction.logical_lower_bound_complex_y && j <= image_reconstruction.logical_upper_bound_complex_y
				 && k >= image_reconstruction.logical_lower_bound_complex_z && k <= image_reconstruction.logical_upper_bound_complex_z)
				{
					weight = (1.0 - fabs(wanted_logical_x_coordinate - i)) * weight_y * weight_z;
					physical_coord = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(i, j, k);
					if (i < 0)
					{
						conjugate = conjf(value_to_insert);
						image_reconstruction.complex_values[physical_coord] = image_reconstruction.complex_values[physical_coord] + conjugate * weight;
					}
					else
					{
						image_reconstruction.complex_values[physical_coord] = image_reconstruction.complex_values[physical_coord] + value_to_insert * weight;
					}
//					ctf_reconstruction[physical_coord] = ctf_reconstruction[physical_coord] + ctf_squared * powf(weight, 2);
					ctf_reconstruction[physical_coord] = ctf_reconstruction[physical_coord] + ctf_squared * weight;
//					weights_reconstruction[physical_coord] = weights_reconstruction[physical_coord] + wanted_weight;
//					number_of_measurements[physical_coord] = number_of_measurements[physical_coord] + 1;
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
	fftwf_complex temp_complex;
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
					image_reconstruction.complex_values[physical_coord_1] = image_reconstruction.complex_values[physical_coord_1] + conjf(image_reconstruction.complex_values[physical_coord_2]);
					image_reconstruction.complex_values[physical_coord_2] = image_reconstruction.complex_values[physical_coord_2] + conjf(temp_complex);
					temp_real = ctf_reconstruction[physical_coord_1];
					ctf_reconstruction[physical_coord_1] = ctf_reconstruction[physical_coord_1] + ctf_reconstruction[physical_coord_2];
					ctf_reconstruction[physical_coord_2] = ctf_reconstruction[physical_coord_2] + temp_real;
/*					temp_real = weights_reconstruction[physical_coord_1];
					weights_reconstruction[physical_coord_1] = weights_reconstruction[physical_coord_1] + weights_reconstruction[physical_coord_2];
					weights_reconstruction[physical_coord_2] = weights_reconstruction[physical_coord_2] + temp_real;
					temp_int = number_of_measurements[physical_coord_1];
					number_of_measurements[physical_coord_1] = number_of_measurements[physical_coord_1] + number_of_measurements[physical_coord_2];
					number_of_measurements[physical_coord_2] = number_of_measurements[physical_coord_2] + temp_int;
*/
				}
			}
		}
		for (j = 1; j <= image_reconstruction.logical_upper_bound_complex_y; j++)
		{
			physical_coord_1 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, j, 0);
			physical_coord_2 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, -j, 0);
			temp_complex = image_reconstruction.complex_values[physical_coord_1];
			image_reconstruction.complex_values[physical_coord_1] = image_reconstruction.complex_values[physical_coord_1] + conjf(image_reconstruction.complex_values[physical_coord_2]);
			image_reconstruction.complex_values[physical_coord_2] = image_reconstruction.complex_values[physical_coord_2] + conjf(temp_complex);
			temp_real = ctf_reconstruction[physical_coord_1];
			ctf_reconstruction[physical_coord_1] = ctf_reconstruction[physical_coord_1] + ctf_reconstruction[physical_coord_2];
			ctf_reconstruction[physical_coord_2] = ctf_reconstruction[physical_coord_2] + temp_real;
/*			temp_real = weights_reconstruction[physical_coord_1];
			weights_reconstruction[physical_coord_1] = weights_reconstruction[physical_coord_1] + weights_reconstruction[physical_coord_2];
			weights_reconstruction[physical_coord_2] = weights_reconstruction[physical_coord_2] + temp_real;
			temp_int = number_of_measurements[physical_coord_1];
			number_of_measurements[physical_coord_1] = number_of_measurements[physical_coord_1] + number_of_measurements[physical_coord_2];
			number_of_measurements[physical_coord_2] = number_of_measurements[physical_coord_2] + temp_int;
*/
		}
// Deal with term at origin
		physical_coord_1 = image_reconstruction.ReturnFourier1DAddressFromLogicalCoord(0, 0, 0);
		image_reconstruction.complex_values[physical_coord_1] = 2.0 * creal(image_reconstruction.complex_values[physical_coord_1]);
		ctf_reconstruction[physical_coord_1] = 2.0 * ctf_reconstruction[physical_coord_1];
//		weights_reconstruction[physical_coord_1] = 2.0 * weights_reconstruction[physical_coord_1];
//		number_of_measurements[physical_coord_1] = 2 * number_of_measurements[physical_coord_1];

		edge_terms_were_added = true;
	}
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
//					this->weights_reconstruction[pixel_counter] = other->weights_reconstruction[pixel_counter];
//					this->number_of_measurements[pixel_counter] = other->number_of_measurements[pixel_counter];
					pixel_counter++;
				}
			}
		}
   }
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
//				this->weights_reconstruction[pixel_counter] += other->weights_reconstruction[pixel_counter];
//				this->number_of_measurements[pixel_counter] += other->number_of_measurements[pixel_counter];
				pixel_counter++;
			}
		}
	}

	return *this;
}

