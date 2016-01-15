#include "core_headers.h"

Reconstruct3d::Reconstruct3d()
{
	logical_x_dimension = 0;
	logical_y_dimension = 0;
	logical_z_dimension = 0;

	ctf_reconstruction = NULL;
}

Reconstruct3d::Reconstruct3d(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension)
{
	ctf_reconstruction = NULL;
	Initialize(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension);
}

Reconstruct3d::~Reconstruct3d()
{
	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
}

void Reconstruct3d::Initialize(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension)
{
	logical_x_dimension = wanted_logical_x_dimension;
	logical_y_dimension = wanted_logical_y_dimension;
	logical_z_dimension = wanted_logical_z_dimension;

	image_reconstruction.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, wanted_logical_z_dimension, false);

	if (ctf_reconstruction != NULL)
	{
		delete [] ctf_reconstruction;
	}
	ctf_reconstruction = new float[image_reconstruction.real_memory_allocated / 2];
	for (int i = 0; i < image_reconstruction.real_memory_allocated / 2; i++)
	{
		ctf_reconstruction[i] = 0.0;
	}

	image_reconstruction.object_is_centred_in_box = false;

	image_reconstruction.SetToConstant(0.0);

	current_ctf_image.Allocate(wanted_logical_x_dimension, wanted_logical_y_dimension, 1, false);
}

void Reconstruct3d::InsertSlice(Image &image_to_insert, CTF &ctf_of_image, AnglesAndShifts &angles_and_shifts_of_image)
{
	int i;
	int j;

	long pixel_counter = 0;

	float x_coordinate_2d;
	float y_coordinate_2d;
	float z_coordinate_2d = 0.0;

	float x_coordinate_3d;
	float y_coordinate_3d;
	float z_coordinate_3d;

	float y_coord_sq;
	float x_coord_sq;

	float frequency_squared;
	float azimuth;

	MyDebugAssertTrue(image_to_insert.logical_x_dimension == logical_x_dimension && image_to_insert.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(image_to_insert.logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(current_ctf_image.is_in_memory, "Memory not allocated for current_ctf_image");
	MyDebugAssertTrue(image_to_insert.is_in_real_space == false, "image not in Fourier space");

	if (ctf_of_image.IsAlmostEqualTo(&current_ctf) == false)
	// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
	{
		current_ctf = ctf_of_image;

		for (j = 0; j <= current_ctf_image.physical_upper_bound_complex_y; j++)
		{
			y_coordinate_2d = current_ctf_image.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * current_ctf_image.fourier_voxel_size_y;
			y_coord_sq = pow(y_coordinate_2d, 2);

			for (i = 0; i <= current_ctf_image.physical_upper_bound_complex_x; i++)
			{
				x_coordinate_2d = i * current_ctf_image.fourier_voxel_size_x;
				x_coord_sq = pow(x_coordinate_2d, 2);

				// Compute the azimuth
				if ( i == 0 && j == 0 ) {
					azimuth = 0.0;
				} else {
					azimuth = atan2(y_coordinate_2d,x_coordinate_2d);
				}

				// Compute the square of the frequency
				frequency_squared = x_coord_sq + y_coord_sq;

				current_ctf_image.complex_values[pixel_counter] = ctf_of_image.Evaluate(frequency_squared,azimuth);
				pixel_counter++;
			}
		}
	}

// Now insert into 3D arrays
	for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
		{
			x_coordinate_2d = i;
			angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter]);
		}
	}
// Now deal with special case of i = 0
	for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		x_coordinate_2d = 0;
		angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
		pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
		AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], current_ctf_image.complex_values[pixel_counter]);
	}
}


void Reconstruct3d::InsertSlice(Image &image_to_insert, AnglesAndShifts &angles_and_shifts_of_image)
{
	int i;
	int j;

	long pixel_counter = 0;

	float x_coordinate_2d;
	float y_coordinate_2d;
	float z_coordinate_2d = 0.0;

	float x_coordinate_3d;
	float y_coordinate_3d;
	float z_coordinate_3d;

	float y_coord_sq;
	float x_coord_sq;

	fftwf_complex ctf_value = 1.0;

	MyDebugAssertTrue(image_to_insert.logical_x_dimension == logical_x_dimension && image_to_insert.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
	MyDebugAssertTrue(image_to_insert.logical_z_dimension == 1, "Error: attempting to insert 3D image into 3D reconstruction");
	MyDebugAssertTrue(image_reconstruction.is_in_memory, "Memory not allocated for image_reconstruction");
	MyDebugAssertTrue(image_to_insert.is_in_real_space == false, "image not in Fourier space");

	for (j = image_to_insert.logical_lower_bound_complex_y; j <= image_to_insert.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		for (i = 1; i <= image_to_insert.logical_upper_bound_complex_x; i++)
		{
			x_coordinate_2d = i;
			angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
			pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
			AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value);
		}
	}
// Now deal with special case of i = 0
	for (j = 0; j <= image_to_insert.logical_upper_bound_complex_y; j++)
	{
		y_coordinate_2d = j;
		x_coordinate_2d = 0;
		angles_and_shifts_of_image.euler_matrix.RotateCoords(x_coordinate_2d, y_coordinate_2d, z_coordinate_2d, x_coordinate_3d, y_coordinate_3d, z_coordinate_3d);
		pixel_counter = image_to_insert.ReturnFourier1DAddressFromLogicalCoord(0,j,0);
		AddByLinearInterpolation(x_coordinate_3d, y_coordinate_3d, z_coordinate_3d, image_to_insert.complex_values[pixel_counter], ctf_value);
	}
}


void Reconstruct3d::AddByLinearInterpolation(float &wanted_logical_x_coordinate, float &wanted_logical_y_coordinate, float &wanted_logical_z_coordinate, fftwf_complex &input_value, fftwf_complex &ctf_value)
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
	fftwf_complex value_to_insert = input_value * ctf_value;
//	fftwf_complex value_to_insert = input_value;

	float ctf_squared = creal(ctf_value * conj(ctf_value));

	int_x_coordinate = int(floor(wanted_logical_x_coordinate));
	int_y_coordinate = int(floor(wanted_logical_y_coordinate));
	int_z_coordinate = int(floor(wanted_logical_z_coordinate));

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
					ctf_reconstruction[physical_coord] = ctf_reconstruction[physical_coord] + ctf_squared * pow(weight, 2);
				}
			}
		}
	}
}


void Reconstruct3d::FinalizeSimple(Image &final3d)
{
	int i;
	int j;
	int k;

	long pixel_counter = 0;
	long physical_coord_1;
	long physical_coord_2;

	fftwf_complex temp_complex;
	float temp_real;

	final3d.Allocate(logical_x_dimension, logical_y_dimension, logical_z_dimension, false);
	final3d.object_is_centred_in_box = false;

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

// Now do the division by the CTF volume
	for (k = 0; k <= image_reconstruction.physical_upper_bound_complex_z; k++)
	{
		for (j = 0; j <= image_reconstruction.physical_upper_bound_complex_y; j++)
		{
			for (i = 0; i <= image_reconstruction.physical_upper_bound_complex_x; i++)
			{
//				final3d.complex_values[pixel_counter] = image_reconstruction.complex_values[pixel_counter];
				if (ctf_reconstruction[pixel_counter] != 0.0)
				{
					final3d.complex_values[pixel_counter] = image_reconstruction.complex_values[pixel_counter]/(ctf_reconstruction[pixel_counter]+1.0);
				}
				else
				{
					final3d.complex_values[pixel_counter] = 0.0;
				}
				pixel_counter++;
			}
		}
	}
}
