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
}

Image::Image( const Image &other_image) // copy constructor
{
	 *this = other_image;
}

Image::~Image()
{
	Deallocate();
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


void Image::CosineMask(float mask_radius, float mask_edge)
{
	int i;
	int j;
	int k;
	int number_of_pixels;

	float x;
	float y;
	float z;

	long pixel_counter = 0;

	float distance_from_center;
	float mask_radius_plus_edge;
	float distance_from_center_squared;
	float mask_radius_squared;
	float mask_radius_plus_edge_squared;
	float edge;
	double pixel_sum;

	float frequency;
	float frequency_squared;

	mask_radius_plus_edge = mask_radius + mask_edge;

	mask_radius_squared = pow(mask_radius, 2);
	mask_radius_plus_edge_squared = pow(mask_radius_plus_edge, 2);

	pixel_sum = 0.0;
	number_of_pixels = 0;
	if (is_in_real_space == true)
	{
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = pow(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = pow(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = pow(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						pixel_sum += real_values[pixel_counter];
						number_of_pixels++;
					}
					pixel_counter++;
				}
				pixel_counter+=padding_jump_value;
			}
		}
		pixel_sum /= number_of_pixels;

		pixel_counter = 0.0;
		for (k = 0; k < logical_z_dimension; k++)
		{
			z = pow(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = pow(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = pow(i - physical_address_of_box_center_x, 2);

					distance_from_center_squared = x + y + z;

					if (distance_from_center_squared >= mask_radius_squared && distance_from_center_squared <= mask_radius_plus_edge_squared)
					{
						distance_from_center = sqrt(distance_from_center_squared);
						edge = (1.0 + cos(PI * (distance_from_center - mask_radius) / mask_edge)) / 2.0;
						real_values[pixel_counter] = real_values[pixel_counter] * edge + (1.0 - edge) * pixel_sum;
					}
					else
					if (distance_from_center_squared >= mask_radius_plus_edge_squared) real_values[pixel_counter] = pixel_sum;

					pixel_counter++;
				}
				pixel_counter+=padding_jump_value;
			}
		}
	}
	else
	{
		for (k = 0; k <= physical_upper_bound_complex_z; k++)
		{
			z = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

			for (j = 0; j <= physical_upper_bound_complex_y; j++)
			{
				y = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

				for (i = 0; i <= physical_upper_bound_complex_x; i++)
				{
					x = pow(i * fourier_voxel_size_x, 2);

					// compute squared radius, in units of reciprocal pixels

					frequency_squared = x + y + z;

					if (frequency_squared >= mask_radius_squared && frequency_squared <= mask_radius_plus_edge_squared)
					{
						frequency = sqrt(frequency_squared);
						edge = (1.0 + cos(PI * (frequency - mask_radius) / mask_edge)) / 2.0;
						complex_values[pixel_counter] *= edge;
					}
					else
					if (frequency_squared >= mask_radius_plus_edge_squared) complex_values[pixel_counter] = 0.0;

					pixel_counter++;
				}
			}
		}
	}
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
	physical_index_of_first_negative_frequency_y = logical_y_dimension / 2 + 1;
	physical_index_of_first_negative_frequency_z = logical_z_dimension / 2 + 1;


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

inline void Image::DivideByConstant(float constant_to_divide_by)
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

void Image::QuickAndDirtyWriteSlice(std::string filename, long slice_to_write)
{
	MyDebugAssertTrue(slice_to_write >0, "Slice is less than 1, first slice is 1");
	MRCFile output_file(filename, false);
	WriteSlice(&output_file, slice_to_write);
}

void Image::QuickAndDirtyReadSlice(std::string filename, long slice_to_read)
{
	MRCFile input_file(filename, false);

	MyDebugAssertTrue(slice_to_read <= input_file.ReturnNumberOfSlices(), "End slices is greater than number of slices in the file!");
	MyDebugAssertTrue(slice_to_read >0, "Slice is less than 1, first slice is 1");

	ReadSlice(&input_file, slice_to_read);
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

float Image::ReturnAverageOfRealValuesOnEdges()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(logical_z_dimension == 1, "Not implemented for volumes yet, sorry");

	double sum;
	long number_of_pixels;
	int pixel_counter;
	int line_counter;
	int plane_counter;
	long address;


	sum = 0.0;
	number_of_pixels = 0;
	pixel_counter = 0;
	line_counter = 0;
	plane_counter = 0;
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
			address += padding_jump_value;
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
				address += padding_jump_value;
				number_of_pixels += 2;
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
		}
	}
}

float Image::ReturnAverageOfRealValues()
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space, "Not in real space");

	double sum = 0.0;
	long number_of_voxels = 0;
	long address = 0;
	int		x;
	int		y;
	int 	z;

	for (z=0;z<logical_z_dimension;z++)
	{
		for (y=0;y<logical_y_dimension;y++)
		{
			for (x=0;x<logical_x_dimension;x++)
			{
				sum += real_values[address];
				//
				address++;
			}
			address += padding_jump_value;
		}
	}

	return sum / (logical_x_dimension * logical_y_dimension * logical_z_dimension);
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
	float min_rad_sq = powf(minimum_radius,2);
	float max_rad_sq = powf(maximum_radius,2);
	float cross_half_width_sq = pow(cross_half_width,2);
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

	MyDebugPrint("Average = %g  Sigma = %g",average,sigma);
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
	const float		lowest_freq = pow(ctf.GetLowestFrequencyForFitting(),2);
	const float		highest_freq = pow(ctf.GetHighestFrequencyForFitting(),2);
	long			address = 0;
	float			current_azimuth;
	float			current_ctf_value;
	const int		central_cross_half_width = 10;
	float			astigmatism_penalty;

	Image			debug_image;
	static int      loc_in_debug_image = 0;

	debug_image.Allocate(logical_x_dimension,logical_y_dimension,true);
	debug_image.SetToConstant(0.0);

	// Loop over half of the image (ignore Friedel mates)
	for (j=0;j<logical_y_dimension;j++)
	{
		j_logi = float(j-physical_address_of_box_center_y)*inverse_logical_y_dimension;
		j_logi_sq = pow(j_logi,2);
		for (i=0;i<physical_address_of_box_center_x;i++)
		{
			i_logi = float(i-physical_address_of_box_center_x)*inverse_logical_x_dimension;
			i_logi_sq = pow(i_logi,2);

			// Where are we?
			current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

			if (current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared < highest_freq)
			{
				current_azimuth = atan2f(j_logi,i_logi);
				current_ctf_value = fabsf(ctf.Evaluate(current_spatial_frequency_squared,current_azimuth));
				debug_image.real_values[address] = current_ctf_value;
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
		astigmatism_penalty = pow(ctf.GetAstigmatism(),2) * 0.5 / pow(ctf.GetAstigmatismTolerance(),2) / float(number_of_values);
	}
	else
	{
		astigmatism_penalty = 0.0;
	}

	//MyDebugPrint("Defocus (%6.1f,%6.1f,%5.1f) gives score %g / sqrt(%g * %g) - %g = %g",ctf.GetDefocus1(),ctf.GetDefocus2(),ctf.GetAstigmatismAzimuth(),cross_product,norm_image,norm_ctf,astigmatism_penalty,cross_product / sqrt(norm_image * norm_ctf) - astigmatism_penalty);
	loc_in_debug_image++;
	debug_image.QuickAndDirtyWriteSlice("dbg_scoring_ctf.mrc",loc_in_debug_image);
	QuickAndDirtyWriteSlice("dbg_scoring_this.mrc",1);

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
		j_sq = pow((j - physical_address_of_box_center_y),2);

		for (i = 0; i < logical_x_dimension; i++)
		{
			i_friedel = 2 * physical_address_of_box_center_x - i;
			i_sq = pow((i - physical_address_of_box_center_x),2);

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
					jj_sq = pow((jj - physical_address_of_box_center_y),2);

					for ( l = - half_box_size; l <= half_box_size; l++)
					{
						ii = i + l;
						if (ii < 0) { ii += logical_x_dimension; }
						if (ii >= logical_x_dimension) { ii -= logical_x_dimension; }
						ii_friedel = 2 * physical_address_of_box_center_x - ii;
						ii_sq = pow((ii - physical_address_of_box_center_x),2);

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
		if (i > 511 || i_friedel > 511 || logical_y_dimension - 1 > 511) { MyDebugPrint("Bad values\n"); exit(-1); }
		output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,0,0)]                     = output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,0,0)];
		output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i,logical_y_dimension-1,0)] = output_image->real_values[ReturnReal1DAddressFromPhysicalCoord(i_friedel,logical_y_dimension - 1,0)];
	}
}

// Taper edges of image so that there are no sharp discontinuities in real space
// This is a re-implementation of the MRC program taperedgek.for (Richard Henderson, 1987)
void Image::TaperEdges()
{
	MyDebugAssertTrue(is_in_memory,"Image not in memory");
	MyDebugAssertTrue(is_in_real_space,"Image not in real space");

	// Private variables
	const int				averaging_strip_width_x	=	100;
	const int				averaging_strip_width_y	=	100;
	const int				averaging_strip_width_z =   100;
	const int				tapering_strip_width_x	=	500;
	const int				tapering_strip_width_y	=	500;
	const int				tapering_strip_width_z	=	500;
	const int				smoothing_half_width_x	=	1;
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
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(logical_x_dimension-i+1,j-1,k-1)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_x);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_x);
						break;
				case(2):
						for (i=1;i<=averaging_strip_width_y;i++)
						{
							average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,i-1,k-1)];
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,logical_y_dimension-i+1,k-1)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_y);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_y);
						break;
				case(3):
						for (i=1;i<=averaging_strip_width_z;i++)
						{
							average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,i-1)];
							average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension] += real_values[ReturnReal1DAddressFromPhysicalCoord(j-1,k-1,logical_z_dimension-i+1)];
						}
						average_for_current_edge_start [(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_z);
						average_for_current_edge_finish[(j-1)+(k-1)*logical_second_dimension]  /= float(averaging_strip_width_z);
						break;
				}
			}
		}
	}

	for (address=0;address<logical_second_dimension*logical_third_dimension;address++)
	{
		average_for_current_edge_average[address] = 0.5 * ( average_for_current_edge_finish[address] - average_for_current_edge_start[address]);
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

}


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
			x_coord_sq = pow(x_coord, 2);

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
		z_coord = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * fourier_voxel_size_z, 2);

		for (j = 0; j <= physical_upper_bound_complex_y; j++)
		{
			y_coord = pow(ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * fourier_voxel_size_y, 2);

			for (i = 0; i <= physical_upper_bound_complex_x; i++)
			{
				x_coord = pow(i * fourier_voxel_size_x, 2);

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

Peak Image::FindPeakWithIntegerCoordinates(float wanted_min_radius, float wanted_max_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");

	int k;
	int j;
	int i;

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

	if (wanted_min_radius > 0.0 && wanted_min_radius < 1.0 && wanted_max_radius > 0.0 && wanted_max_radius < 1.0) radii_are_fractional = true;
	else radii_are_fractional = false;

	wanted_min_radius = pow(wanted_min_radius, 2);
	wanted_max_radius = pow(wanted_max_radius, 2);

	if (radii_are_fractional == true)
	{
		inv_max_radius_sq_x = 1.0 / pow(physical_address_of_box_center_x, 2);
		inv_max_radius_sq_y = 1.0 / pow(physical_address_of_box_center_y, 2);

		if (logical_z_dimension == 1) inv_max_radius_sq_z = 0;
		else inv_max_radius_sq_z = 1.0 / pow(physical_address_of_box_center_z, 2);

		for (k = 0; k < logical_z_dimension; k++)
			{
				z = pow(k - physical_address_of_box_center_z, 2) * inv_max_radius_sq_z;

				for (j = 0; j < logical_y_dimension; j++)
				{
					y = pow(j - physical_address_of_box_center_y, 2) * inv_max_radius_sq_y;

					for (i = 0; i < logical_x_dimension; i++)
					{
						x = pow(i - physical_address_of_box_center_x, 2) * inv_max_radius_sq_x;

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
			z = pow(k - physical_address_of_box_center_z, 2);

			for (j = 0; j < logical_y_dimension; j++)
			{
				y = pow(j - physical_address_of_box_center_y, 2);

				for (i = 0; i < logical_x_dimension; i++)
				{
					x = pow(i - physical_address_of_box_center_x, 2);

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
    denominator = 4.0e0 * coefficient_three * coefficient_six - pow(coefficient_five, 2);

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

        found_peak.value = 4.0e0 * coefficient_one * coefficient_three * coefficient_six - coefficient_one * pow(coefficient_five,2) - pow(coefficient_two,2) * coefficient_six + coefficient_two * coefficient_four*coefficient_five - pow(coefficient_four,2) * coefficient_three;
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





