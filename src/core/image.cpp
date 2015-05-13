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

	physical_index_of_first_negative_frequency_x = 0;
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

	real_values = NULL;
	complex_values = NULL;

	is_in_memory = false;
	real_memory_allocated = 0;

	plan_fwd = NULL;
	plan_bwd = NULL;

	planned = false;
};

Image::~Image()
{
	Deallocate();
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
		if (wanted_x_size == logical_x_dimension && wanted_y_size == logical_y_dimension && wanted_z_size == logical_z_dimension)
		{
			// everything is already done..

			is_in_real_space = should_be_in_real_space;

			return;
		}
		else fftwf_free(real_values);
	}

	// if we got here we need to do allocation..

	SetLogicalDimensions(wanted_x_size, wanted_y_size, wanted_z_size);
	is_in_real_space = should_be_in_real_space;

	// first_x_dimension
	if (IsEven(wanted_x_size) == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;


	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

	printf ("allocating %i reals\n", real_memory_allocated);

	real_values = fftwf_alloc_real(real_memory_allocated);
	complex_values = (fftwf_complex*) real_values;  // Set the complex_values to point at the newly allocated real values;

	is_in_memory = true;

	// Update addresses etc..

    UpdateLoopingAndAddressing();

    // Prepare the plans for FFTW

    if (planned == false)
    {
    	if (logical_z_dimension > 1)
    	{
    		plan_fwd = fftwf_plan_dft_r2c_3d(logical_x_dimension, logical_y_dimension, logical_z_dimension, real_values, complex_values, FFTW_ESTIMATE);
    		plan_bwd = fftwf_plan_dft_c2r_3d(logical_x_dimension, logical_y_dimension, logical_z_dimension, complex_values, real_values, FFTW_ESTIMATE);
    	}
    	else
    	{
    		plan_fwd = fftwf_plan_dft_r2c_2d(logical_x_dimension, logical_y_dimension, real_values, complex_values, FFTW_ESTIMATE);
    	    plan_bwd = fftwf_plan_dft_c2r_2d(logical_x_dimension, logical_y_dimension, complex_values, real_values, FFTW_ESTIMATE);

    	}

    	planned = true;
    }

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

	if (IsEven(logical_x_dimension) == true) physical_upper_bound_complex_x = logical_x_dimension / 2 + 1;
	else physical_upper_bound_complex_x = (logical_x_dimension - 1) / 2 + 1;

	physical_upper_bound_complex_y = logical_y_dimension;
	physical_upper_bound_complex_z = logical_z_dimension;

	UpdatePhysicalAddressOfBoxCenter();

    // In each dimension, the physical index of the first pixel which stores negative frequencies
    // Note that we never actually store negative frequencies for the first dimension. However, it is sometimes useful
    // to pretend as though we did (for example when generating real-space images of CTFs).

	if (IsEven(logical_x_dimension) == true) physical_index_of_first_negative_frequency_x = logical_x_dimension / 2 + 2;
	else physical_index_of_first_negative_frequency_x = (logical_x_dimension + 3) / 2;

	if (IsEven(logical_y_dimension) == true) physical_index_of_first_negative_frequency_y = logical_y_dimension / 2 + 2;
	else physical_index_of_first_negative_frequency_y = (logical_y_dimension + 3) / 2;

	if (IsEven(logical_z_dimension) == true) physical_index_of_first_negative_frequency_z = logical_z_dimension / 2 + 2;
	else physical_index_of_first_negative_frequency_z = (logical_z_dimension + 3) / 2;

    // Update the Fourier voxel size

	fourier_voxel_size_x = 1.0 / float(logical_x_dimension);
	fourier_voxel_size_y = 1.0 / float(logical_y_dimension);
	fourier_voxel_size_z = 1.0 / float(logical_z_dimension);

    // Update the bounds of the logical addresses

	if (IsEven(logical_x_dimension) == true)
	{
        logical_lower_bound_complex_x = -logical_x_dimension / 2;
        logical_upper_bound_complex_x =  logical_x_dimension / 2;
        logical_lower_bound_real_x    = -logical_x_dimension / 2;
        logical_upper_bound_real_x    =  logical_x_dimension / 2 - 1;
	}
	else
	{
        logical_lower_bound_complex_x = -(logical_x_dimension - 1) / 2;
        logical_upper_bound_complex_y = (logical_x_dimension - 1) / 2;
	}

	if (IsEven(logical_y_dimension) == true)
	{
        logical_lower_bound_complex_y = -logical_y_dimension / 2;
        logical_upper_bound_complex_y =  logical_y_dimension /2 - 1;
        logical_lower_bound_real_y    = -logical_y_dimension / 2;
        logical_upper_bound_real_y    =  logical_y_dimension / 2 - 1;

	}
	else
	{
		logical_lower_bound_complex_y = -(logical_y_dimension - 1) / 2;
		logical_upper_bound_complex_y =  (logical_y_dimension - 1) / 2;
		logical_lower_bound_real_y    = -(logical_y_dimension - 1) / 2;
		logical_upper_bound_real_y    =  (logical_y_dimension - 1) / 2;
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
	physical_address_of_box_center_x = logical_x_dimension / 2 + 1;
	physical_address_of_box_center_y = logical_y_dimension / 2 + 1;
	physical_address_of_box_center_z = logical_z_dimension / 2 + 1;
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

inline void Image::MultiplyByConstant(float constant_to_multiply_by)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] *= constant_to_multiply_by;
	}
}

//!> \brief Read a set of slices from disk (FFTW padding is done automatically)

void Image::ReadSlices(MRCFile *input_file, long start_slice, long end_slice)
{
	MyDebugAssertTrue(start_slice <= end_slice, "Start slice larger than end slice!");
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
		input_file->SetYSize(logical_x_dimension);

		if (end_slice > input_file->ReturnNumberOfSlices())
		{
			input_file->SetNumberOfSlices(end_slice);
		}

		input_file->WriteHeader();
	}
	else // if the last slice is bigger than the current max number of slices, increase the max number of slices
	{
		if (end_slice > input_file->ReturnNumberOfSlices())
		{
			input_file->SetNumberOfSlices(end_slice);
		}

		input_file->WriteHeader();
	}


	MyDebugAssertTrue(logical_x_dimension == input_file->ReturnXSize() || logical_y_dimension == input_file->ReturnYSize(), "Image dimensions and file dimensions differ!");

	// check the allocations..

	int number_of_slices = (end_slice - start_slice) + 1;

	RemoveFFTWPadding();

	input_file->WriteSlicesToDisk(start_slice, end_slice, real_values);

	AddFFTWPadding(); // to go back

}

//!> \brief Take a contiguous set of values, and add the FFTW padding.

void Image::AddFFTWPadding()
{
	MyDebugAssertTrue(is_in_memory, "Image not allocated!");

	int x,y,z;

	long current_write_position = real_memory_allocated - 3;
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

			current_write_position -= 2;
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

			current_read_position +=2;
		}
	}


}


