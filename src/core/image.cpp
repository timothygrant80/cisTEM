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
		is_in_real_space = should_be_in_real_space;

		if (wanted_x_size == logical_x_dimension && wanted_y_size == logical_y_dimension && wanted_z_size == logical_z_dimension)
		{
			// everything is already done..

			is_in_real_space = should_be_in_real_space;
			wxPrintf("returning\n");

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
	physical_upper_bound_complex_x = logical_x_dimension / 2;
	physical_upper_bound_complex_y = logical_y_dimension - 1;
	physical_upper_bound_complex_z = logical_z_dimension - 1;

	UpdatePhysicalAddressOfBoxCenter();

	physical_index_of_first_negative_frequency_x = logical_x_dimension / 2 + 1;
	physical_index_of_first_negative_frequency_y = logical_y_dimension / 2 + 1;
	physical_index_of_first_negative_frequency_z = logical_z_dimension / 2 + 1;

    // Update the Fourier voxel size

	fourier_voxel_size_x = 1.0 / float(logical_x_dimension);
	fourier_voxel_size_y = 1.0 / float(logical_y_dimension);
	fourier_voxel_size_z = 1.0 / float(logical_z_dimension);

    logical_lower_bound_complex_x = -logical_x_dimension / 2;
    logical_upper_bound_complex_x =  logical_x_dimension / 2;
    logical_lower_bound_real_x    = -logical_x_dimension / 2;
    logical_upper_bound_real_x    =  logical_x_dimension / 2 - 1;

    logical_lower_bound_complex_y = -logical_y_dimension / 2;
    logical_upper_bound_complex_y =  logical_y_dimension / 2;
    logical_lower_bound_real_y    = -logical_y_dimension / 2;
    logical_upper_bound_real_y    =  logical_y_dimension / 2 - 1;

    logical_lower_bound_complex_z = -logical_z_dimension / 2;
    logical_upper_bound_complex_z =  logical_z_dimension / 2;
    logical_lower_bound_real_z    = -logical_z_dimension / 2;
    logical_upper_bound_real_z    =  logical_z_dimension / 2 - 1;
/*
	if (IsEven(logical_x_dimension) == true) physical_upper_bound_complex_x = logical_x_dimension / 2;
	else physical_upper_bound_complex_x = (logical_x_dimension - 1) / 2;

	physical_upper_bound_complex_y = logical_y_dimension - 1;
	physical_upper_bound_complex_z = logical_z_dimension - 1;

	UpdatePhysicalAddressOfBoxCenter();

    // In each dimension, the physical index of the first pixel which stores negative frequencies
    // Note that we never actually store negative frequencies for the first dimension. However, it is sometimes useful
    // to pretend as though we did (for example when generating real-space images of CTFs).
/*
	if (IsEven(logical_x_dimension) == true) physical_index_of_first_negative_frequency_x = (logical_x_dimension / 2 + 2) - 1;
	else physical_index_of_first_negative_frequency_x = ((logical_x_dimension + 3) / 2) - 1;

	if (IsEven(logical_y_dimension) == true) physical_index_of_first_negative_frequency_y = (logical_y_dimension / 2 + 2) - 1;
	else physical_index_of_first_negative_frequency_y = ((logical_y_dimension + 3) / 2) - 1;

	if (IsEven(logical_z_dimension) == true) physical_index_of_first_negative_frequency_z = (logical_z_dimension / 2 + 2) - 1;
	else physical_index_of_first_negative_frequency_z = ((logical_z_dimension + 3) / 2) - 1;/

	physical_index_of_first_negative_frequency_x = logical_x_dimension / 2;
	physical_index_of_first_negative_frequency_y = logical_y_dimension / 2;
	physical_index_of_first_negative_frequency_z = logical_z_dimension / 2;

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

	max_array_value = (logical_x_dimension * logical_y_dimension * logical_z_dimension) - 1;*/
}

//!>  \brief  Returns the physical address of the image origin

void Image::UpdatePhysicalAddressOfBoxCenter()
{
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

void Image::SetToConstant(float wanted_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	for (long pixel_counter = 0; pixel_counter < real_memory_allocated; pixel_counter++)
	{
		real_values[pixel_counter] = wanted_value;
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

    if (physical_index >= physical_index_of_first_negative_frequency_x)
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

				pixel_counter+=2;
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

void Image::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");

	Image temp_image;

	temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);
	ClipInto(&temp_image, wanted_padding_value);

	CopyFrom(&temp_image);
}

void Image::CopyFrom(Image *other_image)
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

	for (k=0; k < physical_upper_bound_complex_z; k++)
	{
		k_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k);
		phase_z = ReturnPhaseFromShift(wanted_z_shift, k_logical, logical_z_dimension);

		for (j = 0; j < physical_upper_bound_complex_y; j++)
		{
			j_logical = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j);
			phase_y = ReturnPhaseFromShift(wanted_y_shift, j_logical, logical_y_dimension);
		}

		for (i = 0; i < physical_upper_bound_complex_x; i++)
		{
			phase_x = ReturnPhaseFromShift(wanted_x_shift, i, logical_x_dimension);
			total_phase_shift = Return3DPhaseFromIndividualDimensions(phase_x, phase_y, phase_z);

			complex_values[pixel_counter] *= total_phase_shift;

			pixel_counter++;
		}
	}

	if (need_to_fft == true) BackwardFFT();

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

	for (k = 0; k < logical_z_dimension; k++)
	{
		z_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k) * pow(fourier_voxel_size_z,2);

		for (j = 0; j < logical_y_dimension; j++)
		{
			y_coord = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j) * pow(fourier_voxel_size_y, 2);

			for (i = 0; i < physical_upper_bound_complex_x; i++)
			{
				x_coord = i * pow(fourier_voxel_size_x, 2);

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

	for (pixel_counter = 0, logical_y_dimension; pixel_counter++;)
	{
		for (width_counter = 0; width_counter < vertical_half_width; width_counter++)
		{
			complex_values[width_counter + (pixel_counter * physical_upper_bound_complex_x)] = 0.0 + 0.0 * I;
		}
	}


	for (pixel_counter = 0; pixel_counter < physical_upper_bound_complex_x; pixel_counter++)
	{
		for (width_counter = -width_counter; width_counter <  vertical_half_width; width_counter++)
		{
			complex_values[ReturnFourier1DAddressFromPhysicalAddress(pixel_counter, logical_y_dimension / 2 + width_counter, 1)] = 0.0 + 0.0 * I;

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
	long pixel_counter;

	float x_shift_to_apply;
	float y_shift_to_apply;
	float z_shift_to_apply;

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT();
	}

	if (object_is_centred_in_box)
	{
		x_shift_to_apply = float(physical_address_of_box_center_x) - 1.0;
		y_shift_to_apply = float(physical_address_of_box_center_y) - 1.0;
		z_shift_to_apply = float(physical_address_of_box_center_z) - 1.0;
	}
	else
	{
		if (IsEven(logical_x_dimension) == true)
		{
			x_shift_to_apply = float(physical_address_of_box_center_x) - 1.0;
		}
		else
		{
			x_shift_to_apply = float(physical_address_of_box_center_x);
		}

		if (IsEven(logical_y_dimension) == true)
		{
			y_shift_to_apply = float(physical_address_of_box_center_y) - 1.0;
		}
		else
		{
			y_shift_to_apply = float(physical_address_of_box_center_y);
		}

		if (IsEven(logical_z_dimension) == true)
		{
			z_shift_to_apply = float(physical_address_of_box_center_z) - 1.0;
		}
		else
		{
			z_shift_to_apply = float(physical_address_of_box_center_z);
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
	MyDebugAssertTrue(HasSameDimensionsAs(other_image) == false, "Images are in different spaces");

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
	MyDebugAssertTrue(is_in_real_space == false, "Image not in real space");

	int k;
	int j;
	int i;

	int z;
	int y;
	int x;

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

	if (radii_are_fractional == true)
	{
		inv_max_radius_sq_x = 1.0 / pow(physical_address_of_box_center_x, 2);
		inv_max_radius_sq_y = 1.0 / pow(physical_address_of_box_center_y, 2);
		inv_max_radius_sq_z = 1.0 / pow(physical_address_of_box_center_z, 2);

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

						if (distance_from_origin > wanted_min_radius && distance_from_origin < wanted_max_radius)
						{
							if (real_values[pixel_counter] > found_peak.value)
							{
								found_peak.value = real_values[pixel_counter];
								found_peak.x = i - physical_address_of_box_center_x;
								found_peak.y = j - physical_address_of_box_center_y;
								found_peak.z = k - physical_address_of_box_center_z;
							}

						}

						pixel_counter++;
					}
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

					if (distance_from_origin > wanted_min_radius && distance_from_origin < wanted_max_radius)
					{
						if (real_values[pixel_counter] > found_peak.value)
						{
							found_peak.value = real_values[pixel_counter];
							found_peak.x = i - physical_address_of_box_center_x;
							found_peak.y = j - physical_address_of_box_center_y;
							found_peak.z = k - physical_address_of_box_center_z;
						}
					}

					pixel_counter++;
				}
			}
		}
	}

	return found_peak;
}

Peak Image::FindPeakWithParabolaFit(float wanted_min_radius, float wanted_max_radius)
{
	MyDebugAssertTrue(is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(is_in_real_space == false, "Image not in real space");
	MyDebugAssertTrue(logical_z_dimension != 1, "Only 2D images supported for now");

	int x;
	int y;

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
			ImportantSquare[x][y] = ReturnRealPixelFromPhysicalCoord(integer_peak.x - 2 + x, integer_peak.y - 2 + x, 0);
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

	    if (ymax > 1.0 || ymax < -1.0) ymax = 0.0;
	    if (xmax > 1.0 || xmax < -1.0) xmax = 0.0;

	    found_peak.y = integer_peak.y + ymax;
	    found_peak.x = integer_peak.y + xmax;

	    found_peak.value = 4.*c1*c3*c6 - c1*c5*c5 - c2*c2*c6 + c2*c4*c5 - c4*c4*c3;
	    found_peak.value *= (average / denomin);

	    if (fabs((found_peak.value - integer_peak.value) / (found_peak.value + integer_peak.value)) > 0.15) found_peak.value = integer_peak.value;

    }

    return found_peak;
}

inline int Image::ReturnFourier1DAddressFromPhysicalAddress(int wanted_x, int wanted_y, int wanted_z)
{
	MyDebugAssertTrue(wanted_x >= 0 && wanted_x <= physical_address_of_box_center_x && wanted_y >= 0 && wanted_y <= physical_upper_bound_complex_y && wanted_z >= 0 && wanted_z <= physical_upper_bound_complex_z, "Address out of bounds!" )
	return (((physical_upper_bound_complex_x + 1) * (physical_upper_bound_complex_y + 1)) * wanted_z) + ((physical_upper_bound_complex_x + 1) * wanted_y) + wanted_x;
}

inline int Image::ReturnFourier1DAddressFromLogicalCoord(int wanted_x, int wanted_y, int wanted_z)
{
	MyDebugAssertTrue(wanted_x >= logical_lower_bound_complex_x && wanted_x <=logical_upper_bound_complex_x && wanted_y >= logical_lower_bound_complex_y && wanted_y <= logical_upper_bound_complex_y && wanted_z >= logical_lower_bound_complex_z && wanted_z <= logical_upper_bound_complex_z, "Coord out of bounds!")

	int physical_x_address;
	int physical_y_address;
	int physical_z_address;

	if (wanted_x >= 0)
	{
		physical_x_address = wanted_x;

		if (wanted_y >= 0)
		{
			physical_y_address = wanted_y;
		}
		else
		{
			physical_y_address = logical_y_dimension + wanted_y;
		}

		if (wanted_z >= 0)
		{
			physical_z_address = wanted_z;
		}
		else
		{
			physical_z_address = logical_z_dimension + wanted_z;
		}
	}
	else
	{
		physical_x_address = -wanted_x;

		if (wanted_y > 0)
		{
			physical_y_address = logical_y_dimension - wanted_y;
		}
		else
		{
			physical_y_address = -wanted_y;
		}

		if (wanted_z > 0)
		{
			physical_z_address = logical_z_dimension - wanted_z;
		}
		else
		{
			physical_z_address = -wanted_z;
		}
	}

	return ReturnFourier1DAddressFromPhysicalAddress(physical_x_address, physical_y_address, physical_z_address);
}

inline fftw_complex Image::ReturnComplexPixelFromLogicalCoord(int wanted_x, int wanted_y, int wanted_z, float out_of_bounds_value)
{
	if (wanted_x < logical_lower_bound_complex_x || wanted_x > logical_upper_bound_complex_x || wanted_y < logical_lower_bound_complex_y ||wanted_y > logical_upper_bound_complex_y || wanted_z < logical_lower_bound_complex_z || wanted_z > logical_upper_bound_complex_z)
	{
		return out_of_bounds_value;
	}
	else return complex_values[ReturnFourier1DAddressFromLogicalCoord(wanted_x, wanted_y, wanted_z)];
}


