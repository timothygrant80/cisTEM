/*
 * image_extender.cpp
 *
 *  Created on: Sep 16, 2019
 *      Author: himesb
 */

#include "core_headers.h"

ImageExtender::ImageExtender()
{
	is_initialized = false;
}

ImageExtender::~ImageExtender()
{
	Deallocate();
}


ImageExtender::ImageExtender( const ImageExtender &extended_image) // copy constructor
{

	*this = extended_image;
}

ImageExtender & ImageExtender::operator = (const ImageExtender &extended_image)
{
	*this = &extended_image;
	return *this;
}


ImageExtender & ImageExtender::operator = (const ImageExtender *extended_image)
{
	// Check for self assignment
	if(this != extended_image)
	{

//		MyAssertTrue(other_gpu_image->is_in_memory_gpu, "Other image Memory not allocated");

		if (this->is_initialized == true)
		{
			this->Deallocate();
		}
		else
		{
			this->Init(extended_image->n_sub_regions, extended_image->convolution_padding);
		}

	}

   return *this;
}

void ImageExtender::Init(int n_sub_regions, int convolution_padding)
{

	MyAssertTrue(n_sub_regions == 1 || n_sub_regions == 2, "Only 1 or 2 sub-images are currently supported.");

	this->n_sub_regions = n_sub_regions;
	this->convolution_padding = convolution_padding;

	subImage = 	new Image[n_sub_regions];
	coords	=	new SubCoordinates[n_sub_regions];

	is_initialized = true;
}

void ImageExtender::Split(Image &input_image)
{

	nx_original = input_image.logical_x_dimension;
	ny_original = input_image.logical_y_dimension;
	padding_jump_value = input_image.padding_jump_value;

	for (int iSC = 0; iSC < n_sub_regions; iSC++)
	{

		// TODO move these into the coords struct
		if ( nx_original >= ny_original )
		{
			coords[iSC].x_division = n_sub_regions;
			coords[iSC].x_padding = convolution_padding;
			coords[iSC].y_padding = 0;
		}
		else
		{
			coords[iSC].y_division = n_sub_regions;
			coords[iSC].y_padding = convolution_padding;
			coords[iSC].x_padding = 0;
		}

		int factorizable_x = nx_original / coords[iSC].x_division + coords[iSC].x_padding;
		int factorizable_y = ny_original / coords[iSC].y_division + coords[iSC].y_padding;


		bool DO_FACTORIZATION = true;
		bool MUST_BE_POWER_OF_TWO = false; // Required for half-preicision xforms
		int MUST_BE_FACTOR_OF_FOUR = 0; // May be faster
		const int max_number_primes = 6;
		int primes[max_number_primes] = {2,3,5,7,9,13};
		float max_reduction_by_fraction_of_reference = 0.5f;
		float max_increas_by_fraction_of_image = 0.10f;
		int max_padding = 0; // To restrict histogram calculation
		int factor_result_neg;
		int factor_result_pos;

		// for 5760 this will return
		// 5832 2     2     2     3     3     3     3     3     3 - this is ~ 10% faster than the previous solution BUT
		// 6144  2     2     2     2     2     2     2     2     2     2     2     3 is another ~ 5% faster
		if (DO_FACTORIZATION)
		{
		for ( int i = 0; i < max_number_primes; i++ )
		{

			factor_result_neg = ReturnClosestFactorizedLower(nx_original, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
			factor_result_pos = ReturnClosestFactorizedUpper(nx_original, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
			// We don't want to shrink the dimension that has been trimmed;
			if ( coords[iSC].x_division > 1 ) factor_result_neg = 0;

	//		wxPrintf("i, result, score = %i %i %g\n", i, factor_result, logf(float(abs(i) + 100)) * factor_result);
			if ( (float)(nx_original - factor_result_neg) < (float)convolution_padding * max_reduction_by_fraction_of_reference)
			{
				factorizable_x = factor_result_neg;
				break;
			}
			if ((float)(-nx_original + factor_result_pos) < (float)input_image.logical_x_dimension * max_increas_by_fraction_of_image)
			{
				factorizable_x = factor_result_pos;
			break;
			}

		}

		for ( int i = 0; i < max_number_primes; i++ )
		{

			factor_result_neg = ReturnClosestFactorizedLower(ny_original, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
			factor_result_pos = ReturnClosestFactorizedUpper(ny_original, primes[i], true, MUST_BE_FACTOR_OF_FOUR);
			// We don't want to shrink the dimwnsion that has been cut
			if ( coords[iSC].y_division > 1 ) factor_result_neg = 1;

//		wxPrintf("i, result, score = %i %i %g\n", i, factor_result, logf(float(abs(i) + 100)) * factor_result);
			if ( (float)(ny_original - factor_result_neg) < (float)convolution_padding * max_reduction_by_fraction_of_reference)
			{
				factorizable_y = factor_result_neg;
				break;
			}
			if ((float)(-ny_original + factor_result_pos) < (float)input_image.logical_y_dimension * max_increas_by_fraction_of_image)
			{
				factorizable_y = factor_result_pos;
				break;
			}

		}

		if (factorizable_x - nx_original > max_padding) max_padding = factorizable_x - nx_original;
		if (factorizable_y - ny_original > max_padding) max_padding = factorizable_y - ny_original;

	//	// Temp override for profiling:
		//factorizable_x = 1024*4+512;
		//factorizable_y = 1024*4+512;

		wxPrintf("old x, y; new x, y = %i %i %i %i\n", input_image.logical_x_dimension, input_image.logical_y_dimension, factorizable_x, factorizable_y);


		coords[iSC].nx_trimmed = factorizable_x;
		coords[iSC].ny_trimmed = factorizable_y;
		coords[iSC].ox = factorizable_x/2 - nx_original;
		coords[iSC].oy = factorizable_y/2 - ny_original;

#ifdef USEGPU
		subImage[iSC].Allocate(coords[iSC].nx_trimmed, coords[iSC].ny_trimmed, 1, true);
		input_image.ClipInto(&subImage[iSC], 0.0f, false, 1.0f, coords[iSC].ox, coords[iSC].oy, 0);
#else
			input_image.Resize(factorizable_x, factorizable_y, 1, input_image.ReturnAverageOfRealValuesOnEdges());
#endif

		} // end of if on do factorization
		} // end of loop on subregions.

}


void ImageExtender::ReAssemble(Image* output_image, int region_to_insert)
{

	Image buffer;
	buffer.CopyFrom(output_image); // TODO maybe keep this as a member variable if it could be reused.

	int i,j,k;
	long iPixel;
	// the default is to reassemble the full image
	if (region_to_insert < 0)
	{
		for (int iSC = 0; iSC < n_sub_regions; iSC++)
		{
			buffer.SetToConstant(0.0f);
			subImage[iSC].ClipInto(&buffer, 0.0f, false, 1.0f, -1*coords[iSC].ox, -1*coords[iSC].oy, 0);
			iPixel = 0;
			// Zero regions that are not valid
			// This will be more complicated if more than 2 regions are allowed. For now though, the non-convolution padded is already
			// taken care of by ClipInto.


			if (iSC == 0)
			{
				for (k = 0; k < coords[iSC].nz_original; k++)
				{
					for (j = 0; j < coords[iSC].ny_original; j++)
					{
						for (i = 0; i < coords[iSC].nx_original; i++)
						{

							if (coords[iSC].x_padding > 0 && i > nx_original/2) { ;}
							else if (coords[iSC].x_padding > 0 && i < nx_original/2+1)

							iPixel++;
						}
						iPixel += padding_jump_value;
					}
				}
			}
			else
			{

			}
		}


	}






}

void ImageExtender::Deallocate()
{
	if (is_initialized)
	{
		delete [] subImage;
		delete [] coords;
	}
}




