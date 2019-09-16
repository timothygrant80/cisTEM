/*
 * image_extender.h
 *
 *  Created on: Sep 16, 2019
 *      Author: himesb
 */

#ifndef IMAGE_EXTENDER_H_
#define IMAGE_EXTENDER_H_

/*
 * 	increase or decrease image dimensions, with or without padding. Optionally split a large image into smaller tiles, optionally padded for convolution.
 * 	return re-assembled image including only the valid areas after op.
 *
 */

struct  SubCoordinates
{
	// Track the input and ouput image sizes
	int nx_trimmed,  ny_trimmed, nz_trimmed;

	// Logigal indices describing the valid area in the sub-image
	// Use these to zero any padding values prior to re-inserting in the full image with ClipInto()
	int x_lower_bound, y_lower_bound, z_lower_bound;
	int x_upper_bound, y_upper_bound, z_upper_bound;

	int x_division 	= 1;
	int y_division	= 1;
	int z_division	= 1;
	int x_padding	= 0;
	int y_padding	= 0;
	int z_padding	= 0;

	// Shift vector describing the offset between the sub_image origin in the original image
	// and the origin of the original image. The negative of these are used to cut out with ClipInto() while the positive
	// are used to re-insert;
	int ox, oy, oz;
};

class ImageExtender
{

public:

	ImageExtender();
	ImageExtender( const ImageExtender &other_extended_image ); // copy constructor
	virtual ~ImageExtender();

	void Deallocate();

	ImageExtender & operator = (const ImageExtender &t);
	ImageExtender & operator = (const ImageExtender *t);

	int n_sub_regions;
	int convolution_padding;
	int max_padding = 0; // for histogram
	int nx_original, ny_original, nz_original;
	int padding_jump_value;


	bool is_initialized;
	Image* subImage;
	SubCoordinates* coords;

	void Init(int n_sub_regions, int convolution_padding = 0);
	// Note that Split and ReAssemble will simply be a resize if n_sub_regions = 1 & convolution padding > 0;
	void Split(Image &input_image);
	void ReAssemble(Image* output_image, int region_to_insert = -1);


private:

};

#endif /* IMAGE_EXTENDER_H_ */
