#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"


void ConvertImageToBitmap(Image *input_image, wxBitmap *output_bitmap)
{
	MyDebugAssertTrue(input_image->logical_z_dimension == 1, "Only 2D images can be used");
	MyDebugAssertTrue(output_bitmap->GetDepth() == 24, "bitmap should be 24 bit");

	float image_min_value;
	float image_max_value;
	float range;
	float inverse_range;

	int current_grey_value;
	int i, j;
	long address = 0;


	if (input_image->logical_x_dimension != output_bitmap->GetWidth() || input_image->logical_y_dimension != output_bitmap->GetHeight())
	{
		output_bitmap->SetWidth(input_image->logical_x_dimension);
		output_bitmap->SetHeight(input_image->logical_y_dimension);
	}

	input_image->GetMinMax(image_min_value, image_max_value);

	range = image_max_value - image_min_value;
	inverse_range = 1. / range;

	wxNativePixelData pixel_data(*output_bitmap);

	if ( !pixel_data )
	{
	   MyPrintWithDetails("Can't access bitmap data");
	   abort();
	}

	wxNativePixelData::Iterator p(pixel_data);

	for (j = 0; j < input_image->logical_y_dimension; j++)
	{
		for (i = 0; i < input_image->logical_x_dimension; i++)
		{
			current_grey_value = myroundint((input_image->real_values[address] - image_min_value) * inverse_range);

			p.Red() = current_grey_value;
			p.Green() = current_grey_value;
			p.Blue() = current_grey_value;

			p++;
			address++;
		}

		address += input_image->padding_jump_value;
	}

}
