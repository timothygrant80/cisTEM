#include "../../core/core_headers.h"
#include <wx/dir.h>

class
RemoveRelionStripes : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

void SetHorizontalLineMaskToOne(Image &image_to_set, int wanted_line);
void SetVerticalLineMaskToOne(Image &image_to_set, int wanted_line);

float ReturnCorrelationBetweenTwoHorizontalLines(Image &image_to_use, int first_line, int second_line);
float ReturnCorrelationBetweenTwoVerticalLines(Image &image_to_use, int first_line, int second_line);


IMPLEMENT_APP(RemoveRelionStripes)

// override the DoInteractiveUserInput

void RemoveRelionStripes::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("RemoveRelionStripes", 1.0);

	std::string input_filename			=		my_input->GetFilenameFromUser("Input file name", "Filename of input image", "input.mrc", true );
	std::string output_filename			=		my_input->GetFilenameFromUser("Output MRC file name", "Filename of output image which should remove repeated edge pixels", "output.mrc", false );
	float       dot_product_threshold	=		my_input->GetFloatFromUser("Dot Product Threshold", "", "0.9");
	bool		invert_images			= 		my_input->GetYesNoFromUser("Invert image contrast?", "", "YES");


	delete my_input;

	my_current_job.ManualSetArguments("ttfb", input_filename.c_str(), output_filename.c_str(), dot_product_threshold, invert_images);
}

// override the do calculation method which will be what is actually run..

bool RemoveRelionStripes::DoCalculation()
{


	std::string	input_filename 		= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 	= my_current_job.arguments[1].ReturnStringArgument();
	float dot_product_threshold		= my_current_job.arguments[2].ReturnFloatArgument();
	bool invert_images				= my_current_job.arguments[3].ReturnBoolArgument();

	ImageFile input_file;
	MRCFile output_file;

	Image input_image;
	Image buffer_image;
	Image mask_image;

	float current_correlation;
	int line_counter;
	long pixel_counter;

	int i,j;

	input_file.OpenFile(input_filename, false);
	output_file.OpenFile(output_filename, true);

	mask_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	buffer_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);


	wxPrintf("\nRemoving Stripes...\n\n");

	ProgressBar *my_progress = new ProgressBar(input_file.ReturnNumberOfSlices());

	for (int counter = 1; counter <= input_file.ReturnNumberOfSlices(); counter++ )
	{
		input_image.ReadSlice(&input_file, counter);
		if (input_image.ContainsRepeatedLineEdges() == true) input_image.SetToConstant(0.0f);

		/*

		if (invert_images == true) input_image.InvertRealValues();
		mask_image.SetToConstant(0.0f);

		buffer_image.CopyFrom(&input_image);
		buffer_image.ForwardFFT();
		buffer_image.CosineMask(0.45,0.05);
		buffer_image.BackwardFFT();

		buffer_image.QuickAndDirtyWriteSlice("/tmp/junk.mrc", 1);
		// from the top..


		for (line_counter = 2; line_counter < input_image.logical_y_dimension; line_counter++)
		{
			current_correlation = ReturnCorrelationBetweenTwoHorizontalLines(buffer_image, 1, line_counter);
			wxPrintf("line %i = %f\n", line_counter, current_correlation);

			if (current_correlation > dot_product_threshold)
			{
				if (line_counter == 2)
				{
					SetHorizontalLineMaskToOne(mask_image, 0);
					SetHorizontalLineMaskToOne(mask_image, 1);
				}

				SetHorizontalLineMaskToOne(mask_image, line_counter);
			}
			else
			{
				if (line_counter != 2) SetHorizontalLineMaskToOne(mask_image, line_counter);
				break;
			}
		}

		// from the bottom..


		for (line_counter = input_image.logical_y_dimension - 3; line_counter >= 0; line_counter--)
		{
			current_correlation = ReturnCorrelationBetweenTwoHorizontalLines(buffer_image, input_image.logical_y_dimension - 2, line_counter);
			wxPrintf("line %i = %f\n", line_counter, current_correlation);

			if (current_correlation > dot_product_threshold)
			{
				if (line_counter == input_image.logical_y_dimension - 3)
				{
					SetHorizontalLineMaskToOne(mask_image, input_image.logical_y_dimension - 1);
					SetHorizontalLineMaskToOne(mask_image, input_image.logical_y_dimension - 2);
				}

				SetHorizontalLineMaskToOne(mask_image, line_counter);
			}
			else
			{
				if (line_counter != input_image.logical_y_dimension - 3) SetHorizontalLineMaskToOne(mask_image, line_counter);
				break;
			}
		}

		// from the left...


		for (line_counter = 2; line_counter < input_image.logical_x_dimension; line_counter++)
		{
			current_correlation = ReturnCorrelationBetweenTwoVerticalLines(buffer_image, 1, line_counter);
			wxPrintf("line %i = %f\n", line_counter, current_correlation);

			if (current_correlation > dot_product_threshold)
			{
				if (line_counter == 2)
				{
					SetVerticalLineMaskToOne(mask_image, 0);
					SetVerticalLineMaskToOne(mask_image, 1);
				}

				SetVerticalLineMaskToOne(mask_image, line_counter);
			}
			else
			{
				if (line_counter != 2) SetVerticalLineMaskToOne(mask_image, line_counter);
				break;
			}
		}


		// from the right



		for (line_counter = input_image.logical_x_dimension - 3; line_counter >= 0; line_counter--)
		{
			current_correlation = ReturnCorrelationBetweenTwoVerticalLines(buffer_image, input_image.logical_x_dimension - 2, line_counter);
			wxPrintf("line %i = %f\n", line_counter, current_correlation);

			if (current_correlation > dot_product_threshold)
			{
				if (line_counter == input_image.logical_x_dimension - 3)
				{
					SetVerticalLineMaskToOne(mask_image, input_image.logical_x_dimension - 1);
					SetVerticalLineMaskToOne(mask_image, input_image.logical_x_dimension - 2);
				}

				SetVerticalLineMaskToOne(mask_image, line_counter);
			}
			else
			{
				if (line_counter != input_image.logical_x_dimension - 3) SetVerticalLineMaskToOne(mask_image, line_counter);
				break;
			}
		}



		// ok.. should have a mask..

		EmpiricalDistribution current_distribution;
		pixel_counter = 0;

		for (j = 0; j < input_image.logical_y_dimension; j++)
		{
			for (i = 0; i < input_image.logical_x_dimension; i++)
			{
				if (mask_image.real_values[pixel_counter] == 0.0f) current_distribution.AddSampleValue(input_image.real_values[pixel_counter]);
				pixel_counter++;
			}
			pixel_counter += input_image.padding_jump_value;
		}

		// normalize over the correct area..

		if (current_distribution.IsConstant())
		{
			input_image.AddConstant(-current_distribution.GetSampleMean());
		}
		else
		{
			input_image.AddMultiplyConstant(-current_distribution.GetSampleMean(), 1.0f/sqrtf(current_distribution.GetSampleVariance()));
		}


		// now set the weird lines to the average..

		pixel_counter = 0;

		for (j = 0; j < input_image.logical_y_dimension; j++)
		{
			for (i = 0; i < input_image.logical_x_dimension; i++)
			{
				if (mask_image.real_values[pixel_counter] == 1.0f) input_image.real_values[pixel_counter] = 0.0f;
				pixel_counter++;
			}
			pixel_counter += input_image.padding_jump_value;
		}

*/
		input_image.WriteSlice(&output_file, counter);
		my_progress->Update(counter);
	}

	delete my_progress;


	return true;
}

float ReturnCorrelationBetweenTwoHorizontalLines(Image &image_to_use, int first_line, int second_line)
{
	float correlation = 0.0f;
	float ab_buffer = 0.0f;
	float aa_buffer = 0.0f;
	float bb_buffer = 0.0f;

	float first_line_average = 0.0f;
	float second_line_average = 0.0f;

	int counter;

	for (counter = 0; counter < image_to_use.logical_x_dimension; counter++)
	{
		first_line_average += image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)];
		second_line_average += image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)];
	}

	first_line_average /= image_to_use.logical_x_dimension;
	second_line_average /= image_to_use.logical_x_dimension;

	for (counter = 0; counter < image_to_use.logical_x_dimension; counter++)
	{
		ab_buffer += (image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)] - first_line_average) * (image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)] - second_line_average);
		aa_buffer += powf(image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, first_line, 0)] - first_line_average, 2);
		bb_buffer += powf(image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(counter, second_line, 0)] - second_line_average, 2);
	}

	aa_buffer /= image_to_use.logical_x_dimension - 1;
	bb_buffer /= image_to_use.logical_x_dimension - 1;

	correlation = ab_buffer / sqrtf(aa_buffer * bb_buffer);
	return correlation / image_to_use.logical_x_dimension;
}

float ReturnCorrelationBetweenTwoVerticalLines(Image &image_to_use, int first_line, int second_line)
{
	float correlation = 0.0f;
	float ab_buffer = 0.0f;
	float aa_buffer = 0.0f;
	float bb_buffer = 0.0f;

	float first_line_average = 0.0f;
	float second_line_average = 0.0f;

	int counter;

	for (counter = 0; counter < image_to_use.logical_y_dimension; counter++)
	{
		first_line_average += image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)];
		second_line_average += image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)];
	}

	first_line_average /= image_to_use.logical_y_dimension;
	second_line_average /= image_to_use.logical_y_dimension;

	for (counter = 0; counter < image_to_use.logical_y_dimension; counter++)
	{
		ab_buffer += (image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)] - first_line_average) * (image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)] - second_line_average);
		aa_buffer += powf(image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(first_line, counter, 0)] - first_line_average, 2);
		bb_buffer += powf(image_to_use.real_values[image_to_use.ReturnReal1DAddressFromPhysicalCoord(second_line, counter, 0)] - second_line_average, 2);
	}

	aa_buffer /= image_to_use.logical_y_dimension - 1;
	bb_buffer /= image_to_use.logical_y_dimension - 1;

	correlation = ab_buffer / sqrtf(aa_buffer * bb_buffer);
	return correlation  / image_to_use.logical_y_dimension;
}

void SetHorizontalLineMaskToOne(Image &image_to_set, int wanted_line)
{

	for (int counter = 0; counter < image_to_set.logical_x_dimension; counter++)
	{
		image_to_set.real_values[image_to_set.ReturnReal1DAddressFromPhysicalCoord(counter, wanted_line, 0)] = 1.0f;
	}
}

void SetVerticalLineMaskToOne(Image &image_to_set, int wanted_line)
{

	for (int counter = 0; counter < image_to_set.logical_y_dimension; counter++)
	{
		image_to_set.real_values[image_to_set.ReturnReal1DAddressFromPhysicalCoord(wanted_line, counter,  0)] = 1.0f;
	}
}
