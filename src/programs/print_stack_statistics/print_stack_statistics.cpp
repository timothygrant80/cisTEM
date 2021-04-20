#include "../../core/core_headers.h"

class
PrintStackStatistics : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(PrintStackStatistics)

// override the DoInteractiveUserInput

void PrintStackStatistics::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("PrintStackStatistics", 1.0);

	std::string input_filename_one	=		my_input->GetFilenameFromUser("Input image file name", "Filename of stack to print statistics for", "input_stack1.mrc", true );


	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("t", input_filename_one.c_str());

}

// override the do calculation method which will be what is actually run..

bool PrintStackStatistics::DoCalculation()
{
	std::string	input_filename_one						= my_current_job.arguments[0].ReturnStringArgument();

	ImageFile my_input_file_one(input_filename_one,false);

	Image my_image_one;
	float image_sigma;
	float image_average;
	float average_sigma = 0.0f;
	float average_average = 0.0f;

	float image_min;
	float image_max;
	float average_min = 0;
	float average_max = 0;

	wxPrintf("\n\n");

	wxPrintf("Image No.      Min Value.     Max Value.    Average Value.  Sigma.\n\n");

	for ( int image_counter = 0; image_counter < my_input_file_one.ReturnNumberOfSlices(); image_counter++ )
	{
		my_image_one.ReadSlice(&my_input_file_one,image_counter+1);
		image_sigma = sqrtf(my_image_one.ReturnVarianceOfRealValues(0.0, 0.0, 0.0, 0.0, false));
		image_average = my_image_one.ReturnAverageOfRealValues(0.0, false);
		my_image_one.GetMinMax(image_min, image_max);

		average_sigma += image_sigma;
		average_min += image_min;
		average_max += image_max;
		average_average += image_average;

		wxPrintf("  %7i\t%f\t%f\t%f\t%f\n", image_counter + 1, image_min, image_max, image_average, image_sigma);
	}

	wxPrintf ("\n  Average:\t%f\t%f\t%f\t%f\n\n\n", average_min / my_input_file_one.ReturnNumberOfSlices(), average_max /  my_input_file_one.ReturnNumberOfSlices(), average_average /  my_input_file_one.ReturnNumberOfSlices(), average_sigma / my_input_file_one.ReturnNumberOfSlices());


	return true;
}
