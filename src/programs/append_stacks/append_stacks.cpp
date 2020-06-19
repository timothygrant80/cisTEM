#include "../../core/core_headers.h"

class
AppendStacks : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(AppendStacks)

// override the DoInteractiveUserInput

void AppendStacks::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("AppendStacks", 1.0);

	std::string input_filename_one	=		my_input->GetFilenameFromUser("Input image file name #1", "Filename of stack to be appended to", "input_stack1.mrc", true );
	std::string input_filename_two	=		my_input->GetFilenameFromUser("Input image file name #2", "Filename of stack to be append", "input_stack2.mrc", true );

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tt", input_filename_one.c_str(), input_filename_two.c_str());

}

// override the do calculation method which will be what is actually run..

bool AppendStacks::DoCalculation()
{
	std::string	input_filename_one						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	input_filename_two						= my_current_job.arguments[1].ReturnStringArgument();


	MRCFile my_input_file_one(input_filename_one,false);
	ImageFile my_input_file_two(input_filename_two,false);

	int starting_number = my_input_file_one.ReturnNumberOfSlices();

	if (my_input_file_one.ReturnXSize() != my_input_file_two.ReturnXSize() || my_input_file_one.ReturnYSize() != my_input_file_two.ReturnYSize())
	{
		MyPrintfRed("\n\n Error: Image dimensions are not the same\n\n");
		exit(-1);
	}

	Image my_image_two;

	float input_pixel_size = my_input_file_one.ReturnPixelSize();

	wxPrintf("\nAdding Images...\n\n");
	ProgressBar *my_progress = new ProgressBar(my_input_file_two.ReturnNumberOfSlices());

	for ( long image_counter = 0; image_counter < my_input_file_two.ReturnNumberOfSlices(); image_counter++ )
	{
		my_image_two.ReadSlice(&my_input_file_two,image_counter+1);
		my_image_two.WriteSlice(&my_input_file_one,starting_number+image_counter+1);
		my_progress->Update(image_counter + 1);
	}

	delete my_progress;
	wxPrintf("\n\n");

	return true;
}
