#include "../../core/core_headers.h"

class
InvertStack : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(InvertStack)

// override the DoInteractiveUserInput

void InvertStack::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("InvertStack", 1.0);

	std::string input_filename_one	=		my_input->GetFilenameFromUser("Input image file name #1", "Filename of first stack to be multiplied", "input_stack1.mrc", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output image file name", "the inverted result", "output.mrc", false );

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tt", input_filename_one.c_str(), output_filename.c_str());

}

// override the do calculation method which will be what is actually run..

bool InvertStack::DoCalculation()
{
	std::string	input_filename_one						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 						= my_current_job.arguments[1].ReturnStringArgument();


	ImageFile my_input_file_one(input_filename_one,false);
	MRCFile my_output_file(output_filename,true);

	Image my_image_one;
	float input_pixel_size = my_input_file_one.ReturnPixelSize();

	wxPrintf("\nInverting Images...\n\n");
	ProgressBar *my_progress = new ProgressBar(my_input_file_one.ReturnNumberOfSlices());

	for ( long image_counter = 0; image_counter < my_input_file_one.ReturnNumberOfSlices(); image_counter++ )
	{
		my_image_one.ReadSlice(&my_input_file_one,image_counter+1);
		my_image_one.InvertRealValues();
		my_image_one.WriteSlice(&my_output_file,image_counter+1);
		my_progress->Update(image_counter + 1);
	}

	delete my_progress;
	wxPrintf("\n\n");


	my_output_file.SetPixelSize(input_pixel_size);
	my_output_file.WriteHeader();

	return true;
}
