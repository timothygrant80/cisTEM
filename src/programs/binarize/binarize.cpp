#include "../../core/core_headers.h"

class
Binarize : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(Binarize)

// override the DoInteractiveUserInput

void Binarize::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("Binarize", 1.0);

	std::string input_filename		=		my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false );
	float binarize_threshold        =       my_input->GetFloatFromUser("Binarization Threshold", "Values less than this value will be 0.0, values equal to or greater will be 1.","1.0");

	delete my_input;

	my_current_job.Reset(3);
	my_current_job.ManualSetArguments("ttf", input_filename.c_str(), output_filename.c_str(), binarize_threshold);

}

// override the do calculation method which will be what is actually run..

bool Binarize::DoCalculation()
{
	std::string	input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	float binarize_threshold						= my_current_job.arguments[2].ReturnFloatArgument();

	MRCFile my_input_file(input_filename,false);
	MRCFile my_output_file(output_filename,true);

	Image my_image;

	float input_pixel_size = my_input_file.ReturnPixelSize();

	wxPrintf("\nBinarizing Images...\n\n");
	ProgressBar *my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices());

	for ( long image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices(); image_counter++ )
	{
		my_image.ReadSlice(&my_input_file,image_counter+1);
		my_image.Binarise(binarize_threshold);
		my_image.WriteSlice(&my_output_file,image_counter+1);
		my_progress->Update(image_counter + 1);
	}

	delete my_progress;
	wxPrintf("\n\n");


	my_output_file.SetPixelSize(input_pixel_size);
	my_output_file.WriteHeader();

	return true;
}
