#include "../../core/core_headers.h"

class
NormalizeStack : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(NormalizeStack)

// override the DoInteractiveUserInput

void NormalizeStack::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("NormalizeStack", 1.0);

	std::string input_filename_one	=		my_input->GetFilenameFromUser("Input image stack", "Filename of the stack of images to normalize", "input_stack1.mrc", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output image file name", "the Normalised result", "output.mrc", false );
	float		wanted_sigma		=		my_input->GetFloatFromUser("Desired Sigma", "Wanted Sigma for normalization.","1.0");
	bool        should_zero_float   =       my_input->GetYesNoFromUser("Also Zero-float?", "If yes, images will also be zero floated (average value set to 0)", "YES");
	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("ttfb", input_filename_one.c_str(), output_filename.c_str(), wanted_sigma, should_zero_float);

}

// override the do calculation method which will be what is actually run..

bool NormalizeStack::DoCalculation()
{
	std::string	input_filename_one						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 						= my_current_job.arguments[1].ReturnStringArgument();
	float		wanted_sigma							= my_current_job.arguments[2].ReturnFloatArgument();
	bool        should_zero_float  						= my_current_job.arguments[3].ReturnBoolArgument();


	ImageFile my_input_file_one(input_filename_one,false);
	MRCFile my_output_file(output_filename,true);

	Image my_image_one;
	float input_pixel_size = my_input_file_one.ReturnPixelSize();

	wxPrintf("\nNormalizing Images...\n\n");
	ProgressBar *my_progress = new ProgressBar(my_input_file_one.ReturnNumberOfSlices());

	for ( long image_counter = 0; image_counter < my_input_file_one.ReturnNumberOfSlices(); image_counter++ )
	{
		my_image_one.ReadSlice(&my_input_file_one,image_counter+1);
		if (should_zero_float == false) my_image_one.Normalize(wanted_sigma);
		else my_image_one.ZeroFloatAndNormalize(wanted_sigma);

		my_image_one.WriteSlice(&my_output_file,image_counter+1);
		my_progress->Update(image_counter + 1);
	}

	delete my_progress;
	wxPrintf("\n\n");


	my_output_file.SetPixelSize(input_pixel_size);
	my_output_file.WriteHeader();

	return true;
}
