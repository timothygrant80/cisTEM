#include "../../core/core_headers.h"

class
Resize : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(Resize)

// override the DoInteractiveUserInput

void Resize::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("Resize", 1.0);

	std::string input_filename		=		my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output image file name", "Filename of output image", "output.mrc", false );
	bool        is_a_volume         =       my_input->GetYesNoFromUser("Is the input a volume", "Yes if it is a 3D", "NO");
	int         new_x_size          =       my_input->GetIntFromUser("New X-Size", "Wanted new X size", "100",1);
	int         new_y_size          =       my_input->GetIntFromUser("New Y-Size", "Wanted new Y size", "100",1);

	if (is_a_volume == true)
	{
		        new_z_size          =       my_input->GetIntFromUser("New Z-Size", "Wanted new Z size", "100",1);
	}

	bool       should_normalise     =       my_input->GetYesNoFromUser("Normalize and Zero-float Input?", "If yes, images will also be normalized and zero floated", "NO");

	delete my_input;

	my_current_job.Reset(7);
	my_current_job.ManualSetArguments("ttbiiib", input_filename.c_str(), output_filename.c_str(), is_a_volume, new_x_size, new_y_size, new_z_size, should_normalise);
}

// override the do calculation method which will be what is actually run..

bool Resize::DoCalculation()
{

	std::string	input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	bool        is_a_volume                         = my_current_job.arguments[2].ReturnBoolArgument();
	int         new_x_size                          = my_current_job.arguments[3].ReturnIntegerArgument();
	int         new_y_size                          = my_current_job.arguments[4].ReturnIntegerArgument();
	int         new_z_size                          = my_current_job.arguments[5].ReturnIntegerArgument();
	bool        should_normalise                    = my_current_job.arguments[6].ReturnBoolArgument();
	float		pixel_size;

	MRCFile my_input_file(input_filename,false);
	MRCFile my_output_file(output_filename,true);

	Image my_image;

	pixel_size = my_input_file.ReturnPixelSize();

	if (is_a_volume == true)
	{
		wxPrintf("\nResizing Volume...\n\n");
		my_image.ReadSlices(&my_input_file, 1, my_input_file.ReturnNumberOfSlices());

		if (should_normalise == true)
		{
			my_image.ZeroFloatAndNormalize(1);
		}

		my_image.Resize(new_x_size, new_y_size, new_z_size, my_image.ReturnAverageOfRealValuesOnEdges());
		my_image.WriteSlices(&my_output_file,1, new_z_size);

	}
	else
	{
		wxPrintf("\nResizing Images...\n\n");
		ProgressBar *my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices());

		for ( long image_counter = 0; image_counter < my_input_file.ReturnNumberOfSlices(); image_counter++ )
		{
			my_image.ReadSlice(&my_input_file,image_counter+1);

			if (should_normalise == true)
			{
				my_image.ZeroFloatAndNormalize(1);
			}

			my_image.Resize(new_x_size, new_y_size, new_z_size, my_image.ReturnAverageOfRealValuesOnEdges());

			my_image.WriteSlice(&my_output_file,image_counter+1);
			my_progress->Update(image_counter + 1);

		}

		delete my_progress;
		wxPrintf("\n\n");

	}

	my_output_file.SetPixelSize(pixel_size);
	my_output_file.WriteHeader();

	return true;
}
