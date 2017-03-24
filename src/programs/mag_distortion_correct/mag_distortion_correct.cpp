#include "../../core/core_headers.h"

class
MagDistortionCorrectApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(MagDistortionCorrectApp)

// override the DoInteractiveUserInput

void MagDistortionCorrectApp::DoInteractiveUserInput()
{
	std::string input_filename;
	std::string output_filename;
	bool correct_mag_distortion;
	float mag_distortion_angle;
	float mag_distortion_major_scale;
	float mag_distortion_minor_scale;
	bool movie_is_gain_corrected;
	std::string gain_filename;
	bool resample_output;
	int new_x_size;
	int new_y_size;


	UserInput *my_input = new UserInput("mag_distortion_correct", 1.0);

	input_filename = my_input->GetFilenameFromUser("Input image(s) filename", "The input file, containing the images you want to correct", "my_images.mrc", true );
	output_filename = my_input->GetFilenameFromUser("Output distortion corrected image(s)", "The output file, containing the distortion corrected images", "my_images_corrected.mrc", false);
	mag_distortion_angle = my_input->GetFloatFromUser("Distortion Angle (Degrees)", "The distortion angle in degrees", "0.0");
	mag_distortion_major_scale = my_input->GetFloatFromUser("Major Scale", "The major axis scale factor", "1.0", 0.0);
	mag_distortion_minor_scale = my_input->GetFloatFromUser("Minor Scale", "The minor axis scale factor", "1.0", 0.0);;
	movie_is_gain_corrected = my_input->GetYesNoFromUser("Input stack is gain-corrected?", "Are the input frames are already gain-corrected, if no, you can provide a gain to apply", "yes");

	if (movie_is_gain_corrected == false)
	{
		gain_filename = my_input->GetFilenameFromUser("Gain image filename", "The filename of the camera's gain reference image", "my_gain_reference.dm4", true);
	}
	else
	{
		gain_filename = "";
	}

	resample_output = my_input->GetYesNoFromUser("Resample the output?", "If yes, the image will be resampled using Fourier cropping to the desired size", "no");

	if (resample_output == true)
	{
		new_x_size = my_input->GetIntFromUser("New X-Size", "The desired X size after resampling", "3838", 1);
		new_y_size = my_input->GetIntFromUser("New Y-Size", "The desired Y size after resampling", "3708", 1);
	}
	else
	{
		new_x_size = 1;
		new_y_size = 1;
	}

	delete my_input;

	 my_current_job.Reset(10);
	 my_current_job.ManualSetArguments("ttfffbtbii", input_filename.c_str(),
												output_filename.c_str(),
												mag_distortion_angle,
												mag_distortion_major_scale,
												mag_distortion_minor_scale,
												movie_is_gain_corrected,
												gain_filename.c_str(),
												resample_output,
												new_x_size,
												new_y_size);


}

// overide the do calculation method which will be what is actually run..

bool MagDistortionCorrectApp::DoCalculation()
{

	// get the arguments for this job..

	std::string input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
    float       mag_distortion_angle				= my_current_job.arguments[2].ReturnFloatArgument();
    float       mag_distortion_major_scale          = my_current_job.arguments[3].ReturnFloatArgument();
	float       mag_distortion_minor_scale          = my_current_job.arguments[4].ReturnFloatArgument();
	bool 		movie_is_gain_corrected             = my_current_job.arguments[5].ReturnBoolArgument();
	std::string gain_filename						= my_current_job.arguments[6].ReturnStringArgument();
	bool        resample_output                     = my_current_job.arguments[7].ReturnBoolArgument();
	int         new_x_size                          = my_current_job.arguments[8].ReturnIntegerArgument();
	int         new_y_size                          = my_current_job.arguments[9].ReturnIntegerArgument();

	// The Files

	ImageFile input_file(input_filename, false);
	MRCFile output_file(output_filename, true);
	ImageFile gain_file;

	if (! movie_is_gain_corrected) gain_file.OpenFile(gain_filename, false);
	long number_of_input_images = input_file.ReturnNumberOfSlices();
	long slice_byte_size;

	long image_counter;

	Image gain_image;
	Image input_image;

	// Read in gain reference
	if (!movie_is_gain_corrected) { gain_image.ReadSlice(&gain_file,1);	}

	// Read in, gain-correct, correct-distortion and resample all the images..

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		// Read from disk
		input_image.ReadSlice(&input_file,image_counter+1);

		// Gain correction
		if (! movie_is_gain_corrected)
		{
			if (! input_image.HasSameDimensionsAs(&gain_image))
			{
				SendError(wxString::Format("Error: location %i of input file does not have same dimensions as the gain image",image_counter+1));
				ExitMainLoop();
			}
			//if (image_counter == 0) SendInfo(wxString::Format("Info: multiplying %s by gain %s\n",input_filename,gain_filename.ToStdString()));
			input_image.MultiplyPixelWise(gain_image);
		}

		input_image.CorrectMagnificationDistortion(mag_distortion_angle, mag_distortion_major_scale, mag_distortion_minor_scale);

		if (resample_output == true)
		{
			input_image.ForwardFFT();
			input_image.Resize(new_x_size, new_y_size, 1);
			input_image.BackwardFFT();
		}

		input_image.WriteSlice(&output_file, image_counter + 1);
	}


	return true;
}
