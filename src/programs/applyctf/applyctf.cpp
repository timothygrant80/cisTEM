#include "../../core/core_headers.h"

class
ApplyCTFApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(ApplyCTFApp)

// override the DoInteractiveUserInput

void ApplyCTFApp::DoInteractiveUserInput()
{
	std::string input_filename;
	std::string output_filename;
	float pixel_size;
	float acceleration_voltage;
	float spherical_aberration;
	float amplitude_contrast;
	float defocus_1;
	float defocus_2;
	float astigmatism_angle;
	float additional_phase_shift;


	bool set_expert_options;

	UserInput *my_input = new UserInput("ApplyCTF", 1.0);

	input_filename = my_input->GetFilenameFromUser("Input image filename", "The input file, containing one or more images in a stack", "input.mrc", true );
	output_filename = my_input->GetFilenameFromUser("Output filename", "The output file", "output.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (keV)", "Acceleration voltage, in keV", "300.0", 0.0,500.0);
	spherical_aberration = my_input->GetFloatFromUser("Spherical aberration (mm)","Objective lens spherical aberration","2.7",0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast","Fraction of total contrast attributed to amplitude contrast","0.07",0.0);
	defocus_1 = my_input->GetFloatFromUser("Underfocus 1 (A)","In Angstroms, the objective lens underfocus along the first axis","1.2");
	defocus_2 = my_input->GetFloatFromUser("Underfocus 2 (A)","In Angstroms, the objective lens underfocus along the second axis","1.2");
	astigmatism_angle = my_input->GetFloatFromUser("Astigmatism angle","Angle between the first axis and the x axis of the image","0.0");
	additional_phase_shift = my_input->GetFloatFromUser("Additional phase shift (rad)","Additional phase shift relative to undiffracted beam, as introduced for example by a phase plate","0.0");

	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("ttffffffff",     	 input_filename.c_str(),
															 output_filename.c_str(),
															 pixel_size,
															 acceleration_voltage,
															 spherical_aberration,
															 amplitude_contrast,
															 defocus_1,
															 defocus_2,
															 astigmatism_angle,
															 additional_phase_shift);


}

// override the do calculation method which will be what is actually run..

bool ApplyCTFApp::DoCalculation()
{
	long image_counter;

	Image current_image;
	CTF current_ctf;

	// get the arguments for this job..

	std::string input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	float       pixel_size					        = my_current_job.arguments[2].ReturnFloatArgument();
	float 		acceleration_voltage				= my_current_job.arguments[3].ReturnFloatArgument();
	float 		spherical_aberration				= my_current_job.arguments[4].ReturnFloatArgument();
	float 		amplitude_contrast   				= my_current_job.arguments[5].ReturnFloatArgument();
	float		defocus_1							= my_current_job.arguments[6].ReturnFloatArgument();
	float		defocus_2							= my_current_job.arguments[7].ReturnFloatArgument();
	float		astigmatism_angle					= my_current_job.arguments[8].ReturnFloatArgument();
	float		additional_phase_shift				= my_current_job.arguments[9].ReturnFloatArgument();


	//my_current_job.PrintAllArguments();

	// The Files
	MRCFile input_file(input_filename, false);
	MRCFile output_file(output_filename, true);
	long number_of_input_images = input_file.ReturnNumberOfSlices();

	// CTF object
	current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,0.0,0.5,0.0,pixel_size,additional_phase_shift);

	// Loop over input images
	for ( image_counter = 0 ; image_counter < number_of_input_images; image_counter++)
	{
		current_image.ReadSlice(&input_file,image_counter + 1 );
		current_image.ForwardFFT();
		current_image.ApplyCTF(current_ctf);
		current_image.BackwardFFT();
		current_image.WriteSlice(&output_file,image_counter + 1 );
	}

	return true;
}




